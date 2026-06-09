import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class RCPRConsensusEstimator(nn.Module):
    """
    ACRE-only robust consensus prototype estimator for CoSOD Stage-2.
    Inputs:
      feats: [B, HW, C]
      w0:    [B, HW] in [0, 1]
    Outputs:
      proto: [C]
      agree: [B, HW] in [0, 1]
      w:     [B, HW] normalized weights
    """

    def __init__(self, iters=2, beta=0.6, gamma=2.0, eps=1e-6):
        super().__init__()
        self.iters = iters
        self.beta = beta
        self.gamma = gamma
        self.eps = eps
        # TopK v2: inference-only enhancement, default off.
        self.topk_ratio = 0.0
        self.topk_mode = 'legacy'
        self.topk_res_alpha = 0.20
        self.topk_conf_gate = 0.58
        self.topk_mass_min = 0.55
        self.topk_delta_th = 0.035
        self._printed_topk_shape = False
        self._printed_topk_once = False
        self._printed_topk_fallback = False
        self._printed_topk_rtg_once = False

    @staticmethod
    def _cos_sim_tokens_to_proto(feats, proto, eps=1e-6):
        feats_n = F.normalize(feats, dim=-1, eps=eps)
        proto_n = F.normalize(proto, dim=-1, eps=eps)
        return (feats_n * proto_n.view(1, 1, -1)).sum(dim=-1)

    def _apply_topk_on_weights(self, w, ratio):
        """
        Apply row-wise TopK filtering on w.
        Inputs:
          w: [B, HW] normalized weights.
          ratio: scalar in [0, 1].
        Returns:
          w_topk: [B, HW], row-normalized.
        """
        eps = self.eps
        # Clamp ratio into [0, 1] to avoid accidental invalid inputs.
        ratio = 0.0 if ratio is None else float(ratio)
        ratio = max(0.0, min(1.0, ratio))
        if ratio <= 0.0:
            return w

        if w.dim() != 2:
            raise ValueError(f"TopK expects w as [B,HW], got {tuple(w.shape)}")

        bsz, hw = w.shape
        k = int(math.ceil(float(ratio) * float(hw)))
        k = max(1, min(k, hw))

        if not self._printed_topk_once:
            print(f"[Stage2TopK] enabled=1 ratio={float(ratio):.3f} hw={hw} k={k}")
            self._printed_topk_once = True

        if k >= hw:
            return w

        _, idx = torch.topk(w, k, dim=1, largest=True, sorted=False)
        mask = torch.zeros_like(w)
        mask.scatter_(1, idx, 1.0)
        w_masked = w * mask

        denom = w_masked.sum(dim=1, keepdim=True)
        w_norm = w / (w.sum(dim=1, keepdim=True) + eps)
        need_fallback = denom <= eps

        if bool(need_fallback.any().item()) and (not self._printed_topk_fallback):
            fb_rows = int(need_fallback.sum().item())
            print(f"[TopKFallback] rows={fb_rows}/{bsz} denom<=eps -> fallback to original normalized w")
            self._printed_topk_fallback = True

        w_topk = torch.where(need_fallback, w_norm, w_masked / (denom + eps))
        return w_topk

    def _compute_proto_per_group(self, feats, w):
        eps = self.eps
        denom = w.sum(dim=1, keepdim=True).clamp(min=eps)
        return (w.unsqueeze(-1) * feats).sum(dim=1) / denom

    def _apply_topk_rtg(self, feats, w, ratio, conf_vec=None):
        """
        RTG-TopK (risk-controlled TopK):
          1) TopK on each image/group row
          2) Residual convex blending with full weights
          3) Per-image gating by confidence, retained mass and proto drift
        """
        eps = self.eps
        ratio = 0.0 if ratio is None else float(ratio)
        ratio = max(0.0, min(1.0, ratio))
        if ratio <= 0.0:
            return w, None

        if w.dim() != 2:
            raise ValueError(f"TopK expects w as [B,HW], got {tuple(w.shape)}")

        bsz, hw = w.shape
        k = int(math.ceil(float(ratio) * float(hw)))
        k = max(1, min(k, hw))
        if not self._printed_topk_once:
            print(f"[Stage2TopK] enabled=1 mode=rtg ratio={float(ratio):.3f} hw={hw} k={k}")
            self._printed_topk_once = True

        w_norm = w / (w.sum(dim=1, keepdim=True) + eps)
        if k >= hw:
            delta = torch.zeros((bsz,), device=w.device, dtype=w.dtype)
            stats = {
                'applied_rate': 0.0,
                'applied': 0,
                'total': int(bsz),
                'delta_mean': 0.0,
                'delta_p95': 0.0,
                'conf_invalid': int(bsz if conf_vec is None else 0),
                'mass_fail': 0,
                'delta_fail': 0,
            }
            return w_norm, stats

        _, idx = torch.topk(w_norm, k, dim=1, largest=True, sorted=False)
        mask = torch.zeros_like(w_norm)
        mask.scatter_(1, idx, 1.0)
        w_masked = w_norm * mask
        mass_keep = w_masked.sum(dim=1)

        denom = w_masked.sum(dim=1, keepdim=True)
        need_fallback = (denom <= eps).squeeze(1)
        if bool(need_fallback.any().item()) and (not self._printed_topk_fallback):
            fb_rows = int(need_fallback.sum().item())
            print(f"[TopKFallback] rows={fb_rows}/{bsz} denom<=eps -> fallback to original normalized w")
            self._printed_topk_fallback = True

        w_topk = torch.where(need_fallback.unsqueeze(1), w_norm, w_masked / (denom + eps))
        alpha = max(0.0, min(1.0, float(self.topk_res_alpha)))
        # Residual convex blend to avoid abrupt hard-truncation side effects.
        w_res = (1.0 - alpha) * w_topk + alpha * w_norm
        w_res = w_res / (w_res.sum(dim=1, keepdim=True) + eps)

        p_full = self._compute_proto_per_group(feats, w_norm)
        p_topk = self._compute_proto_per_group(feats, w_topk)
        delta = (1.0 - F.cosine_similarity(p_full, p_topk, dim=1, eps=eps)).clamp(min=0.0, max=2.0)

        if conf_vec is None:
            conf = torch.ones((bsz,), device=w.device, dtype=w.dtype)
            conf_invalid = torch.ones((bsz,), device=w.device, dtype=torch.bool)
        else:
            conf = conf_vec.reshape(-1).to(device=w.device, dtype=w.dtype)
            if conf.numel() != bsz:
                conf = torch.ones((bsz,), device=w.device, dtype=w.dtype)
                conf_invalid = torch.ones((bsz,), device=w.device, dtype=torch.bool)
            else:
                conf_invalid = torch.isnan(conf) | torch.isinf(conf)
                conf = torch.where(conf_invalid, torch.ones_like(conf), conf)

        conf_gate = max(0.0, min(1.0, float(self.topk_conf_gate)))
        mass_min = max(0.0, min(1.0, float(self.topk_mass_min)))
        delta_th = max(0.0, float(self.topk_delta_th))
        conf_pass = conf < conf_gate
        mass_pass = mass_keep >= mass_min
        delta_pass = delta <= delta_th
        use_topk = conf_pass & mass_pass & delta_pass & (~need_fallback)

        w_final = torch.where(use_topk.unsqueeze(1), w_res, w_norm)
        w_final = w_final / (w_final.sum(dim=1, keepdim=True) + eps)

        if bsz > 0:
            delta_sorted, _ = torch.sort(delta.detach())
            p95_idx = int(math.ceil(0.95 * bsz)) - 1
            p95_idx = max(0, min(p95_idx, bsz - 1))
            delta_p95 = float(delta_sorted[p95_idx].item())
            delta_mean = float(delta.mean().item())
        else:
            delta_p95 = 0.0
            delta_mean = 0.0

        stats = {
            'applied_rate': float(use_topk.float().mean().item()) if bsz > 0 else 0.0,
            'applied': int(use_topk.sum().item()),
            'total': int(bsz),
            'delta_mean': delta_mean,
            'delta_p95': delta_p95,
            'conf_invalid': int(conf_invalid.sum().item()),
            'mass_fail': int((~mass_pass).sum().item()),
            'delta_fail': int((~delta_pass).sum().item()),
        }
        return w_final, stats

    def forward(self, feats, w0, apply_topk=False, conf_vec=None):
        eps = self.eps
        w0 = w0.clamp(min=0.0)
        bsz, hw, _ = feats.shape
        w0_sum = w0.sum(dim=1, keepdim=True)
        fallback_mask = (w0_sum < eps)  # [B,1], per-group fallback
        uniform_w = torch.full_like(w0, 1.0 / max(hw, 1))
        w = w0 / (w0_sum + eps)
        w = torch.where(fallback_mask, uniform_w, w)

        if apply_topk and (not self._printed_topk_shape):
            print(f"[Stage2TopKShape] feats_shape={tuple(feats.shape)} w0_shape={tuple(w0.shape)}")
            self._printed_topk_shape = True
        if apply_topk:
            if w.shape != w0.shape:
                raise ValueError(f"TopK expects w.shape==w0.shape, got {tuple(w.shape)} vs {tuple(w0.shape)}")

        proto = feats.mean(dim=(0, 1))
        agree = torch.full_like(w, 0.5)
        for _ in range(max(1, self.iters)):
            if bool(fallback_mask.squeeze(1).all().item()):
                # When every group is empty, use global mean proto for stability.
                proto = feats.mean(dim=(0, 1))
            else:
                denom = w.sum(dim=(0, 1)) + eps
                proto = (w.unsqueeze(-1) * feats).sum(dim=(0, 1)) / denom
            cos_raw = self._cos_sim_tokens_to_proto(feats, proto, eps=eps)
            agree = ((cos_raw + 1.0) * 0.5).clamp(0.0, 1.0)
            agree = torch.where(fallback_mask, torch.full_like(agree, 0.5), agree)
            w_new = w0 * (agree ** self.gamma)
            w_new = w_new / (w_new.sum(dim=1, keepdim=True) + eps)
            w_new = torch.where(fallback_mask, uniform_w, w_new)
            w = (1.0 - self.beta) * w + self.beta * w_new
            w = w / (w.sum(dim=1, keepdim=True) + eps)
            w = torch.where(fallback_mask, uniform_w, w)

        # TopK is inference-only and only used in ACRE branch caller.
        if apply_topk and (self.topk_ratio > 0.0):
            topk_mode = str(getattr(self, 'topk_mode', 'legacy')).lower()
            if topk_mode == 'rtg':
                w, rtg_stats = self._apply_topk_rtg(feats, w, self.topk_ratio, conf_vec=conf_vec)
                if (rtg_stats is not None) and (not self._printed_topk_rtg_once):
                    print(
                        f"[TOPK_RTG] applied_rate={rtg_stats['applied_rate']:.3f} "
                        f"applied={rtg_stats['applied']}/{rtg_stats['total']} "
                        f"delta_mean={rtg_stats['delta_mean']:.4f} delta_p95={rtg_stats['delta_p95']:.4f} "
                        f"conf_invalid={rtg_stats['conf_invalid']} "
                        f"mass_fail={rtg_stats['mass_fail']} delta_fail={rtg_stats['delta_fail']}"
                    )
                    self._printed_topk_rtg_once = True
            else:
                w = self._apply_topk_on_weights(w, self.topk_ratio)
            denom = w.sum(dim=(0, 1)) + eps
            proto = (w.unsqueeze(-1) * feats).sum(dim=(0, 1)) / denom

        return proto, agree, w
