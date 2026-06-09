import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.utils as vutils
from torchvision import models
import os
import sys
import numpy as np
from matplotlib import pyplot as plt
import time
import argparse
from tqdm import tqdm
import glob
import pydensecrf.densecrf as dcrf
import scipy.io as sio
from skimage import measure
from sklearn.feature_extraction import image
import warnings
warnings.filterwarnings("ignore")

from utils import *
from loss import *
from rcpr import RCPRConsensusEstimator

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

device = torch.device("cuda")

    
class Encoder_Attentioner(nn.Module):  

    def __init__(self,input_channels=512):
        super().__init__()
        self.scale = 1.0 / (input_channels ** 0.5)
        self.conv = nn.Conv2d(input_channels, input_channels, kernel_size=1, stride=1, padding=0)
        
        self.query_transform = nn.Conv2d(input_channels, input_channels, kernel_size=1, stride=1, padding=0) 
        self.key_transform = nn.Conv2d(input_channels, input_channels, kernel_size=1, stride=1, padding=0)
        
        self.sig = nn.Sigmoid()
        
    def forward(self, x):
        x = torch.permute(x, (0, 3, 1, 2))
        x = self.conv(x)+x
        B, C, H5, W5 = x.size()
        
        x_query = self.query_transform(x).view(B, C, -1)

        x_query = torch.transpose(x_query, 1, 2).contiguous().view(-1, C)  # BHW, C

        x_key = self.key_transform(x).view(B, C, -1)
        x_key = torch.transpose(x_key, 0, 1).contiguous().view(C, -1)  # C, BHW

        x_w1 = torch.matmul(x_query, x_key) * self.scale # BHW, BHW
        
        x_w = x_w1.view(B, H5 * W5, B * H5 * W5)
        
        for i in range(B):
            rep = torch.mean(x_w[i,:,:],1)
            rep = ((rep*B*H5*W5)-torch.sum(x_w[i,:,:][:,i*H5*W5:(i+1)*H5*W5],1))/((B-1)*H5*W5)
            rep = ((rep-torch.min(rep))/(torch.max(rep)-torch.min(rep))).unsqueeze(0)
            
            thresh = 0.65
            rep = self.sig((rep-thresh)/0.15)
            
            if i == 0:
                var = rep
            else:
                var = torch.cat((var,rep),0)

        return var



model_dino = dino_desc()

class SCoSPARC(nn.Module):

    def __init__(self, mode='train'):
        super(SCoSPARC, self).__init__()
        self.device = device
        self.patch_size2 = 8
        self.num_patches2 = int(224/self.patch_size2)
        self.encoder_attn = Encoder_Attentioner(768).cuda()
        # Baseline (Fixed) vs Ours (ACRE-only) switch for Stage-2 prototype.
        self.stage2_use_acre = False
        # Unified Stage-2 similarity threshold in sim01 space.
        self.stage2_tau_sim01 = 0.85
        self.stage2_tau2_delta = 0.0
        self.stage2_tau2_mode = 'fixed'
        self.stage2_tau2_margin_eps = 0.02
        self.stage2_tau2_target_near = 0.08
        self.stage2_tau2_k = 0.5
        self.stage2_tau2_proto_gate = 1
        self.stage2_tau2_proto_th = 0.001
        self.stage2_tau2_delta_min = 0.005
        self.stage2_tau2_delta_max = 0.03
        # Compatibility switch: keep baseline(mean) close to original paper behavior.
        self.stage2_baseline_legacy = 1
        self.stage2_baseline_box_sim = 0.77
        # RPF rounds in inference: 1 (default, ACRE-only) or 2 (feedback refinement).
        self.stage2_rpf_rounds = 1
        self.stage2_rpf_soft_lambda = 0.0
        self._printed_rpf_once = False
        self._printed_rpf_diag_once = False
        self._printed_rpf_diag_v33_once = False
        self._printed_rpf_diag_v34_once = False
        self._printed_rpf_diag_v35_once = False
        self._rpf_diag_v35_groups = 0
        self._rpf_diag_v35_hit_min = 0
        self._rpf_diag_v35_hit_max = 0
        self._rpf_diag_v35_gate_hits = 0
        # ACRE-only consensus estimator (TopK/RPF/etc. are intentionally not used here).
        self.rcpr = RCPRConsensusEstimator(iters=2, beta=0.6, gamma=2.0).cuda()
  
    def forward(self,x,paths,mode,idx,epoch,cut_off_epoch,dataset):
        
        th0 = 0.505
        alpha_c = 1.0
        bm_bar = 0.48
        th_val = 0.15

        cos_dist = torch.nn.CosineSimilarity(dim=0)
        
        self_attn_maps, patch_toks_group = self_attention_module2(x,self.patch_size2,model_dino)

        patch_toks2 = patch_toks_group.reshape(len(x),self.num_patches2,self.num_patches2,768) #512
        self_attn_maps_reshaped = F.interpolate(self_attn_maps.unsqueeze(1), [self.num_patches2, self.num_patches2], mode='bilinear', align_corners=True)
        self_attn_maps2 = self_attn_maps_reshaped.reshape(len(self_attn_maps_reshaped),self.num_patches2*self.num_patches2)
        
        cross_attn_weights = self.encoder_attn(patch_toks2)
        
        cross_attn_weights_reshaped = cross_attn_weights.reshape(len(self_attn_maps),self.num_patches2,self.num_patches2).unsqueeze(1)
        cross_attn_weights = F.interpolate(cross_attn_weights_reshaped, [224, 224], mode='bilinear', align_corners=True)

        self_attn_maps2 = F.interpolate(self_attn_maps_reshaped, [224, 224], mode='bilinear', align_corners=True)


        fg_wts = cross_attn_weights_reshaped.reshape(x.size()[0],self.num_patches2*self.num_patches2).unsqueeze(1) 
        
        caw = cross_attn_weights.clone()

        preds_fin = cross_attn_weights
        preds_fin_noncrf = preds_fin.clone()
        
        crossattwts = preds_fin.clone()
        
        pat_tok = torch.reshape(patch_toks2,(len(patch_toks2),28*28,768))
        
 
        preds_fin_round2_crf = preds_fin.clone()
        
        list1 = []
        avg_conf_vals = []
        conf_vec_topk = []
        for j in range(len(patch_toks2)):
            th_map = preds_fin[j].clone()
            th_map = F.interpolate(th_map.unsqueeze(0), [28, 28], mode='bilinear', align_corners=True)
            th_map = th_map.reshape(-1)
            valid_mask = th_map >= th_val
            if bool(valid_mask.any().item()):
                avg_conf = th_map[valid_mask].mean()
                # Valid confidence participates in TopK gating.
                conf_vec_topk.append(avg_conf)
            else:
                # Empty-support confidence is treated as high-confidence (no TopK trigger).
                avg_conf = torch.tensor(1.0, device=th_map.device, dtype=th_map.dtype)
                # Keep NaN for audit; RCPR will convert invalid conf to conservative high confidence.
                conf_vec_topk.append(torch.tensor(float('nan'), device=th_map.device, dtype=th_map.dtype))
            avg_conf_vals.append(avg_conf)

        if len(avg_conf_vals) > 0:
            avg_tot_conf = float(torch.stack(avg_conf_vals).mean().detach().cpu().item())
            conf_vec_topk = torch.stack(conf_vec_topk).detach()
        else:
            avg_tot_conf = 1.0
            conf_vec_topk = torch.empty((0,), device=pat_tok.device, dtype=pat_tok.dtype)

        sel_th = th0 + alpha_c*((1-avg_tot_conf) - bm_bar)  #Adaptive thresholding
        
        best_threshs = []
        for j in range(len(patch_toks2)): 
            preds_fin[j][preds_fin[j] >= sel_th] = 1
            preds_fin[j][preds_fin[j] < sel_th] = 0
            best_threshs.append(sel_th)
        
        #for testing
        fg_wts_masked = fg_wts.clone()
            
        for j in range(len(fg_wts)):
            fg_wts_masked[j][fg_wts_masked[j] >= best_threshs[j]] = 1
            fg_wts_masked[j][fg_wts_masked[j] < best_threshs[j]] = 0
        fg_embeds, _ = get_embeddings_mask(fg_wts_masked,patch_toks_group)

        # Stage-2 consensus prototype:
        # - Baseline (Fixed): mean(fg_embeds)
        # - Ours (ACRE-only): p* = ACRE(feats, w0)
        if self.stage2_use_acre:
            feats_tok = pat_tok
            w0 = fg_wts.squeeze(1) * fg_wts_masked.squeeze(1)
            avg_embeds, _, _ = self.rcpr(feats_tok, w0, apply_topk=(mode == 'test'), conf_vec=conf_vec_topk)
        else:
            avg_embeds = torch.mean(fg_embeds,0)
        
        fg_interim = preds_fin.clone()
        
        #for testing
        if mode == 'test':
            # Round1 cache for optional RPF round2.
            round1_labels = []
            round1_regions = []
            round1_masks = []
            round1_m1_flat = []
            round1_keep_flags = []
            use_rpf_round2 = self.stage2_use_acre and int(getattr(self, 'stage2_rpf_rounds', 1)) == 2
            lam = float(getattr(self, 'stage2_rpf_soft_lambda', 0.0))
            lam = max(0.0, min(1.0, lam))
            tau2_delta = max(0.0, float(getattr(self, 'stage2_tau2_delta', 0.0)))
            tau2_mode = str(getattr(self, 'stage2_tau2_mode', 'fixed')).lower()
            if tau2_mode not in ['fixed', 'adaptive']:
                tau2_mode = 'fixed'
            tau2_margin_eps = max(1e-6, float(getattr(self, 'stage2_tau2_margin_eps', 0.02)))
            tau2_target_near = max(1e-6, float(getattr(self, 'stage2_tau2_target_near', 0.08)))
            tau2_k = max(0.0, float(getattr(self, 'stage2_tau2_k', 0.5)))
            tau2_proto_gate = 1 if int(getattr(self, 'stage2_tau2_proto_gate', 1)) != 0 else 0
            tau2_proto_th = max(0.0, float(getattr(self, 'stage2_tau2_proto_th', 0.001)))
            tau2_delta_min = max(0.0, float(getattr(self, 'stage2_tau2_delta_min', 0.005)))
            tau2_delta_max = max(tau2_delta_min, float(getattr(self, 'stage2_tau2_delta_max', 0.03)))
            baseline_legacy = (not self.stage2_use_acre) and (int(getattr(self, 'stage2_baseline_legacy', 1)) == 1)
            baseline_box_sim = float(getattr(self, 'stage2_baseline_box_sim', 0.77))
            tau1 = float(self.stage2_tau_sim01)
            near_r1 = 0.0
            delta_eff = tau2_delta
            delta_raw = tau2_delta
            hit_min = 0
            hit_max = 0
            gate_hit = 0
            tau2 = tau1
            round1_m1_soft_flat = [] if (use_rpf_round2 and lam > 1e-12) else None
            round1_scores = []
            diag_num_regions0 = 0
            diag_num_countk0 = 0
            diag_same_as_m1 = 0
            diag_num_fallback_rows = 0
            diag_cross_boundary = 0
            diag_proto_delta = 0.0
            # v3.3b: per-group accumulated diagnostics (no algorithm change).
            diag_all_img_total = 0
            diag_all_same_as_m1_total = 0
            diag_all_regions_total = 0
            diag_all_cross_boundary_total = 0
            diag_all_proto_delta_list = []
            diag_all_margin_list = []

            for i in range(len(preds_fin)):
                blobs_labels = measure.label(preds_fin[i][0].detach().cpu().numpy(), background=0)
                list1 = []
                for j in range(1, len(np.unique(blobs_labels))):
                    blobs_mask = np.zeros_like(blobs_labels)
                    blobs_mask[blobs_labels == j] = 1
                    list_boxes = find_closest_box(torch.from_numpy(blobs_mask).cuda())
                    list1.append(list_boxes)

                scores_list = []
                for k in range(0, len(list1)):
                    blobs_mask = np.zeros_like(blobs_labels)
                    blobs_mask[blobs_labels == k+1] = 1
                    blobs_mask = cv2.resize(np.uint8(blobs_mask), (28, 28), interpolation=cv2.INTER_NEAREST)
                    blobs_mask = torch.from_numpy(blobs_mask).cuda()
                    masked_embs = blobs_mask.unsqueeze(2) * patch_toks2[i]
                    masked_embs = torch.reshape(masked_embs, (28 * 28, 768))
                    if baseline_legacy:
                        # Preserve original baseline scoring semantics.
                        part_mask_emb = torch.sum(masked_embs, 0) / torch.numel(masked_embs == 1)
                    else:
                        den = torch.tensor(float(blobs_mask.sum()), device=masked_embs.device).clamp(min=1.0)
                        part_mask_emb = torch.sum(masked_embs, 0) / den
                    part_mask_sim_raw = cos_dist(avg_embeds, part_mask_emb)
                    if baseline_legacy:
                        score_legacy = int(part_mask_sim_raw.detach().cpu().numpy() * 100)
                        scores_list.append(score_legacy)
                    else:
                        score_sim01 = ((part_mask_sim_raw + 1.0) * 0.5).detach().cpu().numpy().item()
                        scores_list.append(score_sim01)

                if len(scores_list) > 0:
                    scores_list = np.array(scores_list)
                    if baseline_legacy:
                        # Strict legacy semantics: direct normalization by max.
                        scores_eval = scores_list / np.max(scores_list)
                        keep_flags_r1 = (scores_eval >= baseline_box_sim)
                        round1_scores.append(scores_eval.copy())
                    else:
                        scores_eval = scores_list
                        keep_flags_r1 = (scores_eval >= self.stage2_tau_sim01)
                        round1_scores.append(scores_eval.copy())
                    count_k = 0
                    for k in range(0, len(list1)):
                        blobs_mask = np.zeros_like(blobs_labels)
                        blobs_mask[blobs_labels == k+1] = 1
                        if keep_flags_r1[k]:
                            if count_k == 0:
                                blobs_mask_comb_round1 = blobs_mask.copy()
                            else:
                                blobs_mask_comb_round1 += blobs_mask
                            count_k += 1
                    if count_k == 0:
                        if baseline_legacy and len(list1) > 0:
                            # Original baseline fallback: keep last discovered region when all filtered out.
                            blobs_mask_comb_round1 = (blobs_labels == len(list1)).astype(np.uint8)
                        else:
                            best_k = int(np.argmax(scores_eval))
                            blobs_mask_comb_round1 = (blobs_labels == (best_k + 1)).astype(np.uint8)
                    blobs_mask_comb_round1[blobs_mask_comb_round1 > 0] = 1
                else:
                    blobs_mask_comb_round1 = preds_fin_noncrf[i][0].detach().cpu().numpy()
                    blobs_mask_comb_round1 = (blobs_mask_comb_round1 > 0).astype(np.uint8)
                    keep_flags_r1 = np.zeros((0,), dtype=np.bool_)
                    round1_scores.append(np.array([], dtype=np.float32))

                round1_labels.append(blobs_labels)
                round1_regions.append(list1)
                round1_masks.append(blobs_mask_comb_round1.copy())
                round1_keep_flags.append(keep_flags_r1.copy())

                m1_28 = cv2.resize(np.uint8(blobs_mask_comb_round1 > 0), (28, 28), interpolation=cv2.INTER_NEAREST)
                m1_28 = (m1_28 > 0).astype(np.float32)
                round1_m1_flat.append(torch.from_numpy(m1_28.reshape(-1)).to(pat_tok.device))
                if use_rpf_round2 and lam > 1e-12:
                    m1_soft_28 = cv2.resize(preds_fin_noncrf[i][0].detach().cpu().numpy().astype(np.float32), (28, 28), interpolation=cv2.INTER_NEAREST)
                    if float(np.max(m1_soft_28)) > 1.5:
                        m1_soft_28 = m1_soft_28 / 255.0
                    m1_soft_28 = np.clip(m1_soft_28, 0.0, 1.0)
                    round1_m1_soft_flat.append(torch.from_numpy(m1_soft_28.reshape(-1)).to(pat_tok.device))

            # Round2 group-level prototype: p2 = ACRE(F, w1), where w1 is built from all M1.
            if use_rpf_round2:
                valid_scores = [arr for arr in round1_scores if arr.size > 0]
                if len(valid_scores) > 0:
                    all_scores_r1 = np.concatenate(valid_scores, axis=0)
                    near_r1 = float(np.mean(np.abs(all_scores_r1 - tau1) < tau2_margin_eps))
                else:
                    near_r1 = 0.0

                if tau2_mode == 'adaptive' and tau2_delta > 0.0:
                    # v3.5 step1: additive adjustment to avoid near_r1->0 amplification.
                    delta_raw = tau2_delta + tau2_k * (tau2_target_near - near_r1)
                    if delta_raw <= tau2_delta_min:
                        delta_eff = tau2_delta_min
                        hit_min = 1
                        hit_max = 0
                    elif delta_raw >= tau2_delta_max:
                        delta_eff = tau2_delta_max
                        hit_min = 0
                        hit_max = 1
                    else:
                        delta_eff = delta_raw
                        hit_min = 0
                        hit_max = 0
                else:
                    delta_eff = tau2_delta
                    delta_raw = tau2_delta
                    hit_min = 0
                    hit_max = 0
                tau2 = max(0.0, min(1.0, tau1 - delta_eff))

                m1_flat_batch = torch.stack(round1_m1_flat, dim=0)
                # v3.3 hard/soft gating with strict hard fallback when lambda==0.
                if lam <= 1e-12:
                    m_mix = m1_flat_batch
                else:
                    m1_soft_batch = torch.stack(round1_m1_soft_flat, dim=0)
                    m_mix = (1.0 - lam) * m1_flat_batch + lam * m1_soft_batch
                w1 = w0 * m_mix
                p2, _, _ = self.rcpr(pat_tok, w1, apply_topk=(mode == 'test'), conf_vec=conf_vec_topk)
                diag_proto_delta = float((1.0 - cos_dist(avg_embeds, p2)).detach().cpu().item())
                diag_all_proto_delta_list.append(diag_proto_delta)
                if tau2_mode == 'adaptive' and tau2_proto_gate == 1 and diag_proto_delta < tau2_proto_th:
                    delta_eff = tau2_delta
                    tau2 = max(0.0, min(1.0, tau1 - delta_eff))
                    # Gate fallback means no adaptive clamp hit in final decision.
                    hit_min = 0
                    hit_max = 0
                    gate_hit = 1

                if not self._printed_rpf_once:
                    w1_sums = w1.sum(dim=1)
                    num_fb = int((w1_sums < float(self.rcpr.eps)).sum().item())
                    diag_num_fallback_rows = num_fb
                    region_count = int(sum(len(r) for r in round1_regions))
                    print(f"[RPF] reuse_labels=1 regions={region_count}")
                    print(f"[RPF] w1_sum_min={float(w1_sums.min().item()):.6f} w1_sum_max={float(w1_sums.max().item()):.6f} w1_sum_mean={float(w1_sums.mean().item()):.6f} num_fallback_rows={num_fb}")
                    self._printed_rpf_once = True
                else:
                    w1_sums = w1.sum(dim=1)
                    diag_num_fallback_rows = int((w1_sums < float(self.rcpr.eps)).sum().item())
            else:
                p2 = None

            # Final mask assembly: round1 direct output, or round2 re-score with reused labels.
            for i in range(len(preds_fin)):
                blobs_labels = round1_labels[i]
                list1 = round1_regions[i]
                blobs_mask_comb = round1_masks[i].copy()

                if use_rpf_round2:
                    scores_list_r2 = []
                    per_img_cross_boundary = 0
                    for k in range(0, len(list1)):
                        blobs_mask = np.zeros_like(blobs_labels)
                        blobs_mask[blobs_labels == k+1] = 1
                        blobs_mask_28 = cv2.resize(np.uint8(blobs_mask), (28, 28), interpolation=cv2.INTER_NEAREST)
                        blobs_mask_28 = torch.from_numpy(blobs_mask_28).to(patch_toks2[i].device)
                        masked_embs = blobs_mask_28.unsqueeze(2) * patch_toks2[i]
                        masked_embs = torch.reshape(masked_embs, (28 * 28, 768))
                        den = torch.tensor(float(blobs_mask_28.sum()), device=masked_embs.device).clamp(min=1.0)
                        part_mask_emb = torch.sum(masked_embs, 0) / den
                        part_mask_sim_raw = cos_dist(p2, part_mask_emb)
                        score_sim01 = ((part_mask_sim_raw + 1.0) * 0.5).detach().cpu().numpy().item()
                        scores_list_r2.append(score_sim01)

                    # Fallback level-1: empty region set or empty score list -> fallback M1.
                    if len(scores_list_r2) == 0 or len(list1) == 0:
                        diag_num_regions0 += 1
                        blobs_mask_comb = round1_masks[i].copy()
                    else:
                        scores_list_r2 = np.array(scores_list_r2)
                        diag_all_margin_list.extend(np.abs(scores_list_r2 - tau2).tolist())
                        keep_flags_r2 = (scores_list_r2 >= tau2)
                        keep_flags_r1 = round1_keep_flags[i]
                        n_cmp = min(len(keep_flags_r1), len(keep_flags_r2))
                        if n_cmp > 0:
                            per_img_cross_boundary = int(np.logical_xor(keep_flags_r1[:n_cmp], keep_flags_r2[:n_cmp]).sum())
                            diag_cross_boundary += per_img_cross_boundary
                        count_k2 = 0
                        for k in range(0, len(list1)):
                            blobs_mask = np.zeros_like(blobs_labels)
                            blobs_mask[blobs_labels == k+1] = 1
                            if scores_list_r2[k] >= tau2:
                                if count_k2 == 0:
                                    blobs_mask_comb = blobs_mask.copy()
                                else:
                                    blobs_mask_comb += blobs_mask
                                count_k2 += 1

                        # Fallback level-2: all filtered -> fallback M1 first, then keep-max as last resort.
                        if count_k2 == 0:
                            diag_num_countk0 += 1
                            blobs_mask_comb = round1_masks[i].copy()
                            if np.sum(blobs_mask_comb) == 0 and len(scores_list_r2) > 0:
                                best_k2 = int(np.argmax(scores_list_r2))
                                blobs_mask_comb = (blobs_labels == (best_k2 + 1)).astype(np.uint8)

                        blobs_mask_comb[blobs_mask_comb > 0] = 1

                    # Fixed diagnostic comparison protocol:
                    # same size (224x224) + same binary (>0->1) + same dtype(uint8).
                    m2_cmp = cv2.resize(np.uint8(blobs_mask_comb > 0), (224, 224), interpolation=cv2.INTER_NEAREST).astype(np.uint8)
                    m1_cmp = cv2.resize(np.uint8(round1_masks[i] > 0), (224, 224), interpolation=cv2.INTER_NEAREST).astype(np.uint8)
                    is_same = np.array_equal(m2_cmp, m1_cmp)
                    if is_same:
                        diag_same_as_m1 += 1
                    # Per-group accumulated diagnostics.
                    diag_all_img_total += 1
                    diag_all_same_as_m1_total += int(is_same)
                    diag_all_regions_total += int(len(list1))
                    diag_all_cross_boundary_total += int(per_img_cross_boundary)

                if i == 0:
                    preds_fin_round2 = torch.from_numpy(blobs_mask_comb).cuda().unsqueeze(0)
                else:
                    preds_fin_round2 = torch.cat((preds_fin_round2, torch.from_numpy(blobs_mask_comb).cuda().unsqueeze(0)))

            if use_rpf_round2 and (not self._printed_rpf_diag_once):
                print(f"[RPF_DIAG] regions0={diag_num_regions0} countk0={diag_num_countk0} fallback_rows={diag_num_fallback_rows} same_as_M1={diag_same_as_m1} total_imgs={len(preds_fin)}")
                self._printed_rpf_diag_once = True
            if use_rpf_round2:
                proto_arr = np.array(diag_all_proto_delta_list, dtype=np.float32)
                margin_arr = np.array(diag_all_margin_list, dtype=np.float32)
                proto_mean = float(proto_arr.mean()) if proto_arr.size > 0 else 0.0
                proto_p50 = float(np.percentile(proto_arr, 50)) if proto_arr.size > 0 else 0.0
                proto_p90 = float(np.percentile(proto_arr, 90)) if proto_arr.size > 0 else 0.0
                proto_max = float(proto_arr.max()) if proto_arr.size > 0 else 0.0
                margin_min = float(margin_arr.min()) if margin_arr.size > 0 else 0.0
                margin_p10 = float(np.percentile(margin_arr, 10)) if margin_arr.size > 0 else 0.0
                near_tau_frac = float((margin_arr < 0.02).mean()) if margin_arr.size > 0 else 0.0
                same_ratio = float(diag_all_same_as_m1_total) / float(max(1, diag_all_img_total))
                cross_rate = float(diag_all_cross_boundary_total) / float(max(1, diag_all_regions_total))
                print(
                    f"[RPF_DIAG_ALL] imgs={diag_all_img_total} same_ratio={same_ratio:.6f} "
                    f"regions={diag_all_regions_total} cross_rate={cross_rate:.6f} "
                    f"proto_mean={proto_mean:.6f} proto_p50={proto_p50:.6f} proto_p90={proto_p90:.6f} proto_max={proto_max:.6f} "
                    f"margin_min={margin_min:.6f} margin_p10={margin_p10:.6f} near_tau@0.02={near_tau_frac:.6f}"
                )
            if use_rpf_round2 and (not self._printed_rpf_diag_v34_once):
                print(f"[RPF_DIAG_V34] tau1={tau1:.3f} tau2={tau2:.3f} delta={delta_eff:.3f}")
                self._printed_rpf_diag_v34_once = True
            if use_rpf_round2 and tau2_mode == 'adaptive' and (not self._printed_rpf_diag_v35_once):
                print(
                    f"[RPF_DIAG_V35] mode={tau2_mode} k={tau2_k:.6f} near_r1={near_r1:.6f} "
                    f"delta_base={tau2_delta:.6f} delta_raw={delta_raw:.6f} delta_eff={delta_eff:.6f} "
                    f"tau1={tau1:.6f} tau2={tau2:.6f} hit_min={hit_min} hit_max={hit_max} "
                    f"proto_delta={diag_proto_delta:.6f} gate_on={tau2_proto_gate} "
                    f"gate_th={tau2_proto_th:.6f} gate_hit={gate_hit}"
                )
                self._printed_rpf_diag_v35_once = True
            if use_rpf_round2 and tau2_mode == 'adaptive':
                # Step1.1 audit-only counters (no decision logic changes).
                self._rpf_diag_v35_groups += 1
                self._rpf_diag_v35_hit_min += int(hit_min)
                self._rpf_diag_v35_hit_max += int(hit_max)
                self._rpf_diag_v35_gate_hits += int(gate_hit)
                hit_max_rate = float(self._rpf_diag_v35_hit_max) / float(max(1, self._rpf_diag_v35_groups))
                gate_rate = float(self._rpf_diag_v35_gate_hits) / float(max(1, self._rpf_diag_v35_groups))
                print(
                    f"[RPF_DIAG_V35_ALL] groups={self._rpf_diag_v35_groups} "
                    f"hit_min={self._rpf_diag_v35_hit_min} hit_max={self._rpf_diag_v35_hit_max} "
                    f"hit_max_rate={hit_max_rate:.6f} gate_hits={self._rpf_diag_v35_gate_hits} "
                    f"gate_rate={gate_rate:.6f}"
                )
            if use_rpf_round2 and lam > 1e-12 and (not self._printed_rpf_diag_v33_once):
                print(f"[RPF_DIAG_V33] lambda={lam:.3f} proto_delta={diag_proto_delta:.6f} cross_boundary={diag_cross_boundary}")
                self._printed_rpf_diag_v33_once = True
      
            try:
                preds_fin_round2_crf = apply_crf(preds_fin_round2.unsqueeze(1),paths,mode,dataset,'label')
            except:
                print('CRF exception')
                preds_fin_round2_crf = preds_fin_round2

            preds_fin_round3 = preds_fin.clone()
            
            return preds_fin,preds_fin_noncrf,caw,self_attn_maps2, preds_fin_round2_crf,preds_fin_round3,avg_tot_conf,sel_th, fg_interim
            
        else:
            if mode == 'train':
                fg_embeds, bg_embeds = get_embeddings(fg_wts,patch_toks_group)
                fg_sal = get_saliency(fg_wts,self_attn_maps_reshaped)
            
            return fg_embeds, bg_embeds, fg_sal
        
        
            
