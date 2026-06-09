# -*- coding: utf-8 -*-
"""
Evaluate CoSOD predictions with:
  - MAE   (lower is better)
  - Fmax  (max F-measure over thresholds, beta^2=0.3)
  - Emax  (max E-measure over thresholds)
  - S_alpha (S-measure / Structure-measure, Fan et al. ICCV 2017, alpha=0.5)

Defaults are aligned with this repo layout:
  preds: ./predictions/<model_folder>/<dataset>/**/<name>.png
  gts  : ./datasets/<dataset>/groundtruth/**/<name>.png
"""

import os
import argparse
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image

import time
from datetime import datetime

def _fmt_seconds(sec: float) -> str:
    """把秒格式化成 HH:MM:SS.mmm"""
    sec = float(sec)
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = sec % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"

# -----------------------------
# IO helpers
# -----------------------------
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def read_gray(path: str) -> np.ndarray:
    """Read grayscale image as float32 in [0, 1]."""
    im = Image.open(path).convert("L")
    arr = np.asarray(im, dtype=np.float32)
    return arr / 255.0


def list_images_recursive(root: str) -> List[str]:
    out = []
    for dp, _, fns in os.walk(root):
        for fn in fns:
            ext = os.path.splitext(fn)[1].lower()
            if ext in IMG_EXTS:
                out.append(os.path.join(dp, fn))
    return out


def norm_rel(path: str) -> str:
    """Normalize relative path for matching (case-insensitive, forward slashes)."""
    return path.replace("\\", "/").lower()


def build_gt_map(gt_dir: str) -> Dict[str, str]:
    """Map: normalized relative path -> absolute gt path."""
    gt_files = list_images_recursive(gt_dir)
    m = {}
    for p in gt_files:
        rel = norm_rel(os.path.relpath(p, gt_dir))
        m[rel] = p
    return m


def match_gt(pred_path: str, pred_dir: str, gt_map: Dict[str, str]) -> str:
    """
    Try match GT by relative path (with possible extension differences).
    Returns matched gt_path or "" if not found.
    """
    rel = norm_rel(os.path.relpath(pred_path, pred_dir))
    if rel in gt_map:
        return gt_map[rel]

    base, _ = os.path.splitext(rel)
    for ext in [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"]:
        key = base + ext
        if key in gt_map:
            return gt_map[key]

    # fallback: match by basename only (last resort)
    bn = os.path.basename(base)
    candidates = [k for k in gt_map.keys() if os.path.splitext(os.path.basename(k))[0] == bn]
    if len(candidates) == 1:
        return gt_map[candidates[0]]

    return ""


# -----------------------------
# Metrics: MAE / Fmax / Emax
# (dataset-level max over thresholds)
# -----------------------------
def minmax01(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64, copy=False)
    mn = x.min()
    mx = x.max()
    return (x - mn) / (mx - mn + 1e-8)


def pr_counts_over_thresholds(
    pred01: np.ndarray,
    gt_bin: np.ndarray,
    thresholds: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """
    Efficient TP/FP over all thresholds for one image.
    pred01, gt_bin are flattened/handled internally.
    Returns:
      tp[t], fp[t], total_fg, N
    """
    p = pred01.reshape(-1).astype(np.float64, copy=False)
    g = gt_bin.reshape(-1).astype(np.uint8, copy=False)

    N = p.size
    order = np.argsort(p, kind="mergesort")  # deterministic
    p_sorted = p[order]
    g_sorted = g[order]

    prefix_fg = np.cumsum(g_sorted, dtype=np.int64)
    total_fg = int(prefix_fg[-1]) if prefix_fg.size > 0 else 0

    idxs = np.searchsorted(p_sorted, thresholds, side="left")  # first >= th
    fg_before = np.zeros_like(idxs, dtype=np.int64)
    mask = idxs > 0
    fg_before[mask] = prefix_fg[idxs[mask] - 1]

    tp = total_fg - fg_before
    pred_pos = N - idxs
    fp = pred_pos - tp
    return tp.astype(np.float64), fp.astype(np.float64), total_fg, N


def f_measure_curve(tp: np.ndarray, fp: np.ndarray, total_fg: int, beta2: float = 0.3) -> np.ndarray:
    eps = 1e-20
    tp = tp.astype(np.float64, copy=False)
    fp = fp.astype(np.float64, copy=False)

    prec = tp / (tp + fp + eps)
    rec = tp / (total_fg + eps)
    f = (1.0 + beta2) * prec * rec / (beta2 * prec + rec + eps)
    f = np.nan_to_num(f, nan=0.0, posinf=0.0, neginf=0.0)
    return f


def e_measure_curve(tp: np.ndarray, fp: np.ndarray, total_fg: int, N: int) -> np.ndarray:
    """
    Compute E-measure per threshold for binary pred (thresholded) and binary gt,
    using 4-case constant values (no per-pixel loop).
    Mirrors the common enhanced-alignment E-measure used in saliency eval.
    """
    eps = 1e-20
    total_bg = N - total_fg

    fn = total_fg - tp
    tn = total_bg - fp
    pred_pos = tp + fp

    mu_p = pred_pos / (N + eps)
    mu_g = total_fg / (N + eps)  # constant w.r.t threshold (for this image)

    dp1 = 1.0 - mu_p
    dp0 = -mu_p
    dg1 = 1.0 - mu_g
    dg0 = -mu_g

    def align(a, b):
        return 2.0 * a * b / (a * a + b * b + eps)

    align_tp = align(dp1, dg1)
    align_fp = align(dp1, dg0)
    align_fn = align(dp0, dg1)
    align_tn = align(dp0, dg0)

    enh_tp = (align_tp + 1.0) ** 2 / 4.0
    enh_fp = (align_fp + 1.0) ** 2 / 4.0
    enh_fn = (align_fn + 1.0) ** 2 / 4.0
    enh_tn = (align_tn + 1.0) ** 2 / 4.0

    e = (tp * enh_tp + fp * enh_fp + fn * enh_fn + tn * enh_tn) / (N - 1.0 + eps)
    e = np.nan_to_num(e, nan=0.0, posinf=0.0, neginf=0.0)
    return e


# -----------------------------
# S-measure (Fan et al. 2017)
# -----------------------------
def ssim_fan(x: np.ndarray, y: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    x = x.astype(np.float64, copy=False)
    y = y.astype(np.float64, copy=False)

    ux = x.mean()
    uy = y.mean()
    vx = ((x - ux) ** 2).mean()
    vy = ((y - uy) ** 2).mean()
    vxy = ((x - ux) * (y - uy)).mean()

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    num = (2 * ux * uy + C1) * (2 * vxy + C2)
    den = (ux * ux + uy * uy + C1) * (vx + vy + C2)
    return float(num / (den + 1e-8))


def centroid(gt: np.ndarray) -> Tuple[int, int]:
    h, w = gt.shape
    if gt.sum() == 0:
        return w // 2, h // 2
    ys = np.arange(h, dtype=np.float64)[:, None]
    xs = np.arange(w, dtype=np.float64)[None, :]
    cx = int(np.round((xs * gt).sum() / (gt.sum() + 1e-8)))
    cy = int(np.round((ys * gt).sum() / (gt.sum() + 1e-8)))
    cx = int(np.clip(cx, 1, w - 1))
    cy = int(np.clip(cy, 1, h - 1))
    return cx, cy


def object_score(pred: np.ndarray, mask: np.ndarray) -> float:
    x = pred[mask > 0.5]
    if x.size == 0:
        return 0.0
    mu = float(x.mean())
    sigma = float(x.std())
    return float(2.0 * mu / (mu * mu + 1.0 + sigma + 1e-8))


def s_object(pred: np.ndarray, gt: np.ndarray) -> float:
    u = float(gt.mean())
    ofg = object_score(pred, gt)
    obg = object_score(1.0 - pred, 1.0 - gt)
    return u * ofg + (1.0 - u) * obg


def s_region(pred: np.ndarray, gt: np.ndarray) -> float:
    h, w = gt.shape
    cx, cy = centroid(gt)

    gt1, gt2, gt3, gt4 = gt[:cy, :cx], gt[:cy, cx:], gt[cy:, :cx], gt[cy:, cx:]
    pr1, pr2, pr3, pr4 = pred[:cy, :cx], pred[:cy, cx:], pred[cy:, :cx], pred[cy:, cx:]

    area = float(h * w)
    w1 = float(cx * cy) / area
    w2 = float((w - cx) * cy) / area
    w3 = float(cx * (h - cy)) / area
    w4 = float((w - cx) * (h - cy)) / area

    q1 = ssim_fan(pr1, gt1)
    q2 = ssim_fan(pr2, gt2)
    q3 = ssim_fan(pr3, gt3)
    q4 = ssim_fan(pr4, gt4)

    return w1 * q1 + w2 * q2 + w3 * q3 + w4 * q4


def s_measure(pred01: np.ndarray, gt01_bin: np.ndarray, alpha: float = 0.5) -> float:
    gt_mean = float(gt01_bin.mean())
    if gt_mean == 0.0:
        return float(1.0 - pred01.mean())
    if gt_mean == 1.0:
        return float(pred01.mean())

    so = s_object(pred01, gt01_bin)
    sr = s_region(pred01, gt01_bin)
    s = alpha * so + (1.0 - alpha) * sr
    # clip to [0,1] for numerical safety
    return float(np.clip(s, 0.0, 1.0))


# -----------------------------
# Dataset evaluation
# -----------------------------
def evaluate_dataset(pred_dir: str, gt_dir: str, thresholds: np.ndarray) -> Dict[str, float]:
    gt_map = build_gt_map(gt_dir)
    pred_files = list_images_recursive(pred_dir)

    mae_list = []
    s_list = []

    F_sum = None
    E_sum = None
    matched = 0
    skipped = 0

    for pf in pred_files:
        gf = match_gt(pf, pred_dir, gt_map)
        if not gf:
            skipped += 1
            continue

        pred = read_gray(pf)
        gt = read_gray(gf)
        gt_bin = (gt > 0.5).astype(np.float64)

        pred01 = minmax01(pred)
        # (gt is binary already, but keep as float64)
        gt01 = gt_bin

        # MAE
        mae_list.append(float(np.mean(np.abs(pred01 - gt01))))

        # F/E curves (efficient)
        tp, fp, total_fg, N = pr_counts_over_thresholds(pred01, gt01, thresholds)
        f_curve = f_measure_curve(tp, fp, total_fg, beta2=0.3)
        e_curve = e_measure_curve(tp, fp, total_fg, N)

        if F_sum is None:
            F_sum = f_curve.copy()
            E_sum = e_curve.copy()
        else:
            F_sum += f_curve
            E_sum += e_curve

        # S-measure
        s_list.append(s_measure(pred01, gt01, alpha=0.5))

        matched += 1

    if matched == 0:
        raise RuntimeError(f"No matched pairs found. pred_dir={pred_dir}, gt_dir={gt_dir}")

    F_mean = F_sum / matched
    E_mean = E_sum / matched

    return {
        "N": float(matched),
        "MAE": float(np.mean(mae_list)),
        "Fmax": float(np.max(F_mean)),
        "Emax": float(np.max(E_mean)),
        "Salpha": float(np.mean(s_list)),
        "Skipped": float(skipped),
        "PredFiles": float(len(pred_files)),
        "GtFiles": float(len(gt_map)),
    }


def main():
    t_all0 = time.perf_counter()
    start_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"========== EVAL START @ {start_ts} ==========", flush=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_root", default="./predictions", type=str, help="predictions root")
    parser.add_argument("--model_folder", default="checkpoints", type=str, help="subfolder under pred_root")
    parser.add_argument("--gt_root", default="./datasets", type=str, help="datasets root")
    parser.add_argument("--datasets", nargs="+", default=["CoCA", "Cosal2015", "CoSOD3k"], help="dataset names")
    args = parser.parse_args()

    thresholds = np.linspace(0.0, 1.0, 255, dtype=np.float64)

    per_ds_time = {}

    for ds in args.datasets:
        t0 = time.perf_counter()

        pred_dir = os.path.join(args.pred_root, args.model_folder, ds)
        gt_dir = os.path.join(args.gt_root, ds, "groundtruth")

        print("")
        print("==== {} ====".format(ds))
        print("PredDir : {}".format(pred_dir))
        print("GtDir   : {}".format(gt_dir))

        if not os.path.isdir(pred_dir):
            print("ERROR: pred_dir not found.")
            continue
        if not os.path.isdir(gt_dir):
            print("ERROR: gt_dir not found.")
            continue

        r = evaluate_dataset(pred_dir, gt_dir, thresholds)

        print("Matched : {}".format(int(r["N"])))
        print("Skipped : {} (unmatched preds)".format(int(r["Skipped"])))
        print("MAE     : {:.4f}".format(r["MAE"]))
        print("Fmax    : {:.4f}".format(r["Fmax"]))
        print("Emax    : {:.4f}".format(r["Emax"]))
        print("S_alpha : {:.4f}".format(r["Salpha"]))

        t1 = time.perf_counter()
        per_ds_time[ds] = (t1 - t0)
        print(f"Time    : {_fmt_seconds(t1 - t0)}", flush=True)
        print("", flush=True)
    t_all1 = time.perf_counter()
    end_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print("========== PER-DATASET TIME ==========", flush=True)
    for ds in ("CoCA", "Cosal2015", "CoSOD3k"):
        if ds in per_ds_time:
            print(f"{ds:<10s}: {_fmt_seconds(per_ds_time[ds])}", flush=True)

    print("========== EVAL END ==========", flush=True)
    print(f"Total   : {_fmt_seconds(t_all1 - t_all0)}", flush=True)
    print(f"Finish  : {end_ts}", flush=True)


if __name__ == "__main__":
    main()
