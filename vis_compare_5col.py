import argparse
import glob
from pathlib import Path

import cv2
import numpy as np


def find_image(img_root, parent, stem):
    exts = [".jpg", ".png", ".jpeg", ".JPG", ".PNG", ".JPEG"]
    for ext in exts:
        candidate = img_root / parent / (stem + ext)
        if candidate.exists():
            return candidate
    return None


def main():
    parser = argparse.ArgumentParser(description="Export 5-column comparison panels for CoCA.")
    parser.add_argument("--root", default=r"C:\COSOD\SCoSPARC-main", type=str, help="Project root")
    parser.add_argument("--baseline_folder", default="baseline_fixed", type=str, help="Baseline prediction folder name")
    parser.add_argument("--ours_folder", default="ours_acre", type=str, help="Ours prediction folder name")
    parser.add_argument("--dataset", default="CoCA", type=str, help="Dataset name")
    parser.add_argument("--limit", default=10, type=int, help="Number of panels to save")
    parser.add_argument(
        "--out_dir",
        default="vis_compare/baseline_stage1_vs_mean_vs_acre_coca_k10_10",
        type=str,
        help="Output folder (relative to root or absolute)",
    )
    args = parser.parse_args()

    root = Path(args.root)
    img_root = root / "datasets" / args.dataset / "image"
    gt_root = root / "datasets" / args.dataset / "groundtruth"

    baseline_root = root / "predictions" / args.baseline_folder / args.dataset
    ours_root = root / "predictions" / args.ours_folder / args.dataset
    stage1_root = root / "predictions" / args.baseline_folder / (args.dataset + "_stage1")

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(glob.glob(str(baseline_root / "**" / "*.png"), recursive=True))
    saved = 0

    for bf in files:
        bp = Path(bf)
        rel = bp.relative_to(baseline_root)
        stem = bp.stem
        parent = rel.parent

        ours_path = ours_root / rel
        stage1_path = stage1_root / rel
        gt_path = gt_root / parent / (stem + ".png")
        img_path = find_image(img_root, parent, stem)

        if (not ours_path.exists()) or (not stage1_path.exists()) or (not gt_path.exists()) or (img_path is None):
            continue

        img = cv2.imread(str(img_path))
        s1 = cv2.imread(str(stage1_path), 0)
        bm = cv2.imread(str(bp), 0)
        om = cv2.imread(str(ours_path), 0)
        gt = cv2.imread(str(gt_path), 0)
        if img is None or s1 is None or bm is None or om is None or gt is None:
            continue

        h, w = img.shape[:2]
        s1 = cv2.resize(s1, (w, h), interpolation=cv2.INTER_NEAREST)
        bm = cv2.resize(bm, (w, h), interpolation=cv2.INTER_NEAREST)
        om = cv2.resize(om, (w, h), interpolation=cv2.INTER_NEAREST)
        gt = cv2.resize(gt, (w, h), interpolation=cv2.INTER_NEAREST)

        panel = np.hstack(
            [
                img,
                cv2.cvtColor(s1, cv2.COLOR_GRAY2BGR),
                cv2.cvtColor(bm, cv2.COLOR_GRAY2BGR),
                cv2.cvtColor(om, cv2.COLOR_GRAY2BGR),
                cv2.cvtColor(gt, cv2.COLOR_GRAY2BGR),
            ]
        )
        cv2.imwrite(str(out_dir / ("{0:02d}_{1}.png".format(saved, stem))), panel)
        saved += 1
        if saved >= args.limit:
            break

    print("saved", saved, "to", out_dir)


if __name__ == "__main__":
    main()
