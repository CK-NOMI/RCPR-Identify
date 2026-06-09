# vis_compare_coca.py
# 只读：读取 Input / Baseline / Ours / GT，拼成一张横向 panel 输出
# 自动识别输入图像后缀：.jpg/.png/.jpeg（大小写都支持）
# 默认输出 10 张对比图
# Compatible with Python 3.7+

import glob
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


def find_image_with_any_ext(imgroot: Path, parent: Path, stem: str) -> Optional[Path]:
    exts = [".jpg", ".png", ".jpeg", ".JPG", ".PNG", ".JPEG"]
    for e in exts:
        cand = imgroot / parent / (stem + e)
        if cand.exists():
            return cand
    return None


def main():
    root = Path(r"C:\COSOD\SCoSPARC-main")

    broot = root / "predictions" / "baseline_fixed" / "CoCA"
    oroot = root / "predictions" / "ours_acre" / "CoCA"

    imgroot = root / "datasets" / "CoCA" / "image"
    gtroot = root / "datasets" / "CoCA" / "groundtruth"

    out = root / "vis_compare" / "baseline_vs_ours_coca_k10_10"
    out.mkdir(parents=True, exist_ok=True)

    files = sorted(glob.glob(str(broot / "**" / "*.png"), recursive=True))

    if len(files) == 0:
        print("[ERROR] No baseline prediction png found under:", broot)
        print("        Please confirm you have run K=10 predictions and the folder exists.")
        return

    saved = 0
    for bf in files:
        p = Path(bf)
        rel = p.relative_to(broot)          # e.g., class/stem.png
        parent = rel.parent                 # class/
        stem = p.stem

        of = oroot / rel                    # ours pred path
        gf = gtroot / parent / (stem + ".png")
        imgp = find_image_with_any_ext(imgroot, parent, stem)

        if (not of.exists()) or (imgp is None) or (not gf.exists()):
            continue

        bi = cv2.imread(str(imgp))
        bm = cv2.imread(str(p), 0)
        om = cv2.imread(str(of), 0)
        gt = cv2.imread(str(gf), 0)

        if bi is None or bm is None or om is None or gt is None:
            continue

        H, W = bi.shape[:2]
        bm = cv2.resize(bm, (W, H), interpolation=cv2.INTER_NEAREST)
        om = cv2.resize(om, (W, H), interpolation=cv2.INTER_NEAREST)
        gt = cv2.resize(gt, (W, H), interpolation=cv2.INTER_NEAREST)

        b3 = cv2.cvtColor(bm, cv2.COLOR_GRAY2BGR)
        o3 = cv2.cvtColor(om, cv2.COLOR_GRAY2BGR)
        g3 = cv2.cvtColor(gt, cv2.COLOR_GRAY2BGR)

        panel = np.hstack([bi, b3, o3, g3])
        out_path = out / f"{saved:02d}_{parent.as_posix().replace('/', '_')}__{stem}.png"
        cv2.imwrite(str(out_path), panel)

        saved += 1
        if saved >= 10:
            break

    print("saved", saved, "to", out)


if __name__ == "__main__":
    main()