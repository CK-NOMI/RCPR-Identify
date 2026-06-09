import torch
import torch.nn as nn
import torch.optim as optim
from util import Logger, AverageMeter, save_checkpoint, save_tensor_img, set_seed
import os
import numpy as np
from matplotlib import pyplot as plt
import time
import argparse
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
from scipy.stats import kurtosis, skew
from scipy.stats import entropy
from sklearn.linear_model import LinearRegression
import time
from sklearn.cluster import KMeans
import seaborn as sns

from models import *
from utils import *
from dataset import get_loader


os.environ['CUDA_VISIBLE_DEVICES'] = '0'

parser = argparse.ArgumentParser(description='')

parser.add_argument('--loss',
                    default='IoU_loss',
                    type=str,
                    help="Options: '', ''")
parser.add_argument('--data_split',
                    default=2,
                    type=str,
                    help="Options: '', ''")
parser.add_argument('--lab_im_path',
                    default= './datasets/COCO9213/img_bilinear_224',
                    type=str,
                    help="Options: '', ''")
parser.add_argument('--lab_gt_path',
                    default='./datasets/COCO9213/gt_bilinear_224',
                    type=str,
                    help="Options: '', ''")                     
parser.add_argument('--bs', '--batch_size', default=1, type=int)
parser.add_argument('--lr',
                    '--learning_rate',
                    default=1*1e-4,
                    type=float,
                    help='Initial learning rate')
parser.add_argument('--resume',
                    default=None,
                    type=str,
                    help='path to latest checkpoint')
parser.add_argument('--epochs', default=350, type=int)
parser.add_argument('--start_epoch',
                    default=0,
                    type=int,
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--trainset',
                    default='CoCo',
                    type=str,
                    help="Options: 'CoCo'")
parser.add_argument('--size',
                    default=224,
                    type=int,
                    help='input size')
parser.add_argument('--tmp', default='./sup_sam2', help='Temporary folder')

# args = parser.parse_args()


def main(args):
    
    device = torch.device("cuda")
    model = SCoSPARC()
    model = model.to(device)

    modelname = './checkpoints/'+ args.checkpoint_name #E.g. of model_checkpoint:'model.pt'
    model1 = torch.load(modelname)
    print('loaded', modelname)

    model.to(device)
    model.load_state_dict(model1)
    if args.stage2_proto == 'mean':
        model.stage2_use_acre = False
    else:
        model.stage2_use_acre = True
    # TopK v2: only active for ACRE branch in inference.
    model.rcpr.topk_ratio = float(args.topk_ratio) if args.stage2_proto == 'acre' else 0.0
    model.rcpr.topk_mode = str(args.topk_mode).lower()
    model.rcpr.topk_res_alpha = max(0.0, min(1.0, float(args.topk_res_alpha)))
    model.rcpr.topk_conf_gate = max(0.0, min(1.0, float(args.topk_conf_gate)))
    model.rcpr.topk_mass_min = max(0.0, min(1.0, float(args.topk_mass_min)))
    model.rcpr.topk_delta_th = max(0.0, float(args.topk_delta_th))
    model.stage2_rpf_rounds = int(args.rpf_rounds)
    lam = max(0.0, min(1.0, float(args.rpf_soft_lambda)))
    model.stage2_rpf_soft_lambda = lam
    tau2_delta = max(0.0, float(args.tau2_delta))
    model.stage2_tau2_delta = tau2_delta
    model.stage2_tau2_mode = str(args.tau2_mode).lower()
    model.stage2_tau2_margin_eps = max(1e-6, float(args.tau2_margin_eps))
    model.stage2_tau2_target_near = max(1e-6, float(args.tau2_target_near))
    model.stage2_tau2_delta_min = max(0.0, float(args.tau2_delta_min))
    model.stage2_tau2_delta_max = max(model.stage2_tau2_delta_min, float(args.tau2_delta_max))
    model.stage2_tau2_k = max(0.0, float(args.tau2_k))
    model.stage2_tau2_proto_gate = 1 if int(args.tau2_proto_gate) != 0 else 0
    model.stage2_tau2_proto_th = max(0.0, float(args.tau2_proto_th))
    model.stage2_baseline_legacy = 1 if int(args.baseline_legacy) != 0 else 0
    model.stage2_baseline_box_sim = float(args.baseline_box_sim)
    # Only affects ACRE branch; keep mean path fully unchanged.
    if args.stage2_proto == 'acre':
        model.stage2_tau_sim01 = float(args.tau1_sim)
    print(f"[Stage2Proto] {args.stage2_proto.upper()} | folder={args.model_folder} | ckpt={args.checkpoint_name} | K={args.max_group_images}")
    print(f"[Stage2TopK] ratio={model.rcpr.topk_ratio:.3f} mode={model.rcpr.topk_mode} enabled={model.rcpr.topk_ratio>0} (ACRE inference-only)")
    topk_rtg_enabled = int(args.stage2_proto == 'acre' and model.rcpr.topk_ratio > 0 and model.rcpr.topk_mode == 'rtg')
    print(
        f"[TOPK_RTG_CFG] mode={model.rcpr.topk_mode} alpha={model.rcpr.topk_res_alpha:.3f} "
        f"c_gate={model.rcpr.topk_conf_gate:.3f} mass_min={model.rcpr.topk_mass_min:.3f} "
        f"delta_th={model.rcpr.topk_delta_th:.3f} enabled={topk_rtg_enabled}"
    )
    print(f"[RPF] rounds={model.stage2_rpf_rounds}")
    rpf_v33_enabled = int(args.stage2_proto == 'acre' and args.rpf_rounds == 2 and lam > 0.0)
    print(f"[RPF_V33] lambda={lam:.3f} enabled={rpf_v33_enabled}")
    rpf_v34_enabled = int(args.stage2_proto == 'acre' and args.rpf_rounds == 2 and tau2_delta > 0.0)
    print(f"[RPF_V34] tau2_delta={tau2_delta:.3f} enabled={rpf_v34_enabled}")
    rpf_v35_enabled = int(
        args.stage2_proto == 'acre' and
        args.rpf_rounds == 2 and
        str(args.tau2_mode).lower() == 'adaptive' and
        tau2_delta > 0.0
    )
    print(
        f"[RPF_V35] mode={model.stage2_tau2_mode} base={tau2_delta:.3f} "
        f"margin_eps={model.stage2_tau2_margin_eps:.3f} target_near={model.stage2_tau2_target_near:.3f} "
        f"min={model.stage2_tau2_delta_min:.3f} max={model.stage2_tau2_delta_max:.3f} "
        f"k={model.stage2_tau2_k:.3f} proto_gate={model.stage2_tau2_proto_gate} "
        f"proto_th={model.stage2_tau2_proto_th:.6f} "
        f"enabled={rpf_v35_enabled}"
    )
    print(f"[BaselineLegacy] enabled={model.stage2_baseline_legacy} box_sim={model.stage2_baseline_box_sim:.3f}")
    acre_calib_enabled = int(args.stage2_proto == 'acre')
    print(f"[ACRE_CALIB] tau1={float(args.tau1_sim):.3f} score_norm=none enabled={acre_calib_enabled}")
    model.eval()

    save_root = './predictions/'+args.model_folder+'/'
    custom_img_path = str(getattr(args, 'custom_img_path', '') or '').strip()
    custom_gt_path = str(getattr(args, 'custom_gt_path', '') or '').strip()
    if custom_img_path:
        os.environ['COSOD_CUSTOM_IMAGE_ROOT'] = custom_img_path
    else:
        os.environ.pop('COSOD_CUSTOM_IMAGE_ROOT', None)
    
    if args.datasets.lower() == 'all':
        testsets = ['CoCA', 'Cosal2015', 'CoSOD3k']
    else:
        testsets = [d.strip() for d in args.datasets.split(',') if d.strip()]

    for testset in testsets:
        
        print('=============================================')
        if testset == 'CoCA':
            test_img_path = custom_img_path if custom_img_path else './datasets/CoCA/image/' # CoCA image folder path
            test_gt_path = custom_gt_path if custom_gt_path else './datasets/CoCA/groundtruth/' # CoCA ground truth folder path
            saved_root = os.path.join(save_root, 'CoCA')
            stage1_saved_root = os.path.join(save_root, 'CoCA_stage1')

        elif testset == 'CoSOD3k':
            test_img_path = './datasets/CoSOD3k/image/' # CoSOD3k image folder path
            test_gt_path = './datasets/CoSOD3k/groundtruth/' # CoSOD3k ground truth folder path
            saved_root = os.path.join(save_root, 'CoSOD3k')
            stage1_saved_root = os.path.join(save_root, 'CoSOD3k_stage1')

        elif testset == 'Cosal2015':
            test_img_path = './datasets/Cosal2015/image/' # CoSal2015 image folder path
            test_gt_path = './datasets/Cosal2015/groundtruth/' # CoSal2015 ground truth folder path
            saved_root = os.path.join(save_root, 'Cosal2015')
            stage1_saved_root = os.path.join(save_root, 'Cosal2015_stage1')
            
            
        else:
            print('Unknown test dataset')
            print(args.dataset)
        
        test_loader = get_loader(
            test_img_path,
            test_gt_path,
            args.size,
            1,
            istrain=False,
            shuffle=False,
            num_workers=args.test_num_workers,
            pin=True,
        )

        
        count = 0
        count_l,time_t = 0,0
        for batch in test_loader:
            inputs = batch[0].to(device).squeeze(0)
            gts = batch[1].to(device).squeeze(0)
            subpaths = batch[2]
            ori_sizes = batch[3]
            if args.max_group_images > 0:
                # v1.3: deterministic truncation for smoke test stability.
                # Keep the first K samples in each group (no randomness).
                k = min(args.max_group_images, len(inputs))
                inputs = inputs[:k]
                gts = gts[:k]
                subpaths = subpaths[:k]
                ori_sizes = ori_sizes[:k]
            t0 = time.time()
            
            scaled_preds_m,scaled_preds_nocrf,corr_maps,sa_maps,scaled_preds,preds_nocrf,avg_conf,avg_th,fg_interim = model(inputs,subpaths,'test',0,0,50,testset)

            count_l += len(inputs)
            count +=1   
                
            num = gts.shape[0]
            
            os.makedirs(os.path.join(saved_root, subpaths[0][0].split('/')[0]), exist_ok=True)
            os.makedirs(os.path.join(stage1_saved_root, subpaths[0][0].split('/')[0]), exist_ok=True)

            for inum in range(num):
                subpath = subpaths[inum][0]
                ori_size = (ori_sizes[inum][0].item(), ori_sizes[inum][1].item())
                if custom_img_path and testset == 'CoCA':
                    orig = cv2.imread(os.path.join(custom_img_path, subpath[:-4] + '.jpg'))
                else:
                    orig = cv2.imread('./datasets/'+testset+'/image/'+subpath[:-4]+'.jpg')
                
                res = scaled_preds[inum].detach().cpu().numpy()
                res = np.uint8(res*255)
                res = cv2.resize(np.uint8(res),(ori_size[1],ori_size[0]))
                cv2.imwrite(os.path.join(saved_root, subpath),res)

                stage1_res = fg_interim[inum].detach().cpu().numpy()
                if stage1_res.ndim == 3:
                    stage1_res = np.squeeze(stage1_res, axis=0)
                stage1_res = np.clip(stage1_res, 0.0, 1.0)
                stage1_res = np.uint8(stage1_res * 255.0)
                stage1_res = cv2.resize(stage1_res, (ori_size[1], ori_size[0]), interpolation=cv2.INTER_NEAREST)
                cv2.imwrite(os.path.join(stage1_saved_root, subpath), stage1_res)

        print(f"[Stage1Saved] {stage1_saved_root}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--size',
                        default=224,
                        type=int,
                        help='input size')
    parser.add_argument('--model_folder', default='checkpoints', type=str, help='model folder')
    parser.add_argument('--checkpoint_name', default='model_combined.pt', type=str, help='Checkpoint name')
    parser.add_argument('--test_num_workers', default=8, type=int, help='dataloader workers for test')
    parser.add_argument('--datasets', default='all', type=str,
                        help="test datasets: 'all' or comma-separated list, e.g. 'CoCA' or 'CoCA,Cosal2015'")
    parser.add_argument('--max_group_images', default=0, type=int,
                        help='max images per group for test; 0 means no truncation')
    parser.add_argument('--stage2_proto', default='mean', choices=['mean', 'acre'], type=str,
                        help="Stage-2 prototype branch: 'mean' for Baseline(Fixed), 'acre' for Ours(ACRE-only)")
    parser.add_argument('--topk_ratio', default=0.0, type=float,
                        help='TopK ratio for ACRE weights in inference-only mode. 0 disables.')
    parser.add_argument('--topk_mode', default='legacy', choices=['legacy', 'rtg'], type=str,
                        help="TopK mode in ACRE branch: 'legacy' keeps hard TopK; 'rtg' enables risk-controlled gating.")
    parser.add_argument('--topk_res_alpha', default=0.20, type=float,
                        help='RTG residual convex alpha in [0,1]: w_res=(1-alpha)*w_topk + alpha*w_full.')
    parser.add_argument('--topk_conf_gate', default=0.58, type=float,
                        help='RTG confidence gate c_gate: only conf_i<c_gate can use TopK.')
    parser.add_argument('--topk_mass_min', default=0.55, type=float,
                        help='RTG minimum retained mass after TopK.')
    parser.add_argument('--topk_delta_th', default=0.035, type=float,
                        help='RTG proto drift upper bound delta_th.')
    parser.add_argument('--rpf_rounds', default=1, choices=[1, 2], type=int,
                        help='RPF rounds in Stage-2 inference. 1=ACRE-only, 2=ACRE+RPF feedback.')
    parser.add_argument('--rpf_soft_lambda', default=0.0, type=float,
                        help='Soft-RPF mixing lambda in [0,1]. 0 keeps hard-RPF(v3.2) behavior.')
    parser.add_argument('--tau2_delta', default=0.0, type=float,
                        help='Round2 threshold offset: tau2=tau1-delta (default 0.015 is v3.4 recommended).')
    parser.add_argument('--tau2_mode', default='fixed', choices=['fixed', 'adaptive'], type=str,
                        help="Round2 threshold mode: 'fixed' keeps v3.4; 'adaptive' enables v3.5 per-group delta.")
    parser.add_argument('--tau2_margin_eps', default=0.02, type=float,
                        help='Near-threshold window eps for adaptive mode.')
    parser.add_argument('--tau2_target_near', default=0.08, type=float,
                        help='Target near-threshold ratio in adaptive mode.')
    parser.add_argument('--tau2_k', default=0.5, type=float,
                        help='Adaptive additive gain k (>=0): delta_eff=clip(delta_base+k*(target_near-near_r1),min,max).')
    parser.add_argument('--tau2_proto_gate', default=1, type=int,
                        help='Adaptive proto gate switch. 1 enables proto_delta gate, 0 disables.')
    parser.add_argument('--tau2_proto_th', default=0.001, type=float,
                        help='Proto gate threshold. If proto_delta<th, force delta_eff=delta_base.')
    parser.add_argument('--tau2_delta_min', default=0.005, type=float,
                        help='Minimum effective tau2_delta in adaptive mode.')
    parser.add_argument('--tau2_delta_max', default=0.03, type=float,
                        help='Maximum effective tau2_delta in adaptive mode.')
    parser.add_argument('--baseline_legacy', default=1, type=int,
                        help='Baseline(mean) compatibility switch. 1=use original Stage-2 scoring semantics.')
    parser.add_argument('--baseline_box_sim', default=0.77, type=float,
                        help='Legacy baseline normalized region score threshold.')
    parser.add_argument('--tau1_sim', default=0.85, type=float,
                        help='Stage-2 region threshold in sim01 space; only active when --stage2_proto acre.')
    parser.add_argument('--custom_img_path', default='', type=str,
                        help='Optional image root override for CoCA-compatible inference.')
    parser.add_argument('--custom_gt_path', default='', type=str,
                        help='Optional groundtruth root override for CoCA-compatible inference; can be missing in test mode.')

    args = parser.parse_args()

    main(args)
