import argparse
import os
from os.path import exists, join as join_paths
import torch
import numpy as np
from torchvision.transforms import functional as FF
from metrics import *
import warnings
from torchvision.utils import save_image, make_grid
from tqdm import tqdm
from torch.utils.data import DataLoader
# from dataloader import *
# from dataloader_TEST import *
from dataloader_all_TEST import *
from collections import OrderedDict
# model-------------------------------------------
# from xwdesnow.xw_Snowformer import snowformer
# from xwdesnow.xw_SnowFormer_exp9_globalQ_morelocalK_filpK import exp9_globalQ_morelocalK_filpK
# from exp_SAM_desnow.Table_2.Baseline_Finetune import Baseline_Finetune
# from exp_SAM_desnow.Table_2.Baseline_Finetune_MSPM_DARM_LoRA_QKV import Baseline_Finetune_MSPM_DARM_QKV
# from exp_SAM_desnow.Transweather import Transweather
# from exp_SAM_desnow.Desnow_Baseline_SIE_adapter_LoRAQKV_relpos_cat_SkipCupsample_CTOF import Desnow_Baseline_SIE_Adapter_LoRAQKV_relpos_cat_SkipCupsample_CTOF
from exp_SAM_desnow.Table_2.Baseline_Finetune import Baseline_Finetune

# Rain100L
# from exp_SAM_desnow.Table_7.AcLM_NAFNet_woDAR import AcLM_NAFNet_woDAR
# from exp_SAM_desnow.Table_7.Baseline_NAFNet import Table7_Baseline_NAFNet
# from exp_SAM_desnow.Table_7.AcLM_NAFNet import AcLM_NAFNet
# from exp_SAM_desnow.Table_7.Baseline_Uformer import Table7_Baseline_Uformer
# from exp_SAM_desnow.Table_7.AcLM_Uformer import AcLM_Uformer
# from exp_SAM_desnow.Table_7.Baseline_Restormer import Table7_Baseline_Restormer
# from exp_SAM_desnow.Table_7.AcLM_Restormer import AcLM_Restormer

# from exp_SAM_desnow.Table_9.Baseline_Finetune_MSPM_16_78_910_1112 import Table5_Baseline_Finetune_MSPM_16_78_910_1112
from PIL import Image
warnings.filterwarnings("ignore")
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('--cuda', action='store_true', help='use GPU computation')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--tile', type=int, default=256, help='Tile size, None for no tile during testing (testing as a whole)')
parser.add_argument('--tile_overlap', type=int, default=32, help='Overlapping of different tiles')
parser.add_argument('--scale', type=int, default=1, help='scale factor: 1, 2, 3, 4, 8') 
parser.add_argument('--dataset_type', type=str, default='CSD', help='CSD/SRRS/Snow100K/Snow100K_real/Rain100L/RESIDE6K/GoPro/Kitti/Cityscape')
parser.add_argument('--dataset_Kitti', type=str, default='/mnt/Temp_new/dxw/DATASET/snow_dataset_kitti_city/Kitti-Dataset/test', help='path of Kitti dataset')
parser.add_argument('--dataset_Cityscape', type=str, default='/mnt/Temp_new/dxw/DATASET/snow_dataset_kitti_city/Cityscape-Dataset/test', help='path of Cityscape dataset')
parser.add_argument('--dataset_Rain100L', type=str, default='/mnt/Temp_new/dxw/DATASET/Rain_dataset/Rain200L/test', help='path of Rain100L dataset')
parser.add_argument('--dataset_RESIDE6K', type=str, default='/mnt/Temp_new/dxw/DATASET/Haze_Dataset/RESIDE6K/test', help='path of RESIDE6K dataset')
parser.add_argument('--dataset_GoPro', type=str, default='/mnt/Temp_new/dxw/DATASET/GoPro_Image/test/', help='path of GoPro dataset')
parser.add_argument('--dataset_real', type=str, default='/mnt/Temp_new/dxw/DATASET/snow_dataset/CSD/Test', help='path of CSD dataset')
parser.add_argument('--dataset_CSD', type=str, default='/mnt/Temp_new/dxw/DATASET/snow_dataset/CSD/Test', help='path of CSD dataset')
parser.add_argument('--dataset_SRRS', type=str, default='/mnt/Temp_new/dxw/DATASET/snow_dataset/SRRS-2021/', help='path of SRRS dataset')
parser.add_argument('--dataset_Snow100K', type=str, default='/mnt/Temp_new/dxw/DATASET/snow_dataset/Snow100K/realistic/',  help='path of Snow100k dataset')
parser.add_argument('--savepath', type=str, default='./out/Without_PMSP_CSD', help='path of output image')
parser.add_argument('--model_path', type=str,
                    default='/mnt/Temp_new/dxw/code/SAM_desnow/AcLM_Ablexp_checkpoint_log/Table2/Table2_Baseline_Finetune/version_0/checkpoints/CSD-v5-epoch99-psnr31.878-ssim0.952.ckpt', help='path of SnowFormer checkpoint')

opt = parser.parse_args()
if opt.dataset_type == 'CSD':
    snow_test = DataLoader(dataset=CSD_Dataset(opt.dataset_CSD, train=False, size=256, rand_inpaint=False, rand_augment=None), batch_size=1, shuffle=False, num_workers=4)
if opt.dataset_type == 'SRRS':
    snow_test = DataLoader(dataset=SRRS_Dataset_13000(opt.dataset_SRRS, train=False, size=256, rand_inpaint=False, rand_augment=None), batch_size=1, shuffle=False, num_workers=4)
if opt.dataset_type == 'Snow100K':
    snow_test = DataLoader(dataset=Snow100K_test_Dataset(opt.dataset_Snow100K, train=False, size=256, rand_inpaint=False, rand_augment=None), batch_size=1, shuffle=False, num_workers=4)
if opt.dataset_type == 'Snow100K_real':
    snow_test = DataLoader(dataset=snow_real_Dataset(opt.dataset_Snow100K, train=False, size=256, rand_inpaint=False, rand_augment=None), batch_size=1, shuffle=False, num_workers=4)
if opt.dataset_type == 'Rain100L':
    snow_test = DataLoader(dataset=Rain100_Dataset_test(opt.dataset_Rain100L, train=False, size=256, rand_inpaint=False, rand_augment=None), batch_size=1, shuffle=False, num_workers=4)
if opt.dataset_type == 'RESIDE6K':
    snow_test = DataLoader(dataset=RESIDE_6k_Dataset_test(opt.dataset_RESIDE6K, train=False, size=256, rand_inpaint=False, rand_augment=None), batch_size=1, shuffle=False, num_workers=4)
if opt.dataset_type == 'GoPro':
    snow_test = DataLoader(dataset=GoPro_Dataset_test(opt.dataset_GoPro, train=False, size=256, rand_inpaint=False, rand_augment=None), batch_size=1, shuffle=False, num_workers=4)
if opt.dataset_type == 'Kitti':
    snow_test = DataLoader(dataset=SnowKitti_Dataset_Medium_test(opt.dataset_Kitti, train=False, size=256, rand_inpaint=False, rand_augment=None), batch_size=1, shuffle=False, num_workers=4)
if opt.dataset_type == 'Cityscape':
    snow_test = DataLoader(dataset=SnowCityScapes_Dataset_Small_test(opt.dataset_Cityscape, train=False, size=256, rand_inpaint=False, rand_augment=None), batch_size=1, shuffle=False, num_workers=4)
# def mae(image1, image2):
#
#     # 计算MAE
#     mae = np.mean(np.abs(image1 - image2))
#     return mae

# netG_1 = snowformer().cuda()
netG_1 = Baseline_Finetune().cuda()


if __name__ == '__main__':   

    ssims = []
    psnrs = []
    rmses = []

    maes = []

    # 测试加载.ckpt模型权重
    new_state_dict = OrderedDict()
    g1ckpt1 = opt.model_path
    checkpoint = torch.load(g1ckpt1)
    for k in checkpoint['state_dict']:
        print(k)
        if k[:6] != 'model.':
            continue
        name = k[6:]
        print(name)
        new_state_dict[name] = (checkpoint['state_dict'][k])
    netG_1.load_state_dict(new_state_dict)
    #
    # save_pth = "snow100k_pth.pth"
    # torch.save(netG_1.state_dict(), save_pth)
    # print("ok")
    # print("ok")
    # print("checkpoint:", checkpoint)
    # print("ok")
    # netG_1.load_state_dict(checkpoint[''])

    # g1ckpt1 = opt.model_path
    # ckpt = torch.load(g1ckpt1)
    # netG_1.load_state_dict(ckpt)

    savepath_dataset = os.path.join(opt.savepath, opt.dataset_type)
    if not os.path.exists(savepath_dataset):
        os.makedirs(savepath_dataset)
    loop = tqdm(enumerate(snow_test), total=len(snow_test))

    for idx, (haze, clean, name) in loop:
        # print(haze.shape)
        # print(clean.shape)
        # print(name)
        # print("ok")
        
        with torch.no_grad():
                
                haze = haze.cuda(); clean = clean.cuda()
                # print("shape:", haze.shape)
                # print("ok")
                b, c, h, w = haze.size()

                tile = min(opt.tile, h, w)
                tile_overlap = opt.tile_overlap
                sf = opt.scale

                stride = tile - tile_overlap
                h_idx_list = list(range(0, h-tile, stride)) + [h-tile]
                w_idx_list = list(range(0, w-tile, stride)) + [w-tile]
                E1 = torch.zeros(b, c, h*sf, w*sf).type_as(haze)
                W1 = torch.zeros_like(E1)

                for h_idx in h_idx_list:
                    for w_idx in w_idx_list:
                        in_patch = haze[..., h_idx:h_idx+tile, w_idx:w_idx+tile]
                        out_patch1 = netG_1(in_patch)
                        out_patch_mask1 = torch.ones_like(out_patch1)
                        E1[..., h_idx*sf:(h_idx+tile)*sf, w_idx*sf:(w_idx+tile)*sf].add_(out_patch1)
                        W1[..., h_idx*sf:(h_idx+tile)*sf, w_idx*sf:(w_idx+tile)*sf].add_(out_patch_mask1)
                dehaze = E1.div_(W1)

                save_image(dehaze, os.path.join(savepath_dataset, '%s.png'%(name)), normalize=False)

                # dehaze=netG_1(haze)
                ssim1 = SSIM(dehaze, clean).item()
                psnr1 = PSNR(dehaze, clean)

                # mae1 = mae(dehaze, clean)

                ssims.append(ssim1)
                psnrs.append(psnr1)
                # maes.append(mae1)
                # print('Generated images %04d of %04d' % (idx+1, len(snow_test)))
                # print('ssim:', (ssim1))
                # print('psnr:', (psnr1))

    ssim = np.mean(ssims)
    psnr = np.mean(psnrs)
    # mae = maes

    print('ssim_avg:', ssim)
    print('psnr_avg:', psnr)
    # print('mae_avg:', mae)
 