import torch
from collections import OrderedDict
# from SnowFormer import *
from xw_desnow_models.xw_dualenc_cnn_superwavevit_xwfusion_refinev2_CTOF_vit_wvit_gQlKVvit import dualenc_cnn_superwavevit_xwfusion_refinev2_CTOF_vit_wvit_gQlKVvit

# 测试加载.ckpt模型权重
netG_1 = dualenc_cnn_superwavevit_xwfusion_refinev2_CTOF_vit_wvit_gQlKVvit()
new_state_dict = OrderedDict()
g1ckpt1 = "/mnt/ai2022_tr/dxw/DeSnow/checkpoint/checkpoints/CSD-v5-epoch850-psnr38.712-ssim0.983.ckpt"
checkpoint = torch.load(g1ckpt1)
for k in checkpoint['state_dict']:
    print(k)
    if k[:6] != 'model.':
        continue
    name = k[6:]
    print(name)
    new_state_dict[name] = (checkpoint['state_dict'][k])

netG_1.load_state_dict(new_state_dict)

save_pth = "Ensample/CSD-WaveFrSnow-v5-epoch850-psnr38.712-ssim0.983.pth"
torch.save(netG_1.state_dict(), save_pth)


# checkpoint_path = "/mnt/ai2022_tr/dxw/DeSnow/desnow_intergral_training/snowformer_alltrain_256/version_1/checkpoints/CSD-v5-epoch00-psnr19.545-ssim0.772.ckpt"
# checkpoint = torch.load(checkpoint_path)
#
# # 创建只包含全权重的新字典
# weights_dict= {}
# for key in checkpoint['state_dict']:
#     if 'weight' in key:
#         weights_dict[key] = checkpoint['state_dict'][key]
#
# # 保存权重为.pth文件
# save_path = "snowformer128_0epoch.pth"
# torch.save(weights_dict, save_path)

