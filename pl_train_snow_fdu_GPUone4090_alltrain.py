from pickle import FALSE
import pytorch_lightning as pl
import os
import numpy as np
import random
import matplotlib.pyplot as plt
from pytorch_lightning import callbacks
from pytorch_lightning.accelerators import accelerator
from pytorch_lightning.core.hooks import CheckpointHooks
from pytorch_lightning.utilities import distributed
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.callbacks import ModelCheckpoint
# from dataloader import *
from dataloader_all_train import *
#from ema import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
import torchvision
import torchvision.transforms as transforms
from loss.CL1 import L1_Charbonnier_loss, PSNRLoss 
from loss.perceptual import PerceptualLoss2
from argparse import Namespace

from pytorch_lightning import seed_everything
from metrics import PSNR, SSIM

# from EDRAIN.Net import KPN
import os

# from model.NAFNet_arch import NAFNet
# from model.transweather_model import Transweather
# from SnowFormer import *

# from xwdesnow.xw_Snowformer import snowformer
# from xw_desnow_models.xw_cnn_vit_cnn import cnn_vit_cnn
# from xw_desnow_models.xw_dualenc_cnn_cat_wavevit_vit_wvit_gQlKVvit_encdecrefineProjhead import dualenc_cnn_cat_wavevit_vit_wvit_gQlKVvit_encdecrefineProjhead
from exp_SAM_desnow.Desnow_Baseline_SIE_adapter_LoRAQKV_relpos_cat_SkipCupsample_CTOF import Desnow_Baseline_SIE_Adapter_LoRAQKV_relpos_cat_SkipCupsample_CTOF

# Set seed
seed = 42 # Global seed set to 42
seed_everything(seed)
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer
# from utils import *
from pytorch_lightning.loggers import TensorBoardLogger
# wandb_logger = WandbLogger(project="SnowFormer")
# import tensorboardX
wandb_logger = TensorBoardLogger('/mnt/ai2022/dxw/SAM_desnow/Bigexp_SAMdesnow_log', name='SRRS_Desnow_Baseline_SIE_Adapter_LoRAQKV_relpos_cat_SkipCupsample_CTOF')
# load_ext tensorboard

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
img_channel = 3
width = 32
class EMA(nn.Module):
    """ Model Exponential Moving Average V2 from timm"""
    def __init__(self, model, decay=0.9999):
        super(EMA, self).__init__()
        # make a copy of the model for accumulating moving average of weights
        self.module = deepcopy(model)
        self.module.eval()
        self.decay = decay

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        self._update(model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)

enc_blks = [2, 2, 2, 2]
middle_blk_num = 2
dec_blks = [2, 2, 2, 2]

class CoolSystem(pl.LightningModule):
    
    def __init__(self, hparams):
        super(CoolSystem, self).__init__()

        self.params = hparams
        
        # train/val/test datasets
        self.train_datasets = self.params.train_datasets
        self.train_batchsize = self.params.train_bs
        self.test_datasets = self.params.test_datasets
        self.test_batchsize = self.params.test_bs
        self.validation_datasets = self.params.val_datasets
        self.val_batchsize = self.params.val_bs

        #Train setting
        self.initlr = self.params.initlr #initial learning
        self.weight_decay = self.params.weight_decay #optimizers weight decay
        self.crop_size = self.params.crop_size #random crop size
        self.num_workers = self.params.num_workers

        #loss_function
        self.loss_f = PSNRLoss()
        self.loss_per = PerceptualLoss2()
        self.model = Desnow_Baseline_SIE_Adapter_LoRAQKV_relpos_cat_SkipCupsample_CTOF()
        #self.model_ema = EMA(self.model, decay=0.999) 
        #self.ema.register()
        #self.model.load_state_dict(torch.load('/root/autodl-tmp/transformer/tb_logs/my_model/version_0/checkpoints/Allweather-Dualformer-epoch1000-psnr35.017_.pth'))#NAFNet(img_channel=img_channel, width=width, middle_blk_num=middle_blk_num, 
                #  enc_blk_nums=enc_blks, dec_blk_nums=dec_blks)
        #self.K_aug = K_Augmentation()
    def forward(self, x):
        y = self.model(x)
        #self.ema.update() 
        return y
    
    def configure_optimizers(self):
        # REQUIRED
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.initlr, betas=[0.9, 0.999])#,weight_decay=self.weight_decay)
        #scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
     	#											lr_lambda=LambdaLR(2000, 0, 350).step)#torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=5,T_mult=3)
        scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer,base_lr=self.initlr, max_lr=1.2*self.initlr,cycle_momentum=False)

 
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        # REQUIRED
        x, y,_= batch
        # x = self.K_aug(x.cuda())
        # y = self.K_aug(y.cuda(),params = self.K_aug._params)
        
        y2 = self.forward(x)          
        loss = self.loss_f(y, y2)+0.2*self.loss_per(y, y2)
        self.log('train_loss', loss)
        
        return {'loss': loss}
    

    def validation_step(self, batch, batch_idx):
        # OPTIONAL
        x, y, _ = batch
        #self.ema.apply_shadow()
        y_hat = self.forward(x)#self.forward(x)
        loss = self.loss_f(y, y_hat)+0.2*self.loss_per(y, y_hat)
        psnr = PSNR(y_hat, y)
        #self.ema.restore()
        ssim = SSIM(y_hat, y)
        self.log('val_loss', loss)
        self.log('psnr', psnr)
        self.log('ssim', ssim)
        #out = torch.cat([x,y_hat,y],2)
        #save_image(out, r'C:\Users\deep01\Desktop\dual_former\gc_dehazing\out' + '\%s.png'%(batch_idx),normalize=False)
        self.trainer.checkpoint_callback.best_model_score #save the best score model

        return {'val_loss': loss, 'psnr': psnr,'ssim':ssim}


    def train_dataloader(self):
        # REQUIRED
        train_set = SRRS_Dataset_13000(self.train_datasets,train=True,size=self.crop_size)
        #train_set = RealWorld_Dataset(self.train_datasets,train=True,size=self.crop_size)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=self.train_batchsize, shuffle=True, num_workers=self.num_workers)

        return train_loader
    
    def val_dataloader(self):
        val_set = SRRS_Dataset_13000(self.validation_datasets, train=False, size=256)
        val_loader = torch.utils.data.DataLoader(val_set, batch_size=self.val_batchsize, shuffle=False, num_workers=self.num_workers)
        
        return val_loader
def main():
    resume = False,
    resume_checkpoint_path = r'C:\Users\deep01\Desktop\dual_former\gc_dehazing\tb_logs\my_model\version_24\checkpoints\CSD-v5-epoch1387-psnr32.950-ssim0.976.ckpt'
    
    args = {
    'epochs': 2100,
    #datasetsw
    'train_datasets':'/mnt/ai2022/dxw/DATASET/snow_dataset/SRRS-2021/',
    'test_datasets':None,
    'val_datasets':'/mnt/ai2022/dxw/DATASET/snow_dataset/SRRS-2021/',
    #bs
    'train_bs':8,
     #'train_bs':4,
    'test_bs':2,
    'val_bs':2,
    'initlr':0.0002,
    'weight_decay':0.01,
    'crop_size':256,
    'num_workers':4,
    #Net
    'model_blocks':5,
    'chns':64
    }   

    ddp = DDPStrategy(process_group_backend="nccl")
    hparams = Namespace(**args)

    model = CoolSystem(hparams)

    checkpoint_callback = ModelCheckpoint(
    monitor='psnr',
    #dirpath='/mnt/data/yt/Documents/TSANet-underwater/snapshots',
    filename='CSD-v5-epoch{epoch:02d}-psnr{psnr:.3f}-ssim{ssim:.3f}',
    auto_insert_metric_name=False,   
    every_n_epochs=1,
    save_top_k=6,
    mode = "max"
    )

    if resume==True:
        trainer = pl.Trainer(
            strategy = ddp,
            max_epochs=hparams.epochs,
            resume_from_checkpoint = resume_checkpoint_path,
            gpus= [0],
            logger=wandb_logger,
            amp_backend="apex",
            amp_level='01',
            #accelerator='ddp',
            #precision=16,
            callbacks = [checkpoint_callback],
        ) 
    else:
        trainer = pl.Trainer(
            strategy = ddp,
            max_epochs=hparams.epochs,
            gpus= [0],
            logger=wandb_logger,
            amp_backend="apex",
            amp_level='01',
            #accelerator='ddp',
            #precision=16,
            callbacks = [checkpoint_callback],
        )  

    trainer.fit(model)

if __name__ == '__main__':
	#your code
    main()