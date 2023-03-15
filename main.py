import configparser as CP
import time
from torchsummary import summary

import pytorch_lightning as pl

import build
from layers import *
from dataloaders import *

pl.seed_everything(0)

init_time = time.time()
##################################################################################
############################ Read Config #########################################
##################################################################################
def load_cfg():
    config = CP.ConfigParser()
    config.read('config.cfg', encoding='UTF-8')

    param = dict()

    param['use_ckpt'] = config.getboolean('data','use_ckpt')
    param['data_hours'] = config.getint('data','data_hours')
    param['dataset'] = config.get('data','dataset')

    param['WORKERS'] = config.getint('model','WORKERS')
    param['batch_size'] = config.getint('model', 'BATCH_SIZE')
    param['learning_rate'] = config.getfloat('model', 'LEARNING_RATE')
    param['max_epoch'] = config.getint('model', 'MAX_EPOCH')
    param['gradient_clip_norm'] = config.getfloat('model', 'GRADIENT_CLIP_NORM')
    param['precision'] = config.getint('model', 'PRECISION')

    param['B'] = config.getint('Net', 'B')
    param['H'] = config.getint('Net', 'H')
    param['L'] = config.getint('Net', 'L')
    param['ffn_dim'] = config.getint('Net', 'ffn_dim')
    param['hidden_dim'] = config.getint('Net', 'hidden_dim')

    return param


if __name__ == '__main__':
    ########################## Load Config ##################################
    cfg = load_cfg()

    ########################## Build DataLoader ###############################
    if cfg['dataset']=='DNS':
        train_loader, val_loader, steps_per_epoch = build_on_fly_dataloaders(cfg)
    elif cfg['dataset']=='VBD':
        train_loader, val_loader, steps_per_epoch = build_vbd_dataloaders(cfg)

    ######### TRAINING ######################
    # summary(Net(cfg).cuda(),(47606,))

    if not cfg['use_ckpt']:
        lightning_net = build.LightningNet(cfg,steps_per_epoch)
        trainer = build.build_trainer(cfg)
        trainer.fit(lightning_net,
                    train_dataloaders=train_loader,
                    val_dataloaders=val_loader,
                    ckpt_path=None
                    )