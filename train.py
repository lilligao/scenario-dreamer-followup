import os 
import hydra
from models.scenario_dreamer_autoencoder import ScenarioDreamerAutoEncoder
from models.scenario_dreamer_autoencoder3d import ScenarioDreamerAutoEncoder3D
from models.scenario_dreamer_ldm import ScenarioDreamerLDM
from models.scenario_dreamer_ldm3d import ScenarioDreamerLDM3D

import torch
import torch.multiprocessing as mp
import shutil
torch.set_float32_matmul_precision('medium')
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import ModelCheckpoint, ModelSummary
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.profilers import AdvancedProfiler, SimpleProfiler, PyTorchProfiler
from cfgs.config import CONFIG_PATH
from hydra.utils import instantiate
from omegaconf import OmegaConf
from utils.train_helpers import cache_latent_stats, set_latent_stats


def train_ldm(cfg, cfg_ae, save_dir=None, profiler=None):
    """ Train the Scenario Dreamer Latent Diffusion Model."""
    # check if latent stats are cached, if not, compute them
    if not os.path.exists(cfg.dataset.latent_stats_path):
        cache_latent_stats(cfg)
    cfg = set_latent_stats(cfg)

    datamodule = instantiate(cfg.datamodule, dataset_cfg=cfg.dataset)
    
    monitor = 'val_loss'
    if cfg.train.save_top_k > 0:
        model_checkpoint = ModelCheckpoint(monitor=monitor, save_last=True, save_top_k=cfg.train.save_top_k, dirpath=save_dir)
    else:
        # we always track the last epoch checkpoint for evaluation or resume training.   
        model_checkpoint = ModelCheckpoint(filename='model', save_last=True, save_top_k=cfg.train.save_top_k, dirpath=save_dir)
    
    lr_monitor = LearningRateMonitor(logging_interval='step')
    model_summary = ModelSummary(max_depth=-1)
    wandb_logger = WandbLogger(
        project=cfg.train.wandb_project,
        name=cfg.train.run_name,
        entity=cfg.train.wandb_entity,
        log_model=False,
        save_dir=save_dir
    )
    if cfg.train.track:
        logger = wandb_logger 
    else:
        logger = None
    
    # resume training
    files_in_save_dir = os.listdir(save_dir)
    ckpt_path = None
    for file in files_in_save_dir:
        if file.endswith('.ckpt') and 'last' in file:
            ckpt_path = os.path.join(save_dir, file)
            backup_ckpt_path = os.path.join(save_dir, 'backup.ckpt')
            dummy = torch.load(ckpt_path, map_location='cpu')
            print("Successfully loaded last.ckpt")
            shutil.copyfile(ckpt_path, backup_ckpt_path)
            print("Resuming from checkpoint: ", ckpt_path)
            del dummy

    trainer = pl.Trainer(accelerator=cfg.train.accelerator,
                         devices=cfg.train.devices,
                         strategy=DDPStrategy(find_unused_parameters=True, gradient_as_bucket_view=True),
                         callbacks=[model_summary, model_checkpoint, lr_monitor],
                         max_steps=cfg.train.max_steps,
                         check_val_every_n_epoch=cfg.train.check_val_every_n_epoch,
                         precision=cfg.train.precision,
                         limit_train_batches=cfg.train.limit_train_batches,
                         limit_val_batches=cfg.train.limit_val_batches,
                         gradient_clip_val=cfg.train.gradient_clip_val,
                         logger=logger,
                         profiler=profiler
                        )
    
    # hack to avoid gpu memory issues when loading from checkpoint
    if '3d' in cfg.model_name:
        ldm_model = ScenarioDreamerLDM3D
    else:
        ldm_model = ScenarioDreamerLDM
        
    if ckpt_path is not None:
        model = ldm_model.load_from_checkpoint(ckpt_path, cfg=cfg, cfg_ae=cfg_ae, map_location='cpu')
    else:
        model = ldm_model(cfg=cfg, cfg_ae=cfg_ae)
    try:
        trainer.fit(model, datamodule, ckpt_path=ckpt_path)
    finally:
        profiler.describe()


def train_autoencoder(cfg, save_dir=None, profiler=None):
    """ Train the Scenario Dreamer AutoEncoder model."""
    datamodule = instantiate(cfg.datamodule, dataset_cfg=cfg.dataset)

    if '3d' in cfg.model_name:
        model = ScenarioDreamerAutoEncoder3D(cfg)
    else:
        model = ScenarioDreamerAutoEncoder(cfg)
    # we always track the last epoch checkpoint for evaluation or resume training.   
    model_checkpoint = ModelCheckpoint(filename='model', save_last=True, save_top_k=0, dirpath=save_dir)
    
    lr_monitor = LearningRateMonitor(logging_interval='step')
    model_summary = ModelSummary(max_depth=-1)
    wandb_logger = WandbLogger(
        project=cfg.train.wandb_project,
        name=cfg.train.run_name,
        entity=cfg.train.wandb_entity,
        log_model=False,
        save_dir=save_dir
    )
    if cfg.train.track:
        logger = wandb_logger 
    else:
        logger = None
    
    # resume training
    files_in_save_dir = os.listdir(save_dir)
    ckpt_path = None
    for file in files_in_save_dir:
        if file.endswith('.ckpt') and 'last' in file:
            ckpt_path = os.path.join(save_dir, file)
            backup_ckpt_path = os.path.join(save_dir, 'backup.ckpt')
            dummy = torch.load(ckpt_path) # this is to check if the checkpoint is valid
            print("Successfully loaded last.ckpt")
            shutil.copyfile(ckpt_path, backup_ckpt_path)
            print("Resuming from checkpoint: ", ckpt_path)
            del dummy
    
    trainer = pl.Trainer(accelerator=cfg.train.accelerator,
                         devices=cfg.train.devices,
                         strategy=DDPStrategy(find_unused_parameters=True, gradient_as_bucket_view=True),
                         callbacks=[model_summary, model_checkpoint, lr_monitor],
                         max_steps=cfg.train.max_steps,
                         check_val_every_n_epoch=cfg.train.check_val_every_n_epoch,
                         precision=cfg.train.precision,
                         limit_train_batches=cfg.train.limit_train_batches,
                         limit_val_batches=cfg.train.limit_val_batches,
                         gradient_clip_val=cfg.train.gradient_clip_val,
                         logger=logger
                        )
    
    trainer.fit(model, datamodule, ckpt_path=ckpt_path)


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="config3dtemp")
def main(cfg):
    # need to track whether we are training a nuplan or waymo model as 
    # nuplan predicts lane types (lane/green light/red light) and waymo does not
    dataset_name = cfg.dataset_name.name
    if 'autoencoder' in cfg.model_name:
        model_name = cfg.model_name
        cfg = cfg.ae
        # not the cleanest solution, but need to track dataset name
        OmegaConf.set_struct(cfg, False)   # unlock to allow setting dataset name
        cfg.dataset_name = dataset_name
        cfg.model_name = model_name
        OmegaConf.set_struct(cfg, True)    # relock
    else:
        model_name = cfg.model_name
        cfg_ae = cfg.ae
        cfg = cfg.ldm
        OmegaConf.set_struct(cfg, False)   # unlock to allow setting dataset name
        OmegaConf.set_struct(cfg_ae, False)
        cfg.dataset_name = dataset_name
        cfg_ae.dataset_name = dataset_name
        cfg.model_name = model_name
        OmegaConf.set_struct(cfg, True)    # relock
        OmegaConf.set_struct(cfg_ae, True)
    
    pl.seed_everything(cfg.train.seed, workers=True)

    # checkpoints saved here
    save_dir = os.path.join(cfg.train.save_dir, cfg.train.run_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    # Profiler
    # Option 1: Simple wall-time profiler
    profiler = SimpleProfiler()
    # Option 2: Detailed Python function-level profiler
    # profiler = AdvancedProfiler(dirpath=".", filename="profiler.txt")
    # profiler = PyTorchProfiler(
    #     dirpath=".",
    #     filename="pytorch_profiler",
    #     schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1), #None,   # or use dict for warmup/active cycles
    #     record_shapes=True,
    #     profile_memory=True,
    #     with_stack=True
    # )
    
    if 'autoencoder' in model_name:
        train_autoencoder(cfg, save_dir, profiler)
    elif 'ldm' in model_name:
        train_ldm(cfg, cfg_ae, save_dir, profiler)

if __name__ == '__main__':
    main()
