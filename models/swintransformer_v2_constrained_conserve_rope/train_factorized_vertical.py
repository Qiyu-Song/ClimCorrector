import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import modulus
from modulus.metrics.general.mse import mse
from modulus.utils import StaticCaptureTraining, StaticCaptureEvaluateNoGrad
from omegaconf import DictConfig
from modulus.launch.logging import (
    PythonLogger,
    LaunchLogger,
    initialize_wandb,
)
from utils.data_utils import *
from climsim_datapip_processed_h5 import climsim_dataset_processed_h5

from swintransformer_modulus_polepadding_conserve_rope import SwinTransformerV2CrModulus_polepadding_conserve
from factorized_vertical import FactorizedVerticalCorrector
import swintransformer_modulus_polepadding_conserve_rope as swintransformer_modulus_polepadding_conserve_rope

#from swintransformer_modulus_polepadding_conserve import SwinTransformerV2CrModulus_polepadding_conserve
#import swintransformer_modulus_polepadding_conserve as swintransformer_modulus_polepadding_conserve

#from swintransformer_modulus_polepadding import SwinTransformerV2CrModulus_polepadding
#import swintransformer_modulus_polepadding as swintransformer_modulus_polepadding

import hydra
from torch.nn.parallel import DistributedDataParallel
from modulus.distributed import DistributedManager
from torch.utils.data.distributed import DistributedSampler
import gc
from torch.nn.utils import clip_grad_norm_
import shutil
import random
from soap import SOAP
from modulus.launch.utils import load_checkpoint, save_checkpoint

torch.set_float32_matmul_precision("high")
# The vertical block runs attention over B*H*W (~31k at bs2) sequences. The flash and
# cuDNN SDPA kernels cannot launch at that batch ("invalid configuration argument" /
# mha_graph execute failure); the mem-efficient / math backends handle it fine.
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_cudnn_sdp(False)

# Ensure a valid TLS CA bundle inside the container: some images set REQUESTS_CA_BUNDLE/SSL_CERT_FILE
# to a missing path (e.g. /etc/ssl/certs/ca-bundle.crt), which breaks wandb media/image uploads.
# Point requests/ssl at certifi's bundle when the configured path is absent. Harmless otherwise.
import os as _os
try:
    import certifi as _certifi
    _cab = _certifi.where()
    for _v in ("REQUESTS_CA_BUNDLE", "SSL_CERT_FILE", "CURL_CA_BUNDLE"):
        if not _os.path.exists(_os.environ.get(_v, "")):
            _os.environ[_v] = _cab
except Exception:
    pass

@hydra.main(version_base="1.2", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> float:

    DistributedManager.initialize()
    dist = DistributedManager()

    # Set the random seed (configurable; default 42 reproduces the original hardcoded value).
    # Vary it for seed-repeat / noise-floor runs. Set before model & dataloader creation.
    seed = cfg.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    train_dataset_path = cfg.train_dataset_path
    val_dataset_path = cfg.val_dataset_path
    
    # check if train_dataset_path/**/train_input.h5 exists
    train_input_path = glob.glob(f'{train_dataset_path}/**/train_input.h5', recursive=True)
    if not train_input_path:
        raise FileNotFoundError("No 'train_input.h5' files found under the specified parent path.")
    # check if val_dataset_path/val_input.h5 exists
    # note that here I assumed there is one or a few subfolders under val_dataset_path that contains val_input.h5
    val_input_path =glob.glob(f'{val_dataset_path}/val_input.h5')
    if not val_input_path:
        raise FileNotFoundError("No 'val_input.h5' file found under the specified parent path.")

    # create the distributed validation dataloader below
    val_dataset = climsim_dataset_processed_h5(parent_path=val_dataset_path,stage='val',target_filename=cfg.target_filename)
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if dist.distributed else None
    val_loader = DataLoader(val_dataset, 
                            batch_size=cfg.batch_size, 
                            shuffle=False,
                            sampler=val_sampler,
                            num_workers=cfg.num_workers,
                            persistent_workers=cfg.num_workers > 0)
    
    # create the distributed training dataloader below
    train_dataset = climsim_dataset_processed_h5(parent_path=train_dataset_path,stage='train',target_filename=cfg.target_filename)
    # optional random sample-fraction subsampling for data-amount / learning-curve experiments (default 1.0 = all samples, previous behavior)
    _subfrac = cfg.get('train_subsample_frac', 1.0)
    if _subfrac < 1.0:
        _n = len(train_dataset); _k = max(1, int(_n * _subfrac))
        _idx = np.sort(np.random.default_rng(cfg.get('seed', 42)).choice(_n, size=_k, replace=False)).tolist()
        train_dataset = torch.utils.data.Subset(train_dataset, _idx)
        if (not dist.distributed) or dist.rank == 0:
            print(f"[train_subsample_frac={_subfrac}] using {_k}/{_n} train samples", flush=True)
    train_sampler = DistributedSampler(train_dataset) if dist.distributed else None
    train_loader = DataLoader(train_dataset, 
                                batch_size=cfg.batch_size, 
                                shuffle=False if dist.distributed else True,
                                sampler=train_sampler,
                                drop_last=True,
                                pin_memory=torch.cuda.is_available(),
                                num_workers=cfg.num_workers,
                                persistent_workers=cfg.num_workers > 0)
    
    # input_mean = np.load(cfg.input_mean)
    # input_std = np.load(cfg.input_std)
    # target_mean = np.load(cfg.target_mean)
    # target_std = np.load(cfg.target_std)
    # ds_grid = xr.open_dataset(cfg.climcorr_path+'utils/grid_info.nc')
    # hyai = ds_grid.hyai.values
    # hybi = ds_grid.hybi.values
    # gw = ds_grid.gw.values

    # create model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FactorizedVerticalCorrector(
        img_size=(96, 144),
        in_chans=cfg.swin.in_chans,
        out_chans=cfg.swin.out_chans,
        d=cfg.swin.embed_dim,
        depth=cfg.swin.depths,
        num_heads=cfg.swin.num_heads,
        window_size=(cfg.swin.window_size_x, cfg.swin.window_size_y),
        nlev=26,
        n3d_vars=7,
        mlp_ratio=cfg.swin.mlp_ratio,
        pos_encoding=cfg.swin.pos_encoding,
        rope_theta=cfg.swin.rope_theta,
        pole_padding=cfg.swin.pole_padding,
        pole_padding_value=cfg.swin.pole_padding_value,
        pole_tqmean=cfg.swin.pole_tqmean,
        grad_checkpoint=cfg.swin.checkpoint_stages,
    ).to(dist.device)

    # create optimizer
    if cfg.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)
    elif cfg.optimizer == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    elif cfg.optimizer == 'soap':
        optimizer = SOAP(model.parameters(), lr = cfg.learning_rate, betas=(.95, .95), weight_decay=.01, precondition_frequency=10)
    else:
        raise ValueError('Optimizer not implemented')
    
    # create scheduler
    if cfg.scheduler_name == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=cfg.scheduler.step.step_size, gamma=cfg.scheduler.step.gamma)
    elif cfg.scheduler_name == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=cfg.scheduler.plateau.factor, patience=cfg.scheduler.plateau.patience, verbose=True)
    elif cfg.scheduler_name == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.scheduler.cosine.T_max, eta_min=cfg.scheduler.cosine.eta_min)
    elif cfg.scheduler_name == 'cosine_warmup':
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=cfg.scheduler.cosine_warmup.T_0, T_mult=cfg.scheduler.cosine_warmup.T_mult, eta_min=cfg.scheduler.cosine_warmup.eta_min)
    else:
        raise ValueError('Scheduler not implemented')

    # create paths to save model weights
    # save_path will save the final model weights. In my code, this final saved model will be the one that has the best validation score
    # save_path_ckpt will save model weights after every epoch
    # save_path_ckpt_full will save the entire training state after every epoch: including model weights, optimizer and scheduler state. 
    # Such full training state and allow the training to restart after any break without the need to warmup the optimizer

    save_path = os.path.join(cfg.save_path, cfg.expname) #cfg.save_path + cfg.expname
    save_path_ckpt = os.path.join(save_path, 'ckpt')
    save_path_ckpt_full = os.path.join(save_path_ckpt, 'ckpt_full')
    if dist.rank == 0:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if not os.path.exists(save_path_ckpt):
            os.makedirs(save_path_ckpt)
        if not os.path.exists(save_path_ckpt_full):
            os.makedirs(save_path_ckpt_full)

    # if cfg.restart_path is set, then the model will load the model weight in the restart_path
    # Note that in this restart method, the optimizer is not warmed up (e.g., for adam optimizer, you typically need some training steps for the adam optimizer to start perform well)
    # below there is also another restart method by setting cfg.restart_full_ckpt, which will load both model weights and optimizer/scheduler state if you saved such checkpoint
    if len(cfg.restart_path) > 0:
        print("Restarting from checkpoint: " + cfg.restart_path)
        if dist.distributed:
            model_restart = modulus.Module.from_checkpoint(cfg.restart_path).to(dist.device)
            if dist.rank == 0:
                model.load_state_dict(model_restart.state_dict(), strict=False)
                torch.distributed.barrier()
            else:
                torch.distributed.barrier()
                model.load_state_dict(model_restart.state_dict(), strict=False)
        else:
            model_restart = modulus.Module.from_checkpoint(cfg.restart_path).to(dist.device)
            model.load_state_dict(model_restart.state_dict(), strict=False)

    # Speed: the model is memory-bandwidth bound (profiled: matmuls ~8% of kernel time,
    # copies/elementwise/layer-norm dominate), so operator fusion is what helps. Compile the
    # blocks INDIVIDUALLY -- torch.compile on the whole model traces through
    # torch.utils.checkpoint, defeats it, and OOMs. Measured 2.90x per sample with bf16,
    # numerics unchanged (30-step loss agreed to 0.000%).
    if cfg.get('compile_blocks', False):
        # Dynamo's DDPOptimizer splits the compiled graph along DDP's gradient all-reduce
        # buckets to overlap comms with backward. That split changes which intermediates are
        # materialised, so torch.utils.checkpoint(use_reentrant=False) sees a different
        # saved-tensor count on the forward vs the recompute and raises CheckpointError
        # (killed job 44212002 48s into epoch 1; single-GPU never hits this).
        # Measured on 2-GPU DDP: fixes it at 1.99x, peak 23.3GB, loss matches the uncompiled
        # control to 2.9e-05 max rel diff over 30 steps with no drift. Costs ~2% (lost
        # comms/compute overlap), negligible for a 12.7M-param model.
        # NB: alias the import. A bare `import torch._dynamo` binds the name `torch`, which
        # makes it a LOCAL for all of main() and breaks torch.manual_seed() at the top.
        import torch._dynamo as _dynamo
        _dynamo.config.optimize_ddp = False
        _n = 0
        # Compile the BOUND FORWARD, not the module. Replacing a child module with an
        # OptimizedModule renames every weight to `<blk>._orig_mod.*` in state_dict; the
        # restart path above loads with strict=False, so a compiled checkpoint would
        # silently load NOTHING instead of erroring. Compiling .forward leaves the module
        # tree untouched. Measured identical speed (0.3462 vs 0.3459 s/sample, H200).
        for _i in range(len(model.vert)):
            model.vert[_i].forward = torch.compile(model.vert[_i].forward); _n += 1
        for _i in range(len(model.horiz)):
            model.horiz[_i].forward = torch.compile(model.horiz[_i].forward); _n += 1
        if dist.rank == 0:
            print(f"[speed] torch.compile applied to {_n} blocks", flush=True)

    # Set up DistributedDataParallel if using more than a single process.
    # The `distributed` property of DistributedManager can be used to
    # check this.
    if dist.distributed:
        ddps = torch.cuda.Stream()
        with torch.cuda.stream(ddps):
            model = DistributedDataParallel(
                model,
                device_ids=[dist.local_rank],  # Set the device_id to be
                                               # the local rank of this process on
                                               # this node
                output_device=dist.device,
                broadcast_buffers=dist.broadcast_buffers,
                find_unused_parameters=dist.find_unused_parameters,
            )
        torch.cuda.current_stream().wait_stream(ddps)

    # if cfg.restart_full_ckpt is set, then the code below will load the model weights as well as the optimizer/scheduler state
    # you won't need to set cfg.restart_path if you set the cfg.restart_full_ckpt
    if cfg.restart_full_ckpt:
        print("Restarting from full checkpoint: " + save_path_ckpt_full)
        loaded_epoch = load_checkpoint(
            save_path_ckpt_full,
            models=model,
            optimizer=optimizer,
            scheduler=None if cfg.restart_full_ckpt_reset_lrscheduler else scheduler,
            device="cuda",
        )
        if cfg.restart_full_ckpt_reset_lrscheduler:
            loaded_epoch = 0
            for param_group in optimizer.param_groups:
                param_group['lr'] = cfg.learning_rate  # Set your desired new LR here
            # create scheduler
            if cfg.scheduler_name == 'step':
                scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=cfg.scheduler.step.step_size, gamma=cfg.scheduler.step.gamma)
            elif cfg.scheduler_name == 'plateau':
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=cfg.scheduler.plateau.factor, patience=cfg.scheduler.plateau.patience, verbose=True)
            elif cfg.scheduler_name == 'cosine':
                scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.scheduler.cosine.T_max, eta_min=cfg.scheduler.cosine.eta_min)
            elif cfg.scheduler_name == 'cosine_warmup':
                scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=cfg.scheduler.cosine_warmup.T_0, T_mult=cfg.scheduler.cosine_warmup.T_mult, eta_min=cfg.scheduler.cosine_warmup.eta_min)
            else:
                raise ValueError('Scheduler not implemented')

    else:
        loaded_epoch = 0

    
    # create loss function (reduction='none' so we can optionally area-weight over latitude)
    if cfg.loss == 'mse':
        criterion = nn.MSELoss(reduction='none')
    elif cfg.loss == 'mae':
        criterion = nn.L1Loss(reduction='none')
    elif cfg.loss == 'huber':
        criterion = nn.SmoothL1Loss(reduction='none')
    else:
        raise ValueError('Loss function not implemented')

    # optional latitude/area weighting (grid_info gw = FV cell-area weights over the 96 lats).
    # w_lat has mean 1, so the loss scale matches the unweighted case; loss_area_weight=False
    # reproduces the previous behavior exactly.
    if cfg.loss_area_weight:
        gw = xr.open_dataset(cfg.climcorr_path + 'utils/grid_info.nc').gw.values  # (96,)
        w_lat = torch.tensor(gw / gw.mean(), dtype=torch.float32, device=device).view(1, 1, 96, 1)
    else:
        w_lat = None

    # --- optional additions (cfg.get so missing keys reproduce previous behavior exactly) ---
    loss_spectral = bool(cfg.get('loss_spectral', False))
    loss_spectral_weight = float(cfg.get('loss_spectral_weight', 0.1))
    log_val_r2 = bool(cfg.get('log_val_r2', False))
    r2_plot_freq = int(cfg.get('r2_plot_freq', 5))

    def spectral_term(output, target):
        # Cheap "spectral" loss: match spatial gradients of pred vs target. A blurred/under-dispersed
        # prediction has too-weak gradients, so penalizing the gradient mismatch fights the damping
        # of extremes. Train-time only (never part of the scripted/exported graph).
        gx_o = output[:, :, :, 1:] - output[:, :, :, :-1]; gx_t = target[:, :, :, 1:] - target[:, :, :, :-1]
        gy_o = output[:, :, 1:, :] - output[:, :, :-1, :]; gy_t = target[:, :, 1:, :] - target[:, :, :-1, :]
        lx = (gx_o - gx_t) ** 2; ly = (gy_o - gy_t) ** 2
        if w_lat is not None:
            lx = lx * w_lat                # (1,1,96,1) broadcasts over the (W-1) lon dim
            ly = ly * w_lat[:, :, 1:, :]   # lat dim reduced by 1
        return lx.mean() + ly.mean()

    def compute_loss(output, target):
        """Returns (total_loss, components_dict). components_dict always has 'loss_base';
        with loss_spectral it also has 'loss_spectral'. loss_spectral=False -> total == base."""
        per_elem = criterion(output, target)
        if w_lat is not None:
            per_elem = per_elem * w_lat
        base = per_elem.mean()
        total = base
        comps = {"loss_base": base.detach()}
        if loss_spectral:
            spec = spectral_term(output, target)
            total = base + loss_spectral_weight * spec
            comps["loss_spectral"] = (loss_spectral_weight * spec).detach()
        return total, comps

    # grid (lat/lev + sin-edge area weights) for optional validation-R2 logging
    if log_val_r2:
        _dsg = xr.open_dataset(cfg.climcorr_path + 'utils/grid_info.nc')
        r2_lat = _dsg.lat.values.astype(np.float64); r2_lev = _dsg.lev.values.astype(np.float64)
        _edges = np.empty(r2_lat.size + 1); _edges[1:-1] = 0.5 * (r2_lat[:-1] + r2_lat[1:]); _edges[0] = -90.0; _edges[-1] = 90.0
        r2_W = np.abs(np.sin(np.deg2rad(_edges[1:])) - np.sin(np.deg2rad(_edges[:-1]))); r2_W = r2_W / r2_W.sum()  # (96,)
    
    
    # Initialize the console logger
    logger = PythonLogger("main")  # General python logger
    if cfg.logger == 'wandb':
        # Initialize the MLFlow logger
        initialize_wandb(
            project=cfg.wandb.project,
            name=cfg.expname,
            entity="qiyusong-harvard-university",
            mode="online",
        )
        LaunchLogger.initialize(use_wandb=True)



    # set how many top checkpoints to track
    if cfg.save_top_ckpts<=0:
        logger.info("Checkpoints should be set >0, setting to 1")
        num_top_ckpts = 1
    else:
        num_top_ckpts = cfg.save_top_ckpts
    if cfg.top_ckpt_mode == 'min':
        top_checkpoints = [(float('inf'), None)] * num_top_ckpts
    elif cfg.top_ckpt_mode == 'max':
        top_checkpoints = [(-float('inf'), None)] * num_top_ckpts
    else:
        raise ValueError('Unknown top_ckpt_mode')
    
    if dist.world_size > 1:
        torch.distributed.barrier()
    
    # # the following training_step and eval_step_forward are some optimized version of the training function that can take use of things like Cuda graphs, Jit, mixed precision.
    # they are provided by the modulus library. However, there are some issue using them with the swintransformer which I have not debugged successfully. So I commented out their usage here.
    # there usage can be found in this example: https://docs.nvidia.com/deeplearning/modulus/modulus-core/tutorials/simple_training_example.html#optimized-training-workflow

    # @StaticCaptureTraining(
    #     model=model,
    #     optim=optimizer,
    #     # cuda_graph_warmup=13,
    # )
    # def training_step(model, data_input, target):
    #     output = model(data_input)
    #     loss = criterion(output, target)
    #     return loss
    
    # @StaticCaptureEvaluateNoGrad(model=model, use_graphs=False)
    # def eval_step_forward(my_model, invar):
    #     return my_model(invar)
    
    #training block
    logger.info("Starting Training!")
    # Basic training block with tqdm for progress tracking
    for epoch in range(max(1,loaded_epoch+1),cfg.epochs+1):
        if dist.distributed:
            train_sampler.set_epoch(epoch)

        with LaunchLogger("train", epoch=epoch, mini_batch_log_freq=cfg.mini_batch_log_freq) as launchlog:
            model.train()

            total_iterations = len(train_loader)
            # Wrap train_loader with tqdm for a progress bar
            train_loop = tqdm(train_loader, desc=f'Epoch {epoch}')
            current_step = 0
            for iteration, (data_input, target) in enumerate(train_loop):
                # here I added an early stop option. if you set cfg.early_stop_step>0 (default is -1), then each training epoch will only have cfg.early_stop_step steps. 
                if cfg.early_stop_step > 0 and current_step > cfg.early_stop_step:
                    break
                data_input, target = data_input.to(device), target.to(device)
                data_input = data_input.permute(0, 3, 1, 2)
                target = target.permute(0, 3, 1, 2)
                optimizer.zero_grad()
                with torch.autocast('cuda', dtype=torch.bfloat16, enabled=cfg.get('amp_bf16', False)):
                    output = model(data_input)
                    loss, comps = compute_loss(output, target)
                loss.backward()

                # # for debug only, check if any parameter has None grad/not used in the backward pass
                # for name, param in model.named_parameters():
                #     if param.grad is None:
                #         print(name)

                # below is an option to do gradient clipping, which can be useful if you found very large gradient at some steps (e.g., when there is a outlier in data) that can potentially disrupt the training.
                if cfg.clip_grad:
                    clip_grad_norm_(model.parameters(), max_norm=cfg.clip_grad_norm)

                optimizer.step()
                if cfg.scheduler_name == 'cosine_warmup':
                    # here in cosine_warmup scheduler, I wanted to let the learning rate to change every training steps instead of every epoch. Just an arbitrary choice.
                    scheduler.step(epoch + iteration / total_iterations)

                launchlog.log_minibatch({"loss_train": loss.detach().cpu().numpy(), "lr": optimizer.param_groups[0]["lr"],
                                         **{k: float(v) for k, v in comps.items()}})

                # Update the progress bar description with the current loss
                train_loop.set_description(f'Epoch {epoch}')
                train_loop.set_postfix(loss=loss.item())
                current_step += 1
            
            model.eval()
            val_loss = 0.0
            num_samples_processed = 0
            vcomp_sums = {}                      # sample-weighted sums of loss components on val
            r2_res = r2_sy = r2_sy2 = None; r2_n = 0   # streaming R2 accumulators (per chan,lat,lon)
            val_loop = tqdm(val_loader, desc=f'Epoch {epoch}/1 [Validation]')
            for data_input, target in val_loop:
                data_input, target = data_input.to(device), target.to(device)
                data_input = data_input.permute(0, 3, 1, 2)
                target = target.permute(0, 3, 1, 2)
                with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16, enabled=cfg.get('amp_bf16', False)):
                    output = model(data_input)
                    loss, vcomps = compute_loss(output, target)
                    bs = data_input.size(0)
                    val_loss += loss.item() * bs
                    num_samples_processed += bs
                    for k, v in vcomps.items():
                        vcomp_sums[k] = vcomp_sums.get(k, 0.0) + float(v) * bs
                    if log_val_r2:
                        d2 = ((target - output) ** 2).sum(dim=0)      # (C,96,144)
                        sy = target.sum(dim=0); sy2 = (target ** 2).sum(dim=0)
                        if r2_res is None:
                            r2_res = torch.zeros_like(d2); r2_sy = torch.zeros_like(sy); r2_sy2 = torch.zeros_like(sy2)
                        r2_res += d2; r2_sy += sy; r2_sy2 += sy2; r2_n += bs

                # Calculate and update the current average loss
                current_val_loss_avg = val_loss / num_samples_processed
                val_loop.set_postfix(loss=current_val_loss_avg)

            
            # for validation, we need to calculate the validation score over the entire validation dataset.
            # since each gpu only usage a fraction of the validation set, we need to gather the validation score on each gpu and do an average using the all_reduce method below
            # per-rank mean of each loss component on val (mirrors the val-loss averaging below)
            vcomp_mean = {("val_" + k): (s / max(1, num_samples_processed)) for k, s in vcomp_sums.items()}
            if dist.world_size > 1:
                current_val_loss_avg = torch.tensor(current_val_loss_avg, device=dist.device)
                torch.distributed.all_reduce(current_val_loss_avg)
                current_val_loss_avg = current_val_loss_avg.item() / dist.world_size
                for k in list(vcomp_mean.keys()):
                    t = torch.tensor(vcomp_mean[k], device=dist.device)
                    torch.distributed.all_reduce(t); vcomp_mean[k] = t.item() / dist.world_size
                if log_val_r2 and r2_res is not None:
                    for t in (r2_res, r2_sy, r2_sy2):
                        torch.distributed.all_reduce(t)          # SUM across ranks
                    r2_n_t = torch.tensor(float(r2_n), device=dist.device)
                    torch.distributed.all_reduce(r2_n_t); r2_n = r2_n_t.item()

            if dist.rank == 0:
                epoch_log = {"loss_valid": current_val_loss_avg}
                epoch_log.update(vcomp_mean)
                if log_val_r2 and r2_res is not None:
                    ss_res = r2_res; ss_tot = r2_sy2 - r2_sy ** 2 / max(1.0, r2_n)
                    r2_3d = (1.0 - ss_res / (ss_tot + 1e-12)).cpu().numpy()      # (C,96,144)
                    r2_zonal = np.nanmean(r2_3d, axis=2).T                        # (96,C)
                    r2_perlevel = np.nansum(r2_zonal * r2_W[:, None], axis=0)     # (C,)
                    epoch_log["val_r2_mean"] = float(np.nanmean(r2_perlevel))
                    for i, v in enumerate(["S", "Q", "U", "V"]):
                        epoch_log[f"val_r2_{v}"] = float(np.nanmean(r2_perlevel[i*26:(i+1)*26]))
                launchlog.log_epoch(epoch_log)

                # full zonal/per-level R2 image plots every r2_plot_freq epochs
                if log_val_r2 and r2_res is not None and (epoch % max(1, r2_plot_freq) == 0):
                    try:
                        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt; import wandb
                        VARS = ["SDIFF", "QDIFF", "UDIFF", "VDIFF"]
                        f1, ax1 = plt.subplots(1, 4, figsize=(16, 4), constrained_layout=True)
                        for i, ax in enumerate(ax1):
                            ax.plot(r2_perlevel[i*26:(i+1)*26], r2_lev, marker="."); ax.set_title(VARS[i])
                            ax.set_xlabel("area-wtd R2"); ax.invert_yaxis()
                        f2, ax2 = plt.subplots(1, 4, figsize=(16, 4), constrained_layout=True)
                        for i, ax in enumerate(ax2):
                            ax.pcolormesh(r2_lat, r2_lev, r2_zonal[:, i*26:(i+1)*26].T, vmin=0.0, vmax=0.8, shading="auto")
                            ax.set_title(VARS[i]); ax.set_xlabel("lat"); ax.invert_yaxis()
                        wandb.log({"val/perlevel_r2": wandb.Image(f1), "val/zonal_r2": wandb.Image(f2), "epoch": epoch})
                        plt.close(f1); plt.close(f2)
                    except Exception as e:
                        logger.info(f"val R2 plot logging skipped: {e}")

                current_metric = current_val_loss_avg
                # Save the top checkpoints
                # this part of the code is a bit arbitrary. I choose to save the top n=cfg.save_top_ckpts checkpoints based on their validation score.
                if cfg.top_ckpt_mode == 'min':
                    is_better = current_metric < max(top_checkpoints, key=lambda x: x[0])[0]
                elif cfg.top_ckpt_mode == 'max':
                    is_better = current_metric > min(top_checkpoints, key=lambda x: x[0])[0]
                
                if len(top_checkpoints) == 0 or is_better:
                    ckpt_path = os.path.join(save_path_ckpt, f'ckpt_epoch_{epoch}_metric_{current_metric:.4f}.mdlus')
                    if dist.distributed:
                        model.module.save(ckpt_path)
                    else:
                        model.save(ckpt_path)

                    top_checkpoints.append((current_metric, ckpt_path))
                    # Sort and keep top n=cfg.save_top_ckpts checkpoints
                    if cfg.top_ckpt_mode == 'min':
                        top_checkpoints.sort(key=lambda x: x[0], reverse=False)
                    elif cfg.top_ckpt_mode == 'max':
                        top_checkpoints.sort(key=lambda x: x[0], reverse=True)
                    # delete the worst checkpoint if the saved checkpoints exceed cfg.save_top_ckpts
                    if len(top_checkpoints) > num_top_ckpts:
                        worst_ckpt = top_checkpoints.pop()
                        print(f"Removing worst checkpoint: {worst_ckpt[1]}")
                        if worst_ckpt[1] is not None:
                            os.remove(worst_ckpt[1])
                            
            if cfg.scheduler_name == 'plateau':
                # note that the reduceonplateau learning rate scheduler needs to adjust the learning rate based on if the validation score has not been improved over certain epochs.
                scheduler.step(current_val_loss_avg)
            elif cfg.scheduler_name == 'cosine_warmup':
                pass # handled in the optimizer.step() in training loop
            else:
                scheduler.step()
            
            if dist.world_size > 1:
                torch.distributed.barrier()

            # add saving full checkpoint including optimizer/scheduler state
            if cfg.restart_full_ckpt:
                if dist.rank == 0:
                    save_checkpoint(
                        save_path_ckpt_full,
                        models=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        epoch=epoch,
                    )
                
    if dist.rank == 0:
        logger.info("Start recovering the model from the top checkpoint")         
        #recover the model weight to the top checkpoint
        model = modulus.Module.from_checkpoint(top_checkpoints[0][1]).to(device)

        # Save the model at save_path
        save_file = os.path.join(save_path, 'model.mdlus')
        model.save(save_file)

        logger.info("saved model to: " + save_path)
        logger.info("Training complete!")

    return current_val_loss_avg

if __name__ == "__main__":
    main()
