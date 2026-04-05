from __future__ import print_function

import os
import time
import random
import argparse
import pathlib
import numpy as np
import torch
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import glob
from pynvml import *
from tensorboardX import SummaryWriter
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
import csv
from datetime import timedelta
from utils import *

from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

world_size, rank = 1, 0

def setup_distributed():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            timeout=timedelta(hours=2)
        )
        return True, local_rank
    return False, 0


def load_val_volume(folder, name):
    for ext in SUPPORTED_EXT:
        path = os.path.join(folder, f"{name}.{ext}")
        if os.path.exists(path):
            vol, spacing, origin, direction = load_volume(path)
            return vol, spacing, origin, direction, ext
    raise FileNotFoundError(f"{name} not found as {name}.nii.gz or {name}.mha in {folder}")


class GPUMonitor:
    def __init__(self):
        nvmlInit()
        self.device_count = nvmlDeviceGetCount()
        self.util = []
        self.mem = []

    def update(self):
        total_util = 0.0
        total_mem = 0.0
        for i in range(self.device_count):
            handle = nvmlDeviceGetHandleByIndex(i)
            util = nvmlDeviceGetUtilizationRates(handle).gpu
            meminfo = nvmlDeviceGetMemoryInfo(handle)
            total_util += util
            total_mem += meminfo.used / (1024**3)
        self.util.append(total_util / self.device_count)
        self.mem.append(total_mem / self.device_count)

    def summary(self):
        return (
            np.mean(self.util) if self.util else 0,
            np.mean(self.mem) if self.mem else 0
        )

    def shutdown(self):
        nvmlShutdown()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True, help="dataset root (each subject folder)")
    parser.add_argument("--output_dir", required=True, help="where predictions saved")
    parser.add_argument('--gpu', type=str, default='0, 1')
    parser.add_argument('--input_nc', type=int, default=1)
    parser.add_argument('--output_nc', type=int, default=1)
    parser.add_argument('--D_model', type=str, default='wave3DDiscriminator')
    parser.add_argument('--G_model', type=str, default='Med2Transformer')
    parser.add_argument('--ngf', type=int, default=16)
    parser.add_argument('--ndf', type=int, default=16)
    parser.add_argument('--n_layers_D', type=int, default=2)
    parser.add_argument('--G_norm', type=str, default='instance')
    parser.add_argument('--D_norm', type=str, default='instance')
    parser.add_argument('--init_type', type=str, default='normal')
    parser.add_argument('--init_gain', type=float, default=0.02)
    parser.add_argument('--no_dropout', action='store_true')
    parser.add_argument('--num_threads', default=16, type=int)
    parser.add_argument('--batch_size', type=int, default=12)
    parser.add_argument('--depthSize', type=int, default=48)
    parser.add_argument('--ImageSize', type=int, default=128)
    parser.add_argument('--load_name', type=str, default='latest')
    parser.add_argument('--lambda_L1', type=float, default=20)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--VGG_loss', action='store_false')
    parser.add_argument('--isTrain', action='store_false')
    parser.add_argument('--Npatch', type=int, default=24)
    parser.add_argument('--print_freq_num', type=int, default=4)
    parser.add_argument('--continue_train', action='store_true')
    parser.add_argument('--epoch_count', type=int, default=1)
    parser.add_argument('--phase', type=str, default='train')
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--lr_max', type=float, default=0.0002)
    parser.add_argument('--gan_mode', type=str, default='vanilla')
    parser.add_argument('--loss_pre_dir', type=str, default='weights/vgg19-dcbb9e9d.pth')
    parser.add_argument('--Max_CT', type=int, default=2000)
    parser.add_argument('--disx', type=int, default=10120)

    opt = parser.parse_args()
    opt.image_dir = os.path.join(opt.data_root, "train")
    opt.val_dir   = os.path.join(opt.data_root, "val")
    
    opt.code_dir = os.getcwd()
    opt.results_dir = os.path.join(
        opt.code_dir,
        opt.output_dir
    )

    opt.model_results = os.path.join(opt.results_dir, 'model_checkpoints')
    opt.file_name_txt = os.path.join(opt.results_dir, 'train_message.txt')
    opt.pretrain_model_path = os.path.join(opt.code_dir, opt.loss_pre_dir)
    opt.sample_img_dir = os.path.join(opt.results_dir, "sample_images")


    os.makedirs(opt.model_results, exist_ok=True)
    os.makedirs(opt.sample_img_dir, exist_ok=True)

    is_distributed, local_rank = setup_distributed()

    if torch.cuda.is_available():
        if is_distributed:
            device = torch.device(f"cuda:{local_rank}")
            opt.gpu_ids = [local_rank]
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
            device = torch.device("cuda:0")
            opt.gpu_ids = [0]
    else:
        device = torch.device("cpu")
        opt.gpu_ids = []

    opt.device = device

    if is_distributed:
        world_size = dist.get_world_size()
        rank = dist.get_rank()
    else:
        world_size, rank = 1, 0

    is_main_process = (rank == 0)
    if is_main_process:
        print_options(opt)

    np.random.seed(opt.seed)
    random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(opt.seed)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

    train_set = DatasetFromFolder_train(opt)
    train_sampler = DistributedSampler(
        train_set,
        world_size,
        rank,
        shuffle=True,
        drop_last=True
    ) if is_distributed else None

    per_rank_workers = max(4, opt.num_threads // world_size)

    train_dataloader = DataLoader(
        dataset=train_set,
        num_workers=per_rank_workers,
        batch_size=opt.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(per_rank_workers > 0),
        prefetch_factor=2 if per_rank_workers > 0 else None,
        drop_last=True
    )

    model = GANclass(opt).to(device)
    
    if not opt.isTrain or opt.continue_train:
        load_networks(opt, model)
    
    # Wrap sub-networks, NOT the whole GAN controller
    if is_distributed:
        model.netG = DDP(
            model.netG,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False
        )
    
        if opt.isTrain and hasattr(model, "netD"):
            model.netD = DDP(
                model.netD,
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=False
            )

    scaler = GradScaler(enabled=(device.type == "cuda"))

    if is_main_process:
        train_writer = SummaryWriter(os.path.join(opt.results_dir, 'log/train'))
        val_writer = SummaryWriter(os.path.join(opt.results_dir, 'log/val'), flush_secs=2)
    else:
        train_writer = None
        val_writer = None

    best_MAE = 1e9
    best_val_SSIM = 0
    best_val_PSNR = 0

    if is_main_process:
        gpu_monitor = GPUMonitor()
        run_start_time = time.time()
        epoch_times = []
        print("Training started")

    total_iters = 0
    for epoch in range(opt.epoch_count, opt.max_epochs + 1):
        epoch_start_time = time.time()
        if is_distributed and train_sampler is not None:
            train_sampler.set_epoch(epoch)

        train_size = len(train_dataloader)
        opt.print_freq = max(1, int(train_size / opt.print_freq_num))

        for i, data in enumerate(train_dataloader):
            if is_main_process:
                gpu_monitor.update()

            total_iters += 1

            # controller = underlying GAN object (for accessing optimizers, nets, etc.)
            batch = {
                k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                for k, v in data.items()
            }
            
            model.set_input(batch)
            
            with autocast(enabled=(device.type == "cuda")):
                model(epoch, scaler=scaler)
            
            if total_iters % opt.print_freq == 0 and is_main_process:
                losses = get_current_losses(model)
                lr = model.optimizer_G.param_groups[0]['lr']
            
                msg = f"(epoch: {epoch}, iters: {i}/{train_size}, lr: {lr:.6f}) "
                msg += " ".join([f"{k}:{v:.3f}" for k, v in losses.items()])
                print(msg)
            
                if train_writer is not None:
                    train_writer.add_scalar("learning_rate", lr, total_iters)
                    for k, v in losses.items():
                        train_writer.add_scalar(k, v, total_iters)


        update_learning_rate(model, opt.max_epochs, epoch, opt.lr_max)
        
        if is_distributed:
            dist.barrier()
        
        if epoch % 10 == 0:
            netG = model.netG.module if isinstance(model.netG, DDP) else model.netG
            was_training = netG.training
            netG.eval()
        
            local_sum_MAE = 0.0
            local_sum_SSIM = 0.0
            local_sum_PSNR = 0.0
            local_count = 0
        
            image_filenames = sorted(glob.glob(os.path.join(opt.val_dir, '*')))
            if is_distributed:
                val_subjects = image_filenames[rank::world_size]
            else:
                val_subjects = image_filenames
        
            patch_size = opt.ImageSize
            patch_deep = opt.depthSize
        
            with torch.no_grad():
                for local_idx, sub in enumerate(val_subjects):
                    MR, spacing, origin, direction, ext_mr = load_val_volume(sub, "mr")
                    CT, _, _, _, _ = load_val_volume(sub, "ct")
                    MASK, _, _, _, _ = load_val_volume(sub, "mask")
        
                    MR = normalization(MR, 0, 255).astype(np.float32)
                    z, y, x = np.where(MASK > 0)
        
                    if len(z) < 1:
                        continue
        
                    z_edge1 = np.where((z + patch_deep / 2) > MR.shape[0])
                    z[z_edge1] = MR.shape[0] - patch_deep / 2
                    z_edge2 = np.where((z - patch_deep / 2) < 0)
                    z[z_edge2] = patch_deep / 2
        
                    y_edge1 = np.where((y + patch_size / 2) > MR.shape[1])
                    y[y_edge1] = MR.shape[1] - patch_size / 2
                    y_edge2 = np.where((y - patch_size / 2) < 0)
                    y[y_edge2] = patch_size / 2
        
                    x_edge1 = np.where((x + patch_size / 2) > MR.shape[2])
                    x[x_edge1] = MR.shape[2] - patch_size / 2
                    x_edge2 = np.where((x - patch_size / 2) < 0)
                    x[x_edge2] = patch_size / 2
        
                    MR_ch = MR[None, :, :, :]
        
                    output = np.zeros_like(MASK, dtype=np.float32)
                    count_used = np.zeros_like(MASK, dtype=np.float32) + 0.0001
                    dis = opt.disx
        
                    for num in range(0, len(x), dis):
                        if num % dis == 0:
                            deep = z[num]
                            height = y[num]
                            width = x[num]
        
                            z0 = int(deep - patch_deep / 2)
                            z1 = int(deep + patch_deep / 2)
                            y0 = int(height - patch_size / 2)
                            y1 = int(height + patch_size / 2)
                            x0 = int(width - patch_size / 2)
                            x1 = int(width + patch_size / 2)
        
                            X_MR = MR_ch[:, z0:z1, y0:y1, x0:x1]
                            X_MR = torch.from_numpy(X_MR).unsqueeze(0).to(device, dtype=torch.float32)
        
                            with autocast(enabled=(device.type == "cuda")):
                                CT_pred = netG(X_MR)
        
                            CT_pred = np.squeeze(CT_pred.detach().float().cpu().numpy())
                            CT_pred[CT_pred < -1] = -1
                            CT_pred[CT_pred > 1] = 1
        
                            output[z0:z1, y0:y1, x0:x1] += CT_pred
                            count_used[z0:z1, y0:y1, x0:x1] += 1.0
        
                    output = output / count_used
                    output[MASK == 0] = -1
                    output_hu = inverser_norm_ct(output, opt.Max_CT, -1000)
                    output_hu = np.clip(output_hu, -1000, opt.Max_CT)
        
                    ct_fg = CT[MASK > 0].astype(np.float32)
                    pr_fg = output_hu[MASK > 0].astype(np.float32)
        
                    MAE = np.mean(np.abs(pr_fg - ct_fg))
                    data_range = CT.max() - CT.min() if CT.max() > CT.min() else opt.Max_CT + 1000
        
                    SSIM = structural_similarity(
                        CT.astype(np.float32),
                        output_hu.astype(np.float32),
                        data_range=data_range,
                        channel_axis=None,
                    )
                    PSNR = peak_signal_noise_ratio(
                        CT.astype(np.float32),
                        output_hu.astype(np.float32),
                        data_range=data_range,
                    )
        
                    print(
                        f"[rank {rank}] epoch[{epoch}] sub[{local_idx}/{len(val_subjects)}] "
                        f"MAE={MAE:.3f} SSIM={SSIM:.3f} PSNR={PSNR:.3f}"
                    )
        
                    local_sum_MAE += float(MAE)
                    local_sum_SSIM += float(SSIM)
                    local_sum_PSNR += float(PSNR)
                    local_count += 1
        
            netG.train(was_training)
        
            if is_distributed:
                stats = torch.tensor(
                    [local_sum_MAE, local_sum_SSIM, local_sum_PSNR, local_count],
                    device=device,
                    dtype=torch.float64,
                )
                dist.all_reduce(stats, op=dist.ReduceOp.SUM)
                total_sum_MAE, total_sum_SSIM, total_sum_PSNR, total_count = stats.tolist()
            else:
                total_sum_MAE = local_sum_MAE
                total_sum_SSIM = local_sum_SSIM
                total_sum_PSNR = local_sum_PSNR
                total_count = local_count
        
            if is_main_process:
                if total_count > 0:
                    mean_MAE = total_sum_MAE / total_count
                    mean_SSIM = total_sum_SSIM / total_count
                    mean_PSNR = total_sum_PSNR / total_count
                else:
                    mean_MAE = mean_SSIM = mean_PSNR = 0.0
        
                if val_writer:
                    val_writer.add_scalar("MAE", mean_MAE, epoch)
                    val_writer.add_scalar("SSIM", mean_SSIM, epoch)
                    val_writer.add_scalar("PSNR", mean_PSNR, epoch)
        
                save_model = model
        
                if mean_MAE < best_MAE:
                    best_MAE = mean_MAE
                    save_networks(opt, "best_MAE", save_model, epoch)
        
                best_val_SSIM = max(best_val_SSIM, mean_SSIM)
                best_val_PSNR = max(best_val_PSNR, mean_PSNR)
        
                save_networks(opt, "latest", save_model, epoch)
                print(f"[Checkpoint Saved] MAE={mean_MAE:.4f}")
                
        if is_distributed:
            dist.barrier()

        if is_main_process:
            elapsed = time.time() - epoch_start_time
            epoch_times.append(elapsed)
            print(f"End of epoch {epoch}/{opt.max_epochs} | Time: {elapsed:.1f}s")

    if is_main_process:
        if train_writer:
            train_writer.close()
        if val_writer:
            val_writer.close()
        total_time = time.time() - run_start_time
        util_mean, mem_mean = gpu_monitor.summary()
        gpu_monitor.shutdown()

        results_csv = os.path.join(opt.results_dir, "results.csv")
        file_exists = os.path.exists(results_csv)

        with open(results_csv, "a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow([
                    "method", "dataset", "gpus", "batch_size", "total_epochs",
                    "total_time_sec", "avg_epoch_time_sec",
                    "gpu_util_mean", "gpu_mem_mean_gb",
                    "best_val_MAE", "best_val_SSIM", "best_val_PSNR"
                ])

            writer.writerow([
                f"{opt.G_model}+{opt.D_model}",
                opt.image_dir,
                world_size,
                opt.batch_size,
                opt.max_epochs,
                round(total_time, 2),
                round(np.mean(epoch_times), 2),
                round(util_mean, 2),
                round(mem_mean, 2),
                round(best_MAE, 4),
                round(best_val_SSIM, 4),
                round(best_val_PSNR, 4),
            ])

        print("\n=== Training Summary Saved ===")
        print(f"Total time: {total_time/3600:.2f}h")
        print(f"GPU Avg Util: {util_mean:.1f}%")

    if is_distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

