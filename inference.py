# +
import os
import glob
import argparse
import numpy as np
import torch
from tqdm import tqdm
from torch.cuda.amp import autocast

from utils import (
    SUPPORTED_EXT,
    load_volume,
    normalization,
    inverser_norm_ct,
    NiiDataWrite,
    GANclass,
    load_networks,
)

def build_options():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_dir", required=True, help="dataset root (each subject folder)")
    parser.add_argument("--output_dir", required=True, help="where predictions saved")

    parser.add_argument("--checkpoint", type=str, default=None,
                        help="single checkpoint file (.pth)")
    parser.add_argument("--checkpoint_dir", type=str, default=None,
                        help="directory containing checkpoints")
    parser.add_argument("--checkpoints", nargs="+", default=["latest"],
                        help="checkpoint names inside checkpoint_dir")
    parser.add_argument("--gpu", default="0")
    
    parser.add_argument("--input_nc", type=int, default=1)
    parser.add_argument("--output_nc", type=int, default=1)
    parser.add_argument("--G_model", default="Med2Transformer")
    parser.add_argument("--D_model", default="wave3DDiscriminator")
    parser.add_argument("--ngf", type=int, default=16)
    parser.add_argument("--ndf", type=int, default=16)
    parser.add_argument("--n_layers_D", type=int, default=2)
    parser.add_argument("--G_norm", default="instance")
    parser.add_argument("--D_norm", default="instance")
    parser.add_argument("--init_type", default="normal")
    parser.add_argument("--init_gain", type=float, default=0.02)
    parser.add_argument("--depthSize", type=int, default=48)
    parser.add_argument("--ImageSize", type=int, default=128)
    parser.add_argument("--stride", type=int, default=10000)
    parser.add_argument("--lambda_L1", type=float, default=20)
    parser.add_argument("--VGG_loss", action="store_false")
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument("--no_dropout", action="store_true")
    parser.add_argument("--loss_pre_dir", type=str, default="weight/vgg19-dcbb9e9d.pth")

    parser.add_argument("--Max_CT", type=int, default=2000)

    opt = parser.parse_args()

    opt.isTrain = False
    opt.continue_train = False

    # device
    if torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
        opt.device = torch.device("cuda:0")
        opt.gpu_ids = [0]
    else:
        opt.device = torch.device("cpu")
        opt.gpu_ids = []

    return opt


def load_case(folder, name):
    for ext in SUPPORTED_EXT:
        f = os.path.join(folder, f"{name}.{ext}")
        if os.path.exists(f):
            vol, spacing, origin, direction = load_volume(f)
            return vol, spacing, origin, direction, ext
    raise FileNotFoundError(f"{name} not found in {folder}")


def load_generator_from_ckpt(opt, ckpt_path):
    model = GANclass(opt).to(opt.device)

    # trick loader: reuse existing function
    opt.model_results = os.path.dirname(ckpt_path)
    opt.load_name = os.path.splitext(os.path.basename(ckpt_path))[0]

    load_networks(opt, model)

    netG = model.netG.eval()
    return netG

@torch.no_grad()
def run_subject(netG, opt, subject_path, save_dir):

    pid = os.path.basename(subject_path)

    MR, spacing, origin, direction, ext = load_case(subject_path, "mr")
    MASK, _, _, _, _ = load_case(subject_path, "mask")

    MR = normalization(MR, 0, 255).astype(np.float32)

    patch_d = opt.depthSize
    patch_h = opt.ImageSize
    stride = opt.stride

    z, y, x = np.where(MASK > 0)
    if len(z) == 0:
        return

    z = z.copy()
    y = y.copy()
    x = x.copy()

    z[z + patch_d // 2 > MR.shape[0]] = MR.shape[0] - patch_d // 2
    z[z - patch_d // 2 < 0] = patch_d // 2

    y[y + patch_h // 2 > MR.shape[1]] = MR.shape[1] - patch_h // 2
    y[y - patch_h // 2 < 0] = patch_h // 2

    x[x + patch_h // 2 > MR.shape[2]] = MR.shape[2] - patch_h // 2
    x[x - patch_h // 2 < 0] = patch_h // 2

    MR_ch = MR[None]

    output = np.zeros_like(MASK, np.float32)
    count = np.zeros_like(MASK, np.float32) + 1e-4

    for idx in range(0, len(x), stride):

        zz, yy, xx = int(z[idx]), int(y[idx]), int(x[idx])

        z0 = zz - patch_d // 2
        z1 = zz + patch_d // 2
        y0 = yy - patch_h // 2
        y1 = yy + patch_h // 2
        x0 = xx - patch_h // 2
        x1 = xx + patch_h // 2

        patch = MR_ch[:, z0:z1, y0:y1, x0:x1]

        if patch.shape[1:] != (patch_d, patch_h, patch_h):
            continue

        inp = torch.from_numpy(patch).unsqueeze(0).float().to(opt.device)

        with autocast(enabled=(opt.device.type == "cuda")):
            pred = netG(inp)

        pred = pred.squeeze().cpu().numpy()
        pred = np.clip(pred, -1, 1)

        output[z0:z1, y0:y1, x0:x1] += pred
        count[z0:z1, y0:y1, x0:x1] += 1

    output = output / count
    output[MASK == 0] = -1

    output_hu = inverser_norm_ct(output, opt.Max_CT, -1000)

    save_path = os.path.join(save_dir, f"{pid}.{ext}")
    NiiDataWrite(save_path, output_hu, spacing, origin, direction)


def run_dataset(opt, netG, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    subjects = sorted(glob.glob(os.path.join(opt.input_dir, "*")))
    for s in tqdm(subjects):
        run_subject(netG, opt, s, save_dir)


def main():

    opt = build_options()

    if opt.checkpoint:
        ckpts = [opt.checkpoint]
    else:
        ckpts = [os.path.join(opt.checkpoint_dir, f"{n}.pth") for n in opt.checkpoints]

    for ckpt in ckpts:

        print(f"\nRunning checkpoint: {ckpt}")

        netG = load_generator_from_ckpt(opt, ckpt)

        name = os.path.splitext(os.path.basename(ckpt))[0]
        save_dir = os.path.join(opt.output_dir, name)

        run_dataset(opt, netG, save_dir)

    print("\nInference finished.")


if __name__ == "__main__":
    main()

