import os
import argparse
import numpy as np
import torch
import glob
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
from utils import *

# Environment setup
os.environ['PYTHONHASHSEED'] = '8'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--test_dir", type=str,
                        default="../../../2023/Task1/brain/test")

    parser.add_argument("--checkpoint_path", type=str,
                        default="results/model_checkpoints/best_MAE.pth")

    parser.add_argument("--save_dir", type=str,
                        default="inference_latest")

    parser.add_argument("--G_model", type=str,
                        default="Med2Transformer")

    parser.add_argument("--input_nc", type=int, default=1)
    parser.add_argument("--output_nc", type=int, default=1)
    parser.add_argument("--ngf", type=int, default=16)
    parser.add_argument("--G_norm", type=str, default="instance")

    parser.add_argument("--depthSize", type=int, default=48)
    parser.add_argument("--ImageSize", type=int, default=128)

    parser.add_argument("--Max_CT", type=int, default=2000)
    parser.add_argument("--disx", type=int, default=10120)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build Generator 
    norm_layer = get_norm_layer(args.G_norm)

    if args.G_model == 'Med2Transformer':
        netG = Med2Transformer(
            input_nc=args.input_nc,
            output_nc=args.output_nc,
            ngf=args.ngf,
            norm_layer=norm_layer,
            resolution=[args.depthSize, args.ImageSize, args.ImageSize]
        )
    else:
        raise NotImplementedError("Only Med2Transformer supported")


    # Load checkpoint
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    netG.load_state_dict(checkpoint['netG_state_dict'], strict=True)

    netG = netG.to(device)
    netG.eval()

    print(f"Loaded generator from {args.checkpoint_path} on {device}")

    os.makedirs(args.save_dir, exist_ok=True)

    subjects = sorted(glob.glob(os.path.join(args.test_dir, "*")))

    patch_size = args.ImageSize
    patch_deep = args.depthSize

    for idx, sub in enumerate(subjects, 1):

        mr_path = os.path.join(sub, "mr.nii.gz")
        mask_path = os.path.join(sub, "mask.nii.gz")

        if not os.path.exists(mr_path) or not os.path.exists(mask_path):
            print(f"Skipping {sub} (missing MR or MASK)")
            continue

        # Load data
        MR, spacing, origin, direction = NiiDataRead(mr_path)
        MASK, _, _, _ = NiiDataRead(mask_path)

        MR = normalization(MR, 0, 255).astype(np.float32)

        z, y, x = np.where(MASK > 0)

        if len(z) == 0:
            print(f"Skipping {sub} (empty mask)")
            continue

        z = z.astype(np.float32)
        y = y.astype(np.float32)
        x = x.astype(np.float32)

        z[z + patch_deep/2 > MR.shape[0]] = MR.shape[0] - patch_deep/2
        z[z - patch_deep/2 < 0] = patch_deep/2

        y[y + patch_size/2 > MR.shape[1]] = MR.shape[1] - patch_size/2
        y[y - patch_size/2 < 0] = patch_size/2

        x[x + patch_size/2 > MR.shape[2]] = MR.shape[2] - patch_size/2
        x[x - patch_size/2 < 0] = patch_size/2

        MR_ch = MR[None, :, :, :]

        output = np.zeros_like(MR, dtype=np.float32)
        count_map = np.zeros_like(MR, dtype=np.float32) + 1e-4

        dis = args.disx

        with torch.no_grad():
            for num in range(0, len(x), dis):

                deep = int(z[num])
                height = int(y[num])
                width = int(x[num])

                z0 = int(deep - patch_deep/2)
                z1 = int(deep + patch_deep/2)
                y0 = int(height - patch_size/2)
                y1 = int(height + patch_size/2)
                x0 = int(width - patch_size/2)
                x1 = int(width + patch_size/2)

                patch = MR_ch[:, z0:z1, y0:y1, x0:x1]
                patch = torch.from_numpy(patch).unsqueeze(0).to(device, dtype=torch.float32)

                # CORRECT forward
                ct_pred = netG(patch)

                ct_pred = ct_pred.squeeze().cpu().numpy()
                ct_pred = np.clip(ct_pred, -1, 1)

                output[z0:z1, y0:y1, x0:x1] += ct_pred
                count_map[z0:z1, y0:y1, x0:x1] += 1.0

        # Average overlapping patches
        output = output / count_map

        # Mask outside
        output[MASK == 0] = -1

        # Convert to HU
        output_hu = inverser_norm_ct(output, args.Max_CT, -1000)
        output_hu = np.clip(output_hu, -1000, args.Max_CT)

        # Save
        base_name = os.path.basename(sub)
        save_path = os.path.join(args.save_dir, f"{base_name}_predCT.nii.gz")
        NiiDataWrite(save_path, output_hu, spacing, origin, direction)

        print(f"[{idx}/{len(subjects)}] Saved: {save_path}")

        gt_path = os.path.join(sub, "ct.nii.gz")

        if os.path.exists(gt_path):
            CT, _, _, _ = NiiDataRead(gt_path)

            MAE = np.mean(np.abs(output_hu[MASK > 0] - CT[MASK > 0]))

            data_range = CT.max() - CT.min() if CT.max() > CT.min() else args.Max_CT + 1000

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

            print(f"  Metrics: MAE={MAE:.3f}, SSIM={SSIM:.3f}, PSNR={PSNR:.3f}")

    print("Inference complete.")


if __name__ == '__main__':
    main()