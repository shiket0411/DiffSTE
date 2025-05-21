import os

from einops import rearrange
import numpy as np
import torch
from PIL import Image
from omegaconf import OmegaConf
from argparse import ArgumentParser
from torchvision.utils import make_grid
from pytorch_lightning import seed_everything
from torchvision.transforms import ToPILImage, ToTensor
from src.trainers import CharInpaintTrainer
from src.dataset import prepare_style_chars
from src.dataset.utils import prepare_npy_image_mask, normalize_image


def create_parser():
    parser = ArgumentParser()
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--in_image", type=str, required=True)
    parser.add_argument("--in_mask", type=str, required=True)
    parser.add_argument("--out_dir", default="output")
    parser.add_argument("--text", type=str)
    parser.add_argument("--font", type=str, default="")
    parser.add_argument("--color", type=str, default="")
    parser.add_argument("--instruction", type=str)
    parser.add_argument("--num_inference_steps", default=30, type=int)
    parser.add_argument("--num_sample_per_image", default=3, type=int)
    parser.add_argument("--guidance_scale", default=7.5, type=float)
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--return_intermediates", action="store_true")
    return parser


def main(opt):
    model = CharInpaintTrainer.load_from_checkpoint(opt.ckpt_path)
    device = "cpu" if opt.no_cuda else "cuda"
    model = model.to(device)

    image = Image.open(opt.in_image)
    mask = Image.open(opt.in_mask).convert("1")
    raw_image, mask, masked_image, mask_coordinate = prepare_npy_image_mask(
        image, mask
    )

    if opt.instruction is not None:
        style = opt.instruction
        char = opt.text
    else:
        char = opt.text
        color = opt.color
        font = opt.font
        style = prepare_style_chars(char, [font, color])
    
    torch.manual_seed(opt.seed)
    batch = {
        "image": torch.from_numpy(raw_image).unsqueeze(0).to(device),
        "mask": torch.from_numpy(mask).unsqueeze(0).to(device),
        "masked_image": torch.from_numpy(masked_image).unsqueeze(0).to(device),
        "coordinate": [mask_coordinate],
        "chars": [char],
        "style": [style],
    }

    generation_kwargs = {
        "num_inference_steps": opt.num_inference_steps,
        "num_sample_per_image": opt.num_sample_per_image,
        "guidance_scale": opt.guidance_scale,
        "generator": torch.Generator(model.device).manual_seed(opt.seed),
        "return_intermediates": opt.return_intermediates
    }

    with torch.no_grad():
        results, intermediate_images = model.log_images(batch, generation_kwargs)

    outpath = opt.out_dir
    os.makedirs(outpath, exist_ok=True)

    process_grid_path = os.path.join(outpath, "process")
    os.makedirs(process_grid_path, exist_ok=True)
    process_grid_count = len(os.listdir(process_grid_path))
    
    keys = results.keys()
    for i, k in enumerate(keys):
        img = torch.cat([
            ((batch["image"][i:i+1].cpu()) / 2. + 0.5).clamp(0., 1.),
            ((batch["masked_image"][i:i+1].cpu()) / 2. + 0.5).clamp(0., 1.),
            results[k]
        ])
        grid = make_grid(img, nrow=5, padding=1)
        ToPILImage()(grid).save(
            f"{outpath}/{k}-grid-step-{opt.num_inference_steps}.png"
        )

    # 拡散過程を一つの画像にして保存
    entire_grid = torch.stack(intermediate_images, 0)

    for i in range(opt.num_sample_per_image):
        
        # process_grid = rearrange(entire_grid, 'n b c h w -> (n b) c h w')
        process_grid = entire_grid[:, i]
        process_grid = make_grid(process_grid, nrow=5)

        # to image
        process_grid = 255. * rearrange(process_grid, 'c h w -> h w c').cpu().numpy()
        img = Image.fromarray(process_grid.astype(np.uint8))
        img.save(os.path.join(process_grid_path, f'process_grid-{process_grid_count:04}-step-{opt.num_inference_steps}.png'))
        process_grid_count += 1


if __name__ == "__main__":
    parser = create_parser()
    opt = parser.parse_args()
    main(opt)