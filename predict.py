""" Photo inpainting using Cog """
# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

import tempfile
from typing import Optional

from cog import BaseModel, BasePredictor, Input, Path

"""Generate images using pretrained network pickle."""
import glob
import os
import random
import re
from typing import List, Optional

import click
import cv2
import numpy as np
import PIL.Image
import pyspng
import torch
import torch.nn.functional as F
from PIL import Image

import dnnlib
import legacy
from datasets.mask_generator_512 import RandomMask
from generate_image import *
from networks.mat import Generator


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""

        self.device = torch.device("cuda")

        places_model = "./pretrained/Places_512_FullData.pkl"
        celeba_model = "./pretrained/CelebA-HQ_512.pkl"
        resolution = 512

        print(f"Loading Places network")
        with dnnlib.util.open_url(places_model) as f:
            self.G_saved_places = legacy.load_network_pkl(f)["G_ema"].to(self.device).eval().requires_grad_(False)  # type: ignore
        self.G_places = (
            Generator(
                z_dim=512, c_dim=0, w_dim=512, img_resolution=resolution, img_channels=3
            )
            .to(self.device)
            .eval()
            .requires_grad_(False)
        )
        copy_params_and_buffers(self.G_saved_places, self.G_places, require_all=True)

        print(f"Loading CelebA network")
        with dnnlib.util.open_url(celeba_model) as f:
            self.G_saved_celeba = legacy.load_network_pkl(f)["G_ema"].to(self.device).eval().requires_grad_(False)  # type: ignore
        self.G_celeba = (
            Generator(
                z_dim=512, c_dim=0, w_dim=512, img_resolution=resolution, img_channels=3
            )
            .to(self.device)
            .eval()
            .requires_grad_(False)
        )
        copy_params_and_buffers(self.G_saved_celeba, self.G_celeba, require_all=True)

    def predict(
        self,
        image: Path = Input(
            description="Input image to inpaint; must be 512x512 in size. You can crop your image to size here: https://www.iloveimg.com/crop-image"
        ),
        mask: Path = Input(
            description="Optional mask (also 512x512) that overlays image. Should be black in areas that you wish to inpaint. If left blank, a random mask will be generated over the image.",
            default=None,
        ),
        model: str = Input(
            description="Select which model to use: Places or CelebA-HQ-512",
            choices=["places", "celeba"],
            default="places",
        ),
        truncation_psi: float = Input(
            description="Truncation psi. Improve image quality at the cost of output diversity/variation; truncation psi ψ = 1 means no truncation, ψ = 0 means no diversity.",
            default=1,
        ),
        noise_mode: str = Input(
            description="Noise mode",
            choices=["const", "random", "none"],
            default="const",
        ),
        seed: int = Input(
            default=-1,
            description="Seed for random number generator to encourage diverse results. If -1, a random seed will be chosen. (minimum: -1; maximum: 4294967295)",
            ge=-1,
            le=(2**32 - 1),
        ),
    ) -> Path:
        """Run a single prediction on the model"""

        dpath = str(image)
        if mask:
            mpath = str(mask)
        else:
            mpath = None

        # set other args
        model = str(model)
        noise_mode = str(noise_mode)
        outdir = "./"
        resolution = 512
        device = self.device

        # set seed
        seed = int(seed)
        if seed == -1:
            seed = random.randint(0, 2**32)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        print(f"Using seed {seed}......")

        print(f"Loading data from: {dpath}")
        img_list = [dpath]

        if mpath is not None:
            print(f"Loading mask from: {mpath}")
            mask_list = [mpath]
            assert len(img_list) == len(mask_list), "illegal mapping"

        os.makedirs(outdir, exist_ok=True)

        print("Setting loaded models")
        # choose loaded model
        if model == "places":
            print("Using Places Model....")
            G = self.G_places
        else:
            assert model == "celeba"
            print("Using CelebA model.....")
            G = self.G_celeba

        label = torch.zeros([1, G.c_dim], device=device)

        def read_image(image_path):
            with open(image_path, "rb") as f:
                if pyspng is not None and image_path.endswith(".png"):
                    image = pyspng.load(f.read())
                else:
                    image = np.array(PIL.Image.open(f))
            if image.ndim == 2:
                image = image[:, :, np.newaxis]  # HW => HWC
                image = np.repeat(image, 3, axis=2)
            image = image.transpose(2, 0, 1)  # HWC => CHW
            image = image[:3]
            return image

        def to_image(image, lo, hi):
            image = np.asarray(image, dtype=np.float32)
            image = (image - lo) * (255 / (hi - lo))
            image = np.rint(image).clip(0, 255).astype(np.uint8)
            image = np.transpose(image, (1, 2, 0))
            if image.shape[2] == 1:
                image = np.repeat(image, 3, axis=2)
            return image

        if resolution != 512:
            noise_mode = "random"

        print("Performing model inference")
        with torch.no_grad():
            for i, ipath in enumerate(img_list):
                iname = os.path.basename(ipath).replace(".jpg", ".png")
                print(f"Processing: {iname}")
                image = read_image(ipath)
                image = (
                    torch.from_numpy(image).float().to(device) / 127.5 - 1
                ).unsqueeze(0)

                if mpath is not None:
                    mask = (
                        cv2.imread(mask_list[i], cv2.IMREAD_GRAYSCALE).astype(
                            np.float32
                        )
                        / 255.0
                    )
                    mask = (
                        torch.from_numpy(mask)
                        .float()
                        .to(device)
                        .unsqueeze(0)
                        .unsqueeze(0)
                    )
                else:
                    mask = RandomMask(
                        resolution
                    )  # adjust the masking ratio by using 'hole_range'
                    mask = torch.from_numpy(mask).float().to(device).unsqueeze(0)

                z = torch.from_numpy(np.random.randn(1, G.z_dim)).to(device)
                output = G(
                    image,
                    mask,
                    z,
                    label,
                    truncation_psi=truncation_psi,
                    noise_mode=noise_mode,
                )
                output = (
                    (output.permute(0, 2, 3, 1) * 127.5 + 127.5)
                    .round()
                    .clamp(0, 255)
                    .to(torch.uint8)
                )
                output = output[0].cpu().numpy()

                print("Saving output image")
                output_path = Path(tempfile.mkdtemp()) / f"output.png"
                PIL.Image.fromarray(output, "RGB").save(str(output_path))

                return output_path
