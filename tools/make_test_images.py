#!/usr/bin/env python3
"""Build test images in the exact form PIRM's calc_scores feeds to the metrics:
Y channel of MATLAB rgb2ycbcr (uint8), with `shave` pixels removed from every border.

Sources: peppers.png from the official sr-metric release and a few scikit-image
sample images (different sizes, including dimensions with large prime factors),
plus two bicubic x4 down/up versions that look like weak super-resolution outputs.
"""
import argparse
import os

import numpy as np
from PIL import Image
from skimage import data


def matlab_rgb2y_uint8(rgb):
    """MATLAB rgb2ycbcr for uint8 input, Y channel: round(16 + (65.481 R + 128.553 G + 24.966 B) / 255)."""
    rgb = rgb.astype(np.float64)
    y = 65.481 / 255 * rgb[..., 0] + 128.553 / 255 * rgb[..., 1] + 24.966 / 255 * rgb[..., 2] + 16
    y = np.floor(y + 0.5)  # MATLAB round: half away from zero (values are positive)
    return np.clip(y, 0, 255).astype(np.uint8)


def bicubic_down_up(rgb, factor=4):
    im = Image.fromarray(rgb)
    w, h = im.size
    small = im.resize((w // factor, h // factor), Image.BICUBIC)
    return np.asarray(small.resize((w // factor * factor, h // factor * factor), Image.BICUBIC))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--peppers", required=True)
    ap.add_argument("--out", default="tests/images")
    ap.add_argument("--shave", type=int, default=4)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    sources = {
        "peppers": np.asarray(Image.open(a.peppers).convert("RGB")),
        "astronaut": data.astronaut(),
        "coffee": data.coffee(),
        "chelsea": data.chelsea(),
        "rocket": data.rocket(),
    }
    sources["coffee_x4bicubic"] = bicubic_down_up(sources["coffee"])
    sources["astronaut_x4bicubic"] = bicubic_down_up(sources["astronaut"])

    s = a.shave
    for name, rgb in sources.items():
        rgb_path = os.path.join(a.out, f"{name}_rgb.png")
        Image.fromarray(rgb).save(rgb_path)
        y = matlab_rgb2y_uint8(rgb)[s:-s, s:-s]
        Image.fromarray(y).save(os.path.join(a.out, f"{name}_y.png"))
        print(f"{name}: rgb {rgb.shape[:2]} -> y {y.shape}")


if __name__ == "__main__":
    main()
