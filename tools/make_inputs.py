#!/usr/bin/env python3
"""Prepare the images used by the experiments. Everything is written under data/.

Subcommands
  inputs    the two fixed inputs of the scaling experiments:
              crop1080x800  1080x800 (width x height) centre crop of DIV2K validation 0801
              full2k        the complete image 0801
            Without DIV2K (--hr missing) a synthetic stand-in of the same sizes is built by
            mirror-tiling a bundled test image, so the pipeline still runs offline; the
            manifest records which source was used. --quick writes a second manifest that
            points at two bundled test images instead.
  bicubic   x4 "super-resolution" outputs of a directory of HR images: the image is cropped
            (top-left) to a multiple of 4, reduced by 4 and enlarged by 4 again.
            Resampler: Pillow Image.resize(..., Image.BICUBIC) in both directions, applied
            to 8-bit RGB (Keys cubic kernel, a = -0.5, with Pillow's antialiasing support
            scaling when reducing). This is NOT MATLAB's imresize, so scores differ slightly
            from numbers published with MATLAB-generated bicubic images.
  extract   unpack a .zip / .tar(.gz) archive safely (no paths outside the target),
            optionally only the first N image members in name order.
  collect   copy the images of every directory of an unpacked tree into
            data/hr/<prefix>_<dir> or data/sr/<prefix>_<dir>/<method>.

The outputs are deterministic for a given Pillow version, which is recorded in the manifests.
"""
import argparse
import json
import os
import re
import shutil
import sys
import tarfile
import zipfile

from PIL import Image
import PIL

IMAGE_EXT = (".png", ".bmp", ".jpg", ".jpeg")
CROP_W, CROP_H = 1080, 800
FULL_W, FULL_H = 2040, 1356  # size of the synthetic stand-in for a 2K image


def is_image(name):
    return name.lower().endswith(IMAGE_EXT) and not os.path.basename(name).startswith(".")


def list_images(folder):
    if not os.path.isdir(folder):
        return []
    return sorted(os.path.join(folder, f) for f in os.listdir(folder) if is_image(f))


def centre_crop(im, w, h):
    if im.width < w or im.height < h:
        raise SystemExit(f"image {im.width}x{im.height} is smaller than the {w}x{h} crop")
    left, top = (im.width - w) // 2, (im.height - h) // 2
    return im.crop((left, top, left + w, top + h))


def mirror_tile(im, w, h):
    """Fill a w x h canvas with reflected copies of im (no seams, deterministic)."""
    flips = [[im, im.transpose(Image.FLIP_LEFT_RIGHT)],
             [im.transpose(Image.FLIP_TOP_BOTTOM), im.transpose(Image.ROTATE_180)]]
    out = Image.new("RGB", (w, h))
    for j in range(0, (h + im.height - 1) // im.height):
        for i in range(0, (w + im.width - 1) // im.width):
            out.paste(flips[j % 2][i % 2], (i * im.width, j * im.height))
    return out


def entry(ident, path, source):
    with Image.open(path) as im:
        return {"id": ident, "path": os.path.abspath(path), "width": im.width, "height": im.height, "source": source}


def cmd_inputs(a):
    os.makedirs(a.out, exist_ok=True)
    if a.quick:
        small = os.path.join(a.fallback_dir, "coffee_rgb.png")
        large = os.path.join(a.fallback_dir, "astronaut_rgb.png")
        manifest = {"quick": True, "pillow": PIL.__version__,
                    "inputs": [entry("quick_small", small, "bundled test image"),
                               entry("quick_large", large, "bundled test image")]}
        path = os.path.join(a.out, "manifest_quick.json")
    else:
        crop_path = os.path.join(a.out, "crop1080x800.png")
        full_path = os.path.join(a.out, "full2k.png")
        if a.hr and os.path.isfile(a.hr):
            full = Image.open(a.hr).convert("RGB")
            source = f"DIV2K validation HR {os.path.basename(a.hr)}"
        else:
            seed = os.path.join(a.fallback_dir, "astronaut_rgb.png")
            full = mirror_tile(Image.open(seed).convert("RGB"), FULL_W, FULL_H)
            source = "synthetic stand-in: mirror-tiled bundled image astronaut_rgb.png (DIV2K not available)"
        full.save(full_path)
        centre_crop(full, CROP_W, CROP_H).save(crop_path)
        manifest = {"quick": False, "pillow": PIL.__version__,
                    "inputs": [entry("crop1080x800", crop_path, source + ", centre crop"),
                               entry("full2k", full_path, source)]}
        path = os.path.join(a.out, "manifest.json")
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)
    for e in manifest["inputs"]:
        print(f"input {e['id']}: {e['width']}x{e['height']}  {e['path']}  [{e['source']}]")


def bicubic_x4(src, dst, factor=4):
    im = Image.open(src).convert("RGB")
    w, h = im.width // factor * factor, im.height // factor * factor
    if w < factor or h < factor:
        raise ValueError("image too small")
    im = im.crop((0, 0, w, h))
    small = im.resize((w // factor, h // factor), Image.BICUBIC)
    small.resize((w, h), Image.BICUBIC).save(dst)


def cmd_bicubic(a):
    images = list_images(a.src)
    if a.exclude:
        images = [p for p in images if not re.search(a.exclude, os.path.basename(p))]
    if a.limit:
        images = images[:a.limit]
    os.makedirs(a.dst, exist_ok=True)
    made = 0
    for src in images:
        dst = os.path.join(a.dst, os.path.splitext(os.path.basename(src))[0] + ".png")
        if os.path.isfile(dst) and os.path.getsize(dst) > 0:
            continue
        try:
            bicubic_x4(src, dst + ".tmp.png")
            os.replace(dst + ".tmp.png", dst)
            made += 1
        except (OSError, ValueError) as e:
            print(f"skipped {src}: {e}", file=sys.stderr)
    with open(os.path.join(a.dst, "README.txt"), "w") as f:
        f.write("x4 bicubic outputs generated by tools/make_inputs.py bicubic:\n"
                "crop to a multiple of 4 (top-left), Pillow Image.BICUBIC down by 4, Pillow Image.BICUBIC up by 4,\n"
                f"8-bit RGB, Pillow {PIL.__version__}. Not MATLAB imresize.\nsource: {os.path.abspath(a.src)}\n")
    print(f"bicubic x4: {len(images)} image(s) in {a.dst} ({made} new)")


def safe_target(dest, name):
    target = os.path.realpath(os.path.join(dest, name))
    if not target.startswith(os.path.realpath(dest) + os.sep):
        raise SystemExit(f"archive member escapes the target directory: {name}")
    return target


def cmd_extract(a):
    os.makedirs(a.dest, exist_ok=True)
    count = 0
    if zipfile.is_zipfile(a.archive):
        with zipfile.ZipFile(a.archive) as z:
            names = sorted(n for n in z.namelist() if not n.endswith("/"))
            if a.images_only:
                names = [n for n in names if is_image(n)]
            if a.first:
                names = names[:a.first]
            for n in names:
                target = safe_target(a.dest, os.path.basename(n) if a.flat else n)
                os.makedirs(os.path.dirname(target), exist_ok=True)
                with z.open(n) as src, open(target, "wb") as out:  # CRC is verified while reading
                    shutil.copyfileobj(src, out)
                count += 1
    elif tarfile.is_tarfile(a.archive):
        with tarfile.open(a.archive) as t:
            members = sorted((m for m in t.getmembers() if m.isfile()), key=lambda m: m.name)
            if a.images_only:
                members = [m for m in members if is_image(m.name)]
            if a.first:
                members = members[:a.first]
            for m in members:
                target = safe_target(a.dest, os.path.basename(m.name) if a.flat else m.name)
                os.makedirs(os.path.dirname(target), exist_ok=True)
                with t.extractfile(m) as src, open(target, "wb") as out:
                    shutil.copyfileobj(src, out)
                count += 1
    else:
        raise SystemExit(f"{a.archive}: not a zip or tar archive")
    if count < a.expect:
        raise SystemExit(f"{a.archive}: extracted {count} file(s), expected at least {a.expect}")
    print(f"extracted {count} file(s) into {a.dest}")


def slug(text):
    return re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_").lower()


def cmd_collect(a):
    """Every directory below --src that directly holds images becomes one image set."""
    found = 0
    for root, _dirs, files in sorted(os.walk(a.src)):
        images = sorted(f for f in files if is_image(f))
        if a.match:
            images = [f for f in images if re.search(a.match, f)]
        rel = os.path.relpath(root, a.src)
        if not images or (a.skip_dirs and re.search(a.skip_dirs, rel, re.I)):
            continue
        name = "_".join(x for x in (a.prefix, slug("" if rel == "." else rel)) if x)
        if a.kind == "hr":
            dest = os.path.join(a.data, "hr", name)
        elif a.method:
            dest = os.path.join(a.data, "sr", name, a.method)
        else:  # the directory name is the method, the prefix is the set
            dest = os.path.join(a.data, "sr", a.prefix, slug(rel) or "published")
        os.makedirs(dest, exist_ok=True)
        for f in images:
            target = os.path.join(dest, f)
            if not os.path.isfile(target):
                shutil.copyfile(os.path.join(root, f), target)
        print(f"collected {len(images)} image(s): {dest}")
        found += 1
    if not found:
        raise SystemExit(f"no images found below {a.src}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("inputs")
    p.add_argument("--hr", help="DIV2K validation image 0801.png")
    p.add_argument("--fallback-dir", default="tests/images")
    p.add_argument("--out", default="data/inputs")
    p.add_argument("--quick", action="store_true")
    p.set_defaults(func=cmd_inputs)

    p = sub.add_parser("bicubic")
    p.add_argument("--src", required=True)
    p.add_argument("--dst", required=True)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--exclude", help="regular expression on the file name")
    p.set_defaults(func=cmd_bicubic)

    p = sub.add_parser("extract")
    p.add_argument("archive")
    p.add_argument("dest")
    p.add_argument("--first", type=int, default=0)
    p.add_argument("--flat", action="store_true", help="drop the directories stored in the archive")
    p.add_argument("--images-only", action="store_true")
    p.add_argument("--expect", type=int, default=1, help="minimum number of extracted files")
    p.set_defaults(func=cmd_extract)

    p = sub.add_parser("collect")
    p.add_argument("--src", required=True)
    p.add_argument("--data", default="data")
    p.add_argument("--kind", choices=("hr", "sr"), required=True)
    p.add_argument("--prefix", default="")
    p.add_argument("--method", help="sr: method name (default: the directory name)")
    p.add_argument("--match", help="regular expression a file name must match")
    p.add_argument("--skip-dirs", help="regular expression of directories to ignore")
    p.set_defaults(func=cmd_collect)

    a = ap.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
