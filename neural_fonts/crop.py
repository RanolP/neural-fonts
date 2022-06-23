import os

import click
import numpy as np
from cv2 import bilateralFilter
from PIL import Image, ImageEnhance

ROWS = 12
COLS = 12
HEADER_RATIO = 16.5 / (16.5 + 42)


def crop_image_uniform(src_dir: str, dst_dir: str):
    f = open("399-uniform.txt", "r")
    os.makedirs(dst_dir, exist_ok=True)

    for page in range(1, 4):
        img = Image.open(f"{src_dir}/{page}-uniform.png").convert("L")

        width, height = img.size
        cell_width = width / float(COLS)
        cell_height = height / float(ROWS)
        header_offset = height / float(ROWS) * HEADER_RATIO
        width_margin = cell_width * 0.10
        height_margin = cell_height * 0.10

        for j in range(0, ROWS):
            for i in range(0, COLS):
                left = i * cell_width
                upper = j * cell_height + header_offset
                right = left + cell_width
                lower = (j + 1) * cell_height

                center_x = (left + right) / 2
                center_y = (upper + lower) / 2

                crop_width = right - left - 2 * width_margin
                crop_height = lower - upper - 2 * height_margin

                size = 0
                if crop_width > crop_height:
                    size = crop_height / 2
                else:
                    size = crop_width / 2

                left = center_x - size
                right = center_x + size
                upper = center_y - size
                lower = center_y + size

                code = f.readline()
                if not code:
                    break
                else:
                    name = dst_dir + "/uni" + code.strip() + ".png"
                    cropped_image = img.crop((left, upper, right, lower))
                    cropped_image = cropped_image.resize((128, 128), Image.LANCZOS)
                    # Increase constrast
                    enhancer = ImageEnhance.Contrast(cropped_image)
                    cropped_image = enhancer.enhance(1.5)
                    opencv_image = np.array(cropped_image)
                    opencv_image = bilateralFilter(opencv_image, 9, 30, 30)
                    cropped_image = Image.fromarray(opencv_image)
                    cropped_image.save(name)
        print("Processed uniform page " + str(page))


@click.command()
@click.option(
    "--src-dir",
    type=click.Path(),
    required=True,
    help="directory to read scanned images",
)
@click.option(
    "--dst-dir",
    type=click.Path(),
    required=True,
    help="directory to save character images",
)
def main(src_dir: str, dst_dir: str):
    """
    Crop scanned images to character images
    """
    crop_image_uniform(src_dir, dst_dir)
    # crop_image_frequency(args.src_dir, args.dst_dir)
