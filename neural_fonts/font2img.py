import collections
import os
import click

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from typing import TYPE_CHECKING

KR_CHARSET = None

if TYPE_CHECKING:
    from typing import Tuple, Literal, TypeVar


def get_offset(
    ch: str, font: ImageFont.FreeTypeFont, canvas_size: int
) -> Tuple[float, float]:
    font_size = font.getsize(ch)
    font_offset = font.getoffset(ch)
    offset_x = canvas_size / 2 - font_size[0] / 2 - font_offset[0] / 2
    offset_y = canvas_size / 2 - font_size[1] / 2 - font_offset[1] / 2
    return (offset_x, offset_y)


def draw_single_char(
    ch: str,
    font: ImageFont.FreeTypeFont,
    canvas_size: int,
    x_offset: int,
    y_offset: int,
) -> Image.Image:
    """
    Draw a single character using provided font.
    Canvas size and x, y offset could be adjusted.
    """
    # Make an image in grayscale mode
    img = Image.new("RGB", (canvas_size, canvas_size), (255, 255, 255)).convert("L")

    # Draw text on the image
    draw = ImageDraw.Draw(img)
    draw.text((x_offset, y_offset), ch, 0, font=font)

    return img


def draw_example(
    ch: str,
    src_font: ImageFont.FreeTypeFont,
    dst_font: ImageFont.FreeTypeFont,
    canvas_size: int,
    src_offset: Tuple[int, int],
    dst_offset,
    filter_hashes: set,
):
    dst_img = draw_single_char(ch, dst_font, canvas_size, dst_offset[0], dst_offset[1])
    # check the filter example in the hashes or not
    dst_hash = hash(dst_img.tobytes())
    if dst_hash in filter_hashes:
        return None
    src_img = draw_single_char(ch, src_font, canvas_size, src_offset[0], src_offset[1])
    example_img = Image.new(
        "RGB", (canvas_size * 2, canvas_size), (255, 255, 255)
    ).convert("L")
    example_img.paste(dst_img, (0, 0))
    example_img.paste(src_img, (canvas_size, 0))
    return example_img


def get_font_offset(
    charset: list[str],
    font: ImageFont.FreeTypeFont,
    canvas_size: int,
    filter_hashes: set[int],
) -> np.ndarray:
    # Copy the characters and sample 2000 of them
    copied_charset = charset[:]
    np.random.shuffle(copied_charset)
    sample = copied_charset[:2000]

    # Get the mean of font offset
    font_offset = np.array([0, 0])
    count = 0
    for c in sample:
        font_img = draw_single_char(c, font, canvas_size, 0, 0)
        font_hash = hash(font_img.tobytes())
        if font_hash not in filter_hashes:
            font_offset += get_offset(c, font, canvas_size)
            count += 1
    font_offset /= count
    return font_offset


def filter_recurring_hash(
    charset: list[str],
    font: ImageFont.FreeTypeFont,
    canvas_size: int,
    x_offset: int,
    y_offset: int,
) -> list[int]:
    """
    Filter missing characters on given font by checking the recurring hashes
    """

    # Copy the characters and sample 2000 of them
    copied_charset = charset[:]
    np.random.shuffle(copied_charset)
    sample = copied_charset[:2000]

    # Count the hash of drawn images
    hash_count: collections.defaultdict[int, int] = collections.defaultdict(int)
    for c in sample:
        img = draw_single_char(c, font, canvas_size, x_offset, y_offset)
        hash_count[hash(img.tobytes())] += 1

    # Filter the hash value that appeared more than twice
    recurring_hashes = filter(lambda d: d[1] > 2, hash_count.items())
    return [rh[0] for rh in recurring_hashes]


if TYPE_CHECKING:
    T = TypeVar("T")


def select_sample(charset: list[T]) -> list[T]:
    # this returns 399 samples from KR charset
    # we selected 399 characters to sample as uniformly as possible
    # (the number of each ChoSeong is fixed to 21 (i.e., 21 Giyeok, 21 Nieun ...))
    # Given the designs of these 399 characters, the rest of Hangeul will be generated
    samples: list[T] = []
    for i in range(399):
        samples.append(charset[28 * i + (i % 28)])
    np.random.shuffle(samples)
    return samples


def draw_handwriting(
    ch: str,
    src_font: ImageFont.FreeTypeFont,
    canvas_size: int,
    src_offset: Tuple[int, int],
    dst_folder: str,
) -> None | Image.Image:
    s = ch.encode("unicode-escape").decode("utf-8").replace("\\u", "").upper()
    dst_path = os.path.join(dst_folder, f"uni{s}.png")
    if not os.path.exists(dst_path):
        return
    dst_img = Image.open(dst_path)

    # check the filter example in the hashes or not
    src_img = draw_single_char(ch, src_font, canvas_size, src_offset[0], src_offset[1])
    example_img = Image.new(
        "RGB", (canvas_size * 2, canvas_size), (255, 255, 255)
    ).convert("L")
    example_img.paste(dst_img, (0, 0))
    example_img.paste(src_img, (canvas_size, 0))
    return example_img


def font2img(
    src: str,
    dst: str,
    charset: list[str],
    char_size: int,
    canvas_size: int,
    x_offset: int,
    y_offset: int,
    sample_count: int,
    sample_dir: str,
    label: int = 0,
    filter_by_hash: bool = True,
    fixed_sample: bool = False,
    all_sample: bool = False,
    handwriting_dir: Literal[False] | str = False,
):
    src_font: ImageFont.FreeTypeFont = ImageFont.truetype(src, size=char_size)
    dst_font: ImageFont.FreeTypeFont = ImageFont.truetype(dst, size=char_size)

    dst_filter_hashes = set(filter_recurring_hash(charset, dst_font, canvas_size, 0, 0))
    dst_offset = get_font_offset(charset, dst_font, canvas_size, dst_filter_hashes)
    print("Src font offset : ", [x_offset, y_offset])
    print("Dst font offset : ", dst_offset)

    filter_hashes: set[int] = set()
    if filter_by_hash:
        filter_hashes = set(
            filter_recurring_hash(
                charset, dst_font, canvas_size, dst_offset[0], dst_offset[1]
            )
        )
        print(f"filter hashes -> {','.join([str(h) for h in filter_hashes])}")

    count = 0

    if handwriting_dir:
        if not os.path.exists(sample_dir):
            os.makedirs(sample_dir)
        train_set: list[str] = []
        for c in charset:
            e = draw_handwriting(
                c, src_font, canvas_size, (x_offset, y_offset), handwriting_dir
            )
            if e is None:
                continue
            character_unicode = (
                c.encode("unicode-escape").decode("utf-8").replace("\\u", "").upper()
            )
            e.save(os.path.join(sample_dir, f"{label}_{character_unicode}_train.png"))
            train_set.append(c)
            count += 1
            if count % 100 == 0:
                print(f"processed {count} chars")

        np.random.shuffle(charset)
        count = 0
        for c in charset:
            e = draw_example(
                c,
                src_font,
                dst_font,
                canvas_size,
                (x_offset, y_offset),
                dst_offset,
                filter_hashes=set(),
            )
            if e is None:
                continue
            character_unicode = (
                c.encode("unicode-escape").decode("utf-8").replace("\\u", "").upper()
            )
            e.save(os.path.join(sample_dir, f"{label}_{character_unicode}_val.png"))
            count += 1
            if count % 100 == 0:
                print("processed %d chars" % count)
        return

    if fixed_sample:
        train_set = select_sample(charset)
        for c in train_set:
            e = draw_example(
                c,
                src_font,
                dst_font,
                canvas_size,
                (x_offset, y_offset),
                dst_offset,
                filter_hashes,
            )
            if e:
                e.save(os.path.join(sample_dir, "%d_%04d_train.png" % (label, count)))
                count += 1
                if count % 100 == 0:
                    print("processed %d chars" % count)

        np.random.shuffle(charset)
        count = 0
        for c in charset:
            if count == sample_count:
                break
            if c in train_set:
                continue
            e = draw_example(
                c,
                src_font,
                dst_font,
                canvas_size,
                (x_offset, y_offset),
                dst_offset,
                filter_hashes=set(),
            )
            if e is None:
                continue
            e.save(os.path.join(sample_dir, f"{label}_{count:04}_val.png"))
            count += 1
            if count % 100 == 0:
                print(f"processed {count} chars")
        return

    if all_sample:
        for c in charset:
            e = draw_example(
                c,
                src_font,
                dst_font,
                canvas_size,
                (x_offset, y_offset),
                dst_offset,
                filter_hashes,
            )
            if e:
                e.save(os.path.join(sample_dir, f"{label}_{count:04}.png"))
                count += 1
                if count % 1000 == 0:
                    print(f"processed {count} chars")
        return

    for c in charset:
        if count == sample_count:
            break
        e = draw_example(
            c,
            src_font,
            dst_font,
            canvas_size,
            (x_offset, y_offset),
            dst_offset,
            filter_hashes,
        )
        if e is None:
            continue
        e.save(os.path.join(sample_dir, f"{label}_{count:04}.png"))
        count += 1
        if count % 100 == 0:
            print(f"processed {count} chars")


@click.command()
@click.option(
    "--src-font", type=click.Path(), required=True, help="path of source font"
)
@click.option(
    "--dst-font", type=click.Path(), required=True, help="path of target font"
)
@click.option("--filter", type=int, default=0, help="filter recurring characters")
@click.option(
    "--shuffle", type=bool, default=False, help="shuffle a charset before processings"
)
@click.option("--char-size", type=int, default=80, help="character size")
@click.option("--canvas-size", type=int, default=128, help="canvas size")
@click.option("--x-offset", type=int, default=27, help="x offset")
@click.option("--y-offset", type=int, default=16, help="y offset")
@click.option(
    "--sample-count", type=int, default=1000, help="number of characters to draw"
)
@click.option("--sample-dir", type=click.Path(), help="directory to save examples")
@click.option("--label", type=int, default=0, help="label as the prefix of examples")
@click.option(
    "--fixed-sample",
    type=bool,
    default=False,
    help="pick fixed samples (399 training set, 500 test set). Note that this should not be used with --shuffle.",
)
@click.option(
    "--all-sample",
    type=bool,
    default=False,
    help="pick all possible samples (except for missing characters)",
)
@click.option(
    "--handwriting-dir",
    type=click.Path(),
    default=False,
    help="pick handwriting samples (399 training set). Note that this should not be used with --shuffle.",
)
def main(
    src_font: str,
    dst_font: str,
    filter: bool,
    shuffle: bool,
    char_size: int,
    canvas_size: int,
    x_offset: int,
    y_offset: int,
    sample_count: int,
    sample_dir: str,
    label: int,
    fixed_sample: bool,
    all_sample: bool,
    handwriting_dir: Literal[False] | str,
):
    """
    Convert font to images
    """

    charset: list[str] = []
    for i in range(0xAC00, 0xD7A4):
        charset.append(chr(i))
    for i in range(0x3131, 0x3164):
        charset.append(chr(i))
    if shuffle:
        np.random.shuffle(charset)
    font2img(
        src_font,
        dst_font,
        charset,
        char_size,
        canvas_size,
        x_offset,
        y_offset,
        sample_count,
        sample_dir,
        label,
        filter,
        fixed_sample,
        all_sample,
        handwriting_dir,
    )
