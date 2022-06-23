from PIL import Image, ImageDraw, ImageFont

KR_CHARSET = None


def draw_single_char(
    ch: str, font: ImageFont.FreeTypeFont, char_size: int, x_offset: int, y_offset: int
):
    img = Image.new("RGB", (char_size, char_size), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((x_offset, y_offset), ch, (0, 0, 0), font=font)
    return img


def drawChars(charset: list[str], font: str, char_size: int):
    src_font = ImageFont.truetype(font, 150)
    canvas = Image.new(
        "RGB", (char_size * 21, char_size * 19), (255, 255, 255)
    )  # 42 -> 21
    x_pos = 0
    y_pos = 0
    for c in charset:
        e = draw_single_char(c, src_font, char_size, x_offset=50, y_offset=20)
        canvas.paste(e, (x_pos * char_size, y_pos * char_size))
        x_pos = x_pos + 1
        if x_pos >= 21:  # 42 -> 21
            x_pos = 0
            y_pos = y_pos + 1
    draw = ImageDraw.Draw(canvas)
    for i in range(20):  # 41 -> 20
        draw.line(
            [((i + 1) * char_size, 0), ((i + 1) * char_size, char_size * 19)],
            fill=(0, 0, 0),
            width=5,
        )
    for i in range(18):
        draw.line(
            [(0, (i + 1) * char_size), (char_size * 21, (i + 1) * char_size)],
            fill=(0, 0, 0),
            width=5,
        )  # 42 -> 21

    canvas.save("399_image.png")


def select_sample(charset: list[str]) -> list[str]:
    # this returns 399 samples from KR charset
    # we selected 399 characters to sample as uniformly as possible
    # (the number of each ChoSeong is fixed to 21 (i.e., 21 Giyeok, 21 Nieun ...))
    # Given the designs of these 399 characters, the rest of Hangeul will be generated
    samples: list[str] = []
    for i in range(399):
        samples.append(charset[28 * i + (i % 28)])
    #        samples.append(charset[28*i+(i%28)+14])
    return samples


def main():
    charset: list[str] = []
    for i in range(0xAC00, 0xD7A4):
        charset.append(chr(i))
    charset = select_sample(charset)
    drawChars(charset, "fonts/Gothic.ttf", char_size=256)
