import colorsys
from rich.console import Console
from rich.text import Text
import numpy as np

from vdaworld import __version__

NAME = (
    " __      _______     __          __        _     _ \n"
    + " \\ \\    / /  __ \\   /\\ \\        / /       | |   | |\n"
    + "  \\ \\  / /| |  | | /  \\ \\  /\\  / /__  _ __| | __| |\n"
    + "   \\ \\/ / | |  | |/ /\\ \\ \\/  \\/ / _ \\| '__| |/ _` |\n"
    + "    \\  /  | |__| / ____ \\  /\\  / (_) | |  | | (_| |\n"
    + f"     \\/   |_____/_/    \\_\\/  \\/ \\___/|_|  |_|\\__,_| {__version__}\n"
)


def hex_to_rgb(hex_str):
    """Converts #RRGGBB to (R, G, B) tuple."""
    hex_str = hex_str.lstrip("#")
    return tuple(int(hex_str[i : i + 2], 16) for i in (0, 2, 4))


def print_gradient(text_content, start_hex, end_hex):
    console = Console()
    rich_text = Text()

    start_rgb = hex_to_rgb(start_hex)
    end_rgb = hex_to_rgb(end_hex)

    _, start_s, start_v = colorsys.rgb_to_hsv(
        start_rgb[0] / 255.0, start_rgb[1] / 255.0, start_rgb[2] / 255.0
    )
    # randomise start_h
    start_h = np.random.rand()
    end_h, end_s, end_v = colorsys.rgb_to_hsv(
        end_rgb[0] / 255.0, end_rgb[1] / 255.0, end_rgb[2] / 255.0
    )

    # Nearest transition direction for hue
    if end_h - start_h > 0.5:
        start_h += 1.0
    elif end_h - start_h < -0.5:
        end_h += 1.0

    # We want to interpolate across the total number of characters
    # excluding newlines for a smoother transition
    chars_per_line = len(text_content.split("\n")[0])

    char_count = 0
    for char in text_content:
        if char == "\n":
            rich_text.append("\n")
            char_count = 0
            continue

        # Calculate the interpolation factor (0.0 to 1.0)
        t = char_count / chars_per_line if chars_per_line > 0 else 0
        t = min(max(t, 0), 1)  # Clamp t to [0, 1]

        # Interpolate H, S, and V
        h = (start_h + (end_h - start_h) * t) % 1.0
        s = start_s + (end_s - start_s) * t
        v = start_v + (end_v - start_v) * t

        r, g, b = colorsys.hsv_to_rgb(h, s, v)

        rich_text.append(
            char, style=f"rgb({int(r * 255)},{int(g * 255)},{int(b * 255)})"
        )
        char_count += 1

    console.print(rich_text)


def printname():
    print_gradient(
        NAME,
        "#386AE0",
        "#85B09A",
    )


if __name__ == "__main__":
    printname()
