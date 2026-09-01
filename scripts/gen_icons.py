"""Generate PWA icons (192/512) without external deps (PIL if present, else pure PNG writer)."""
from __future__ import annotations

import struct
import zlib
from pathlib import Path

WEB = Path(__file__).resolve().parents[1] / "web"
GREEN = (11, 110, 79)
WHITE = (255, 255, 255)


def write_png(path: Path, size: int, pixels: list[list[tuple]]) -> None:
    raw = b"".join(b"\x00" + b"".join(b"".join(p) for p in row) for row in pixels)

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))

    png = (b"\x89PNG\r\n\x1a\n"
           + chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(raw, 9))
           + chunk(b"IEND", b""))
    path.write_bytes(png)


def try_pillow(sizes=(192, 512)):
    from PIL import Image, ImageDraw
    for s in sizes:
        img = Image.new("RGB", (s, s), GREEN)
        d = ImageDraw.Draw(img)
        # microphone: capsule + stand
        m = s // 3.2
        cx = s / 2
        d.rounded_rectangle([cx - m / 2, s * 0.18, cx + m / 2, s * 0.52],
                            radius=int(m / 2), fill=WHITE)
        d.arc([s * 0.28, s * 0.32, s * 0.72, s * 0.72], start=0, end=180, fill=WHITE, width=max(3, s // 64))
        d.line([cx, s * 0.72, cx, s * 0.82], fill=WHITE, width=max(3, s // 64))
        d.line([cx - s * 0.12, s * 0.82, cx + s * 0.12, s * 0.82], fill=WHITE, width=max(3, s // 64))
        out = WEB / "icons" / f"icon-{s}.png"
        out.parent.mkdir(exist_ok=True)
        img.save(out)
    return True


def fallback(sizes=(192, 512)):
    for s in sizes:
        r = s // 6  # rounded corner approx by inset circle contrast
        cx, cy = s // 2, s // 2
        mic_w, mic_h = s // 3, s // 2
        pixels = []
        for y in range(s):
            row = []
            for x in range(s):
                in_mic = (cx - mic_w // 2 <= x < cx + mic_w // 2
                          and s // 5 <= y < s // 5 + mic_h)
                c = WHITE if in_mic else GREEN
                row.append(c)
            pixels.append(row)
        out = WEB / "icons" / f"icon-{s}.png"
        out.parent.mkdir(exist_ok=True)
        write_png(out, s, pixels)


if __name__ == "__main__":
    try:
        try_pillow()
        print("icons written via Pillow")
    except ImportError:
        fallback()
        print("icons written via pure-python PNG writer")
    for f in sorted((WEB / "icons").glob("*.png")):
        print(f.name, f.stat().st_size, "bytes")
