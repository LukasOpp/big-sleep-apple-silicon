"""List every <img> in a folder of HTML pages, with the size it is displayed at.

  python extract_img_manifest.py ~/Programming/substrate-exchange/pages manifest.json

Each entry gets a stable id (from page name + position of the tag in the page),
so generated images can later be matched back to their tag: <id>.png.

Size comes from, in order: width/height attributes, inline style width/height
in px, then a page-wide CSS rule for plain `img { width/height }`. If only one is
known the image is square. If neither, --default-size (shrunk to a CSS
`img { max-width }` if the page has one).
Tags smaller than --min-size on either side (spacers, tracking pixels) are skipped.
"""
import argparse
import hashlib
import json
import re
from collections import Counter
from html.parser import HTMLParser
from pathlib import Path

PX = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*(px)?\s*$", re.I)


def px(value):
    if value is None:
        return None
    m = PX.match(value)
    return round(float(m.group(1))) if m else None  # "50%", "auto", "" -> unknown


def style_px(style, prop):
    m = re.search(rf"(?:^|;)\s*{prop}\s*:\s*([^;]+)", style or "", re.I)
    return px(m.group(1)) if m else None


CSS_RULE = re.compile(r"([^{}]+)\{([^}]*)\}")


def page_img_css(html):
    """width/height/max-width (px) from CSS rules whose selector list includes bare `img`."""
    found = {}
    for sel, body in CSS_RULE.findall(html):
        if any(part.strip().lower() == "img" for part in sel.split(",")):
            for prop in ("width", "height", "max-width"):
                v = style_px(body, prop)
                if v:
                    found[prop] = v
    return found


class ImgCollector(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.imgs = []

    def handle_starttag(self, tag, attrs):
        if tag == "img":
            self.imgs.append((dict(attrs), self.get_starttag_text(), self.getpos()[0]))

    handle_startendtag = handle_starttag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pages")
    ap.add_argument("out")
    ap.add_argument("--default-size", default="512x512", help="for tags with no size at all")
    ap.add_argument("--min-size", type=int, default=8)
    args = ap.parse_args()
    dw, dh = map(int, args.default_size.lower().split("x"))

    items, skipped, sources = [], Counter(), Counter()
    for page in sorted(Path(args.pages).glob("*.html")):
        html = page.read_text(encoding="utf-8", errors="replace")
        css = page_img_css(html)
        p = ImgCollector()
        p.feed(html)
        for idx, (a, raw, line) in enumerate(p.imgs):
            w = px(a.get("width")) or style_px(a.get("style"), "width")
            h = px(a.get("height")) or style_px(a.get("style"), "height")
            src = "attrs"
            if not (w or h) and (css.get("width") or css.get("height")):
                w, h, src = css.get("width"), css.get("height"), "page_css"
            if w and h:
                pass
            elif w or h:
                w = h = w or h
                src = "one_side" if src == "attrs" else src
            else:
                w, h, src = dw, dh, "default"
                if css.get("max-width") and css["max-width"] < w:
                    w = h = css["max-width"]
                    src = "default_capped"
            if min(w, h) < args.min_size:
                skipped["too_small"] += 1
                continue
            sources[src] += 1
            items.append({
                "id": hashlib.sha1(f"{page.name}\0{idx}".encode()).hexdigest()[:12],
                "page": page.name, "index": idx, "line": line,
                "width": w, "height": h, "size_source": src,
                "src": a.get("src"), "alt": a.get("alt"), "tag": raw,
            })

    Path(args.out).write_text(json.dumps(items, ensure_ascii=False, indent=0))
    sizes = Counter((i["width"], i["height"]) for i in items)
    print(f"{len(items)} images from {len({i['page'] for i in items})} pages -> {args.out}")
    print("size from:", dict(sources), "| skipped:", dict(skipped))
    print("with alt text:", sum(1 for i in items if i["alt"]))
    print("most common sizes:", ", ".join(f"{w}x{h} ({n})" for (w, h), n in sizes.most_common(12)))
    big = sorted(items, key=lambda i: -i["width"] * i["height"])[:5]
    print("largest:", ", ".join(f"{i['width']}x{i['height']}" for i in big))


if __name__ == "__main__":
    main()
