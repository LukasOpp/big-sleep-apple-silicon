"""Batch Big Sleep generator for a RunPod GPU.

Every image gets its own random:
  - crispness t in [0, 1]: steps 50 -> 1000 and cutouts 8 -> 128 (log scale)
  - prompt mode: unicode / ascii garbage, random CLIP tokens, blends between
    garbage embeddings, or a random direction in CLIP space (no text at all)
  - "anti-ImageNet" settings: banned BigGAN classes, a negative prompt, and a
    weaker one-class penalty so BigGAN can mix classes into in-between things

Writes NNNNN.png plus NNNNN.json (prompt and every setting) into --out.
Images that already exist are skipped, so a restart resumes where it stopped.

  python generate.py --count 1000 --out /workspace/out --worker 0 --workers 2
  python generate.py --preview 20      # print sample prompts only (no GPU, no model download)
"""
import argparse
import json
import math
import random
import time
from pathlib import Path

# ---------------------------------------------------------------- prompts ----

UNICODE_RANGES = [
    (0x2190, 0x21FF),  # arrows
    (0x2200, 0x22FF),  # math operators
    (0x2300, 0x23FF),  # misc technical
    (0x2500, 0x259F),  # box drawing + blocks
    (0x25A0, 0x25FF),  # geometric shapes
    (0x2600, 0x26FF),  # misc symbols
    (0x2700, 0x27BF),  # dingbats
    (0x2800, 0x28FF),  # braille
    (0x16A0, 0x16F0),  # runes
    (0x1680, 0x169C),  # ogham
    (0x0300, 0x036F),  # combining marks (zalgo)
    (0x0370, 0x03FF),  # greek
    (0x0400, 0x04FF),  # cyrillic
    (0x0E00, 0x0E5B),  # thai
    (0x3040, 0x30FF),  # kana
    (0x4E00, 0x9FFF),  # CJK
    (0xA000, 0xA48C),  # yi
    (0x1D400, 0x1D7FF),  # math alphanumerics
    (0x1F300, 0x1F5FF),  # pictographs
]
ASCII = [chr(c) for c in range(33, 127)]

# Things BigGAN/CLIP collapse into when the prompt is meaningless.
NEGATIVE_PROMPTS = [
    "a dog|an animal|fur|a face",
    "a dog|a cat|a bird|an insect",
    "a photo of an animal|eyes|a face",
    "text|letters|a logo",
]

# ImageNet indices: 0-397 are animals, 151-268 of those are dogs.
CLASS_BANS = {
    "none": [],
    "no_dogs": list(range(151, 269)),
    "no_animals": list(range(0, 398)),
}


def _rand_unicode(rng, n):
    out = []
    for _ in range(n):
        lo, hi = rng.choice(UNICODE_RANGES)
        out.append(chr(rng.randint(lo, hi)))
    return "".join(out)


def _rand_ascii(rng, n):
    return "".join(rng.choice(ASCII) for _ in range(n))


def _rand_tokens(rng, tokenizer, n):
    # Real CLIP BPE vocabulary entries, glued together: nonsense words CLIP still "knows".
    ids = [rng.randrange(0, 49406) for _ in range(n)]
    return tokenizer.decode(ids).strip()


def _fit(text, tokenizer):
    # CLIP context is 77 tokens incl. start/end; garbage unicode tokenizes into many bytes.
    while len(tokenizer.encode(text)) > 75:
        text = text[: max(1, int(len(text) * 0.8))]
    return text


def sample_prompt(rng, tokenizer):
    """Returns (mode, texts). texts is a list of 1-2 strings (2 for blend), empty for void."""
    mode = rng.choices(
        ["unicode", "ascii", "tokens", "mixed", "blend", "void"],
        weights=[25, 12, 20, 13, 20, 10],
    )[0]

    def one(kind):
        n = rng.randint(3, 40)
        if kind == "unicode":
            return _fit(_rand_unicode(rng, n), tokenizer)
        if kind == "ascii":
            return _fit(_rand_ascii(rng, n), tokenizer)
        if kind == "tokens":
            return _fit(_rand_tokens(rng, tokenizer, rng.randint(2, 12)), tokenizer)
        # mixed: tokens interleaved with symbol noise
        parts = [_rand_tokens(rng, tokenizer, rng.randint(1, 3)) if rng.random() < 0.5
                 else _rand_unicode(rng, rng.randint(1, 6)) for _ in range(rng.randint(2, 6))]
        return _fit(" ".join(parts), tokenizer)

    if mode == "void":
        return mode, []
    if mode == "blend":
        kinds = ["unicode", "ascii", "tokens", "mixed"]
        return mode, [one(rng.choice(kinds)), one(rng.choice(kinds))]
    return mode, [one(mode)]


def sample_settings(rng, tokenizer):
    t = rng.random()
    mode, texts = sample_prompt(rng, tokenizer)
    return {
        "crispness": round(t, 3),
        "iterations": round(50 * 20 ** t),   # 50 .. 1000
        "num_cutouts": round(8 * 16 ** t),   # 8 .. 128
        "mode": mode,
        "texts": texts,
        "blend_t": round(rng.uniform(0.2, 0.8), 3),
        "embed_noise": round(rng.choice([0.0, 0.0, 0.15, 0.35]), 3),
        "class_ban": rng.choices(list(CLASS_BANS), weights=[30, 30, 40])[0],
        "negative": rng.choice(NEGATIVE_PROMPTS) if rng.random() < 0.5 else "",
        # 1.0 = stock Big Sleep (pushes toward a single ImageNet class);
        # lower lets classes blend -> fewer recognisable dogs, more in-between stuff
        "class_loss_weight": rng.choice([1.0, 0.3, 0.1, 0.0]),
        "seed": rng.randrange(2**31),
    }


# ----------------------------------------------------------------- canvas ----
#
# BigGAN starts every image from a 4x4 grid ("seed") made by one linear layer;
# everything after it is convolutional, so a gh x gw seed grid gives a
# (128*gh) x (128*gw) image with the 512 model. Two ways to fill a bigger grid:
#   stretch: resize the single 4x4 seed to gh x gw (one composition, spread out)
#   tiles:   one learnable 4x4 seed per 4x4 block, side by side (panorama/collage)
# CLIP still looks at square crops, plus a letterboxed view of the whole thing.

GRID = {"gh": 4, "gw": 4, "fill": "stretch"}  # set per image before im.reset()
FULL_VIEW_P = 0.1                             # share of crops that are the whole (letterboxed) image
MAX_LONG_CELLS = 32                           # 4096 px long side cap


def parse_canvas(spec, rng):
    """'1920x1080' | '1920x1080,1080x1920' (random pick) | 'random' -> (W, H)."""
    if spec == "random":
        aspect = math.exp(rng.uniform(math.log(1 / 3), math.log(3)))
        return (round(1024 * aspect), 1024) if aspect >= 1 else (1024, round(1024 / aspect))
    w, h = rng.choice([c.strip() for c in spec.split(",")]).lower().split("x")
    return int(w), int(h)


def plan_grid(W, H, detail):
    """Seed grid for a WxH canvas: `detail` cells (x128 px) on the short side."""
    aspect = W / H
    short = detail
    long_ = max(1, round(short * max(aspect, 1 / aspect)))
    if long_ > MAX_LONG_CELLS:
        short = max(1, round(short * MAX_LONG_CELLS / long_))
        long_ = MAX_LONG_CELLS
    return (short, long_) if aspect >= 1 else (long_, short)


def n_extra_seeds():
    if GRID["fill"] != "tiles":
        return 0
    return math.ceil(GRID["gh"] / 4) * math.ceil(GRID["gw"] / 4) - 1


def patch_big_sleep():
    """Teach big_sleep's BigGAN + cutouts about non-square seed grids."""
    import torch
    import torch.nn.functional as F
    import big_sleep.big_sleep as bs
    from big_sleep.biggan import GenBlock

    orig_latents_init = bs.Latents.__init__

    def latents_init(self, *a, **k):
        orig_latents_init(self, *a, **k)
        n = n_extra_seeds()
        if n:  # extra seed z's; must exist before the EMA wrapper copies the module
            self.seed_extra = torch.nn.Parameter(torch.randn(n, self.normu.shape[1]))
        else:
            self.seed_extra = None

    def grid_biggan(biggan, z, cls, extra):
        embed = biggan.embeddings(cls)
        cond = torch.cat((z, embed), dim=1)
        g = biggan.generator
        C = 16 * g.config.channel_width
        seeds = [cond[0:1]]
        if extra is not None:
            seeds += [torch.cat((extra[i:i + 1], embed[0:1]), dim=1) for i in range(extra.shape[0])]
        blocks = [g.gen_z(s).view(1, 4, 4, C).permute(0, 3, 1, 2) for s in seeds]
        gh, gw = GRID["gh"], GRID["gw"]
        if len(blocks) == 1:
            x = blocks[0]
        else:
            bh, bw = math.ceil(gh / 4), math.ceil(gw / 4)
            x = torch.cat([torch.cat(blocks[r * bw:(r + 1) * bw], dim=3) for r in range(bh)], dim=2)
        if tuple(x.shape[-2:]) != (gh, gw):
            x = F.interpolate(x, size=(gh, gw), mode="bilinear", align_corners=True)
        x = x.contiguous()
        i = 1
        for layer in g.layers:
            if isinstance(layer, GenBlock):
                x = layer(x, cond[i].unsqueeze(0), 1)
                i += 1
            else:
                x = layer(x)
        x = g.conv_to_rgb(g.relu(g.bn(x, 1)))[:, :3]
        return torch.tanh(x)

    def model_forward(self):
        self.biggan.eval()
        normu, classes = self.latents()
        src = self.latents.model if self.latents.training else self.latents.average
        out = grid_biggan(self.biggan, normu, classes, getattr(src, "seed_extra", None))
        return (out + 1) / 2

    def rand_cutout(image, size, center_bias=False, center_focus=2):
        h, w = image.shape[-2:]
        if h != w and random.random() < FULL_VIEW_P:
            s = max(h, w)
            t, l = (s - h) // 2, (s - w) // 2
            return F.pad(image, (l, s - w - l, t, s - h - t), value=0.5)
        size = min(size, h, w)
        oy, ox = random.randint(0, h - size), random.randint(0, w - size)
        return image[:, :, oy:oy + size, ox:ox + size]

    bs.Latents.__init__ = latents_init
    bs.Model.forward = model_forward
    bs.rand_cutout = rand_cutout


# ------------------------------------------------------------- generation ----

def slerp(a, b, t):
    import torch
    a, b = a / a.norm(), b / b.norm()
    omega = torch.acos((a * b).sum().clamp(-1, 1))
    if omega.abs() < 1e-4:
        return a
    return (torch.sin((1 - t) * omega) * a + torch.sin(t * omega) * b) / torch.sin(omega)


def build_target(im, s):
    import torch
    dtype = im.create_text_encoding("x").dtype  # CLIP runs in fp16 on CUDA; match it
    if s["mode"] == "void":
        return torch.randn(1, 512, device="cuda").to(dtype)
    embs = [im.create_text_encoding(t).float() for t in s["texts"]]
    e = slerp(embs[0], embs[1], s["blend_t"]) if len(embs) == 2 else embs[0] / embs[0].norm()
    if s["embed_noise"] > 0:
        noise = torch.randn_like(e)
        e = e + s["embed_noise"] * noise / noise.norm()
    return e.to(dtype)


def render(im, s, path):
    import torch
    from PIL import Image
    from torchvision.transforms.functional import to_pil_image

    gh, gw = s["grid"]
    GRID.update(gh=gh, gw=gw, fill=s["fill"])
    torch.manual_seed(s["seed"])
    random.seed(s["seed"])
    im.reset()  # builds fresh latents (incl. extra tile seeds) for this grid
    im.model.image_size = 128 * min(gh, gw)  # crop sizes are relative to the short side
    # more area -> more crops, so CLIP still sees all of it
    im.model.num_cutouts = round(s["num_cutouts"] * math.sqrt(gh * gw / 16))
    latents = im.model.model.latents  # EMA wrapper around Latents
    banned = CLASS_BANS[s["class_ban"]]

    def apply_ban():
        if banned:
            with torch.no_grad():
                latents.model.cls[:, banned] = -20.0

    target = [build_target(im, s)]
    negs = [im.create_text_encoding(t) for t in s["negative"].split("|")] if s["negative"] else []
    apply_ban()

    for _ in range(s["iterations"]):
        _, (lat_loss, cls_loss, sim_loss) = im.model(target, negs)
        loss = lat_loss + s["class_loss_weight"] * cls_loss + sim_loss
        extra = latents.model.seed_extra
        if extra is not None:  # keep tile seeds ~N(0,1) like big_sleep does for the main one
            loss = loss + (torch.abs(1 - extra.std(dim=1)) + torch.abs(extra.mean(dim=1))).mean()
        loss.backward()
        im.optimizer.step()
        latents.update()
        im.optimizer.zero_grad()
        apply_ban()

    with torch.no_grad():
        latents.eval()
        _, losses = im.model(target, negs)
        best = torch.topk(losses[2], k=1, largest=False)[1]
        image = im.model.model()[best].cpu()
        latents.train()
    image = image.reshape(-1, *image.shape[-3:])[0].clamp(0, 1)
    img = to_pil_image(image)
    # center-crop to the canvas aspect (grid is within half a cell of it), then resize
    W, H = s["canvas"]
    w, h = img.size
    if w / h > W / H:
        cw = round(h * W / H); img = img.crop(((w - cw) // 2, 0, (w - cw) // 2 + cw, h))
    else:
        ch = round(w * H / W); img = img.crop((0, (h - ch) // 2, w, (h - ch) // 2 + ch))
    if img.size != (W, H):
        img = img.resize((W, H), Image.LANCZOS)
    img.save(str(path), format="PNG")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=1000)
    ap.add_argument("--out", default="/workspace/out")
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--run-seed", type=int, default=1234, help="same seed = same set of prompts/settings")
    ap.add_argument("--preview", type=int, default=0, help="print N sampled settings and exit")
    ap.add_argument("--canvas", default="512x512",
                    help="WxH, a comma list to pick from per image, or 'random' (aspect 1:3..3:1)")
    ap.add_argument("--detail", type=int, default=4,
                    help="128px cells on the short side before resizing (4 = 512px; cost ~ cells)")
    ap.add_argument("--manifest", help="JSON list of {id, width, height, alt?} (extract_img_manifest.py); "
                                       "one image per entry, saved as <id>.png; overrides --count/--canvas")
    ap.add_argument("--scale", type=float, default=1.0, help="manifest sizes x this (2 = retina)")
    ap.add_argument("--prompt-source", choices=["garbage", "alt", "mix"], default="garbage",
                    help="alt: use the manifest's alt text when there is one; mix: half of those")
    args = ap.parse_args()

    from big_sleep.clip import _tokenizer as tokenizer

    items = None
    if args.manifest:
        items = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
        args.count = len(items)

    # Settings for image i depend only on (run_seed, i): stable across restarts and workers.
    def settings_for(i):
        s = sample_settings(random.Random(args.run_seed * 1_000_003 + i), tokenizer)
        crng = random.Random(args.run_seed * 7_919 + i)
        if items is None:
            W, H = parse_canvas(args.canvas, crng)
        else:
            it = items[i]
            W, H = max(1, round(it["width"] * args.scale)), max(1, round(it["height"] * args.scale))
            s.update(id=it["id"], page=it.get("page"), src=it.get("src"))
            alt = (it.get("alt") or "").strip()
            if alt and (args.prompt_source == "alt" or (args.prompt_source == "mix" and crng.random() < 0.5)):
                s.update(mode="alt", texts=[_fit(alt, tokenizer)])
        # Small images can't show fine detail: cap crispness so a 50px icon doesn't get 1000 steps.
        t_max = min(1.0, max(0.3, math.log(max(W, H) / 64, 8))) if max(W, H) > 64 else 0.3
        t = s["crispness"] * t_max
        s.update(crispness=round(t, 3), iterations=round(50 * 20 ** t), num_cutouts=round(8 * 16 ** t))
        gh, gw = plan_grid(W, H, args.detail)
        multi_block = gh > 4 or gw > 4
        s.update(canvas=[W, H], grid=[gh, gw],
                 fill=crng.choice(["stretch", "tiles"]) if multi_block else "stretch")
        return s

    if args.preview:
        for i in range(args.preview):
            s = settings_for(i)
            print(f"{i:05d} {s['canvas'][0]}x{s['canvas'][1]} grid={s['grid'][0]}x{s['grid'][1]} {s['fill']:7s} "
                  f"t={s['crispness']:.2f} {s['mode']:7s} ban={s['class_ban']:10s} "
                  f"clsw={s['class_loss_weight']:<4} neg={'y' if s['negative'] else 'n'}  {s['texts']}")
        return

    patch_big_sleep()
    from big_sleep import Imagine

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    im = Imagine(text="warmup", open_folder=False, save_every=10**9)
    import torch
    with torch.no_grad():
        im.model(im.encoded_texts["max"][0])  # same CUDA warmup Big Sleep does

    mine = [i for i in range(args.count) if i % args.workers == args.worker]
    for n, i in enumerate(mine):
        png = out / (f"{items[i]['id']}.png" if items else f"{i:05d}.png")
        if png.exists():
            continue
        s = settings_for(i)
        t0 = time.time()
        render(im, s, png.with_suffix(".tmp.png"))
        png.with_suffix(".tmp.png").rename(png)
        s["seconds"] = round(time.time() - t0, 1)
        png.with_suffix(".json").write_text(json.dumps(s, ensure_ascii=False, indent=1))
        print(f"[w{args.worker}] {n + 1}/{len(mine)} {png.name} {s['mode']} t={s['crispness']} "
              f"{s['seconds']}s", flush=True)


if __name__ == "__main__":
    main()
