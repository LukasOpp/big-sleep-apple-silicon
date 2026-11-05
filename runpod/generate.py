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
    from torchvision.utils import save_image

    torch.manual_seed(s["seed"])
    random.seed(s["seed"])
    im.reset()
    im.model.num_cutouts = s["num_cutouts"]
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
    save_image(image, str(path))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=1000)
    ap.add_argument("--out", default="/workspace/out")
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--run-seed", type=int, default=1234, help="same seed = same set of prompts/settings")
    ap.add_argument("--preview", type=int, default=0, help="print N sampled settings and exit")
    args = ap.parse_args()

    from big_sleep.clip import _tokenizer as tokenizer

    # Settings for image i depend only on (run_seed, i): stable across restarts and workers.
    def settings_for(i):
        return sample_settings(random.Random(args.run_seed * 1_000_003 + i), tokenizer)

    if args.preview:
        for i in range(args.preview):
            s = settings_for(i)
            print(f"{i:05d} t={s['crispness']:.2f} {s['mode']:7s} ban={s['class_ban']:10s} "
                  f"clsw={s['class_loss_weight']:<4} neg={'y' if s['negative'] else 'n'}  {s['texts']}")
        return

    from big_sleep import Imagine

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    im = Imagine(text="warmup", open_folder=False, save_every=10**9)
    import torch
    with torch.no_grad():
        im.model(im.encoded_texts["max"][0])  # same CUDA warmup Big Sleep does

    mine = [i for i in range(args.count) if i % args.workers == args.worker]
    for n, i in enumerate(mine):
        png = out / f"{i:05d}.png"
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
