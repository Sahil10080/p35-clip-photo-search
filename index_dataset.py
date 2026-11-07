"""Download 200+ gallery photos + a portrait set, embed with CLIP, save embeddings.npy / image_metadata.json.

    python index_dataset.py                       # Flickr8k gallery (captions) + FairFace portraits
    python index_dataset.py --source picsum       # Unsplash-licensed photos, no captions
Commit images/, embeddings.npy and image_metadata.json so Streamlit Cloud needs no indexing step.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import requests
import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

ROOT = Path(__file__).parent
IMG = ROOT / "images"
MODEL_ID = "openai/clip-vit-base-patch32"


def fit(img, size=384):
    img = img.convert("RGB")
    img.thumbnail((size, size))
    return img


def gallery_flickr(n):
    from datasets import load_dataset

    ds = load_dataset("jxie/flickr8k", split="train", streaming=True).shuffle(seed=42, buffer_size=1000)
    out = []
    for row in ds:
        caps = [row[k] for k in row if k.startswith("caption") and row[k]]
        out.append((fit(row["image"]), " | ".join(caps), "Flickr8k (research use)"))
        if len(out) >= n:
            break
    return out


def gallery_picsum(n):
    out = []
    for i in tqdm(range(n), desc="picsum"):
        r = requests.get(f"https://picsum.photos/seed/p35-{i}/512/384", timeout=30)
        r.raise_for_status()
        img = Image.open(__import__("io").BytesIO(r.content))
        out.append((fit(img), f"picsum photo {i}", "Picsum / Unsplash License"))
    return out


def portraits_fairface(n):
    """Returns [(img, label, source, gender, race)]. Verify the dataset id/config if this errors."""
    from datasets import load_dataset

    ds = load_dataset("HuggingFaceM4/FairFace", "1.25", split="validation", streaming=True)
    ds = ds.shuffle(seed=42, buffer_size=2000)

    def name(col, v):
        try:
            return ds.features[col].int2str(v)
        except Exception:
            return str(v)

    out = []
    for row in ds:
        out.append((fit(row["image"], 256), "portrait", "FairFace (CC BY 4.0)",
                    name("gender", row["gender"]), name("race", row["race"])))
        if len(out) >= n:
            break
    return out


@torch.no_grad()
def embed(images, model, proc, bs=32):
    vecs = []
    for i in tqdm(range(0, len(images), bs), desc="CLIP"):
        x = proc(images=images[i:i + bs], return_tensors="pt")
        f = model.get_image_features(**x)
        vecs.append((f / f.norm(dim=-1, keepdim=True)).numpy())
    return np.vstack(vecs).astype("float32")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["flickr", "picsum"], default="flickr")
    ap.add_argument("--n-gallery", type=int, default=220)
    ap.add_argument("--n-portraits", type=int, default=140)
    a = ap.parse_args()

    (IMG / "gallery").mkdir(parents=True, exist_ok=True)
    (IMG / "portraits").mkdir(parents=True, exist_ok=True)
    meta, pil = [], []

    gal = gallery_flickr(a.n_gallery) if a.source == "flickr" else gallery_picsum(a.n_gallery)
    for i, (img, cap, src) in enumerate(gal):
        p = f"images/gallery/g{i:04d}.jpg"
        img.save(ROOT / p, quality=85)
        meta.append(dict(id=f"g{i:04d}", set="gallery", path=p, caption=cap, source=src))
        pil.append(img)

    if a.n_portraits > 0:
        try:
            for i, (img, cap, src, gender, race) in enumerate(portraits_fairface(a.n_portraits)):
                p = f"images/portraits/p{i:04d}.jpg"
                img.save(ROOT / p, quality=85)
                meta.append(dict(id=f"p{i:04d}", set="portrait", path=p, caption=cap, source=src,
                                 gender=gender, race=race))
                pil.append(img)
        except Exception as e:
            print(f"[warn] portrait set skipped ({e}). The Bias Audit tab will be empty.")

    model = CLIPModel.from_pretrained(MODEL_ID).eval()
    proc = CLIPProcessor.from_pretrained(MODEL_ID)
    np.save(ROOT / "embeddings.npy", embed(pil, model, proc))
    (ROOT / "image_metadata.json").write_text(json.dumps(meta, indent=1))
    print(f"Indexed {len(meta)} images ({sum(m['set']=='gallery' for m in meta)} gallery).")


if __name__ == "__main__":
    main()
