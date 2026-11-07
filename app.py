"""P35 · CLIP Text-to-Photo Search  (Track C: Multimodal Models)
U1 latent-space plot · U2 LLM query expansion · U3 CLIP retrieval (text & image)
U4 generative fallback · U5 bias audit.  Run: streamlit run app.py
"""
import io
import json
import os
import urllib.parse
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import requests
import streamlit as st
import torch
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from transformers import CLIPModel, CLIPProcessor

MODEL_ID = "openai/clip-vit-base-patch32"
ROOT = Path(__file__).parent
st.set_page_config(page_title="P35 CLIP Photo Search", page_icon="🔎", layout="wide")


# ───────────────────────── config / secrets ─────────────────────────
def secret(name):
    try:
        val = st.secrets.get(name)
    except Exception:
        val = None
    return val or os.environ.get(name)


# ───────────────────────── CLIP core (U3) ─────────────────────────
@st.cache_resource(show_spinner="Loading CLIP (first run downloads ~600 MB)…")
def load_clip():
    return CLIPModel.from_pretrained(MODEL_ID).eval(), CLIPProcessor.from_pretrained(MODEL_ID)


@st.cache_resource(show_spinner="Loading Sentence-BERT…")
def load_sbert():
    from sentence_transformers import SentenceTransformer  # BERT-family baseline (Layer III)
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def _unit(x):
    return (x / x.norm(dim=-1, keepdim=True)).cpu().numpy()


@torch.no_grad()
def embed_text(texts):
    model, proc = load_clip()
    t = proc(text=list(texts), return_tensors="pt", padding=True, truncation=True, max_length=77)
    return _unit(model.get_text_features(**t))


@torch.no_grad()
def embed_image(img):
    model, proc = load_clip()
    return _unit(model.get_image_features(**proc(images=img.convert("RGB"), return_tensors="pt")))


@st.cache_data
def load_index():
    emb = np.load(ROOT / "embeddings.npy")
    meta = json.loads((ROOT / "image_metadata.json").read_text())
    return emb, meta


def subset(kind):
    emb, meta = load_index()
    ids = [i for i, m in enumerate(meta) if m["set"] == kind]
    return emb[ids], [meta[i] for i in ids]


def rank(qvec, emb, k):
    sims = emb @ qvec  # unit vectors → dot product == cosine similarity
    idx = np.argsort(-sims)[:k]
    return idx, sims[idx]


def show_results(idx, sims, meta):
    cols = st.columns(len(idx))
    for c, i, s in zip(cols, idx, sims):
        c.image(Image.open(ROOT / meta[i]["path"]), use_column_width=True)
        c.caption(f"**{s:.3f}** · {meta[i]['caption'].split('|')[0][:70]}")


# ───────────────────────── U2: LLM query expansion ─────────────────────────
SYSTEM = (
    "You rewrite short or vague photo-search queries into ONE vivid, concrete visual description "
    "for an image-retrieval model. Mention subject, setting, lighting, colours and mood. Max 35 words. "
    "No quotes, no preamble. Do not add ethnicity, gender, age or other sensitive attributes "
    "unless the query contains them."
)
FEW_SHOT = [
    ("chilly morning", "A frosty sunrise over a quiet field, pale blue light, mist rising from the grass, bare trees, cold muted colours."),
    ("cozy", "A warm living room with a knitted blanket on an armchair, a steaming mug, soft lamplight and rain on the window."),
]


def expand_query(q, style="zero-shot"):
    key = secret("GROQ_API_KEY")
    if not key:
        raise RuntimeError("GROQ_API_KEY not set in st.secrets / environment")
    from groq import Groq

    msgs = [{"role": "system", "content": SYSTEM}]
    if style == "few-shot":
        for u, a in FEW_SHOT:
            msgs += [{"role": "user", "content": u}, {"role": "assistant", "content": a}]
    msgs.append({"role": "user", "content": q})
    r = Groq(api_key=key).chat.completions.create(
        model=secret("GROQ_MODEL") or "openai/gpt-oss-20b",  # llama-3.1-8b-instant was retired by Groq (16 Aug 2026)
        messages=msgs, temperature=0.4, max_tokens=600,  # reasoning model: leave room for hidden reasoning tokens
        extra_body={"reasoning_effort": "low"},
    )
    text = (r.choices[0].message.content or "").strip()
    if not text:
        raise RuntimeError("the LLM returned an empty answer")
    return text


# ───────────────────────── U4: generative fallback ─────────────────────────
def generate_image(prompt):
    token = secret("HF_TOKEN")
    if token:
        try:
            r = requests.post(
                "https://api-inference.huggingface.co/models/stabilityai/sd-turbo",
                headers={"Authorization": f"Bearer {token}"}, json={"inputs": prompt}, timeout=90,
            )
            if r.ok and r.headers.get("content-type", "").startswith("image"):
                return Image.open(io.BytesIO(r.content)), "Hugging Face Inference · stabilityai/sd-turbo"
        except requests.RequestException:
            pass
    url = ("https://image.pollinations.ai/prompt/" + urllib.parse.quote(prompt)
           + "?width=512&height=512&model=turbo&nologo=true")
    r = requests.get(url, timeout=90)
    r.raise_for_status()
    return Image.open(io.BytesIO(r.content)), "Pollinations.ai (free endpoint)"


# ───────────────────────── U5: safety guardrail ─────────────────────────
BLOCKED = ["nude", "naked", "porn", "nsfw", "gore", "beheading", "child abuse", "suicide method", "make a bomb"]


def unsafe(q):
    return any(b in q.lower() for b in BLOCKED)


# ───────────────────────── sidebar ─────────────────────────
if not (ROOT / "embeddings.npy").exists():
    st.error("`embeddings.npy` not found. Run `python index_dataset.py` first and commit its outputs.")
    st.stop()

PAGES = ["🔎 Search", "🗺️ Latent Space (U1)", "⚖️ Bias Audit (U5)", "📊 Benchmark", "ℹ️ About"]
page = st.sidebar.radio("Navigate", PAGES)
st.sidebar.divider()
top_k = st.sidebar.slider("Top-K results", 1, 10, 5)
threshold = st.sidebar.slider("U4 · fallback threshold (cosine)", 0.10, 0.35, 0.22, 0.01)
use_gen = st.sidebar.toggle("U4 · Generate image if no match", True)
use_llm = st.sidebar.toggle("U2 · Expand query with LLM", False)
style = st.sidebar.selectbox("U2 · Prompt technique", ["zero-shot", "few-shot"])
combine = st.sidebar.radio("U2 · Query vector", ["expanded", "average", "original"])
st.sidebar.caption("Keys: `GROQ_API_KEY` (U2), optional `HF_TOKEN` (U4) via st.secrets.")


def query_vector(q, expanded):
    if not expanded or combine == "original":
        return embed_text([q])[0]
    if combine == "expanded":
        return embed_text([expanded])[0]
    v = embed_text([q, expanded]).mean(0)
    return v / np.linalg.norm(v)


# ───────────────────────── pages ─────────────────────────
def page_search():
    st.title("🔎 CLIP Text-to-Photo Search")
    emb_g, meta_g = subset("gallery")
    st.caption(f"{len(meta_g)} photos indexed with `{MODEL_ID}`")
    mode = st.radio("Search mode", ["Text → Photo", "Image → Photo"], horizontal=True)

    if mode.startswith("Text"):
        q = st.text_input("Describe the photo you want", placeholder="chilly morning")
        if not q:
            return
        if unsafe(q):
            st.error("🚫 Blocked by the safety guardrail (see Model Card §Safety).")
            return
        expanded = None
        if use_llm:
            try:
                with st.spinner("Expanding query…"):
                    expanded = expand_query(q, style)
                st.info(f"**LLM-expanded ({style}):** {expanded}")
            except Exception as e:
                st.warning(f"Expansion unavailable ({e}). Using the original query.")
        idx, sims = rank(query_vector(q, expanded), emb_g, top_k)
        show_results(idx, sims, meta_g)
        st.metric("Top-1 cosine similarity", f"{sims[0]:.3f}", delta=f"{sims[0]-threshold:+.3f} vs threshold")
        if sims[0] < threshold:
            st.warning("No suitable photo found in the collection.")
            if use_gen:
                try:
                    with st.spinner("Generating a photo instead…"):
                        img, src = generate_image(expanded or q)
                    st.image(img, caption=f"AI-generated · {src} · prompt: {expanded or q}", width=400)
                except Exception as e:
                    st.error(f"Generation failed: {e}")
    else:
        up = st.file_uploader("Upload an example photo", type=["jpg", "jpeg", "png", "webp"])
        if up:
            img = Image.open(up)
            st.image(img, caption="Query photo", width=250)
            idx, sims = rank(embed_image(img)[0], emb_g, top_k)
            show_results(idx, sims, meta_g)
            st.caption("Image-to-image similarities are naturally higher than text-to-image, so the U4 threshold applies to text search only.")


def page_latent():
    st.title("🗺️ Shared CLIP Latent Space (U1)")
    emb_g, meta_g = subset("gallery")
    prompts = st.text_area(
        "Text prompts to place in the space (one per line)",
        "a dog running on grass\na snowy mountain\npeople on a beach\na city street at night\nchilly morning\na plate of food",
    ).splitlines()
    prompts = [p.strip() for p in prompts if p.strip()]
    method = st.radio("Projection", ["PCA", "t-SNE"], horizontal=True)
    if not prompts:
        return
    T = embed_text(prompts)
    X = np.vstack([emb_g, T])
    if method == "PCA":
        pts = PCA(n_components=2, random_state=0).fit_transform(X)
    else:
        pts = TSNE(n_components=2, perplexity=min(30, len(X) - 1), random_state=0, init="pca").fit_transform(X)
    df = pd.DataFrame({
        "x": pts[:, 0], "y": pts[:, 1],
        "kind": ["image"] * len(meta_g) + ["text prompt"] * len(prompts),
        "label": [m["caption"].split("|")[0][:60] for m in meta_g] + prompts,
    })
    fig = px.scatter(df, x="x", y="y", color="kind", symbol="kind", hover_name="label",
                     color_discrete_map={"image": "#4C78A8", "text prompt": "#E45756"}, height=620)
    fig.update_traces(marker=dict(size=9, opacity=0.8))
    fig.update_traces(selector=dict(name="text prompt"), marker=dict(size=15, line=dict(width=1, color="black")))
    st.plotly_chart(fig, use_container_width=True)
    gap = 1 - float(np.dot(emb_g.mean(0), T.mean(0)) / (np.linalg.norm(emb_g.mean(0)) * np.linalg.norm(T.mean(0))))
    st.metric("Modality gap (1 − cos of centroids)", f"{gap:.3f}")
    st.caption("Text and image points share one 512-d space; each prompt should sit nearest its matching photos. "
               "A systematic offset between the two clouds is CLIP's known *modality gap*.")


def page_bias():
    st.title("⚖️ Bias Audit & Responsible AI (U5)")
    emb_p, meta_p = subset("portrait")
    if not meta_p:
        st.warning("No portrait set indexed. Re-run `index_dataset.py` with portraits enabled.")
        return
    dfm = pd.DataFrame(meta_p)
    queries = [q.strip() for q in st.text_area(
        "Occupational / sensitive queries (one per line)",
        "a doctor\na nurse\na software engineer\na criminal\na CEO\na teacher").splitlines() if q.strip()]
    n = st.slider("Top-N portraits counted per query", 5, min(60, len(meta_p)), min(20, len(meta_p)))
    attr = st.radio("Attribute (dataset annotation)", ["gender", "race"], horizontal=True)
    pool = dfm[attr].value_counts(normalize=True)
    rows = [dict(query="⟂ pool baseline", group=g, top_share=s, pool_share=s, ratio=1.0, mean_sim=np.nan)
            for g, s in pool.items()]
    for q, v in zip(queries, embed_text(queries)):
        sims = emb_p @ v
        share = dfm.iloc[np.argsort(-sims)[:n]][attr].value_counts(normalize=True)
        for g in pool.index:
            rows.append(dict(query=q, group=g, top_share=share.get(g, 0.0), pool_share=pool[g],
                             ratio=share.get(g, 0.0) / pool[g], mean_sim=sims[(dfm[attr] == g).values].mean()))
    res = pd.DataFrame(rows)
    st.plotly_chart(px.bar(res, x="group", y="top_share", color="query", barmode="group", height=480,
                           labels={"top_share": f"share of top-{n}"}), use_container_width=True)
    st.dataframe(res.round(3), use_container_width=True, hide_index=True)
    summ = res[res["query"] != "⟂ pool baseline"].groupby("query")["ratio"].agg(["max", "min"])
    summ["disparity (max/min)"] = summ["max"] / summ["min"].replace(0, np.nan)
    st.subheader("Disparity summary")
    st.dataframe(summ.round(2), use_container_width=True)
    st.download_button("Download CSV", res.to_csv(index=False), "bias_audit.csv")
    st.caption("ratio = share in top-N ÷ share in the portrait pool (1.0 = parity). Labels are coarse dataset "
               "annotations, not ground truth about identity; results describe CLIP, not people.")


DEFAULT_BENCH = """a dog running on grass | dog
children playing in water | water
a man climbing a rock | climb
snowy mountain | snow
people on a beach | beach
a bicycle on a road | bike
a girl in a pink dress | pink
a football player | ball
a boy jumping | jump
a black dog in the snow | snow
a surfer on a wave | wave
a crowd in a street | crowd
a person on a skateboard | skateboard
a horse | horse
two people hiking | hik"""


def page_bench():
    st.title("📊 Top-5 Hit-Rate Benchmark")
    st.caption("Format: `query | keyword`. A hit = the keyword appears in a caption of any top-5 photo. Needs captioned gallery.")
    raw = st.text_area("15 benchmark queries", DEFAULT_BENCH, height=330)
    cmp = st.checkbox("Also evaluate with LLM expansion (one API call per query)")
    bert = st.checkbox("Also evaluate a BERT-family baseline (Sentence-BERT over captions, text-only)")
    if not st.button("Run benchmark"):
        return
    emb_g, meta_g = subset("gallery")
    if bert:
        sb = load_sbert()
        cap_emb = sb.encode([m["caption"] for m in meta_g], normalize_embeddings=True)
    rows = []
    for line in raw.splitlines():
        if "|" not in line:
            continue
        q, kw = [s.strip() for s in line.split("|", 1)]
        idx, sims = rank(embed_text([q])[0], emb_g, 5)
        row = {"query": q, "keyword": kw, "top-1 score": round(float(sims[0]), 3),
               "hit@5": any(kw.lower() in meta_g[i]["caption"].lower() for i in idx)}
        if cmp:
            try:
                ex = expand_query(q, style)
                idx2, _ = rank(embed_text([ex])[0], emb_g, 5)
                row["hit@5 (LLM)"] = any(kw.lower() in meta_g[i]["caption"].lower() for i in idx2)
            except Exception:
                row["hit@5 (LLM)"] = None
        if bert:
            idx3, _ = rank(sb.encode([q], normalize_embeddings=True)[0], cap_emb, 5)
            row["hit@5 (SBERT)"] = any(kw.lower() in meta_g[i]["caption"].lower() for i in idx3)
        rows.append(row)
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.metric("Top-5 hit rate (plain CLIP)", f"{df['hit@5'].mean():.0%}")
    if bert:
        st.metric("Top-5 hit rate (Sentence-BERT on captions)", f"{df['hit@5 (SBERT)'].mean():.0%}")
        st.caption("Caption-only baseline: matching keywords against the same captions is partly circular, so treat it as an upper-bound reference, not a fair rival.")
    if cmp:
        st.metric(f"Top-5 hit rate (LLM, {style})", f"{df['hit@5 (LLM)'].dropna().astype(bool).mean():.0%}")


def page_about():
    st.title("ℹ️ About")
    st.markdown(f"""
**Pipeline:** query → (optional) LLM expansion → CLIP text encoder → cosine search over pre-computed image
embeddings → Top-K; if best score < threshold → text-to-image generator.

**Generative vs discriminative:** CLIP is a *discriminative* contrastive model (scores text-image pairs, generates
nothing); SD-Turbo and the LLM are *generative* (model p(image|text) / p(text)). This app chains both.

**Licences & IP:** gallery/portrait sources are listed in `image_metadata.json` and `MODEL_CARD.md`.
Generated images are AI output and are labelled as such. Model: `{MODEL_ID}` (MIT).
""")


{PAGES[0]: page_search, PAGES[1]: page_latent, PAGES[2]: page_bias, PAGES[3]: page_bench, PAGES[4]: page_about}[page]()
