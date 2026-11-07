# P35 · CLIP Text-to-Photo Search

Streamlit app (Generative AI mini-project, Track C) that searches a photo collection by words or by an example image using CLIP (`openai/clip-vit-base-patch32`).

| Task | Feature |
|---|---|
| U1 | Latent Space page: PCA / t-SNE of photo and text embeddings |
| U2 | LLM query expansion (Groq, zero-shot / few-shot) |
| U3 | Text-to-photo and image-to-photo retrieval over 200+ photos, Top-K by cosine similarity; Sentence-BERT baseline in Benchmark |
| U4 | Generative fallback (SD-Turbo / Pollinations) when the best score is below the threshold |
| U5 | Bias Audit page and keyword safety guardrail |

**Live app:** https://https://p35-clip-photo-search.streamlit.app/

## Run locally
```
py -3.11 -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python index_dataset.py        # builds images/, embeddings.npy, image_metadata.json
streamlit run app.py
```

## Secrets (never commit these)
Create `.streamlit/secrets.toml` locally, or paste into the Secrets box on Streamlit Community Cloud:
```
GROQ_API_KEY = "..."
HF_TOKEN = "..."        # optional, for SD-Turbo via Hugging Face
GROQ_MODEL = "openai/gpt-oss-20b"   # optional override
```

## Files
`app.py` · `index_dataset.py` · `requirements.txt` · `MODEL_CARD.md` · `embeddings.npy` · `image_metadata.json` · `images/`

See `MODEL_CARD.md` for intended use, limitations, bias and licence notes.
