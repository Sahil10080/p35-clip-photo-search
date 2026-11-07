# Model Card — P35 CLIP Text-to-Photo Search

## 1. System overview
A Streamlit app that retrieves photos from a ~220-image collection using natural-language or example-image queries,
optionally expands vague queries with an LLM, and generates an image when nothing in the collection matches.

## 2. Components
| Component | Role | Details |
|---|---|---|
| `openai/clip-vit-base-patch32` | Retrieval core (U1, U3, U5) | ViT-B/32 image encoder + Transformer text encoder, 512-d shared space, contrastive training on ~400M web image-text pairs. MIT licence. |
| LLM (`llama-3.1-8b-instant` via Groq free tier) | Query expansion (U2) | Rewrites short queries into ≤35-word visual descriptions (zero-shot or few-shot). Llama 3.1 Community Licence; subject to Groq terms. |
| `stabilityai/sd-turbo` (HF Inference API) / Pollinations.ai | Generative fallback (U4) | Distilled text-to-image diffusion model, 1-4 step sampling. Check current SD-Turbo licence terms (Stability AI Community/non-commercial). Pollinations is a free third-party endpoint. |

## 3. Data
- **Gallery:** [FILL: Flickr8k subset (research use only) *or* Picsum/Unsplash-licensed photos]; resized to ≤384 px.
- **Portrait audit set:** FairFace validation subset (CC BY 4.0), with dataset-provided gender and race-group annotations.
- Sources are recorded per image in `image_metadata.json`.

## 4. Intended use
Academic coursework demonstration of multimodal retrieval, query expansion, generative fallback and bias auditing.
**Not intended** for surveillance, identification or profiling of people, hiring, policing, or any decision about individuals.

## 5. Limitations
- CLIP truncates text at 77 tokens, handles counting, negation ("no people") and spatial relations poorly, and is
  weak at fine-grained or text-in-image queries.
- Small gallery: "no match" often reflects collection coverage, not model failure.
- Cosine scores are not calibrated probabilities; the 0.22 threshold is empirical and should be tuned on a validation set.
- A modality gap exists between text and image embeddings; text-image scores sit in a narrow band (~0.15-0.35).
- LLM expansion can hallucinate details that steer retrieval away from the user's intent.
- Generated images may be unrealistic, biased, or inconsistent with the prompt.

## 6. Ethical considerations & bias
- CLIP inherits social biases from web data (occupational and gender/race associations). The **Bias Audit** tab reports,
  per query, the share of top-N portraits by group versus the pool baseline. Results are findings about the model on a
  small sample, not claims about people. Annotation categories are coarse and imperfect.
- **Safety guardrail:** keyword blocklist refuses explicit/violent/self-harm queries before search, expansion and
  generation. The LLM system prompt forbids adding sensitive attributes. A keyword list is a minimal control; it is
  bypassable and not a complete moderation system.
- Generated images are labelled "AI-generated" with their source. Do not present them as real photographs.
- Uploaded images are processed in memory only and not stored. Do not upload images of people without consent.

## 7. Licence & IP note
Code: [FILL: MIT]. Images retain their original licences (see §3). Flickr8k is for research use only and is not
royalty-free; for a public repo prefer Unsplash/Pexels/CC0 images. Model licences as in §2. API keys live in
`st.secrets`, never in the repo.

## 8. Evaluation summary
[FILL from REPORT.md: top-5 hit rate, bias disparity findings, failure case.]
