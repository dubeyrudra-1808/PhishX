# PhishX — AI-powered URL Phishing Detector

> **PhishX** is an explainable, production-minded project that detects phishing URLs using a hybrid approach:
> a calibrated LightGBM model trained on engineered URL features (lexical, host, structural) combined with a character-level convolutional neural network (Char‑CNN) exported to ONNX for fast inference.  
> This README contains everything you need to understand the scientific intuition, data features, model design, and how to run, train, evaluate, and deploy PhishX.

---

## Quick navigation

- [Overview](#overview)
- [Why PhishX?](#why-phishx)
- [High-level architecture](#high-level-architecture)
- [Core concepts & theory](#core-concepts--theory)
  - [What is phishing?](#what-is-phishing)
  - [Why URL-only detection? pros/cons](#why-url-only-detection-proscons)
  - [Feature families explained](#feature-families-explained)
  - [Char‑CNN explained (intuition)](#char-cnn-explained-intuition)
  - [LightGBM & calibration](#lightgbm--calibration)
  - [Ensembling strategy](#ensembling-strategy)
  - [Evaluation metrics & practices](#evaluation-metrics--practices)
- [Project layout (what's in this repo)](#project-layout-whats-in-this-repo)
- [Installation & environment](#installation--environment)
- [How to use — quick start](#how-to-use---quick-start)
  - [Run API (FastAPI)](#run-api-fastapi)
  - [Use Python inference functions](#use-python-inference-functions)
  - [Run the Colab / Notebook](#run-the-colab--notebook)
- [Training from scratch](#training-from-scratch)
- [Data: where it comes from & how to prepare](#data-where-it-comes-from--how-to-prepare)
- [Interpreting predictions & explanations](#interpreting-predictions--explanations)
- [Deployment notes & tips](#deployment-notes--tips)
- [Troubleshooting & FAQ](#troubleshooting--faq)
- [Contributing, license, credits](#contributing-license-credits)

---

## Overview

PhishX classifies whether a URL is malicious (phishing) or benign (safe) using URL-only signals. It is intended to be:

- Fast and robust for production inference (ONNX for Char‑CNN; joblib pickles for LightGBM).
- Explainable: returns feature vectors and per-feature values alongside a final probability.
- Deployment-friendly: includes a FastAPI server for serving predictions, utilities for feature extraction, and training scripts.

Use cases: email scanners, browser extensions (server-side), triage systems, SOC pipelines.

---

## Why PhishX?

Phishing remains one of the most effective and persistent cyber threats. Full-page analysis (fetching HTML/JS) yields richer signals but is expensive and risky (drive-by downloads, CORS, scraping complexity). PhishX focuses on **URL-only** inference:

- **Pros:** fast, inexpensive, privacy-preserving, safe (no remote fetch), suitable for real-time pipelines.
- **Cons:** may miss phishing pages that rely only on page content or images; URL camouflage can be sophisticated.

PhishX mitigates these limitations by combining **hand-engineered features** (which capture suspicious patterns) with a **char-level neural model** that detects subtle string patterns.

---

## High-level architecture

1. **Feature extraction module** (`src/feature_extraction.py`):
   - Extracts lexical and host-level features from URLs (lengths, token counts, presence of IP, special characters, entropy proxies, TLD, subdomain statistics, percent-encoding, etc.)
   - Optionally performs lightweight network checks (DNS) — *disabled by default for safety*.

2. **Tabular model**:
   - LightGBM classifier trained on extracted features.
   - Post-training probability calibration (e.g., isotonic / sigmoid) for well calibrated outputs.

3. **Character-level model (Char‑CNN)**:
   - Small ConvNet over character sequences (fixed-length) that learns patterns like brand misspellings, homoglyphs, or concatenations that are hard to capture with rules.

4. **Ensemble & serving** (`app/inference_utils.py`, `app/main.py`):
   - At inference, both models produce probabilities which are combined with configurable weights (default around `0.6` tabular, `0.4` char).
   - FastAPI serves a prediction endpoint and returns probability, binary label, and the feature vector for explainability.

---

## Core concepts & theory

### What is phishing?

Phishing is a social-engineering attack where an attacker obtains sensitive information by masquerading as a trustworthy entity — often via URLs that imitate legitimate domains.

### Why URL-only detection? pros/cons

(see "Why PhishX?" above)

### Feature families explained

PhishX extracts several families of features. Here are the most important ones with intuition.

1. **Lexical features (URL tokens & structure)**
   - `length` — very long URLs can indicate obfuscated or encoded payloads.
   - `count_digits`, `count_hyphens`, `count_params` — unusual counts can be suspicious.
   - `has_at_symbol` (`@`) — indicates potential redirection.
   - `percent_encoded_count` — encoded characters frequently used to hide content.

2. **Host features**
   - `is_ip_host` — direct IP address in host is suspicious.
   - `tld` and `tld_suspicious` — uncommon or recently-registered TLDs sometimes used.
   - `subdomain_depth` — many nested subdomains can mask the real host.

3. **Statistical / entropy-like features**
   - `char_entropy_approx` or `ratio_non_alnum` — high randomness suggests autogenerated domains.

4. **Bag-of-words / token features**
   - Presence of brand names, login, secure, accounts, update, etc. — certain tokens appear frequently in phishing URLs.

5. **Char-sequence (for Char‑CNN)**
   - The URL is converted to a fixed-length integer sequence mapped from characters. CNN learns local motifs (e.g., `goog1e`, `paypa1`) that mimic legitimate brands.

### Char‑CNN explained (intuition)

- Operates over characters, not words.
- Convolutional filters capture n-gram-like patterns at character-level (e.g., `bank`, `b@nk`, `bank-login`), helping catch obfuscations like homoglyphs or insertion of extra tokens.
- Exporting the trained PyTorch model to ONNX provides low-latency inference in many runtimes.

### LightGBM & calibration

- LightGBM provides strong baseline performance on tabular features.
- Calibration (isotonic or Platt scaling) ensures probabilities reflect real-world likelihood, which is important when the output is used to make thresholded alerts.

### Ensembling strategy

- Weighted average of calibrated LightGBM probability and Char‑CNN probability:
  ```
  p_final = w_lgb * p_lgb + w_char * p_char
  ```
- Weights can be tuned on a validation set; defaults are conservative (e.g., 0.6 / 0.4).

### Evaluation metrics & practices

- **ROC-AUC** — overall ranking ability.
- **Precision / Recall / F1** — critical when you care about false positives (blocking legit URLs) vs false negatives (letting phishing pass).
- **Precision@k** or **Recall at fixed FPR** — practical for alerting budgets.
- **Confusion matrix** and **calibration plots** help diagnose miscalibration or class imbalance issues.

---

## Project layout (what's in this repo)

```
phishx_project/              # extracted artifact (source)
├─ app/
│  ├─ main.py                # FastAPI app
│  ├─ inference_utils.py     # model loading, predict_ensemble API
│  └─ requirements.txt
├─ src/
│  ├─ feature_extraction.py  # URL-to-feature pipeline used by training + API
│  └─ train_models.py        # training & export (LightGBM + Char‑CNN)
├─ models/                   # trained artifacts (lgb, charcnn.onnx, metadata)
├─ data/
│  ├─ raw/                   # raw datasets (not included here)
│  └─ processed/             # processed features
├─ notebooks/
│  └─ AI_Phishing_Detector.ipynb  # analysis & demo notebook
├─ frontend.html             # small demo frontend
└─ README.md (original)      # original README (replaced by this file)
```

---

## Installation & environment

**Recommended:** create a virtual environment (venv / conda) and install dependencies from the `app/requirements.txt`.

```bash
# create venv
python -m venv .venv
source .venv/bin/activate    # Linux / macOS
# or for Windows: .venv\Scripts\activate

# install requirements (server + common libs)
pip install -r phishx_project/app/requirements.txt
# optionally install training deps (pytorch, lightgbm, onnxruntime) if you plan to train:
pip install lightgbm joblib numpy pandas scikit-learn onnxruntime torch
```

> Note: GPU training requires appropriate PyTorch CUDA builds; ONNX export requires `onnx` and `onnxruntime`. The training script tries to export ONNX when available.

---

## How to use — quick start

### Run API (FastAPI)

From repository root:

```bash
cd phishx_project
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

**Example `curl` request:**

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"url": "http://example.com/login"}'
```

**Example response (JSON):**
```json
{
  "url": "http://example.com/login",
  "is_phishing": false,
  "probability": 0.031,
  "features": { "url_length": 18, "count_digits": 0, ... },
  "produced_feature_vector": [18.0, 0.0, ...],
  "produced_feature_count": 64,
  "produced_feature_names": ["url_length","count_digits", ...]
}
```

### Use Python inference functions directly

```python
from app.inference_utils import predict_ensemble
from src.feature_extraction import build_features_for_df

# create a simple DataFrame
import pandas as pd
df = pd.DataFrame({"url": ["http://example.com/login", "http://malicious-example.com/bank"]})
X_tab, char_seqs, feature_names = build_features_for_df(df["url"].tolist(), save=False)

# Get probabilities
probs = predict_ensemble(X_tab, char_seqs)
print(probs)  # array of probabilities per URL
```

### Run the Colab / Notebook

Open `notebooks/AI_Phishing_Detector.ipynb` (or the root `AI_Phishing_Detector.ipynb` if present) and run cells. Notebook contains exploratory analysis, training examples and demo inference code.

---

## Training from scratch

High-level steps implemented in `src/train_models.py`:

1. Prepare / gather labeled dataset of URLs (phishing vs benign). See Data section below.
2. Run the feature extraction pipeline to produce tabular features and char sequences.
3. Train LightGBM on extracted features; validate and calibrate probabilities.
4. Train a Char‑CNN (PyTorch) on char sequences; export to ONNX if desired.
5. Save artifacts to `models/`:
   - `lgb_raw.pkl`, `lgb_calibrated.pkl`, `charcnn.pt`, `charcnn.onnx`, `ensemble_metadata.json`, `test_predictions.csv`

Example:

```bash
python src/train_models.py --data data/raw/your_dataset.csv --out models/
```

> The training script includes safe defaults and will skip ONNX export if dependencies are missing.

---

## Data: where it comes from & how to prepare

Phishing datasets can come from multiple sources:

- Public datasets: PhishTank, OpenPhish (subject to license), various academic datasets.
- Crawled benign domains: Common crawl or Alexa top sites (be careful with licensing).
- Internal telemetry (corporate URLs) — valuable for production performance.

**Important data hygiene notes:**

- Keep train/validation/test splits **time-aware** (simulate real-world release of new malicious domains).
- Avoid data leakage: if multiple URLs share the same domain, ensure they are not split across train/test incorrectly.
- Maintain class balance or use appropriate sample weighting and metric selection.

**Minimal input format** (used by `train_models.py` and feature extractor):

CSV with at least columns:
```
url,label
http://example.com,0
http://malicious.example/phish,1
```
`label` should be `1` for phishing/malicious and `0` for benign.

---

## Interpreting predictions & explanations

PhishX returns:

- `probability` (0.0–1.0): ensemble probability that the URL is phishing.
- `is_phishing` (bool): thresholded label (default threshold `0.5`).
- `features` (dict): numeric feature values that contributed to the tabular model.
- `produced_feature_names` and `produced_feature_vector`: useful for post-hoc explainers (SHAP, LIME).

**Explainability tips**

- Use per-feature differences between a suspicious URL and a benign baseline to craft human-readable explanations.
- For automated alerts, consider a two-tier system:
  - `probability >= 0.9` -> high confidence — auto-block
  - `0.6 <= probability < 0.9` -> human review
  - `probability < 0.6` -> monitor/low priority

Adjust thresholds to your operational tolerance for false positives.

---

## Deployment notes & tips

- **Model files**: keep `models/` in a secure, versioned artifact store. Use hashing to ensure integrity.
- **Runtime**: ONNXRuntime and joblib are lightweight. CPU inference should be fast (ms to tens of ms per URL depending on batch).
- **Scaling**: run the FastAPI app behind a production ASGI server (Uvicorn/Gunicorn or container orchestrator).
- **Security**: never fetch or render remote URLs in the inference pipeline by default (this repo's default is URL-only). If you enable network checks, run them in sandboxed environments.
- **Monitoring**: log model input distributions and prediction drift; periodically re-evaluate models.

---

## Troubleshooting & FAQ

**Q: Models don't load — errors at startup**
A: Ensure the `models/` directory contains `lgb_calibrated.pkl` and `charcnn.onnx` (if used). The `app.inference_utils` module does best-effort loads; missing artifacts are logged and will error at predict-time.

**Q: I get incorrect probabilities / miscalibration**
A: Re-run calibration stage on a hold-out validation set. Check class imbalance and sampling.

**Q: Can I use only the tabular model?**
A: Yes — `predict_ensemble` accepts weights; set `weight_char=0.0` to use LightGBM only.

**Q: Is this safe for production?**
A: The project is designed to be production-friendly but requires standard production hardening: secrets, secure storage, monitoring, CI, and model governance.

---

## Contribution & extending PhishX

- Suggested extensions:
  - Add domain age & WHOIS-based features (careful with privacy).
  - Add URL semantic embeddings (BPE or subword tokenizers).
  - Implement adversarial training and robustness checks.
  - Continuous learning: daily retraining pipelines with careful monitoring.

To contribute:

1. Fork the project.
2. Create a feature branch.
3. Open PR with tests and rationale.
4. Sign the CLA if required by your organization.

---

## License & credits

PhishX (this repository) is provided under the MIT License — adapt as needed for your use. Credit to original codebase (AI Phishing Detector) and contributors included in repository metadata.

---

## Final notes (practical checklist)

- [ ] Create virtualenv and install `app/requirements.txt`.
- [ ] Place model artifacts (`lgb_calibrated.pkl`, `charcnn.onnx`) into `models/` or train using `src/train_models.py`.
- [ ] Run `uvicorn app.main:app --reload` and test `/predict`.
- [ ] Tune thresholds and weights for your production risk tolerance.

---

### Appendix — quick reference commands

Run server:
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Test URL:
```bash
python -c "from app.inference_utils import predict_ensemble; import pandas as pd; from src.feature_extraction import build_features_for_df; df=pd.Series(['http://example.com']); X,char,names=build_features_for_df(df.tolist()); print(predict_ensemble(X,char))"
```

Train (example):
```bash
python src/train_models.py --data data/raw/your_dataset.csv --out models/
```

---

If you want, I can:
- produce a compact `README.md` zipped together with the server ready for GitHub, or
- generate example notebooks demonstrating model explanations (SHAP) and deployment-ready Dockerfile.