import sys
import warnings
from pathlib import Path
from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# import project inference and feature utilities
try:
    from app.inference_utils import predict_ensemble
except Exception as e:
    raise RuntimeError("Failed to import predict_ensemble from app.inference_utils. Original error: " + str(e)) from e

try:
    from src.feature_extraction import (
        extract_lexical_features,
        encode_url_to_char_indices,
        DEFAULT_MAX_LEN,
    )
except Exception as e:
    raise RuntimeError("Failed to import feature extraction utilities from src.feature_extraction. Original error: " + str(e)) from e

app = FastAPI(title="AI Phishing Detector", version="1.0")
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictURLRequest(BaseModel):
    url: str


# Exact feature order the model expects
MODEL_FEATURES: List[str] = [
    "url_length", "host_length", "path_length", "num_dots", "num_path_segments",
    "num_query_params", "count_digits", "count_letters", "digit_letter_ratio",
    "count_special_chars", "num_hyphens", "has_at", "has_ip_in_host",
    "suspicious_tld", "entropy_host", "entropy_path", "percent_encoded_frac",
    "vowel_frac", "has_https", "has_port", "has_fragment", "token_login",
    "token_signin", "token_secure", "token_webscr", "token_bank", "token_verify",
    "token_update", "token_account", "token_confirm", "url_len_bucket_<50",
    "url_len_bucket_50_100", "url_len_bucket_100_200", "url_len_bucket_>=200",
    "dns_ip_count", "dns_resolves"
]

TRUSTED_DOMAINS = {
    "mmmut.ac.in",
    "www.mmmut.ac.in",
    "mmmut.samarth.edu.in",
    # add other known safe domains here
}

TRUSTED_TLDS = {
    "edu.in", "ac.in", "gov.in"
}

def is_trusted_url(url: str) -> bool:
    """Return True if URL matches trusted domains or TLDs."""
    from urllib.parse import urlparse
    host = urlparse(url).netloc.lower()
    if host in TRUSTED_DOMAINS:
        return True
    # crude TLD check for domains like example.ac.in
    parts = host.split(".")
    if len(parts) >= 2:
        tld_candidate = ".".join(parts[-2:])
        if tld_candidate in TRUSTED_TLDS:
            return True
    return False
def _safe_num(v):
    """Return numeric value or 0.0 for None/NaN/unconvertible."""
    try:
        if v is None:
            return 0.0
        if isinstance(v, (float, int, np.floating, np.integer)):
            if np.isnan(v):
                return 0.0
            return float(v)
        return float(v)
    except Exception:
        return 0.0


@app.get("/")
def root():
    return {"status": "ok", "service": "AI Phishing Detector"}


@app.post("/predict_url")
def predict_url(req: PredictURLRequest):
    url = (req.url or "").strip()
    if not url:
        raise HTTPException(status_code=400, detail="Empty url provided")

    # ---- 1) Extract lexical features (always do this so we can return feature values) ----
    try:
        feats = extract_lexical_features(url, perform_network_checks=False)
        if not isinstance(feats, dict):
            raise ValueError("extract_lexical_features must return a dict")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Feature extraction failed: {e}")

    # ---- 2) Sanitize & build numeric feature row (exact order MODEL_FEATURES) ----
    try:
        row = {k: _safe_num(feats.get(k, 0.0)) for k in MODEL_FEATURES}
        X_tab = np.array([list(row.values())], dtype=float)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Building numeric feature vector failed: {e}")

    # If this is a trusted URL, short-circuit and return sanitized features + safe probability
    if is_trusted_url(url):
        produced_vector = X_tab.reshape(-1).tolist()
        return {
            "url": url,
            "is_phishing": False,
            "probability": 0.0,
            "features": {k: float(row.get(k, 0.0)) for k in MODEL_FEATURES},
            "produced_feature_vector": produced_vector,
            "produced_feature_count": int(X_tab.shape[1]),
            "produced_feature_names": MODEL_FEATURES,
            "note": "Trusted domain (whitelisted) — returned safe."
        }

    # ---- 3) Build char sequence ----
    try:
        maxlen = int(DEFAULT_MAX_LEN) if isinstance(DEFAULT_MAX_LEN, int) else 200
        charseq = encode_url_to_char_indices(url, maxlen=maxlen)
        if charseq.ndim == 1:
            charseq = charseq.reshape(1, -1)
        charseq = charseq.astype(np.int64)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Charseq encoding failed: {e}")

    # ---- 4) Sanity check feature length ----
    expected_features = len(MODEL_FEATURES)
    if X_tab.shape[1] != expected_features:
        raise HTTPException(status_code=500, detail={
            "error": "Produced feature count mismatch",
            "produced_feature_count": int(X_tab.shape[1]),
            "expected_feature_count": expected_features,
            "produced_feature_names": MODEL_FEATURES[:min(X_tab.shape[1], expected_features)],
            "produced_feature_vector": X_tab.reshape(-1).tolist()
        })

    # ---- 5) Call ensemble model ----
    try:
        probs = predict_ensemble(X_tab, charseq)
    except Exception as e:
        err_msg = str(e)
        raise HTTPException(status_code=500, detail={
            "error": err_msg,
            "produced_feature_count": int(X_tab.shape[1]),
            "produced_feature_names": MODEL_FEATURES,
            "produced_feature_vector": X_tab.reshape(-1).tolist(),
            "charseq_shape": list(charseq.shape),
        })

    # ---- 6) Post-process prediction safely ----
    try:
        arr = np.asarray(probs).reshape(-1)
        if arr.size == 0:
            raise ValueError("Model returned empty array")
        raw0 = float(arr[0])
        # Apply sigmoid if out of [0,1]
        if 0.0 <= raw0 <= 1.0:
            prob = raw0
        else:
            prob = 1.0 / (1.0 + np.exp(-raw0)) if raw0 >= 0 else np.exp(raw0) / (1.0 + np.exp(raw0))
        prob = float(np.clip(prob, 0.0, 1.0))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Post-processing model output failed: {e}")

    label = bool(prob >= 0.5)

    # ---- 7) Return everything frontend needs (features dict + produced vector) ----
    return {
        "url": url,
        "is_phishing": label,
        "probability": prob,
        "features": {k: float(row.get(k, 0.0)) for k in MODEL_FEATURES},
        "produced_feature_vector": X_tab.reshape(-1).tolist(),
        "produced_feature_count": int(X_tab.shape[1]),
        "produced_feature_names": MODEL_FEATURES
    }
