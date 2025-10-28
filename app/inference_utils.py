# app/inference_utils.py
"""
Robust inference utilities for AI Phishing Detector.

Design goals:
- Same usage as before: call predict_ensemble(X_tab, char_seqs)
- Load models at import but do not crash the process if files are missing; log and defer errors to predict time.
- Handle logits vs probabilities from ONNX safely.
- Validate and sanitize inputs and outputs.
"""

from pathlib import Path
import json
import logging
import numpy as np
import joblib
from src.feature_extraction import DEFAULT_MAX_LEN

# optional import
try:
    import onnxruntime as ort # type: ignore
    _HAS_ONNX = True
except Exception:
    ort = None
    _HAS_ONNX = False

logger = logging.getLogger("inference_utils")
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))
    logger.addHandler(ch)
logger.setLevel(logging.INFO)

# Paths (project root relative)
BASE = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE / "models"
META_PATH = BASE / "data" / "features" / "live_dataset_meta.json"

# Lazy-loaded model objects (attempt to load at import; set to None on failure)
lgb_model = None
onnx_sess = None
vocab_size = 128  # default

def _safe_sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid for numpy arrays."""
    x = np.asarray(x, dtype=float)
    with np.errstate(over='ignore'):
        pos = x >= 0
        out = np.empty_like(x, dtype=float)
        out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
        exp_x = np.exp(x[~pos])
        out[~pos] = exp_x / (1.0 + exp_x)
    return out

def _looks_like_probability_array(arr: np.ndarray, tol: float = 1e-6) -> bool:
    """Return True if arr values are (approximately) in [0,1] and no nan/inf."""
    a = np.asarray(arr, dtype=float)
    if a.size == 0:
        return False
    if np.isnan(a).any() or np.isinf(a).any():
        return False
    return (a.min() >= -tol) and (a.max() <= 1.0 + tol)

def _ensure_2d_float(X) -> np.ndarray:
    """Convert input to 2D float numpy array, replace nan/inf with safe values."""
    arr = np.asarray(X)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array-like input, got shape {arr.shape}")
    arr = arr.astype(float)
    arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
    return arr

def _ensure_charseq_array(char_seqs, maxlen: int) -> np.ndarray:
    """Make char_seqs into 2D int64 array, pad/clip to maxlen if needed."""
    arr = np.asarray(char_seqs)
    if arr.ndim == 1:
        # treat as single sequence
        if arr.size != maxlen:
            # pad or trim
            a = np.zeros(maxlen, dtype=int)
            a[:min(maxlen, arr.size)] = arr.flatten()[:min(maxlen, arr.size)]
            arr = a
        arr = arr.reshape(1, -1)
    elif arr.ndim == 2:
        # ensure each row has length maxlen: pad/truncate per-row if needed
        n, L = arr.shape
        if L != maxlen:
            new = np.zeros((n, maxlen), dtype=int)
            for i in range(n):
                row = arr[i]
                Lr = row.size
                new[i, :min(maxlen, Lr)] = row[:min(maxlen, Lr)]
            arr = new
    else:
        raise ValueError(f"char_seqs must be 1D or 2D array-like, got ndim={arr.ndim}")
    arr = np.clip(arr, 0, np.iinfo(np.int64).max).astype(np.int64)
    return arr

def _load_meta(path: Path = None):
    global vocab_size
    p = Path(path) if path else META_PATH
    if p.exists():
        try:
            with open(p, "r") as f:
                meta = json.load(f)
            vocab_size = int(meta.get("vocab_size", vocab_size))
            logger.info(f"Loaded meta from {p} (vocab_size={vocab_size})")
        except Exception as e:
            logger.warning(f"Failed to load meta {p}: {e} — using defaults (vocab_size={vocab_size})")
    else:
        logger.info(f"Meta file {p} not found — using default vocab_size={vocab_size}")

def reload_models(lgb_path: Path = None, onnx_path: Path = None, meta_path: Path = None):
    """Explicitly reload models (useful in dev / after model updates)."""
    global lgb_model, onnx_sess
    # LGB
    lp = Path(lgb_path) if lgb_path else MODELS_DIR / "lgb_calibrated.pkl"
    try:
        lgb_model = joblib.load(lp)
        logger.info(f"Loaded LightGBM model from {lp}")
    except Exception as e:
        lgb_model = None
        logger.warning(f"Failed to load LightGBM model from {lp}: {e}")

    # ONNX
    if _HAS_ONNX:
        op = Path(onnx_path) if onnx_path else MODELS_DIR / "charcnn.onnx"
        try:
            onnx_sess = ort.InferenceSession(str(op), providers=["CPUExecutionProvider"])
            logger.info(f"Loaded ONNX session from {op}")
        except Exception as e:
            onnx_sess = None
            logger.warning(f"Failed to load ONNX model from {op}: {e}")
    else:
        onnx_sess = None
        logger.info("onnxruntime not available; ONNX model will not be used.")

    # meta
    _load_meta(meta_path)

# load at import time (best-effort)
try:
    reload_models()
except Exception as e:
    # reload_models handles its own warnings; catch any unexpected exceptions
    logger.exception("Unexpected error during initial model load: %s", e)


def _predict_lgb_prob(X_tab: np.ndarray) -> np.ndarray:
    """Return 1D array of probabilities from LightGBM model (or raise)."""
    if lgb_model is None:
        raise RuntimeError("LightGBM model not loaded (lgb_model is None). Call reload_models() with correct path.")
    X_tab = _ensure_2d_float(X_tab)
    # prefer predict_proba
    if hasattr(lgb_model, "predict_proba"):
        out = lgb_model.predict_proba(X_tab)
        out = np.asarray(out)
        if out.ndim == 2 and out.shape[1] >= 2:
            probs = out[:, 1]
        elif out.ndim == 1:
            probs = out.reshape(-1)
        else:
            raise RuntimeError(f"Unexpected LightGBM predict_proba output shape: {out.shape}")
    elif hasattr(lgb_model, "predict"):
        out = lgb_model.predict(X_tab)
        probs = np.asarray(out).reshape(-1)
        if not _looks_like_probability_array(probs):
            probs = _safe_sigmoid(probs)
    else:
        raise RuntimeError("Loaded LightGBM object has neither predict_proba nor predict")
    probs = np.nan_to_num(probs, nan=0.0, posinf=1.0, neginf=0.0)
    return np.clip(probs, 0.0, 1.0)


def _predict_char_prob(onnx_session, char_seqs: np.ndarray, bs: int = 256, input_name: str = None, vocab: int = None) -> np.ndarray:
    """Run ONNX char-CNN and return 1D array of probabilities in [0,1]."""
    if onnx_session is None:
        raise RuntimeError("ONNX session is not loaded (onnx_sess is None).")
    if vocab is None:
        vocab = vocab_size
    arr = np.asarray(char_seqs)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    arr = np.clip(arr, 0, vocab - 1).astype(np.int64)

    # infer input name if not provided
    if input_name is None:
        try:
            input_name = onnx_session.get_inputs()[0].name
        except Exception:
            input_name = "input"

    outputs = []
    for i in range(0, arr.shape[0], bs):
        xb = arr[i:i+bs]
        out = onnx_session.run(None, {input_name: xb})[0]
        out = np.atleast_1d(np.asarray(out))
        # flatten to 1D probabilities/logits per-row
        if out.ndim > 1 and out.shape[1] == 1:
            out = out.reshape(-1)
        out = out.reshape(-1)
        # if out not in [0,1], apply sigmoid (assume logits)
        if not _looks_like_probability_array(out):
            out = _safe_sigmoid(out)
        outputs.extend(out.tolist())

    probs = np.asarray(outputs, dtype=float)
    probs = np.nan_to_num(probs, nan=0.0, posinf=1.0, neginf=0.0)
    return np.clip(probs, 0.0, 1.0)


def predict_ensemble(X_tab, char_seqs, weight_lgb: float = 0.6, weight_char: float = 0.4, bs: int = 256):
    """
    Main public API.

    - X_tab: array-like (n, num_tab_features) or (num_tab_features,) for single sample
    - char_seqs: array-like (n, char_maxlen) or (char_maxlen,) for single sample
    - returns: 1D numpy array of probabilities shape (n,)
    """
    # normalize weights
    w_sum = float(weight_lgb) + float(weight_char)
    if w_sum <= 0:
        raise ValueError("Sum of weights must be > 0")
    weight_lgb = float(weight_lgb) / w_sum
    weight_char = float(weight_char) / w_sum

    # prepare inputs
    X_tab_arr = _ensure_2d_float(X_tab)
    # ensure vocab_size loaded
    if vocab_size is None:
        _load_meta()

    char_arr = _ensure_charseq_array(char_seqs, maxlen=DEFAULT_MAX_LEN)  # default maxlen placeholder
    # If you want to respect meta maxlen, caller should ensure encode_url_to_char_indices uses same length;
    # here we only standardize shape. If your char encoder maxlen differs, set char_arr accordingly.

    # get lgb probs
    lgb_probs = _predict_lgb_prob(X_tab_arr)

    # get char probs (if ONNX available)
    try:
        if _HAS_ONNX and onnx_sess is not None:
            char_probs = _predict_char_prob(onnx_sess, char_arr, bs=bs, vocab=vocab_size)
        else:
            # fallback: zeros
            char_probs = np.zeros_like(lgb_probs)
    except Exception as e:
        logger.warning("ONNX char-CNN inference failed; falling back to zeros. Error: %s", e)
        char_probs = np.zeros_like(lgb_probs)

    # align lengths
    if char_probs.shape[0] != lgb_probs.shape[0]:
        if char_probs.size == 1:
            char_probs = np.full_like(lgb_probs, float(char_probs.item()))
        else:
            # pad/truncate char_probs to match lgb_probs length
            n = lgb_probs.shape[0]
            cp = np.zeros(n, dtype=float)
            cp[:min(n, char_probs.shape[0])] = char_probs[:min(n, char_probs.shape[0])]
            char_probs = cp

    final = weight_lgb * lgb_probs + weight_char * char_probs
    final = np.nan_to_num(final, nan=0.0, posinf=1.0, neginf=0.0)
    final = np.clip(final, 0.0, 1.0)
    return final

# Attempt to load models/subcomponents when module imported (best-effort)
try:
    reload_models()
except Exception as e:
    logger.warning("Initial model reload had an error (continuing): %s", e)
