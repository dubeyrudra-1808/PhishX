# /content/drive/MyDrive/AI_Phishing_Detector/src/train_models.py
"""
Robust training script (LightGBM + Char-CNN) with calibration and ensemble.
This version will attempt to install ONNX/ONNXRuntime if not present and export the Char-CNN to ONNX.
Saves:
 - models/lgb_calibrated.pkl
 - models/lgb_raw.pkl
 - models/charcnn.pt
 - models/charcnn.onnx (best-effort)
 - models/ensemble_metadata.json
 - models/test_predictions.csv
"""
import os
import sys
import json
import time
import random
import joblib
import subprocess
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# ML libs
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_recall_fscore_support, confusion_matrix,
    accuracy_score
)
from sklearn.calibration import CalibratedClassifierCV

# Torch for Char-CNN
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Repro
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Paths
BASE = Path("/content/drive/MyDrive/AI_Phishing_Detector")
FEATURES_DIRS = [
    BASE / "data" / "features",
    BASE / "data" / "processed"
]
MODELS_DIR = BASE / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Find files helper
def find_file(names):
    for d in FEATURES_DIRS:
        for name in names:
            p = d / name
            if p.exists():
                return p
    return None

LGB_READY = find_file(["live_features_for_lgb.csv", "live_features_for_lgb.csv"])
FEATURES_FULL = find_file(["live_dataset_features.csv", "live_features.csv"])
CHAR_NPY = find_file(["live_dataset_char_seqs.npy", "live_char_seqs.npy", "live_dataset_char_seqs.npy"])
CHAR_META = find_file(["live_dataset_meta.json", "live_char_meta.json", "live_dataset_meta.json"])
DATASET_CSV = BASE / "data" / "processed" / "live_dataset.csv"
TRAIN_IDX = BASE / "data" / "processed" / "train_idx.npy"
TEST_IDX = BASE / "data" / "processed" / "test_idx.npy"

if LGB_READY is None and FEATURES_FULL is None:
    raise FileNotFoundError("No features CSV found. Ensure earlier feature extraction saved live_features_for_lgb.csv or live_dataset_features.csv in data/features or data/processed.")
if CHAR_NPY is None:
    raise FileNotFoundError("Char seqs .npy not found. Ensure live_dataset_char_seqs.npy or live_char_seqs.npy exists under data/features.")

print("Using feature file (LGB-ready):", LGB_READY if LGB_READY else "(will derive from full features)")
print("Using full features file:", FEATURES_FULL)
print("Char seqs file:", CHAR_NPY)
print("Char meta file (optional):", CHAR_META)
print("Labels file:", DATASET_CSV if DATASET_CSV.exists() else "(not found; labels must be in features CSV)")

# Hyperparams
CHAR_MAXLEN = 200
CHAR_BATCH_SIZE = 256 if torch.cuda.is_available() else 64
CHAR_EPOCHS = int(os.environ.get("CHAR_EPOCHS", "12")) if torch.cuda.is_available() else int(os.environ.get("CHAR_EPOCHS", "6"))
CHAR_LR = 1e-3
CHAR_WEIGHT = float(os.environ.get("CHAR_WEIGHT", 0.6))
LGB_WEIGHT = float(os.environ.get("LGB_WEIGHT", 0.4))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}; CHAR_BATCH_SIZE={CHAR_BATCH_SIZE}; CHAR_EPOCHS={CHAR_EPOCHS}")

# Utility
def safe_read_csv(path):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    return pd.read_csv(p)

# Load data
print("Loading datasets...")
if LGB_READY and Path(LGB_READY).exists():
    df_lgb = safe_read_csv(LGB_READY)
else:
    df_full = safe_read_csv(FEATURES_FULL)
    for col in ["registered_domain", "tld"]:
        if col in df_full.columns:
            df_full = df_full.drop(columns=[col])
    df_full = df_full.fillna(0)
    df_lgb = df_full

if "label" not in df_lgb.columns:
    if DATASET_CSV.exists():
        labels = safe_read_csv(DATASET_CSV)["label"].values
        if len(labels) != len(df_lgb):
            raise ValueError("Label length mismatch")
        df_lgb["label"] = labels
    else:
        raise ValueError("No 'label' column and no live_dataset.csv found")

char_seqs = np.load(CHAR_NPY)
N = len(char_seqs)
if len(df_lgb) != N:
    raise ValueError(f"Row count mismatch: tabular={len(df_lgb)} vs char_seqs={N}")

y = df_lgb["label"].values
X_tab = df_lgb.drop(columns=["label"])

# Splits (use saved indices if available)
if TRAIN_IDX.exists() and TEST_IDX.exists():
    train_idx = np.load(TRAIN_IDX)
    test_idx = np.load(TEST_IDX)
    tr_idx, val_idx = train_test_split(train_idx, test_size=0.125, random_state=SEED, stratify=y[train_idx])
else:
    tr_idx, test_idx = train_test_split(np.arange(N), test_size=0.2, random_state=SEED, stratify=y)
    tr_idx, val_idx = train_test_split(tr_idx, test_size=0.125, random_state=SEED, stratify=y[tr_idx])

print(f"Sizes -> train: {len(tr_idx)}, val: {len(val_idx)}, test: {len(test_idx)}")
print("Class counts (train):", np.bincount(y[tr_idx].astype(int)))

# LightGBM training (sklearn API)
def train_lightgbm_sklearn(X_tab, y, tr_idx, val_idx):
    print("Training LightGBM (sklearn API)...")
    X_train = X_tab.iloc[tr_idx]
    y_train = y[tr_idx]
    X_val = X_tab.iloc[val_idx]
    y_val = y[val_idx]

    clf = lgb.LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=64,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=SEED,
        n_jobs=-1
    )

    try:
        clf.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric='auc',
            early_stopping_rounds=50,
            verbose=100
        )
    except TypeError:
        print("Warning: early_stopping_rounds not supported in this LightGBM version; running fit without early stopping.")
        clf.fit(X_train, y_train)

    joblib.dump(clf, MODELS_DIR / "lgb_raw.pkl")
    print("Saved raw LGB sklearn model:", MODELS_DIR / "lgb_raw.pkl")

    print("Calibrating LightGBM probabilities (sigmoid) using validation set...")
    try:
        calibrator = CalibratedClassifierCV(clf, cv='prefit', method='sigmoid')
        calibrator.fit(X_val, y_val)
    except Exception as e:
        print("Calibrator prefit failed:", str(e))
        calibrator = CalibratedClassifierCV(base_estimator=clf, method='sigmoid', cv=3)
        calibrator.fit(np.vstack([X_train, X_val]), np.concatenate([y_train, y_val]))

    joblib.dump(calibrator, MODELS_DIR / "lgb_calibrated.pkl")
    print("Saved calibrated LGB model:", MODELS_DIR / "lgb_calibrated.pkl")
    return calibrator

# Char-CNN model + training
class CharDataset(torch.utils.data.Dataset):
    def __init__(self, seqs, labels, indices):
        self.seqs = seqs[indices]
        self.labels = labels[indices].astype(np.float32)
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        return torch.from_numpy(self.seqs[idx]).long(), torch.tensor(self.labels[idx]).float()

class CharCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim=16, num_filters=128, fc_dim=64, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim, padding_idx=0)
        self.conv3 = nn.Conv1d(in_channels=embed_dim, out_channels=num_filters, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(in_channels=embed_dim, out_channels=num_filters, kernel_size=5, padding=2)
        self.conv7 = nn.Conv1d(in_channels=embed_dim, out_channels=num_filters, kernel_size=7, padding=3)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(num_filters * 3, fc_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_dim, 1)
        )

    def forward(self, x):
        emb = self.embedding(x)
        emb = emb.permute(0, 2, 1)
        o3 = self.pool(self.relu(self.conv3(emb))).squeeze(-1)
        o5 = self.pool(self.relu(self.conv5(emb))).squeeze(-1)
        o7 = self.pool(self.relu(self.conv7(emb))).squeeze(-1)
        cat = torch.cat([o3, o5, o7], dim=1)
        out = self.fc(cat).squeeze(-1)
        return out

def ensure_onnx_installed():
    """Try importing onnx; if missing, attempt pip install (best-effort)."""
    try:
        import onnx  # noqa: F401
        return True
    except Exception:
        print("onnx not installed; attempting to install onnx and onnxruntime via pip...")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "onnx", "onnxruntime"], check=True, stdout=subprocess.PIPE)
            import onnx  # noqa: F401
            return True
        except Exception as e:
            print("Failed to install onnx packages:", e)
            return False

def export_charcnn_to_onnx(model, vocab_size, onnx_path):
    """Attempt ONNX export with new exporter when possible, fallback otherwise."""
    try:
        installed = ensure_onnx_installed()
        if not installed:
            print("ONNX not available; skipping ONNX export.")
            return False
        # prepare cpu copy
        model_cpu = CharCNN(vocab_size=vocab_size)
        model_cpu.load_state_dict({k.replace('module.',''):v for k,v in model.state_dict().items()})
        model_cpu.eval()
        dummy = torch.randint(0, vocab_size, (1, CHAR_MAXLEN), dtype=torch.long)
        # Try new exporter (dynamo) first
        try:
            torch.onnx.export(
                model_cpu,
                dummy,
                str(onnx_path),
                input_names=["input"],
                output_names=["output"],
                opset_version=14,
                dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
                verbose=False,
                export_params=True,
                do_constant_folding=True,
                training=torch.onnx.TrainingMode.EVAL,
                dynamo=True
            )
            print("Exported Char-CNN to ONNX (dynamo):", onnx_path)
            return True
        except TypeError as e:
            # older torch versions or unsupported arg; fallback to legacy export
            print("dynamo export failed or not supported, falling back to legacy ONNX export:", e)
        except Exception as e:
            print("dynamo export raised exception, will try legacy exporter:", e)

        try:
            torch.onnx.export(
                model_cpu,
                dummy,
                str(onnx_path),
                input_names=["input"],
                output_names=["output"],
                opset_version=14,
                dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
                verbose=False,
                export_params=True,
                do_constant_folding=True
            )
            print("Exported Char-CNN to ONNX (legacy):", onnx_path)
            return True
        except Exception as e:
            print("Legacy ONNX export failed:", e)
            return False
    except Exception as e:
        print("ONNX export overall failed:", e)
        return False

def train_charcnn(seqs, labels, tr_idx, val_idx, device):
    # local imports to avoid any odd scoping/resolution issues in some runtimes
    import torch
    import torch.nn as nn
    print("Training Char-CNN...")
    vocab_size = int(seqs.max()) + 2
    model = CharCNN(vocab_size=vocab_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=CHAR_LR)
    criterion = nn.BCEWithLogitsLoss()

    train_ds = CharDataset(seqs, labels, tr_idx)
    val_ds = CharDataset(seqs, labels, val_idx)
    num_workers = 2 if os.name != 'nt' else 0
    train_loader = DataLoader(train_ds, batch_size=CHAR_BATCH_SIZE, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=CHAR_BATCH_SIZE*2, shuffle=False, num_workers=num_workers, pin_memory=True)

    best_auc = 0.0
    best_state = None
    patience = 3
    wait = 0
    for epoch in range(1, CHAR_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        cnt = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
            cnt += xb.size(0)
        epoch_loss = total_loss / max(1, cnt)

        model.eval()
        preds = []
        trues = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                logits = model(xb)
                probs = torch.sigmoid(logits).cpu().numpy()
                preds.append(probs)
                trues.append(yb.cpu().numpy())
        preds = np.concatenate(preds).ravel()
        trues = np.concatenate(trues).ravel()
        try:
            auc = roc_auc_score(trues, preds)
        except Exception:
            auc = 0.0
        print(f"Epoch {epoch}/{CHAR_EPOCHS} — loss: {epoch_loss:.4f}, val_auc: {auc:.4f}")
        if auc > best_auc + 1e-4:
            best_auc = auc
            best_state = {k:v.cpu() for k,v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print("Early stopping Char-CNN (patience reached).")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save(model.state_dict(), MODELS_DIR / "charcnn.pt")
    print("Saved Char-CNN state to", MODELS_DIR / "charcnn.pt")

    # ONNX export attempt (best-effort)
    onnx_path = MODELS_DIR / "charcnn.onnx"
    exported = export_charcnn_to_onnx(model, int(seqs.max()) + 2, onnx_path)
    if not exported:
        print("ONNX export skipped or failed. If you want ONNX, install 'onnx' and 'onnxruntime' then re-run export.")
    return model

# Evaluation helpers
def evaluate_predictions(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    return dict(accuracy=float(acc), roc_auc=float(auc), pr_auc=float(pr_auc), precision=float(p), recall=float(r), f1=float(f1), cm=cm.tolist())

def ensemble_predict(lgb_calib, char_model, X_tab, char_seqs_array, indices, device):
    batch_idx = np.array(indices)
    X_subset = X_tab.iloc[batch_idx]
    lgb_prob = lgb_calib.predict_proba(X_subset)[:,1]
    char_probs = []
    char_model.eval()
    bs = 1024
    with torch.no_grad():
        for i in range(0, len(batch_idx), bs):
            idxs = batch_idx[i:i+bs]
            xb = torch.from_numpy(char_seqs_array[idxs]).long().to(device)
            logits = char_model(xb)
            probs = torch.sigmoid(logits).cpu().numpy()
            char_probs.append(probs)
    char_prob = np.concatenate(char_probs).ravel()
    final_prob = CHAR_WEIGHT * char_prob + LGB_WEIGHT * lgb_prob
    return final_prob, char_prob, lgb_prob

# Main
def main():
    t0 = time.time()
    calibrator = train_lightgbm_sklearn(X_tab, y, tr_idx, val_idx)
    char_model = train_charcnn(char_seqs, y, tr_idx, val_idx, DEVICE)

    meta_out = {
        "date": datetime.now().astimezone().isoformat(),
        "seed": SEED,
        "char_maxlen": CHAR_MAXLEN,
        "char_epochs": CHAR_EPOCHS,
        "char_batch_size": CHAR_BATCH_SIZE,
        "ensemble_weights": {"char": CHAR_WEIGHT, "lgb": LGB_WEIGHT}
    }
    with open(MODELS_DIR / "ensemble_metadata.json", "w") as f:
        json.dump(meta_out, f, indent=2)
    print("Saved ensemble metadata.")

    print("Evaluating on test set...")
    final_prob, char_prob, lgb_prob = ensemble_predict(calibrator, char_model, X_tab, char_seqs, test_idx, DEVICE)
    metrics = evaluate_predictions(y[test_idx], final_prob, threshold=0.5)
    print("ENSEMBLE METRICS:")
    print(json.dumps(metrics, indent=2))

    out_df = pd.DataFrame({
        "url_index": test_idx,
        "y_true": y[test_idx],
        "prob_ensemble": final_prob,
        "prob_char": char_prob,
        "prob_lgb": lgb_prob
    })
    out_df.to_csv(MODELS_DIR / "test_predictions.csv", index=False)
    print("Saved test predictions to", MODELS_DIR / "test_predictions.csv")
    print("Elapsed (minutes):", (time.time() - t0)/60.0)

if __name__ == "__main__":
    main()
