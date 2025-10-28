import re
import os
import math
import json
import hashlib
import socket
from urllib.parse import urlparse, unquote
from collections import Counter
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd
import tldextract

# ---------------------------
# Configuration / constants
# ---------------------------
CHAR_VOCAB = (
    list("abcdefghijklmnopqrstuvwxyz") +
    list("0123456789") +
    list(":/?&=.%-_+#@~")  # a compact set of common URL chars
)
# Add uppercase mapping to lowercase in tokenizer
CHAR_TO_INDEX = {c: i+1 for i, c in enumerate(CHAR_VOCAB)}  # reserve 0 for padding
UNK_CHAR_INDEX = len(CHAR_TO_INDEX) + 1
DEFAULT_MAX_LEN = 200

# Suspicious TLDs (expandable). Use as a binary indicator.
SUSPICIOUS_TLDS = {
    'xyz', 'top', 'club', 'online', 'site', 'website', 'pw', 'tk', 'ml', 'ga', 'cf', 'gq'
}

# Regexes
RE_IPv4 = re.compile(r'^(?:\d{1,3}\.){3}\d{1,3}$')
RE_IP_IN_HOST = re.compile(r'(^|\[)(\d{1,3}\.){3}\d{1,3}(\]|$)')
RE_PORT = re.compile(r':\d+$')
RE_PERCENT_ENCODE = re.compile(r'%[0-9a-fA-F]{2}')
RE_NON_ALNUM = re.compile(r'[^A-Za-z0-9]')

# ---------------------------
# Utility functions
# ---------------------------
def normalize_url(url: str) -> str:
    """Lowercase scheme & host. Strip surrounding whitespace. Keep path/query intact."""
    if not isinstance(url, str):
        return ""
    url = url.strip()
    if not url:
        return ""
    # ensure scheme present (default to https)
    if not re.match(r'^[a-zA-Z]+://', url):
        url = 'http://' + url  # use http to allow parsing; we keep original scheme presence as feature
    parsed = urlparse(url)
    # Rebuild with normalized netloc
    netloc = parsed.netloc.lower()
    rebuilt = parsed._replace(netloc=netloc).geturl()
    return rebuilt

def is_ip_host(host: str) -> int:
    if not host:
        return 0
    # strip possible port
    host_no_port = host.split(':')[0]
    if RE_IPv4.match(host_no_port):
        return 1
    # also check for encoded/ip formats inside host
    if RE_IP_IN_HOST.search(host):
        return 1
    return 0

def shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    counts = Counter(s)
    probs = [v/len(s) for v in counts.values()]
    return -sum(p * math.log2(p) for p in probs) if probs else 0.0

def count_special_chars(s: str) -> int:
    return len(RE_NON_ALNUM.findall(s))

def has_port(netloc: str) -> int:
    # netloc may contain ':port'
    if ':' in netloc and netloc.split(':')[-1].isdigit():
        return 1
    return 0

def extract_tld_info(domain: str) -> Tuple[str, str]:
    """Return (registered_domain, tld). registered_domain is e.g. example.co.uk -> example.co.uk"""
    if not domain:
        return "", ""
    te = tldextract.extract(domain)
    registered = ".".join(part for part in [te.domain, te.suffix] if part)
    return registered, te.suffix.lower() if te.suffix else ""

def percent_encoded_fraction(s: str) -> float:
    if not s:
        return 0.0
    matches = RE_PERCENT_ENCODE.findall(s)
    return len(matches) / max(1, len(s))

def count_digits(s: str) -> int:
    return sum(ch.isdigit() for ch in s)

def count_letters(s: str) -> int:
    return sum(ch.isalpha() for ch in s)

def vowel_fraction(s: str) -> float:
    if not s:
        return 0.0
    return sum(ch in 'aeiou' for ch in s.lower()) / max(1, len(s))

# ---------------------------
# Optional network checks (use sparingly)
# ---------------------------
def try_resolve_domain(domain: str, timeout: float = 2.0) -> Dict[str, Optional[int]]:
    """Try resolving domain to IP(s). Returns dict with ip_count and resolved_one (0/1). Safe: exceptions handled."""
    result = {"ip_count": None, "resolves": None}
    if not domain:
        return result
    try:
        # set default timeout for sockets for safety
        orig_timeout = socket.getdefaulttimeout()
        socket.setdefaulttimeout(timeout)
        infos = socket.getaddrinfo(domain, None)
        ips = {info[4][0] for info in infos if info and info[4]}
        result['ip_count'] = len(ips)
        result['resolves'] = 1 if ips else 0
    except Exception:
        result['ip_count'] = 0
        result['resolves'] = 0
    finally:
        socket.setdefaulttimeout(orig_timeout)
    return result

# ---------------------------
# Core feature extraction per URL
# ---------------------------
def extract_lexical_features(url: str, perform_network_checks: bool = False) -> Dict[str, object]:
    """
    Extract a dict of features for a single URL.
    perform_network_checks: if True, includes lightweight DNS resolution features (may slow down).
    """
    record = {}
    if not url or not isinstance(url, str):
        # return zeroed features
        keys = [
            'url_length','host_length','path_length','num_dots','num_path_segments','num_query_params',
            'count_digits','count_letters','digit_letter_ratio','count_special_chars','num_hyphens','has_at',
            'has_ip_in_host','tld','suspicious_tld','entropy_host','entropy_path','percent_encoded_frac',
            'vowel_frac','has_https','has_port','has_fragment'
        ]
        return {k: 0 for k in keys}

    norm = normalize_url(url)
    parsed = urlparse(norm)
    scheme = parsed.scheme.lower()
    netloc = parsed.netloc
    path = parsed.path or ""
    query = parsed.query or ""
    fragment = parsed.fragment or ""
    hostname = netloc.split('@')[-1].split(':')[0]  # strip possible credentials and port

    # basic lengths
    record['url_length'] = len(norm)
    record['host_length'] = len(hostname)
    record['path_length'] = len(path)
    record['num_dots'] = hostname.count('.')
    # path segments
    record['num_path_segments'] = len([p for p in path.split('/') if p])
    # query params count (rough)
    record['num_query_params'] = len([p for p in query.split('&') if '=' in p]) if query else 0

    # counts
    record['count_digits'] = count_digits(norm)
    record['count_letters'] = count_letters(norm)
    record['digit_letter_ratio'] = (record['count_digits'] / (record['count_letters'] + 1))
    record['count_special_chars'] = count_special_chars(norm)
    record['num_hyphens'] = norm.count('-')
    record['has_at'] = 1 if '@' in norm else 0

    # host checks
    record['has_ip_in_host'] = is_ip_host(hostname)
    registered_domain, tld = extract_tld_info(hostname)
    record['registered_domain'] = registered_domain
    record['tld'] = tld
    record['suspicious_tld'] = 1 if (tld in SUSPICIOUS_TLDS) else 0

    # entropy & encoding
    record['entropy_host'] = shannon_entropy(hostname)
    record['entropy_path'] = shannon_entropy(path)
    record['percent_encoded_frac'] = percent_encoded_fraction(norm)

    # misc heuristics
    record['vowel_frac'] = vowel_fraction(hostname)
    record['has_https'] = 1 if scheme == 'https' else 0
    record['has_port'] = has_port(netloc)
    record['has_fragment'] = 1 if fragment else 0

    # heuristic suspicious tokens in URL/domain
    suspicious_tokens = ['login', 'signin', 'secure', 'webscr', 'bank', 'verify', 'update', 'account', 'confirm']
    url_lower = norm.lower()
    for tok in suspicious_tokens:
        record[f'token_{tok}'] = 1 if tok in url_lower else 0

    # length buckets (useful as categorical-ish numeric)
    record['url_len_bucket_<50'] = 1 if record['url_length'] < 50 else 0
    record['url_len_bucket_50_100'] = 1 if 50 <= record['url_length'] < 100 else 0
    record['url_len_bucket_100_200'] = 1 if 100 <= record['url_length'] < 200 else 0
    record['url_len_bucket_>=200'] = 1 if record['url_length'] >= 200 else 0

    # network checks (optional & cached by user)
    if perform_network_checks:
        netinfo = try_resolve_domain(hostname)
        record['dns_ip_count'] = netinfo.get('ip_count', 0)
        record['dns_resolves'] = netinfo.get('resolves', 0)
    else:
        record['dns_ip_count'] = None
        record['dns_resolves'] = None

    return record

# ---------------------------
# Batch processing helpers
# ---------------------------
def build_features_for_df(url_series: pd.Series,
                          perform_network_checks: bool = False,
                          char_maxlen: int = DEFAULT_MAX_LEN,
                          save_prefix: Optional[str] = None
                         ) -> Tuple[pd.DataFrame, np.ndarray, Dict]:
    """
    Given a pandas Series of URLs, returns:
      - features_df: DataFrame with extracted features (one row per URL)
      - char_seqs: numpy array shape (n_urls, char_maxlen) with integer indices for Char-CNN
      - meta: dict containing char_vocab mapping and any notes

    If save_prefix provided (string path prefix), will save:
      - {save_prefix}_features.csv
      - {save_prefix}_char_seqs.npy
      - {save_prefix}_meta.json
    """
    urls = url_series.fillna("").astype(str).tolist()
    n = len(urls)

    # Extract lexical features
    feats = []
    for i, u in enumerate(urls):
        feats.append(extract_lexical_features(u, perform_network_checks=perform_network_checks))
        if (i + 1) % 5000 == 0:
            print(f"Processed {i+1}/{n} URLs...")

    features_df = pd.DataFrame(feats)

    # Drop columns that are non-numeric or not needed for ML; keep registered_domain and tld optionally
    # We'll keep registered_domain for reference but ML models should not get it raw (hash or drop later)
    # Convert None to np.nan as LightGBM handles nan
    features_df = features_df.replace({None: np.nan})

    # Build char sequences
    char_seqs = np.zeros((n, char_maxlen), dtype=np.int32)
    for i, u in enumerate(urls):
        seq = encode_url_to_char_indices(u, maxlen=char_maxlen)
        char_seqs[i, :] = seq

    meta = {
        "char_to_index": CHAR_TO_INDEX,
        "unk_char_index": UNK_CHAR_INDEX,
        "maxlen": char_maxlen
    }

    # Optional saving
    if save_prefix:
        os.makedirs(os.path.dirname(save_prefix), exist_ok=True)
        features_csv = f"{save_prefix}_features.csv"
        char_npy = f"{save_prefix}_char_seqs.npy"
        meta_json = f"{save_prefix}_meta.json"
        features_df.to_csv(features_csv, index=False)
        np.save(char_npy, char_seqs)
        with open(meta_json, "w") as f:
            json.dump(meta, f)
        print(f"Saved features to {features_csv}, char seqs to {char_npy}, meta to {meta_json}")

    return features_df, char_seqs, meta

# ---------------------------
# Character encoding for Char-CNN
# ---------------------------
def encode_url_to_char_indices(url: str, maxlen: int = DEFAULT_MAX_LEN) -> np.ndarray:
    """
    Turn URL into fixed-length vector of integer token ids.
    - Lowercases before mapping
    - Unknown chars map to UNK_CHAR_INDEX
    - Right-pad with zeros (index 0 reserved for padding)
    """
    if not isinstance(url, str):
        url = ""
    # normalize a bit but keep raw path/params signs
    url = normalize_url(url)
    url = url.lower()
    seq = np.zeros(maxlen, dtype=np.int32)
    # truncate from left → keep rightmost part (commonly path + file)
    u = url[-maxlen:]
    for i, ch in enumerate(u):
        idx = CHAR_TO_INDEX.get(ch)
        if idx is None:
            idx = UNK_CHAR_INDEX
        seq[i] = idx
    return seq

# ---------------------------
# Small helper for hashing domains (for logging without storing raw URL)
# ---------------------------
def salted_hash(value: str, salt: str = "static_salt_for_demo") -> str:
    if not isinstance(value, str):
        value = ""
    return hashlib.sha256((salt + value).encode()).hexdigest()

# ---------------------------
# Quick test utility
# ---------------------------
if __name__ == "__main__":
    # Simple local smoke test (not heavy)
    sample = [
        "http://192.168.0.1/login?user=abc",
        "https://www.paypal.com/signin",
        "http://secure-google-accounts.xyz/verify",
        "https://accounts.google.com/ServiceLogin",
        "phishing-domain.tk/login.php?acc=1"
    ]
    df = pd.DataFrame({'url': sample})
    feats, chars, meta = build_features_for_df(df['url'], perform_network_checks=False, save_prefix=None)
    print("Features head:")
    print(feats.head())
    print("Char seqs shape:", chars.shape)
