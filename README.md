# 🛡️ AI Phishing Detector

Welcome to the **AI Phishing Detector** project! 🚀 This project helps identify phishing URLs using a combination of **machine learning** and **intelligent analysis**.

---

## Table of Contents 📑
1. [Overview](#overview-)
2. [Features](#features-)
3. [Installation](#installation-)
4. [Usage](#usage-)
5. [API Endpoints](#api-endpoints-)
6. [How It Works](#how-it-works-)
7. [Example Predictions](#example-predictions-)
8. [Contributing](#contributing-)
9. [License](#license-)

---

## Overview 📝
Phishing is one of the most common cyber threats today. Malicious actors create fake websites or emails to steal sensitive data like **passwords**, **credit card info**, and **personal details**.  

The **AI Phishing Detector** leverages:
- **Machine Learning** (LightGBM + Char-CNN)
- **Feature Engineering** (lexical, domain, path-based)
- **Optional intelligent API integration** (like Gemini) for enhanced decision-making  

It predicts whether a URL is **safe** or **phishing** with a probability score. 🎯

---

## Features ✨
- ✅ **Detect phishing URLs in real-time**
- ✅ **Ensemble model** using both tabular features and character-level CNN
- ✅ **Lexical analysis**:
  - URL length
  - Number of dots
  - Digits vs letters ratio
  - Suspicious tokens (`login`, `verify`, `secure`)
- ✅ **Domain analysis**:
  - TLD check (`.xyz`, `.top`, `.tk`, etc.)
  - Entropy of host and path
  - IP address detection
- ✅ **Optional DNS checks**
- ✅ **Clean API interface** via FastAPI
- ✅ **CORS enabled for frontend integration**

---

## Installation 🛠️

```bash
# Clone repository
git clone https://github.com/yourusername/ai-phishing-detector.git
cd ai-phishing-detector

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
.\.venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

---

## Usage 🚀

Start the API server:

```bash
uvicorn app.main:app --reload
```

Open in browser or use `curl` to test:

```bash
curl -X POST http://127.0.0.1:8000/predict_url \
-H 'Content-Type: application/json' \
-d '{"url": "http://example-phishing.com/login"}'
```

Expected output:
```json
{
  "url": "http://example-phishing.com/login",
  "is_phishing": true,
  "probability": 0.987,
  "produced_feature_count": 36,
  "produced_feature_names": [ ... ]
}
```

---

## API Endpoints 🌐

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check |
| `/predict_url` | POST | Predicts whether a URL is phishing |

**Request JSON:**
```json
{ "url": "https://secure-example.com" }
```

**Response JSON:**
```json
{
  "url": "https://secure-example.com",
  "is_phishing": false,
  "probability": 0.056,
  "produced_feature_count": 36,
  "produced_feature_names": [ ... ]
}
```

---

## How It Works ⚙️

1. **Feature Extraction** 🧩
   - Lexical features: URL length, path segments, special characters
   - Domain features: TLD, IP presence, entropy
   - Tokens indicating phishing (`login`, `secure`, `verify`, etc.)

2. **Character-level CNN** 🧠
   - Encodes raw URL as a sequence of character indices
   - Captures patterns like `paypal-login.xyz`

3. **Ensemble Model** ⚖️
   - Combines LightGBM and Char-CNN predictions
   - Weighted average output gives final probability

4. **Optional Intelligence Layer** 💡
   - Gemini API (or similar) can provide additional verification
   - Adjusts probability for high-risk URLs

---

## Example Predictions 🔍

| URL | Prediction | Probability |
|-----|-----------|------------|
| `http://notificacionbcr.0hi.me/` | Phishing | 99.87% |
| `https://secure-paypal-login.com` | Safe | 5.60% |
| `https://www.mmmut.ac.in/` | Safe | 0.16% |

---

## Contributing 🤝

Contributions are welcome! Please:
1. Fork the repository
2. Create a branch (`git checkout -b feature-name`)
3. Commit changes (`git commit -m 'Add new feature'`)
4. Push branch (`git push origin feature-name`)
5. Create a Pull Request

---

## License 📄

This project is licensed under the MIT License. See `LICENSE` for details.

---

## Notes 🧠
- The ML model is **not 100% foolproof**; always double-check suspicious URLs.
- For production deployment, consider integrating **Gemini or other intelligent agents**.
- Frontend integration can be done using any web framework (React, Vue, Angular).

---

Made with ❤️ by Rudra Dubey