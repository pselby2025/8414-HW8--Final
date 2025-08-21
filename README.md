# Cognitive SOAR: From Prediction to Attribution

## 🎯 Project Goal
This project enhances a basic phishing URL detection tool by adding threat actor attribution using unsupervised learning. It mimics a real-world SOAR (Security Orchestration, Automation, and Response) environment.

## 🧠 Dual-Model Architecture

- **Classifier:** Uses PyCaret to detect if a URL is benign or malicious.
- **Clustering Model:** For malicious URLs, a second model predicts the most likely threat actor:
  - State-Sponsored
  - Organized Cybercrime
  - Hacktivist

## 🚀 How It Works

1. User inputs URL features into the Streamlit UI.
2. Model 1 (Classifier): Labels the URL as BENIGN or MALICIOUS.
3. If MALICIOUS:
   - Model 2 (Clustering): Assigns the URL to one of three threat actor profiles.
   - The result is shown in the "Threat Attribution" tab.

## 🛠️ Stack

- Python
- Streamlit
- PyCaret
- Docker
- GitHub Actions

## 📦 Usage

```bash
# Run locally
streamlit run app.py
```

Or use Docker:
```bash
make build
make up
```

Access the app at: http://localhost:8501
