# Installation Instructions

## 🧰 Prerequisites

- Python 3.8+
- Docker (optional)
- Git

## 🐍 Local Setup

1. Clone the repository:
```bash
git clone https://github.com/your-username/cognitive-soar.git
cd cognitive-soar
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Train the model:
```bash
python train_model.py
```

4. Run the app:
```bash
streamlit run app.py
```

## 🐳 Docker Setup

1. Build and start the app:
```bash
make build
make up
```

App will be available at: http://localhost:8501
