# Outfit Recommender (FastAPI)

This project is a **FastAPI-based API** that recommends outfits (tops, bottoms, shoes, bags, etc.) based on a given context such as **emotion, weather, temperature, and style preferences**.

---

## Features

* `POST /recommend`: returns outfit items with image URLs.
* `GET /`: health check endpoint.
* Serves outfit images using FastAPI static files.
* Uses a PyTorch model (ResNet50 + context embeddings).
* Applies simple weather & style rules to improve results.

---

## Requirements

Python 3.10+ recommended. Install dependencies:

```bash
pip install -r requirements.txt
```

`requirements.txt` includes:

```
fastapi==0.112.2
uvicorn[standard]==0.30.6
pydantic==2.8.2
torch>=2.6.0
torchvision>=0.21.0
pillow
pandas
numpy
scikit-learn
```

> For GPU, install the correct PyTorch build from [pytorch.org](https://pytorch.org/).

---

## Configuration

Edit `settings.py` to set paths and parameters:

```python
BUNDLE_PATH = Path("outfit_reco.pth")
IMG_DIR = Path("C:/Users/Asus/Desktop/polyvore_outfits/images")
CSV_PATH = Path("C:/Users/Asus/Desktop/polyvore_outfits/augmented_items2.csv")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
STATIC_ROUTE = "/images"
STATIC_DIR = IMG_DIR
TOPK_POOL = 30
IMG_SIZE = 224
```

---

## Running the API

1. Activate virtual environment (optional):

   ```bash
   python -m venv .venv
   source .venv/bin/activate   # Linux/macOS
   .venv\\Scripts\\activate    # Windows
   ```
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```
3. Start the server:

   ```bash
   uvicorn app:app --reload --port 8000
   ```
4. Open in browser:

   * Swagger docs: [http://localhost:8000/docs](http://localhost:8000/docs)
   * Health check: [http://localhost:8000/](http://localhost:8000/)

---

## Example Usage

### Request

```json
POST /recommend
{
  "emotion": "happy",
  "weather_condition": "Ensoleillé",
  "temperature": 24,
  "preferred_style": "casual",
  "preferred_preference": "modern"
}
```

### Response

```json
{
  "tops": {
    "item_name": "linen crop tee",
    "image_url": "/images/tops/tee_001.jpg"
  },
  "shoes": {
    "item_name": "canvas sneakers",
    "image_url": "/images/shoes/sneaker_132.png"
  }
}
```

---

## How It Works

1. **Context encoding**: Converts emotion, weather, style, preference, and temperature into embeddings.
2. **Image encoding**: Extracts features with a pretrained ResNet50.
3. **Scoring**: Matches context and image features + applies rules (e.g., no sandals in rain).
4. **Output**: Returns best item per category with image URL.

---

## Notes

* Images must exist under `IMG_DIR`.
* `CSV_PATH` should include `semantic_category`, `items`, and image references.
* Weather conditions are normalized (e.g., `"Ensoleillé"` → `"Clear"`).
