# CropCare AI

GitHub-ready Flask website for AI-assisted plant disease screening.

## Run locally

```bash
pip install -r requirements.txt
python app.py
```

Open http://127.0.0.1:5000

## Model

Uses **BiernyVR/crop-disease-classifier** with EfficientNetV2-S and a limited PlantVillage class set.

The model is not an all-species/all-disease model and cannot guarantee exact field diagnosis. CropCare AI returns **UNSURE** when confidence, prediction margin or test-time augmentation agreement is insufficient.

## Production start command

```bash
gunicorn --bind 0.0.0.0:$PORT app:app
```

The original `main` branch is intentionally unchanged. This complete website is in `cropcareai-complete`.
