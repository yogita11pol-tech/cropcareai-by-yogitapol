# 🌿 CropCare AI — V2 Accurate Branch

This is a **new GitHub branch** for CropCare AI. The original `main` version is intentionally unchanged.

## What changed

- Uses the real **EfficientNetV2-S** plant-disease model instead of a brightness/demo classifier.
- Uses test-time augmentation (original, mirror, ±7° rotation) and averages predictions.
- Adds image-quality checks.
- Requires confidence + TTA agreement + prediction margin before presenting a specific result.
- Shows top-3 predictions.
- Explicitly returns **UNSURE** rather than forcing a diagnosis when evidence is weak.
- Provides general next-step guidance.

## Model

The app uses `BiernyVR/crop-disease-classifier`, an EfficientNetV2-S model trained on the PlantVillage benchmark with 38 classes. Its model card reports 99.89% validation accuracy on that benchmark; this should **not** be interpreted as 99.89% accuracy on arbitrary real-world field photos.

Model source: https://huggingface.co/BiernyVR/crop-disease-classifier

## Why the app can refuse a diagnosis

A model can produce a high softmax probability even for an image that is outside its training distribution. CropCare AI therefore combines:

1. model probability,
2. agreement across image transformations,
3. separation between the first and second prediction,
4. basic image-quality checks.

If these do not meet the acceptance rule, the app asks for another photo instead of pretending the result is exact.

## Important coverage limitation

The current 38-class model does **not** cover every plant or every disease. For example, its Raspberry class is healthy-only. Therefore the app must not claim to diagnose every raspberry disease or every plant disease.

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy this branch

On Streamlit Community Cloud, select:

- Repository: `yogita11pol-tech/cropcareai-by-yogitapol`
- Branch: `cropcareai-v2-accurate`
- Main file: `app.py`

The original `main` branch remains available as the first version.

## Accuracy statement

No image-only plant disease website should promise an exact diagnosis for every field photograph. For research/Avishkar use, the correct claim is **AI-assisted plant disease screening with an uncertainty safeguard** unless the model has been independently validated on representative field images.
