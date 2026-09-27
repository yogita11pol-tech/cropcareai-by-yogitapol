# 🌿 CropCare AI

AI-powered plant leaf disease screening using a GitHub-hosted Streamlit application and a pretrained EfficientNetV2-S ONNX model.

## Model
CropCare AI uses **BiernyVR/crop-disease-classifier**, an EfficientNetV2-S model trained on the PlantVillage benchmark with 38 crop/disease/healthy classes. The model card reports 99.89% validation accuracy, but that is a benchmark result on PlantVillage and should not be interpreted as 99.89% real-world field accuracy.

Model: https://huggingface.co/BiernyVR/crop-disease-classifier

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

The model files are downloaded automatically from Hugging Face on first inference.

## Deploy from GitHub

1. Open Streamlit Community Cloud.
2. Connect GitHub.
3. Select `yogita11pol-tech/cropcareai-by-yogitapol`.
4. Main file: `app.py`.
5. Deploy.

## Accuracy safeguards
- Top-3 predictions are shown.
- Results below 70% probability are explicitly marked uncertain.
- Users are encouraged to provide a clearer/second image.
- The app does not claim laboratory-level diagnosis.
- Treatment advice is deliberately general; pesticide dose recommendations are not generated.

## Supported classes
Apple, blueberry, cherry, corn, grape, orange, peach, bell pepper, potato, raspberry, soybean, squash, strawberry and tomato classes represented in the PlantVillage model.

## Important limitation
PlantVillage contains curated leaf images. Real-world photographs can have different lighting, backgrounds, occlusion and multiple simultaneous problems. External field validation is required before making strong accuracy claims.
