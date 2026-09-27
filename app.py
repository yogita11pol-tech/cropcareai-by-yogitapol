import json
import numpy as np
import streamlit as st
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import onnxruntime as ort
from huggingface_hub import hf_hub_download

st.set_page_config(page_title="CropCare AI — Accurate", page_icon="🌿", layout="centered")

MODEL_REPO = "BiernyVR/crop-disease-classifier"
MODEL_FILE = "efficientnet_v2_s_best.onnx"
MODEL_DATA = "efficientnet_v2_s_best.onnx.data"
CLASSES_FILE = "classes.json"
IMG_SIZE = 224

@st.cache_resource
def load_model():
    model_path = hf_hub_download(MODEL_REPO, MODEL_FILE)
    hf_hub_download(MODEL_REPO, MODEL_DATA)
    classes_path = hf_hub_download(MODEL_REPO, CLASSES_FILE)
    with open(classes_path, "r", encoding="utf-8") as f:
        classes = json.load(f)["classes"]
    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    return session, classes

def preprocess(img):
    img = img.convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.Resampling.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array(
        [0.229, 0.224, 0.225], dtype=np.float32
    )
    return np.transpose(arr, (2, 0, 1))[None, ...].astype(np.float32)

def softmax(x):
    x = x - np.max(x)
    p = np.exp(x)
    return p / np.sum(p)

def tta_images(img):
    return [
        img,
        ImageOps.mirror(img),
        img.rotate(7, resample=Image.Resampling.BILINEAR, expand=False),
        img.rotate(-7, resample=Image.Resampling.BILINEAR, expand=False),
    ]

def predict_tta(img, session, classes):
    input_name = session.get_inputs()[0].name
    probs = []
    for variant in tta_images(img):
        logits = session.run(None, {input_name: preprocess(variant)})[0][0]
        probs.append(softmax(logits))
    mean_probs = np.mean(np.stack(probs), axis=0)
    order = np.argsort(mean_probs)[::-1]
    top = [(classes[int(i)], float(mean_probs[i])) for i in order[:5]]

    # TTA agreement: how consistently the variants choose the same top class.
    top_ids = []
    for p in probs:
        top_ids.append(int(np.argmax(p)))
    agreement = sum(i == top_ids[0] for i in top_ids) / len(top_ids)
    return top, agreement

def image_quality(img):
    rgb = np.asarray(img.convert("RGB").resize((512, 512)), dtype=np.float32) / 255.0
    gray = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    gx = np.diff(gray, axis=1)
    gy = np.diff(gray, axis=0)
    sharpness = float(np.var(gx) + np.var(gy))
    brightness = float(gray.mean())
    return sharpness, brightness

def pretty(label):
    return label.replace("___", " — ").replace("_", " ").replace("(", "").replace(")", "")

def parse_class(label):
    parts = label.split("___", 1)
    crop = parts[0].replace("_", " ").replace("(including sour)", "").strip()
    condition = parts[1].replace("_", " ").strip() if len(parts) > 1 else "Unknown"
    return crop, condition

def is_healthy(label):
    return "healthy" in label.lower()

def guidance(label):
    l = label.lower()
    if "healthy" in l:
        return [
            "Continue routine monitoring.",
            "Keep good airflow and avoid prolonged leaf wetness.",
            "If symptoms appear later, upload a new close-up image."
        ]
    if "rust" in l:
        return [
            "Remove heavily affected leaves where practical and dispose of them safely.",
            "Improve airflow and avoid prolonged leaf wetness.",
            "Confirm the disease with a local agriculture/plant pathology service before applying a treatment."
        ]
    if "blight" in l:
        return [
            "Remove severely affected foliage where appropriate.",
            "Improve airflow and avoid overhead irrigation.",
            "Confirm the diagnosis before selecting any crop-protection product."
        ]
    if "powdery_mildew" in l:
        return [
            "Improve air circulation and reduce prolonged humidity around foliage.",
            "Remove heavily affected tissue where appropriate.",
            "Confirm the diagnosis before treatment."
        ]
    if "bacterial" in l or "spot" in l:
        return [
            "Avoid handling the crop while leaves are wet.",
            "Sanitize cutting tools and remove severely affected tissue where appropriate.",
            "Use locally approved management practices after confirmation."
        ]
    if "virus" in l or "mosaic" in l:
        return [
            "Inspect for insect vectors and manage them using locally approved methods.",
            "Remove severely affected plants if local guidance recommends it.",
            "Do not assume fungicides will cure a viral disease."
        ]
    return [
        "Inspect nearby leaves for the same symptom pattern.",
        "Maintain appropriate irrigation, sanitation and airflow.",
        "Confirm the result with a local agronomist or plant clinic before treatment."
    ]

st.markdown("""
<style>
.stApp { background: #f5f8f2; }
.hero { padding: 24px 0 8px; }
.title { font-size: 46px; font-weight: 850; color: #163522; }
.title span { color: #4f9659; }
.sub { font-size: 18px; color: #627067; line-height: 1.55; }
.badge { display:inline-block; padding:6px 10px; border-radius:999px; background:#e6f2e6; color:#28633a; font-weight:700; font-size:13px; }
</style>
""", unsafe_allow_html=True)

st.markdown(
    '<div class="hero"><div class="title">🌿 CropCare <span>AI</span></div>'
    '<div class="sub">Real trained-model plant disease screening from a leaf photograph.</div>'
    '<br><span class="badge">EfficientNetV2-S + test-time augmentation</span></div>',
    unsafe_allow_html=True,
)

st.info(
    "This version uses a real trained EfficientNetV2-S model covering 38 PlantVillage classes. "
    "It does not claim 100% field accuracy. When the image is unclear or the model is uncertain, "
    "CropCare AI will refuse to guess."
)

uploaded = st.file_uploader(
    "Upload a clear leaf image",
    type=["jpg", "jpeg", "png", "webp"],
    help="Use a close, well-lit image with the affected leaf clearly visible."
)

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded leaf", use_container_width=True)

    sharpness, brightness = image_quality(image)
    quality_ok = image.width >= 256 and image.height >= 256 and sharpness >= 0.0015 and 0.12 <= brightness <= 0.93

    if not quality_ok:
        st.warning(
            "Image quality may be too low for reliable classification. "
            "Use a sharper, well-lit close-up of one leaf."
        )

    if st.button("🔍 Analyze leaf", type="primary", use_container_width=True):
        if not quality_ok:
            st.error("Please upload a clearer image before analysis.")
            st.stop()

        with st.spinner("Analyzing with the trained plant-disease model…"):
            try:
                session, classes = load_model()
                predictions, agreement = predict_tta(image, session, classes)
                top_label, top_conf = predictions[0]
                second_conf = predictions[1][1]
                margin = top_conf - second_conf
                crop, condition = parse_class(top_label)

                # Conservative acceptance rule. A high softmax score alone is not treated
                # as proof of correctness; TTA agreement and class separation are required.
                accepted = top_conf >= 0.80 and agreement >= 0.75 and margin >= 0.20

                if accepted:
                    if is_healthy(top_label):
                        st.success(f"Likely healthy — {crop}")
                    else:
                        st.success(f"Likely condition: {condition} — {crop}")
                    st.metric("Model probability", f"{top_conf * 100:.1f}%")
                    st.caption(f"TTA agreement: {agreement * 100:.0f}%")
                else:
                    st.warning(
                        "UNSURE — CropCare AI is not confident enough to give a specific diagnosis. "
                        "This is intentional: it is safer to request a better image than to guess."
                    )
                    st.write(
                        f"Highest model probability: **{top_conf * 100:.1f}%** · "
                        f"Prediction agreement: **{agreement * 100:.0f}%**"
                    )

                st.markdown("### Top model results")
                for label, conf in predictions[:3]:
                    st.write(f"**{pretty(label)}** — {conf * 100:.1f}%")

                st.markdown("### Suggested next steps")
                for item in guidance(top_label):
                    st.write("• " + item)

                st.caption(
                    "Coverage: 38 PlantVillage classes across 14 crop species. "
                    "A disease outside these classes may be returned as uncertain or misclassified. "
                    "PlantVillage benchmark accuracy is not the same as real-field accuracy."
                )
            except Exception as e:
                st.error("The trained model could not be loaded. Please retry.")
                st.exception(e)
