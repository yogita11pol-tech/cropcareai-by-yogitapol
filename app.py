import json, os
import numpy as np
import streamlit as st
from PIL import Image
import onnxruntime as ort
from huggingface_hub import hf_hub_download

st.set_page_config(page_title="CropCare AI", page_icon="🌿", layout="centered")

MODEL_REPO="BiernyVR/crop-disease-classifier"
MODEL_FILE="efficientnet_v2_s_best.onnx"
MODEL_DATA="efficientnet_v2_s_best.onnx.data"
CLASSES_FILE="classes.json"

@st.cache_resource
def load_model():
    model=hf_hub_download(MODEL_REPO, MODEL_FILE)
    # External ONNX data file is downloaded to the same HF cache location.
    hf_hub_download(MODEL_REPO, MODEL_DATA)
    classes_path=hf_hub_download(MODEL_REPO, CLASSES_FILE)
    with open(classes_path,"r",encoding="utf-8") as f:
        classes=json.load(f)["classes"]
    session=ort.InferenceSession(model, providers=["CPUExecutionProvider"])
    return session, classes

def predict(img, session, classes):
    img=img.convert("RGB").resize((224,224),Image.Resampling.BILINEAR)
    arr=np.asarray(img,dtype=np.float32)/255.0
    arr=(arr-np.array([0.485,0.456,0.406],dtype=np.float32))/np.array([0.229,0.224,0.225],dtype=np.float32)
    x=np.transpose(arr,(2,0,1))[None,...].astype(np.float32)
    name=session.get_inputs()[0].name
    logits=session.run(None,{name:x})[0][0]
    p=np.exp(logits-np.max(logits)); p=p/p.sum()
    idx=np.argsort(p)[::-1][:5]
    return [(classes[int(i)],float(p[i])) for i in idx]

def pretty(label):
    return label.replace("___"," — ").replace("_"," ").replace("(","").replace(")","")

def guidance(label):
    l=label.lower()
    if "healthy" in l: return ["Keep monitoring the plant regularly.","Maintain good airflow and avoid unnecessary leaf wetness.","Remove severely damaged leaves only if needed."]
    if "early_blight" in l: return ["Remove badly affected leaves and dispose of them away from the crop.","Improve airflow and avoid overhead irrigation.","Confirm the diagnosis locally before applying any crop-protection product."]
    if "late_blight" in l: return ["Separate affected plants where practical.","Keep foliage dry and improve airflow.","Seek local agricultural guidance promptly because late blight can spread quickly."]
    if "powdery_mildew" in l: return ["Improve air circulation and reduce prolonged leaf humidity.","Remove heavily affected tissue.","Confirm the diagnosis before selecting a permitted treatment."]
    if "rust" in l: return ["Remove heavily infected leaves where practical.","Improve airflow and avoid prolonged leaf wetness.","Monitor nearby plants for similar symptoms."]
    if "bacterial" in l or "spot" in l: return ["Avoid handling plants when foliage is wet.","Remove severely affected tissue and sanitize tools.","Use locally approved management practices after confirming the diagnosis."]
    if "virus" in l or "mosaic" in l or "yellow_leaf_curl" in l: return ["Control insect vectors according to local agricultural guidance.","Remove severely affected plants if recommended locally.","Do not assume chemical treatment will cure a viral infection."]
    if "spider_mites" in l: return ["Inspect the undersides of leaves for mites and webbing.","Reduce plant stress and maintain appropriate irrigation.","Confirm the pest before choosing a control method."]
    return ["Inspect the plant and nearby leaves for matching symptoms.","Improve sanitation, airflow and appropriate irrigation.","Confirm the result with a local agronomist or plant clinic before treatment."]

st.markdown("""<style>
.main{background:#f6f8f2}.hero{padding:30px 0 10px}.title{font-size:46px;font-weight:800;color:#173426}.title span{color:#4d8c57}.sub{font-size:18px;color:#65736a;line-height:1.6}.card{padding:22px;border:1px solid #dfe7dc;border-radius:18px;background:white}
</style>""",unsafe_allow_html=True)

st.markdown('<div class="hero"><div class="title">🌿 CropCare <span>AI</span></div><div class="sub">Real plant-disease image classification from a leaf photo.</div></div>',unsafe_allow_html=True)
st.info("Supported model: 38 PlantVillage crop/disease classes. Use a clear, close leaf photo. Field conditions can reduce accuracy.")

uploaded=st.file_uploader("Upload a leaf image",type=["jpg","jpeg","png","webp"])
if uploaded:
    image=Image.open(uploaded)
    st.image(image,caption="Uploaded leaf",use_container_width=True)
    if st.button("🔍 Analyze leaf",type="primary",use_container_width=True):
        with st.spinner("Running the plant-disease model…"):
            try:
                session,classes=load_model()
                preds=predict(image,session,classes)
                top_label,top_conf=preds[0]
                if top_conf < 0.70:
                    st.warning(f"Uncertain result ({top_conf*100:.1f}% model probability). Please upload a sharper photo or a second angle.")
                else:
                    st.success(f"Prediction: {pretty(top_label)}")
                    st.metric("Model confidence",f"{top_conf*100:.1f}%")
                st.markdown("### Top predictions")
                for label,conf in preds[:3]:
                    st.write(f"**{pretty(label)}** — {conf*100:.1f}%")
                st.markdown("### Suggested next steps")
                for item in guidance(top_label):
                    st.write("• "+item)
                st.caption("CropCare AI is an assistive screening tool, not a laboratory or agronomist diagnosis. The model is trained on PlantVillage-style images and may perform differently on real field photographs.")
            except Exception as e:
                st.error("Model setup failed. Please retry. Details: "+str(e))
