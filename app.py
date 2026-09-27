import json,os
from io import BytesIO
import numpy as np
from PIL import Image,ImageOps
from flask import Flask,jsonify,request,send_from_directory
import onnxruntime as ort
from huggingface_hub import hf_hub_download
app=Flask(__name__)
REPO="BiernyVR/crop-disease-classifier"; MODEL="efficientnet_v2_s_best.onnx"; DATA="efficientnet_v2_s_best.onnx.data"; CLASSES="classes.json"; SIZE=224
def load():
    model=hf_hub_download(REPO,MODEL); hf_hub_download(REPO,DATA); cp=hf_hub_download(REPO,CLASSES)
    with open(cp,encoding="utf-8") as f: classes=json.load(f)["classes"]
    return ort.InferenceSession(model,providers=["CPUExecutionProvider"]),classes
SESSION,CLASSES=load(); INPUT=SESSION.get_inputs()[0].name
def prep(img):
    a=np.asarray(img.convert("RGB").resize((SIZE,SIZE),Image.Resampling.BILINEAR),dtype=np.float32)/255
    a=(a-np.array([.485,.456,.406],dtype=np.float32))/np.array([.229,.224,.225],dtype=np.float32)
    return np.transpose(a,(2,0,1))[None].astype(np.float32)
def softmax(x):
    x=x-np.max(x); e=np.exp(x); return e/e.sum()
def infer(img):
    vs=[img,ImageOps.mirror(img),img.rotate(7,Image.Resampling.BILINEAR),img.rotate(-7,Image.Resampling.BILINEAR)]
    ps=np.stack([softmax(SESSION.run(None,{INPUT:prep(v)})[0][0]) for v in vs]); mean=ps.mean(axis=0); order=np.argsort(mean)[::-1]
    preds=[{"label":CLASSES[int(i)],"confidence":float(mean[i])} for i in order[:5]]
    top=[int(np.argmax(p)) for p in ps]; agree=sum(x==top[0] for x in top)/len(top)
    return preds,float(mean[order[0]]),float(mean[order[1]]),agree
def parse(label):
    p=label.split("___",1); return p[0].replace("_"," "),((p[1] if len(p)>1 else p[0]).replace("_"," "))
def quality(img):
    if img.width<256 or img.height<256:return False
    a=np.asarray(img.resize((256,256)).convert("RGB"),dtype=np.float32)/255; g=.299*a[:,:,0]+.587*a[:,:,1]+.114*a[:,:,2]
    return .08<g.mean()<.96 and (np.var(np.diff(g,axis=1))+np.var(np.diff(g,axis=0)))>.001
def steps(d):
    l=d.lower()
    if "healthy" in l:return ["Continue regular crop monitoring.","Maintain appropriate irrigation, nutrition and airflow."]
    if "rust" in l:return ["Inspect nearby plants for similar symptoms.","Avoid prolonged leaf wetness and confirm the diagnosis before treatment."]
    if "blight" in l:return ["Remove severely affected tissue where appropriate.","Improve airflow and avoid unnecessary overhead irrigation.","Confirm the diagnosis before applying a crop-protection product."]
    if "mildew" in l:return ["Improve air circulation.","Reduce prolonged humidity around foliage.","Confirm the diagnosis before treatment."]
    if "virus" in l or "mosaic" in l:return ["Inspect for insect vectors.","Follow local agricultural guidance; chemical treatment does not cure viral infection."]
    return ["Inspect nearby plants for similar symptoms.","Maintain sanitation, airflow and appropriate irrigation.","Confirm important diagnoses with a local agronomist or plant clinic."]
@app.route("/")
def home(): return send_from_directory("public","index.html")
@app.route("/health")
def health(): return jsonify(status="online",model="EfficientNetV2-S",classes=len(CLASSES))
@app.route("/predict",methods=["POST"])
def predict():
    try:
        if "file" not in request.files:return jsonify(error="No image file uploaded"),400
        img=Image.open(BytesIO(request.files["file"].read())).convert("RGB")
        if not quality(img):return jsonify(plant="Unknown",disease="UNSURE",scientific_name="—",confidence=0,uncertain=True,predictions=[],management=["Upload a sharper, well-lit close-up leaf image."])
        preds,top,second,agree=infer(img); crop,disease=parse(preds[0]["label"]); reliable=top>=.70 and top-second>=.12 and agree>=.75
        return jsonify(plant=crop,disease=disease if reliable else "UNSURE",scientific_name="—",confidence=top,uncertain=not reliable,agreement=agree,predictions=preds,management=steps(disease))
    except Exception as e:return jsonify(error="Prediction failed",details=str(e)),500
if __name__=="__main__": app.run(host="0.0.0.0",port=int(os.environ.get("PORT",5000)))
