import { pipeline, env } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1/+esm";
env.allowLocalModels=false;
env.useBrowserCache=true;

const MIN_CONFIDENCE=0.72;
const MIN_MARGIN=0.12;
const MODEL="onnx-community/mobilenet_v2_1.0_224-plant-disease-identification-ONNX";
const $=s=>document.querySelector(s);
let file=null,model=null,objectUrl=null;

const guidance={
"Apple___Apple_scab":"Remove infected fallen leaves and improve canopy airflow. Follow locally approved disease-control guidance.",
"Apple___Black_rot":"Prune infected tissue and mummified fruit and improve orchard sanitation.",
"Potato___Early_blight":"Manage infected residue, rotate crops and follow locally approved disease-control guidance.",
"Potato___Late_blight":"Act quickly, isolate affected material and follow local late-blight alerts and approved treatment guidance.",
"Tomato___Early_blight":"Remove lower infected leaves, improve airflow and reduce soil splash.",
"Tomato___Late_blight":"Remove affected material, improve airflow and follow local late-blight alerts.",
"Tomato___Bacterial_spot":"Use clean planting material, avoid overhead irrigation and remove severely infected tissue.",
"Tomato___Leaf_Mold":"Increase ventilation and reduce humidity and leaf wetness.",
"Tomato___Septoria_leaf_spot":"Remove infected lower leaves, mulch to reduce soil splash and improve airflow.",
"Tomato___Tomato_Yellow_Leaf_Curl_Virus":"Manage whitefly vectors and use resistant varieties where available.",
"Tomato___Tomato_mosaic_virus":"Remove infected plants/tools from production areas and sanitize equipment.",
"healthy":"No disease pattern was classified. Continue routine scouting, balanced nutrition, irrigation management and sanitation."
};

const pretty=s=>s.replaceAll("___"," — ").replaceAll("_"," ").replaceAll("(","").replaceAll(")","").trim();
const crop=s=>s.split("___")[0].replaceAll("_"," ");

async function getModel(){
 if(model)return model;
 $("#status").textContent="Loading the trained plant-disease AI model. The first scan can take a little longer.";
 $("#modelStatus").textContent="AI model: downloading (first time only)";
 model=await pipeline("image-classification",MODEL,{dtype:"q8",device:"wasm"});
 $("#modelStatus").textContent="AI model: loaded and ready";
 return model;
}

function select(f){
 if(!f||!f.type.startsWith("image/")){
   $("#status").textContent="Please choose a JPG, PNG or WEBP image.";
   return;
 }
 file=f;
 if(objectUrl)URL.revokeObjectURL(objectUrl);
 objectUrl=URL.createObjectURL(f);
 const img=$("#preview");
 img.onload=()=>{$("#status").textContent="Image ready. Press Detect disease.";};
 img.onerror=()=>{$("#status").textContent="This image could not be opened. Please choose another photo.";};
 img.src=objectUrl;
 img.hidden=false;
 $("#uploadView").hidden=true;
 $("#actions").hidden=false;
 $("#result").hidden=true;
}

$("#browseBtn").onclick=()=>$("#fileInput").click();
$("#cameraBtn").onclick=()=>$("#cameraInput").click();
$("#fileInput").onchange=e=>select(e.target.files[0]);
$("#cameraInput").onchange=e=>select(e.target.files[0]);

$("#dropzone").onclick=e=>{
 if(!e.target.closest("button")&&!e.target.closest("img"))$("#fileInput").click();
};
["dragenter","dragover"].forEach(x=>$("#dropzone").addEventListener(x,e=>{
 e.preventDefault();$("#dropzone").classList.add("drag");
}));
["dragleave","drop"].forEach(x=>$("#dropzone").addEventListener(x,e=>{
 e.preventDefault();$("#dropzone").classList.remove("drag");
}));
$("#dropzone").addEventListener("drop",e=>select(e.dataTransfer.files[0]));

$("#clearBtn").onclick=()=>{
 file=null;
 if(objectUrl)URL.revokeObjectURL(objectUrl);
 objectUrl=null;
 $("#fileInput").value="";
 $("#cameraInput").value="";
 $("#preview").removeAttribute("src");
 $("#preview").hidden=true;
 $("#uploadView").hidden=false;
 $("#actions").hidden=true;
 $("#result").hidden=true;
 $("#status").textContent="";
};

$("#scanBtn").onclick=async()=>{
 if(!file){$("#status").textContent="Please upload or capture a leaf image first.";return;}
 const img=$("#preview");
 if(!img.complete||!img.naturalWidth){
   $("#status").textContent="Please wait for the image to finish loading.";
   return;
 }
 $("#scanBtn").disabled=true;
 $("#scanBtn").textContent="Analyzing…";
 $("#status").textContent="CropCare AI is analyzing the leaf…";
 try{
   const classifier=await getModel();
   const out=await classifier(img,{top_k:3});
   if(!Array.isArray(out)||!out.length)throw new Error("The model returned no prediction.");
   const top=out[0],conf=Number(top.score)||0,label=top.label||"Unknown";
   const second=Number(out[1]?.score)||0;
   const margin=conf-second;
   const uncertain=conf<MIN_CONFIDENCE||margin<MIN_MARGIN;
   $("#diseaseName").textContent=uncertain?"Uncertain result — retake the photo":pretty(label);
   $("#cropName").textContent=uncertain?"The model is not confident enough to name a disease.":"Crop: "+crop(label);
   $("#confidenceBadge").textContent=Math.round(conf*100)+"% model confidence";
   $("#scoreValue").textContent=Math.round(conf*100)+"%";
   $("#description").textContent=uncertain
     ?"The image may be unclear, outside the supported classes, or unlike the model's training images. Try one close, well-lit leaf with little background."
     :"AI screening result: "+pretty(label)+". This is a model prediction, not a confirmed diagnosis.";
   $("#solution").textContent=uncertain
     ?"Retake the image with one leaf filling most of the frame. If symptoms persist, confirm the condition with an agronomist."
     :(guidance[label]||"Confirm the result in the field and follow local integrated pest-management guidance.");
   $("#warning").textContent="Top model predictions: "+out.map(x=>pretty(x.label)+" ("+Math.round((Number(x.score)||0)*100)+"%)").join(" · ");
   $("#result").hidden=false;
   $("#result").scrollIntoView({behavior:"smooth",block:"start"});
   $("#status").textContent="Analysis complete.";
 }catch(e){
   console.error(e);
   $("#status").textContent="Detection failed: "+(e?.message||"the AI model could not run")+". Refresh once and try a clear JPG/PNG leaf photo.";
 }finally{
   $("#scanBtn").disabled=false;
   $("#scanBtn").textContent="🔎 Detect disease";
 }
};
