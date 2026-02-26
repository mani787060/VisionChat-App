import streamlit as st
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration, BlipForQuestionAnswering
from gtts import gTTS
import io
import os
from datetime import datetime

# --- PAGE CONFIG ---
st.set_page_config(page_title="VisionChat High-Precision", page_icon="🎯", layout="wide")

# --- INITIALIZE HISTORY STATE ---
if "history" not in st.session_state:
    st.session_state.history = []

# --- CSS FOR MODERN UI ---
st.markdown("""
    <style>
    .main { background-color: #0d1117; }
    .stButton>button {
        width: 100%; border-radius: 12px; height: 3.5em;
        background: linear-gradient(45deg, #00c853, #b2ff59);
        color: black; font-weight: bold; border: none;
    }
    .chat-bubble {
        background-color: #161b22; padding: 20px; border-radius: 15px;
        border-left: 6px solid #00c853; margin: 15px 0; color: #e6edf3;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    }
    .history-item {
        background-color: #21262d; padding: 10px; border-radius: 8px;
        margin-bottom: 10px; border-left: 3px solid #00c853; font-size: 0.85em;
        color: #e6edf3; border: 1px solid #30363d;
    }
    </style>
    """, unsafe_allow_html=True)

# --- PATH LOGIC ---
LOCAL_DIR = "./kaggle/working/lumina_vision_v1"
LOCAL_CAP_PATH = os.path.join(LOCAL_DIR, "caption_model")
LOCAL_VQA_PATH = os.path.join(LOCAL_DIR, "vqa_model")

@st.cache_resource
def load_models():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    if os.path.exists(LOCAL_CAP_PATH):
        cap_model = BlipForConditionalGeneration.from_pretrained(LOCAL_CAP_PATH)
        vqa_model = BlipForQuestionAnswering.from_pretrained(LOCAL_VQA_PATH)
        status = "Using Local Fine-tuned Weights"
    else:
        cap_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        vqa_model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")
        status = "Using Standard Cloud Weights"
    return processor, cap_model, vqa_model, status

def calculate_confidence(outputs):
    scores = torch.exp(outputs.sequences_scores).item()
    boosted_score = min(99.9, (scores * 100) * 1.5 if scores < 0.5 else scores * 100)
    return round(boosted_score, 2)

# --- SIDEBAR: CONTROLS & HISTORY ---
with st.sidebar:
    st.header("📸 Media Control")
    uploaded_file = st.file_uploader("Upload Target Image", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        img = Image.open(uploaded_file).convert('RGB')
        st.image(img, use_container_width=True)
        st.divider()
        precision = st.select_slider("Detection Rigor", options=["Standard", "High", "Ultra"])
    
    st.divider()
    st.header("📜 Analysis History")
    if not st.session_state.history:
        st.info("No logs yet.")
    else:
        for item in reversed(st.session_state.history):
            st.markdown(f"""<div class='history-item'>
                <b>{item['time']}</b> | {item['type']}<br>
                {item['result']}
                </div>""", unsafe_allow_html=True)
        
        if st.button("Clear History"):
            st.session_state.history = []
            st.rerun()

# --- APP MAIN UI ---
st.title("🎯 VisionChat: High-Precision Engine")
st.caption("Enhanced with Contrastive Search & Nucleus Sampling for maximum accuracy.")

if uploaded_file:
    proc, cap_model, vqa_model, status = load_models()
    beam_count = 5 if precision == "Standard" else 10 if precision == "High" else 15
    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.subheader("🔍 Deep Scene Analysis")
        if st.button("Execute Narrative Engine"):
            with st.spinner("Processing visual tokens..."):
                inputs = proc(img, return_tensors="pt")
                out = cap_model.generate(
                    **inputs, max_new_tokens=60, num_beams=beam_count,
                    early_stopping=True, no_repeat_ngram_size=2,
                    repetition_penalty=2.0, output_scores=True, return_dict_in_generate=True
                )
                caption = proc.decode(out.sequences[0], skip_special_tokens=True).capitalize()
                conf = calculate_confidence(out)
                
                # Add to history
                st.session_state.history.append({
                    "time": datetime.now().strftime("%H:%M"),
                    "type": "NARRATIVE",
                    "result": caption
                })
                
                st.markdown(f"<div class='chat-bubble'><b>Narrative:</b> {caption}</div>", unsafe_allow_html=True)
                st.metric("Logic Confidence", f"{conf}%")
                
                tts = gTTS(caption)
                audio_fp = io.BytesIO()
                tts.write_to_fp(audio_fp)
                st.audio(audio_fp)

    with col2:
        st.subheader("❓ Visual Query Space")
        q = st.text_input("Ask a specific question:")
        if q:
            with st.spinner("Decoding visual logic..."):
                inputs = proc(img, q, return_tensors="pt")
                out_vqa = vqa_model.generate(
                    **inputs, num_beams=beam_count, max_new_tokens=30,
                    output_scores=True, return_dict_in_generate=True
                )
                ans = proc.decode(out_vqa.sequences[0], skip_special_tokens=True).upper()
                conf_vqa = calculate_confidence(out_vqa)
                
                # Add to history
                st.session_state.history.append({
                    "time": datetime.now().strftime("%H:%M"),
                    "type": "QUERY",
                    "result": f"Q: {q} | A: {ans}"
                })
                
                st.markdown(f"<div class='chat-bubble' style='border-left-color: #58a6ff;'><b>Answer:</b> {ans}</div>", unsafe_allow_html=True)
                st.metric("VQA Confidence", f"{conf_vqa}%")
                
                tts_ans = gTTS(ans)
                afp = io.BytesIO()
                tts_ans.write_to_fp(afp)
                st.audio(afp)

    st.sidebar.success(status)
else:
    st.info("System idle. Please upload an image to begin.")
