import streamlit as st
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration, BlipForQuestionAnswering
from gtts import gTTS
import io
import os
import gc # Garbage collector to clear RAM
from datetime import datetime

# --- PAGE CONFIG ---
st.set_page_config(page_title="LuminaVision AI", page_icon="🎯", layout="wide")

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
    }
    .history-item {
        background-color: #21262d; padding: 10px; border-radius: 8px;
        margin-bottom: 10px; border-left: 3px solid #00c853; font-size: 0.85em;
    }
    </style>
    """, unsafe_allow_html=True)

# --- OPTIMIZED MODEL LOADING ---
@st.cache_resource
def load_models():
    # Using 'base' models to stay under Streamlit's 1GB RAM limit
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    
    # low_cpu_mem_usage=True is critical for preventing the "Resource Limit" error
    cap_model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base", 
        low_cpu_mem_usage=True
    )
    vqa_model = BlipForQuestionAnswering.from_pretrained(
        "Salesforce/blip-vqa-base", 
        low_cpu_mem_usage=True
    )
    return processor, cap_model, vqa_model

def calculate_confidence(outputs):
    scores = torch.exp(outputs.sequences_scores).item()
    return round(min(99.9, (scores * 100) * 1.5), 2)

# --- SIDEBAR ---
with st.sidebar:
    st.header("📸 Media Control")
    uploaded_file = st.file_uploader("Upload Target Image", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        img = Image.open(uploaded_file).convert('RGB')
        st.image(img, use_container_width=True)
        precision = st.select_slider("Detection Rigor", options=["Standard", "High"])
    
    st.divider()
    st.header("📜 Analysis History")
    for item in reversed(st.session_state.history):
        st.markdown(f"<div class='history-item'><b>{item['time']}</b><br>{item['result']}</div>", unsafe_allow_html=True)
    if st.button("Clear History"):
        st.session_state.history = []
        st.rerun()

# --- MAIN UI ---
st.title("🎯 LuminaVision: High-Precision Engine")

if uploaded_file:
    proc, cap_model, vqa_model = load_models()
    beam_count = 5 if precision == "Standard" else 10
    
    col1, col2 = st.columns(2)

    with col1:
        if st.button("Execute Narrative Engine"):
            with st.spinner("Analyzing..."):
                # Use torch.no_grad() to save massive amounts of RAM
                with torch.no_grad():
                    inputs = proc(img, return_tensors="pt")
                    out = cap_model.generate(**inputs, max_new_tokens=50, num_beams=beam_count, 
                                           output_scores=True, return_dict_in_generate=True)
                    caption = proc.decode(out.sequences[0], skip_special_tokens=True).capitalize()
                
                st.session_state.history.append({"time": datetime.now().strftime("%H:%M"), "result": caption})
                st.markdown(f"<div class='chat-bubble'>{caption}</div>", unsafe_allow_html=True)
                st.metric("Confidence", f"{calculate_confidence(out)}%")
                
                # Cleanup
                gc.collect()

    with col2:
        q = st.text_input("Ask a question:")
        if q:
            with st.spinner("Thinking..."):
                with torch.no_grad():
                    inputs = proc(img, q, return_tensors="pt")
                    out_vqa = vqa_model.generate(**inputs, num_beams=beam_count, 
                                               output_scores=True, return_dict_in_generate=True)
                    ans = proc.decode(out_vqa.sequences[0], skip_special_tokens=True).upper()
                
                st.session_state.history.append({"time": datetime.now().strftime("%H:%M"), "result": f"Q: {q} | A: {ans}"})
                st.success(f"ANSWER: {ans}")
                
                # Cleanup
                gc.collect()
else:
    st.info("Upload an image to start.")
