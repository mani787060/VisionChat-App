# 🎯 LuminaVision AI: High-Precision Narrative Engine

LuminaVision is a professional-grade Multimodal AI application that bridges the gap between Computer Vision (CV) and Natural Language Processing (NLP). By utilizing the **Salesforce BLIP (Bootstrapping Language-Image Pre-training)** architecture, this engine can "see" images and discuss them in natural language.



## ✨ Key Features
- **Deep Scene Analysis**: Generates descriptive narratives of complex visual scenes using Beam Search optimization.
- **Visual Question Answering (VQA)**: Allows users to interrogate images for specific details (e.g., colors, counts, actions).
- **Text-to-Speech (TTS)**: Integrated gTTS (Google Text-to-Speech) for accessibility and audio feedback.
- **High-Precision Controls**: Toggle between Standard, High, and Ultra detection rigor to manage AI confidence.
- **Session History**: A persistent log of all narratives and queries generated during a session.
- **Modern UI**: A sleek, dark-themed Glassmorphic dashboard built with Streamlit.

## 🚀 Tech Stack
- **Framework**: Streamlit (Web UI)
- **Deep Learning**: PyTorch
- **Models**: Salesforce BLIP (Vision Transformer + Language Decoder)
- **Libraries**: Hugging Face Transformers, Pillow (PIL), gTTS

## ⚙️ Installation & Usage

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/YOUR_USERNAME/VisionChat-App.git](https://github.com/YOUR_USERNAME/VisionChat-App.git)
   cd VisionChat-App

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt

3. **Run the Application:**
   ```bash
   streamlit run app.py   

## 🧠 Technical Insight: Accuracy & Confidence
The engine uses Beam Search decoding to ensure high-precision outputs. By exploring multiple word-sequence paths, the model selects the narrative with the highest cumulative probability, significantly reducing "hallucinations" common in smaller AI models.
