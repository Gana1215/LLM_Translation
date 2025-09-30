import os
from pathlib import Path
import json
import streamlit as st
import pandas as pd
from gtts import gTTS
import docx
from PyPDF2 import PdfReader
from datetime import datetime
import google.generativeai as genai
import base64
import asyncio
import edge_tts
import time
import torch
import torchaudio
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from st_audiorec import st_audiorec
import git_lfs  # Python wrapper for Git LFS
import tempfile

# ----------------- Folders -----------------
UPLOAD_FOLDER = os.path.join(os.getcwd(), "Files_To_Upload")
SPEECH_FOLDER = os.path.join(os.getcwd(), "Generated_Speech")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(SPEECH_FOLDER, exist_ok=True)

# ----------------- Load CSS -----------------
def load_css(*files):
    for file_name in files:
        if os.path.exists(file_name):
            with open(file_name, "r") as f:
                st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("style.css", "record.css")

# ----------------- Gemini API -----------------
genai.configure(api_key="AIzaSyBMj0Yshu5o4YxMp2oLImlseU6lV_FiFjI")  # Replace with your key
gen_model = genai.GenerativeModel("gemini-2.0-flash"
# INITIALIZE LFS CLIENT                                  
lfs_client = git_lfs.GitLFSClient(str(model_dir))
# 
# ----------------- Whisper MN-SP-MINI ASR -----------------
@st.cache_resource
def load_mongolian_asr():
    model_dir = Path("./pretrained_models/mn-sp-mini")
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory does not exist: {model_dir.resolve()}")

    # List all JSON files in the model directory
    json_files = [f for f in model_dir.glob("*.json")]
    for config_file in json_files:
        with open(config_file, "r", encoding="utf-8") as f:
            content = f.read().strip()

        # Detect LFS pointer and download real content
        if content.startswith("version https://git-lfs.github.com/spec/v1"):
            pointer = lfs_client.get_pointer(str(config_file))
            content = lfs_client.download_file(pointer).decode("utf-8")

        # Verify JSON validity
        try:
            json.loads(content)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in {config_file.resolve()}: {e}\nContent preview: {content[:200]}")

    # Load processor and model
    processor = WhisperProcessor.from_pretrained(model_dir, sampling_rate=16000)
    model = WhisperForConditionalGeneration.from_pretrained(model_dir)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    return processor, model, device

asr_processor, asr_model, device = load_mongolian_asr()

# ----------------- Session State -----------------
if "user_text" not in st.session_state:
    st.session_state.user_text = ""
if "translated_text" not in st.session_state:
    st.session_state.translated_text = ""
if "audio_data" not in st.session_state:
    st.session_state.audio_data = None
if "show_recorder" not in st.session_state:
    st.session_state.show_recorder = False

# ----------------- Helper Functions -----------------
def translate_text(text, target_language):
    prompt = f"Translate the following text to {target_language} naturally and correctly. Output ONLY the translation:\n{text}"
    try:
        response = gen_model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        st.error(f"Translation failed: {e}")
        return ""

async def generate_wav(text, filename, voice="mn-MN-YesuiNeural"):
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(filename)

def text_to_speech(text, target_lang='en'):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"speech_{timestamp}.mp3"
    file_path = os.path.join(SPEECH_FOLDER, file_name)
    try:
        if target_lang.lower() != "mn":
            tts = gTTS(text=text, lang=target_lang[:2].lower())
            tts.save(file_path)
        else:
            asyncio.run(generate_wav(text, file_path))
            retries = 5
            while not os.path.exists(file_path) and retries > 0:
                time.sleep(0.5)
                retries -= 1
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"TTS file not created: {file_path}")
    except Exception as e:
        st.error(f"Error generating speech: {e}")
        return None
    return file_path

def extract_text_from_file(file_path):
    if not file_path or not os.path.exists(file_path):
        return ""
    file_name = file_path.lower()
    try:
        if file_name.endswith(".txt"):
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        elif file_name.endswith(".pdf"):
            reader = PdfReader(file_path)
            return "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
        elif file_name.endswith((".docx", ".doc")):
            doc = docx.Document(file_path)
            return "\n".join([para.text for para in doc.paragraphs if para.text.strip()])
        elif file_name.endswith((".csv", ".xls", ".xlsx")):
            try:
                if file_name.endswith((".xls", ".xlsx")):
                    df = pd.read_excel(file_path)
                else:
                    df = pd.read_csv(file_path, encoding="utf-8")
            except Exception:
                df = pd.read_csv(file_path, encoding="latin1")
            return "\n".join(df.astype(str).apply(lambda x: " ".join(x), axis=1))
        else:
            st.error("Unsupported file type")
            return ""
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return ""

# ----------------- Streamlit UI -----------------
st.markdown('<h2 class="main-title">🌐 Multi-language Translator & TTS</h2>', unsafe_allow_html=True)
st.markdown('<h3 class="subtitle">✨ Translate, Speak, and Listen</h3>', unsafe_allow_html=True)

# Language Selection
language = st.selectbox(
    "🌍 Select target language for translation:",
    ["English", "French", "Spanish", "German", "Chinese", "Japanese", "Russian", "Mongolian"],
    index=0
)

# Input Method: Direct Text or File Upload
input_option = st.radio(
    "📝 Choose input method:",
    ["Direct Text", "Upload File"],
    horizontal=True
)

if input_option == "Direct Text":
    st.session_state.user_text = st.text_area("✍️ Enter text here", value=st.session_state.user_text, height=150)
elif input_option == "Upload File":
    uploaded_file = st.file_uploader("📁 Upload your file", type=["txt","pdf","docx","doc","csv","xls","xlsx"])
    if uploaded_file is not None:
        save_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.session_state.user_text = extract_text_from_file(save_path)
        st.success(f"✅ File uploaded and text extracted: {uploaded_file.name}")

# ----------------- Recorder Toggle Button -----------------
button_label = "🎤 Click here to record your voice"
if st.button(button_label):
    st.session_state.show_recorder = not st.session_state.show_recorder

# Apply pulsing animation class when recording
if st.session_state.show_recorder:
    st.markdown(
        "<style>.stButton > button:contains('🎤 Click here') { animation: pulse 1.2s infinite; }</style>",
        unsafe_allow_html=True
    )

# ----------------- Show Recorder -----------------
if st.session_state.show_recorder:
    audio_bytes = st_audiorec()
    if audio_bytes is not None and len(audio_bytes) > 0:
        st.session_state.audio_data = audio_bytes
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name
        try:
            waveform, sample_rate = torchaudio.load(tmp_path)
            if waveform.numel() > 0:
                if sample_rate != 16000:
                    resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
                    waveform = resampler(waveform)
                input_features = asr_processor(
                    waveform.squeeze().numpy(), sampling_rate=16000, return_tensors="pt"
                ).input_features
                input_features = input_features.to(device)
                with torch.no_grad():
                    predicted_ids = asr_model.generate(input_features)
                transcription = asr_processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
                st.session_state.user_text = transcription.strip()
            else:
                st.warning("⚠️ Empty audio. Please record again.")
        except Exception as e:
            st.error(f"Audio processing failed: {e}")

# -------- Display recognized/input text --------
if st.session_state.user_text.strip():
    st.markdown("### 📝 Recognized Text")
    st.text_area("Recognized text", st.session_state.user_text, height=150)

# -------- Translate / TTS Buttons ---------
col1, col2 = st.columns(2)
with col1:
    if st.button("🌐 Translate"):
        if st.session_state.user_text.strip() != "":
            st.session_state.translated_text = translate_text(st.session_state.user_text, language)
            if st.session_state.translated_text:
                st.success("✅ Translation completed!")
        else:
            st.error("Please provide text, upload a file, or record voice to translate.")

with col2:
    if st.button("🔊 Convert to Speech"):
        if st.session_state.translated_text:
            target_lang_code = language.lower()[:2] if language.lower() != "mongolian" else "mn"
            file_path = text_to_speech(st.session_state.translated_text, target_lang=target_lang_code)
            if file_path:
                with open(file_path, "rb") as f:
                    audio_bytes = f.read()
                st.audio(audio_bytes, format="audio/mp3")
                b64 = base64.b64encode(audio_bytes).decode()
                href = f'<a href="data:audio/mp3;base64,{b64}" download="{os.path.basename(file_path)}">⬇️ Download Speech</a>'
                st.markdown(href, unsafe_allow_html=True)
                st.success(f"🎧 Speech generated: {os.path.basename(file_path)}")
        else:
            st.warning("⚠️ Please translate text first before converting to speech.")

# -------- Display Translated Text --------
if st.session_state.translated_text:
    st.markdown("### 📝 Translated Text")
    st.text_area("Translated text", st.session_state.translated_text, height=150)
