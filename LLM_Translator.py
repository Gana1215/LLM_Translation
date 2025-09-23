import os
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
import subprocess
import torch
import soundfile as sf
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from st_audiorec import st_audiorec
import tempfile
import librosa

# ----------------- Folders -----------------
UPLOAD_FOLDER = os.path.join(os.getcwd(), "Files_To_Upload")
SPEECH_FOLDER = os.path.join(os.getcwd(), "Downloaded_Speech")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(SPEECH_FOLDER, exist_ok=True)

# ----------------- Load CSS -----------------
def load_css(file_name):
    if os.path.exists(file_name):
        with open(file_name, "r") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("style.css")

# ----------------- Gemini API -----------------
genai.configure(api_key="YOUR_API_KEY")  # replace with your real key
model = genai.GenerativeModel("gemini-1.5-flash")

# ----------------- Wav2Vec2 STT Model -----------------
stt_model_name = "tugstugi/wav2vec2-large-xlsr-53-mongolian"
processor = Wav2Vec2Processor.from_pretrained(stt_model_name)
stt_model = Wav2Vec2ForCTC.from_pretrained(stt_model_name)

# ----------------- Session State -----------------
if "translated_text" not in st.session_state:
    st.session_state.translated_text = ""
if "audio_file" not in st.session_state:
    st.session_state.audio_file = None
if "user_text" not in st.session_state:
    st.session_state.user_text = ""

# ----------------- Helper Functions -----------------
def translate_text(text, target_language):
    prompt = f"Translate the following text to {target_language} naturally and correctly. Output ONLY the translation:\n{text}"
    try:
        response = model.generate_content(prompt)
        if hasattr(response, "text") and response.text:
            return response.text.strip()
        else:
            st.error("⚠️ Translation API returned empty result.")
            return ""
    except Exception as e:
        st.error(f"Translation failed: {e}")
        return ""

async def generate_edge_speech(text, file_path, voice="mn-MN-YesuiNeural"):
    try:
        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(file_path)
        return file_path
    except Exception as e:
        st.error(f"Edge TTS error: {e}")
        return None

def fix_mp3(input_file):
    safe_file = input_file.replace(".mp3", "_fixed.mp3")
    try:
        cmd = [
            "ffmpeg", "-y", "-i", input_file,
            "-ar", "44100", "-b:a", "192k", "-codec:a", "libmp3lame",
            safe_file
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return safe_file
    except Exception as e:
        st.error(f"MP3 re-encode failed: {e}")
        return input_file

def text_to_speech(text, target_lang='en', source_lang='Eng'):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    lang_code = target_lang[:2].capitalize()
    raw_file = os.path.join(SPEECH_FOLDER, f"{source_lang}To{lang_code}{timestamp}.mp3")
    try:
        if target_lang.lower() == "mn":
            voice_code = "mn-MN-YesuiNeural"
            with st.spinner("🎧 Generating Mongolian speech..."):
                asyncio.run(generate_edge_speech(text, raw_file, voice=voice_code))
        else:
            tts = gTTS(text=text, lang=target_lang[:2].lower())
            tts.save(raw_file)
        return fix_mp3(raw_file)
    except Exception as e:
        st.error(f"Error generating speech: {e}")
        return None

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
st.markdown('<h3 class="subtitle">✨ Translate and listen</h3>', unsafe_allow_html=True)

# Language Selection
language = st.selectbox(
    "🌍 Select target language for translation:",
    ["English", "French", "Spanish", "German", "Chinese", "Japanese", "Russian", "Mongolian"],
    index=0
)

# Input Method
input_option = st.radio(
    "📝 Choose input method:",
    ["Direct Text", "Upload File", "Voice Recording"],
    horizontal=True
)

# -------- Direct Text or File Upload or Voice --------
if input_option == "Direct Text":
    st.markdown('<p class="prompt-label">✍️ Enter your text here:</p>', unsafe_allow_html=True)
    st.session_state.user_text = st.text_area("Enter text", value=st.session_state.user_text, height=150)

elif input_option == "Upload File":
    st.markdown('<p class="prompt-label">📁 Upload your file here:</p>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload file", type=["txt","pdf","docx","doc","csv","xls","xlsx"])
    if uploaded_file is not None:
        save_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.session_state.user_text = extract_text_from_file(save_path)
        st.success(f"✅ File uploaded and text extracted: {uploaded_file.name}")

elif input_option == "Voice Recording":
    st.info("🎤 Click **Start** to record and **Stop** to finish.")
    wav_audio_data = st_audiorec()
    if wav_audio_data is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(wav_audio_data)
            tmp_path = tmp.name
        try:
            speech, sr = sf.read(tmp_path)
            if sr != 16000:
                speech = librosa.resample(speech, orig_sr=sr, target_sr=16000)
            input_values = processor(speech, sampling_rate=16000, return_tensors="pt").input_values
            with torch.no_grad():
                logits = stt_model(input_values).logits
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = processor.batch_decode(predicted_ids)[0]

            st.session_state.user_text = transcription
            st.success("✅ Voice transcribed successfully!")
        except Exception as e:
            st.error(f"Audio processing failed: {e}")
    else:
        st.warning("⚠️ No voice input detected yet.")

# -------- Always show recognized/input text --------
if st.session_state.user_text.strip():
    st.markdown("### 📝 Input / Recognized Text")
    st.text_area("Input Text", st.session_state.user_text, height=150)

# -------- Buttons --------
col1, col2 = st.columns(2)
with col1:
    if st.button("🌐 Translate"):
        if st.session_state.user_text.strip() != "":
            st.session_state.translated_text = translate_text(st.session_state.user_text, language)
            if st.session_state.translated_text:
                st.success("✅ Translation completed!")
        else:
            st.error("Please enter text, upload a file, or record voice to translate.")

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
