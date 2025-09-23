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
if "user_text" not in st.session_state:
    st.session_state.user_text = ""
if "wav_audio_data" not in st.session_state:
    st.session_state.wav_audio_data = None

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
        st.error(f"Error reading fi
