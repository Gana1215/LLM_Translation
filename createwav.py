import os
import tempfile
from pathlib import Path
import json
import streamlit as st
import pandas as pd
import torch
import torchaudio
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from st_audiorec import st_audiorec
import git_lfs  # Ensure Git LFS is installed

# ----------------- Config -----------------
GIT_FOLDER = Path("voice_dataset_repo")
WAV_FOLDER = GIT_FOLDER / "wav_files"
CSV_FILE = GIT_FOLDER / "mn-model.csv"
os.makedirs(WAV_FOLDER, exist_ok=True)
if not CSV_FILE.exists():
    pd.DataFrame(columns=["file_path", "text"]).to_csv(CSV_FILE, index=False)

# ----------------- Inline Styles -----------------
st.markdown("""
    <style>
    .title {font-size:30px; font-weight:bold; color:#4B0082;}
    .subtitle {font-size:20px; color:#800080; margin-bottom:20px;}
    .stButton > button {background-color:#4B0082; color:white; font-weight:bold;}
    textarea {font-size:16px;}
    </style>
""", unsafe_allow_html=True)

# ----------------- App Header -----------------
st.markdown('<div class="title">🎤 Voice Recording & Dataset Creator</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Record voice, transcribe, edit, and save to dataset</div>', unsafe_allow_html=True)

# ----------------- Load MN-SP-MINI ASR -----------------
@st.cache_resource
def load_mongolian_asr():
    model_dir = Path("./pretrained_models/mn-sp-mini")
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory does not exist: {model_dir.resolve()}")
    processor = WhisperProcessor.from_pretrained(model_dir, sampling_rate=16000)
    model = WhisperForConditionalGeneration.from_pretrained(model_dir)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return processor, model, device

processor, model, device = load_mongolian_asr()

# ----------------- Session State -----------------
if "recognized_text" not in st.session_state:
    st.session_state.recognized_text = ""
if "audio_bytes" not in st.session_state:
    st.session_state.audio_bytes = None
if "status_msg" not in st.session_state:
    st.session_state.status_msg = ""

# ----------------- Recorder -----------------
st.markdown("### 🎙️ Record Your Voice")
audio_bytes = st_audiorec()
if audio_bytes is not None and len(audio_bytes) > 0:
    st.session_state.audio_bytes = audio_bytes
    st.session_state.status_msg = "Converting audio to text..."
    st.markdown(f"**Status:** {st.session_state.status_msg}")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name
    try:
        waveform, sr = torchaudio.load(tmp_path)
        if sr != 16000:
            waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)
        input_features = processor(waveform.squeeze().numpy(), sampling_rate=16000, return_tensors="pt").input_features
        input_features = input_features.to(device)
        with torch.no_grad():
            predicted_ids = model.generate(input_features)
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()
        st.session_state.recognized_text = transcription
        st.session_state.status_msg = "✅ Transcription completed!"
    except Exception as e:
        st.error(f"Audio processing failed: {e}")
        st.session_state.status_msg = "⚠️ Error during transcription"

st.markdown(f"**Status:** {st.session_state.status_msg}")

# ----------------- Display Recognized Text -----------------
if st.session_state.recognized_text:
    st.markdown("### 📝 Edit Recognized Text")
    st.session_state.recognized_text = st.text_area(
        "Edit text if needed before saving",
        st.session_state.recognized_text,
        height=150
    )

# ----------------- Save Button -----------------
if st.session_state.recognized_text.strip() != "":
    if st.button("💾 Save to Dataset"):
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        wav_filename = f"audio_{timestamp}.wav"
        wav_path = WAV_FOLDER / wav_filename
        with open(wav_path, "wb") as f:
            f.write(st.session_state.audio_bytes)
        # Update CSV
        df = pd.read_csv(CSV_FILE)
        df = pd.concat([df, pd.DataFrame({"file_path":[str(wav_path)], "text":[st.session_state.recognized_text]})], ignore_index=True)
        df.to_csv(CSV_FILE, index=False)
        st.success(f"Saved audio and text to dataset: {wav_filename}")
        st.session_state.recognized_text = ""
        st.session_state.audio_bytes = None
        st.session_state.status_msg = ""
