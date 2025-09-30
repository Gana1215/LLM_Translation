import os
from pathlib import Path
import json
import streamlit as st
import pandas as pd
import torch
import torchaudio
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from st_audiorec import st_audiorec
import tempfile
import git
from datetime import datetime

# ----------------- Config -----------------
GIT_FOLDER = Path("./voice_dataset_repo")
WAV_FOLDER = GIT_FOLDER / "wav_files"
CSV_FILE = GIT_FOLDER / "mn-model.csv"

os.makedirs(WAV_FOLDER, exist_ok=True)
if not CSV_FILE.exists():
    pd.DataFrame(columns=["file_path", "text"]).to_csv(CSV_FILE, index=False)

# ----------------- Load Whisper ASR -----------------
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

asr_processor, asr_model, device = load_mongolian_asr()

# ----------------- Streamlit UI -----------------
st.title("🎤 Mongolian Voice Recorder & Dataset Builder")

# Session State
if "recognized_text" not in st.session_state:
    st.session_state.recognized_text = ""
if "audio_bytes" not in st.session_state:
    st.session_state.audio_bytes = None

# ----------------- Recorder -----------------
st.markdown("## Record your voice")
audio_bytes = st_audiorec()
if audio_bytes and len(audio_bytes) > 0:
    st.session_state.audio_bytes = audio_bytes
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        with st.spinner("⏳ Converting voice to text..."):
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
                st.session_state.recognized_text = transcription.strip()
            else:
                st.warning("⚠️ Empty audio. Please record again.")
    except Exception as e:
        st.error(f"Audio processing failed: {e}")

# ----------------- Display Recognized Text -----------------
st.markdown("## 📝 Recognized Text (editable)")
if st.session_state.recognized_text:
    st.session_state.recognized_text = st.text_area(
        "Fix any mistakes before saving:", value=st.session_state.recognized_text, height=150
    )

# ----------------- Save Button -----------------
if st.session_state.recognized_text.strip() and st.session_state.audio_bytes:
    if st.button("💾 Save to Dataset"):
        # Save wav file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        wav_filename = f"{timestamp}.wav"
        wav_path = WAV_FOLDER / wav_filename
        with open(wav_path, "wb") as f:
            f.write(st.session_state.audio_bytes)

        # Update CSV
        df = pd.read_csv(CSV_FILE)
        df = pd.concat([df, pd.DataFrame([{"file_path": str(wav_filename), "text": st.session_state.recognized_text}])])
        df.to_csv(CSV_FILE, index=False)

        # Git commit & push
        try:
            if not GIT_FOLDER.exists():
                repo = git.Repo.init(GIT_FOLDER)
            else:
                repo = git.Repo(GIT_FOLDER)
            repo.git.add(all=True)
            repo.index.commit(f"Add {wav_filename} with transcription")
            st.success(f"✅ Saved {wav_filename} and updated mn-model.csv")
        except Exception as e:
            st.error(f"Git commit failed: {e}")

else:
    st.info("Record voice and fix recognized text to enable saving.")
