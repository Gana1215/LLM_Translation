import os
import tempfile
from pathlib import Path
import pandas as pd
import streamlit as st
import torch
import torchaudio
import subprocess
import base64
from st_audiorec import st_audiorec
from st_aggrid import AgGrid, GridOptionsBuilder, JsCode
import git_lfs  # Git LFS integration

# ----------------- Ensure transformers is installed -----------------
try:
    from transformers import WhisperProcessor, WhisperForConditionalGeneration
except ImportError:
    subprocess.check_call(["pip", "install", "--upgrade", "transformers>=4.35.0"])
    from transformers import WhisperProcessor, WhisperForConditionalGeneration

# ----------------- Config -----------------
GIT_FOLDER = Path("voice_dataset_repo")
WAV_FOLDER = GIT_FOLDER / "wav_files"
CSV_FILE = GIT_FOLDER / "mn-model.csv"
MODEL_FOLDER = Path("./pretrained_models/mn-sp-mini")
os.makedirs(WAV_FOLDER, exist_ok=True)
if not CSV_FILE.exists():
    pd.DataFrame(columns=["file_path", "text"]).to_csv(CSV_FILE, index=False)

# ----------------- Initialize Git LFS -----------------
lfs_client = git_lfs.GitLFSClient(repo_path=str(GIT_FOLDER))

# ----------------- App Header -----------------
st.markdown("<h2 style='color:#4B0082;'>🎤 Voice Recording & Dataset Manager</h2>", unsafe_allow_html=True)

# ----------------- Load ASR Model via LFS -----------------
@st.cache_resource
def load_asr_lfs():
    if not MODEL_FOLDER.exists():
        os.makedirs(MODEL_FOLDER, exist_ok=True)

    repo_model_folder = GIT_FOLDER / MODEL_FOLDER
    if repo_model_folder.exists():
        for file_path in repo_model_folder.glob("*"):
            pointer = lfs_client.get_pointer(str(file_path))
            content = lfs_client.download_file(pointer)
            with open(MODEL_FOLDER / file_path.name, "wb") as f:
                f.write(content)

    processor = WhisperProcessor.from_pretrained(MODEL_FOLDER, sampling_rate=16000)
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_FOLDER)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return processor, model, device

processor, model, device = load_asr_lfs()

# ----------------- Session State -----------------
if "recognized_text" not in st.session_state:
    st.session_state.recognized_text = ""
if "audio_bytes" not in st.session_state:
    st.session_state.audio_bytes = None

# ----------------- Record Audio -----------------
st.markdown("### 🎙️ Record Your Voice")
audio_bytes = st_audiorec()

if audio_bytes and len(audio_bytes) > 0:
    st.session_state.audio_bytes = audio_bytes
    st.markdown("Converting audio to text...")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        waveform, sr = torchaudio.load(tmp_path)
        if sr != 16000:
            waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)
        input_features = processor(
            waveform.squeeze().numpy(), sampling_rate=16000, return_tensors="pt"
        ).input_features.to(device)
        with torch.no_grad():
            predicted_ids = model.generate(input_features)
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()
        st.session_state.recognized_text = transcription
        st.success("Transcription completed!")
    except Exception as e:
        st.error(f"Audio processing failed: {e}")
    finally:
        os.remove(tmp_path)

# ----------------- Display Recognized Text -----------------
if st.session_state.recognized_text:
    st.session_state.recognized_text = st.text_area(
        "Edit transcription before saving:",
        value=st.session_state.recognized_text,
        height=150
    )

# ----------------- Save Recording -----------------
if st.session_state.recognized_text.strip() != "":
    if st.button("💾 Save to Dataset"):
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        wav_filename = f"audio_{timestamp}.wav"
        wav_path = WAV_FOLDER / wav_filename
        with open(wav_path, "wb") as f:
            f.write(st.session_state.audio_bytes)
        df = pd.read_csv(CSV_FILE)
        df = pd.concat(
            [df, pd.DataFrame({"file_path": [str(wav_path)], "text": [st.session_state.recognized_text]})],
            ignore_index=True,
        )
        df.to_csv(CSV_FILE, index=False)
        st.success(f"Saved: {wav_filename}")
        st.session_state.recognized_text = ""
        st.session_state.audio_bytes = None

# ----------------- AG Grid Table with Filename and Play/Pause Buttons -----------------
st.markdown("### 📄 Dataset Manager")

if CSV_FILE.exists():
    df = pd.read_csv(CSV_FILE)
    df["file_name"] = df["file_path"].apply(lambda x: Path(x).name)

    # Convert WAV files to Base64 for inline play
    def path_to_base64(path):
        if not Path(path).exists():
            return ""
        data = Path(path).read_bytes()
        encoded = base64.b64encode(data).decode()
        return f"data:audio/wav;base64,{encoded}"

    df["play_url"] = df["file_path"].apply(path_to_base64)

    grid_df = df[["file_name", "text", "play_url"]].copy()

    cell_renderer = JsCode("""
    class BtnCellRenderer {
        init(params) {
            this.eGui = document.createElement('button');
            this.eGui.innerText = '▶️ Play';
            this.eGui.style.width = '60px';
            this.eGui.style.height = '28px';
            this.eGui.style.backgroundColor = '#4B0082';
            this.eGui.style.color = 'white';
            this.eGui.style.fontWeight = 'bold';
            this.audio = null;
            this.isPlaying = false;

            this.eGui.onclick = () => {
                if (!this.audio) {
                    this.audio = new Audio(params.value);
                    this.audio.onended = () => { 
                        this.eGui.innerText = '▶️ Play'; 
                        this.isPlaying = false; 
                    };
                }

                if (this.isPlaying) {
                    this.audio.pause();
                    this.eGui.innerText = '▶️ Play';
                    this.isPlaying = false;
                } else {
                    this.audio.play();
                    this.eGui.innerText = '⏸ Pause';
                    this.isPlaying = true;
                }
            };
        }
        getGui() { return this.eGui; }
    }
    """)

    gb = GridOptionsBuilder.from_dataframe(grid_df)
    gb.configure_column("file_name", header_name="WAV File", editable=False, minWidth=150, maxWidth=250)
    gb.configure_column("text", header_name="Transcription", editable=True, wrapText=True, autoHeight=True, minWidth=300)
    gb.configure_column("play_url", header_name="Play", cellRenderer=cell_renderer, editable=False, minWidth=70, maxWidth=70)
    grid_options = gb.build()

    grid_response = AgGrid(
        grid_df,
        gridOptions=grid_options,
        enable_enterprise_modules=False,
        fit_columns_on_grid_load=True,
        height=500,
        update_mode="MODEL_CHANGED",
        allow_unsafe_jscode=True
    )

    edited_df = grid_response["data"].copy()
    edited_df["file_path"] = df["file_path"]
    edited_df = edited_df[["file_path", "text"]]

    if st.button("💾 Save Table Edits"):
        edited_df.to_csv(CSV_FILE, index=False)
        st.success("Dataset updated successfully!")
else:
    st.info("No dataset yet. Record and save some audio first!")
