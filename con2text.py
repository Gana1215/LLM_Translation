import os
import tempfile
from pathlib import Path
import platform
import pandas as pd
import streamlit as st
import torch
import torchaudio
import subprocess
import base64
from st_audiorec import st_audiorec
from st_aggrid import AgGrid, GridOptionsBuilder, JsCode

# ----------------- Ensure transformers is installed -----------------
import subprocess as sp
try:
    from transformers import WhisperProcessor, WhisperForConditionalGeneration
except ImportError:
    sp.check_call(["pip", "install", "--upgrade", "transformers>=4.35.0"])
    from transformers import WhisperProcessor, WhisperForConditionalGeneration

# ----------------- Config -----------------
GIT_FOLDER = Path("voice_dataset_repo")
WAV_FOLDER = GIT_FOLDER / "wav_files"
CSV_FILE = GIT_FOLDER / "mn-model.csv"
MODEL_FOLDER = Path("./pretrained_models/mn-sp-mini")
os.makedirs(WAV_FOLDER, exist_ok=True)
if not CSV_FILE.exists():
    pd.DataFrame(columns=["file_path", "text"]).to_csv(CSV_FILE, index=False)

# ----------------- App Header -----------------
st.markdown("<h2 style='color:#4B0082;'>🎤 Voice Recording & Dataset Manager</h2>", unsafe_allow_html=True)

# ----------------- Git LFS Utilities -----------------
def ensure_git_lfs():
    """Ensure Git LFS is installed and initialized (auto-install on macOS)."""
    try:
        subprocess.run(["git", "lfs", "version"], check=True, capture_output=True)
        st.info("Git LFS is already installed ✅")
    except subprocess.CalledProcessError:
        st.warning("Git LFS not found! Attempting installation...")
        if platform.system() == "Darwin":  # macOS
            try:
                subprocess.run(["brew", "install", "git-lfs"], check=True)
                subprocess.run(["git", "lfs", "install"], check=True)
                st.success("Git LFS installed successfully via Homebrew! 🎉")
            except subprocess.CalledProcessError as e:
                st.error(f"Automatic installation failed: {e}")
        else:
            st.error(
                "Automatic Git LFS installation only implemented for macOS. "
                "Please install manually: https://git-lfs.github.com/"
            )

def fetch_lfs_files(folder: Path):
    """Fetch and checkout LFS files to get real content."""
    try:
        subprocess.run(["git", "lfs", "install"], check=True, capture_output=True)
        subprocess.run(["git", "-C", str(folder.parent), "lfs", "fetch", "--all"], check=True, capture_output=True)
        subprocess.run(["git", "-C", str(folder.parent), "lfs", "checkout"], check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        st.error(f"Git LFS error: {e.stderr.decode()}")
        raise

def is_pointer_file(path: Path):
    """Check if file is still an LFS pointer."""
    try:
        content = path.read_text(encoding="utf-8").strip()
        return content.startswith("version https://git-lfs.github.com/spec/v1")
    except Exception:
        return False

# ----------------- Load ASR Model -----------------
@st.cache_resource
def load_asr():
    if not MODEL_FOLDER.exists():
        raise FileNotFoundError(f"Model directory not found: {MODEL_FOLDER.resolve()}")

    ensure_git_lfs()
    fetch_lfs_files(MODEL_FOLDER)

    # Verify no pointer files remain
    for json_file in MODEL_FOLDER.glob("*.json"):
        if is_pointer_file(json_file):
            raise RuntimeError(f"LFS pointer detected in {json_file}. Fetch/checkout may have failed.")

    # Load processor and model
    processor = WhisperProcessor.from_pretrained(MODEL_FOLDER, sampling_rate=16000)
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_FOLDER)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return processor, model, device

processor, model, device = load_asr()

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
