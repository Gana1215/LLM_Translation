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

# ----------------- Google Drive Setup -----------------
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive

@st.cache_resource
def init_drive():
    gauth = GoogleAuth()
    gauth.LoadCredentialsFile("drive_creds.json")
    if gauth.credentials is None:
        gauth.LocalWebserverAuth()
    elif gauth.access_token_expired:
        gauth.Refresh()
    else:
        gauth.Authorize()
    gauth.SaveCredentialsFile("drive_creds.json")
    return GoogleDrive(gauth)

drive = init_drive()

# ----------------- Folder IDs -----------------
MODEL_FOLDER_ID = "1dSYQDIVT7_FyDqWHTqCgZonMooP7_FuS"      # pretrained_models/mn-sp-mini
VOICE_DATA_FOLDER_ID = "1ER5odh_oFCLwUHg94KUpvzWcZX5x5lh0"  # voice_dataset_repo

# ----------------- Local Paths -----------------
LOCAL_FOLDER = Path("voice_dataset_repo")
WAV_FOLDER = LOCAL_FOLDER / "wav_files"
CSV_FILE = LOCAL_FOLDER / "mn-model.csv"
MODEL_DIR = Path("./pretrained_models/mn-sp-mini")
os.makedirs(WAV_FOLDER, exist_ok=True)
os.makedirs(MODEL_DIR.parent, exist_ok=True)

# ----------------- Ensure transformers -----------------
try:
    from transformers import WhisperProcessor, WhisperForConditionalGeneration
except ImportError:
    subprocess.check_call(["pip", "install", "--upgrade", "transformers>=4.35.0"])
    from transformers import WhisperProcessor, WhisperForConditionalGeneration

# ----------------- Drive Helpers -----------------
def upload_to_drive(local_path, parent_folder_id):
    fname = Path(local_path).name
    f = drive.CreateFile({"title": fname, "parents": [{"id": parent_folder_id}]})
    f.SetContentFile(str(local_path))
    f.Upload()
    return f["id"]

def download_from_drive(file_name, parent_folder_id, local_path):
    file_list = drive.ListFile({'q': f"'{parent_folder_id}' in parents and trashed=false"}).GetList()
    for f in file_list:
        if f['title'] == file_name:
            f.GetContentFile(local_path)
            return True
    return False

def sync_csv():
    upload_to_drive(CSV_FILE, VOICE_DATA_FOLDER_ID)

# ----------------- CSV Initialization -----------------
if not CSV_FILE.exists():
    if download_from_drive("mn-model.csv", VOICE_DATA_FOLDER_ID, CSV_FILE):
        st.success("Loaded CSV from Google Drive ✅")
    else:
        pd.DataFrame(columns=["file_path", "text"]).to_csv(CSV_FILE, index=False)

# ----------------- Load ASR Model -----------------
@st.cache_resource
def load_asr():
    if not MODEL_DIR.exists():
        zip_name = "mn-sp-mini.zip"
        if download_from_drive(zip_name, MODEL_FOLDER_ID, zip_name):
            import zipfile
            with zipfile.ZipFile(zip_name, "r") as zip_ref:
                zip_ref.extractall(MODEL_DIR.parent)
            os.remove(zip_name)
            st.success("Model downloaded from Google Drive ✅")
        else:
            raise FileNotFoundError(f"Model not found locally or on Drive: {MODEL_DIR.resolve()}")
    processor = WhisperProcessor.from_pretrained(MODEL_DIR, sampling_rate=16000)
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_DIR)
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
if "df" not in st.session_state:
    st.session_state.df = pd.read_csv(CSV_FILE)

# ----------------- App Header -----------------
st.markdown("<h2 style='color:#4B0082;'>🎤 Voice Recording & Dataset Manager (Drive Sync)</h2>", unsafe_allow_html=True)

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

# ----------------- Functions -----------------
def add_record(wav_data, text):
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    wav_filename = f"audio_{timestamp}.wav"
    wav_path = WAV_FOLDER / wav_filename

    with open(wav_path, "wb") as f:
        f.write(wav_data)

    upload_to_drive(wav_path, VOICE_DATA_FOLDER_ID)

    df = pd.read_csv(CSV_FILE)
    df = pd.concat([df, pd.DataFrame({"file_path": [str(wav_path)], "text": [text]})], ignore_index=True)
    df.to_csv(CSV_FILE, index=False)
    sync_csv()
    st.session_state.df = df
    st.success(f"Saved and synced: {wav_filename}")
    st.session_state.recognized_text = ""
    st.session_state.audio_bytes = None

# ----------------- Save Recording -----------------
if st.session_state.recognized_text.strip() != "":
    if st.button("💾 Save to Dataset"):
        add_record(st.session_state.audio_bytes, st.session_state.recognized_text)

# ----------------- Refresh from Drive -----------------
if st.button("🔄 Refresh from Drive"):
    if download_from_drive("mn-model.csv", VOICE_DATA_FOLDER_ID, CSV_FILE):
        st.session_state.df = pd.read_csv(CSV_FILE)
        st.success("Dataset refreshed from Google Drive ✅")
    else:
        st.warning("No CSV found on Drive to refresh!")

# ----------------- AG Grid Table -----------------
st.markdown("### 📄 Dataset Manager")

df = st.session_state.df.copy()
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
    sync_csv()
    st.session_state.df = edited_df
    st.success("Dataset updated and synced with Drive ✅")
