import os

import streamlit as st

import pandas as pd

from gtts import gTTS

import pyttsx3

import docx

from PyPDF2 import PdfReader

from datetime import datetime

import google.generativeai as genai

import base64



# ----------------- Folders -----------------

UPLOAD_FOLDER = os.path.join(os.getcwd(), "Files_To_Upload")

SPEECH_FOLDER = os.path.join(os.getcwd(), "Downloaded_Speech")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

os.makedirs(SPEECH_FOLDER, exist_ok=True)



# ----------------- Load CSS -----------------

def load_css(file_name):

    with open(file_name, "r") as f:

        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)



load_css("style.css")



# ----------------- Gemini API -----------------

genai.configure(api_key="AIzaSyA6f51_kEGieaUmBJ_YngqxIEX9fTSZB1A")  # Replace with your key

model = genai.GenerativeModel("gemini-1.5-flash")



# ----------------- Session State Init -----------------

if "translated_text" not in st.session_state:

    st.session_state.translated_text = ""

if "audio_file" not in st.session_state:

    st.session_state.audio_file = None



# ----------------- Helper Functions -----------------

def translate_text(text, target_language):

    prompt = f"Translate the following text to {target_language}:\n{text}"

    response = model.generate_content(prompt)

    return response.text



def text_to_speech(text, target_lang='en', source_lang='Eng'):

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")

    lang_code = target_lang[:2].capitalize()

    file_name = f"{source_lang}To{lang_code}{timestamp}.mp3"

    file_path = os.path.join(SPEECH_FOLDER, file_name)



    try:

        if target_lang.lower() != "mongolian":

            tts = gTTS(text=text, lang=target_lang[:2].lower())

            tts.save(file_path)

        else:

            engine = pyttsx3.init()

            engine.save_to_file(text, file_path)

            engine.runAndWait()

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

st.markdown('<h3 class="subtitle">✨ Let us translate and listen</h3>', unsafe_allow_html=True)



# Language Selection

language = st.selectbox(

    "🌍 Select target language for translation:",

    ["English", "French", "Spanish", "German", "Chinese", "Japanese", "Russian", "Mongolian"],

    index=0

)



# Input Method

input_option = st.radio("📝 Choose input method:", ["Direct Text", "Upload File"], horizontal=True)



if input_option == "Direct Text":

    st.markdown('<p class="prompt-label">✍️ Enter your text here:</p>', unsafe_allow_html=True)

    user_text = st.text_area("", height=150)

else:

    st.markdown('<p class="prompt-label">📁 Upload your file here:</p>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader("", type=["txt","pdf","docx","doc","csv","xls","xlsx"], key="file_uploader")

    user_text = ""

    if uploaded_file is not None:

        save_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)

        with open(save_path, "wb") as f:

            f.write(uploaded_file.getbuffer())

        user_text = extract_text_from_file(save_path)

        st.success(f"✅ File uploaded and text extracted from: {uploaded_file.name}")



# -------- Buttons --------

col1, col2 = st.columns(2)



with col1:

    if st.button("🌐 Translate", key="translate_btn"):

        if user_text.strip() != "":

            try:

                st.session_state.translated_text = translate_text(user_text, language)

                st.success("✅ Translation completed!")

            except Exception as e:

                st.error(f"Translation failed: {e}")

        else:

            st.error("Please enter text or upload a file to translate.")



with col2:

    if st.button("🔊 Convert to Speech", key="speech_btn"):

        if st.session_state.translated_text:

            target_lang_code = language.lower()[:2] if language.lower() != "mongolian" else "mn"

            file_path = text_to_speech(st.session_state.translated_text, target_lang=target_lang_code)

            if file_path:

                # --- Play audio ---

                st.audio(file_path)



                # --- Mobile-friendly download ---

                with open(file_path, "rb") as f:

                    audio_bytes = f.read()

                b64 = base64.b64encode(audio_bytes).decode()

                href = f'<a href="data:audio/mp3;base64,{b64}" download="{os.path.basename(file_path)}">⬇️ Download Speech</a>'

                st.markdown(href, unsafe_allow_html=True)



                st.success(f"🎧 Speech generated: {os.path.basename(file_path)}")

        else:

            st.warning("⚠️ Please translate text first before converting to speech.")



# -------- Display Translated Text --------

if st.session_state.translated_text:

    st.markdown("### 📝 Translated Text")

    st.text_area("", st.session_state.translated_text, height=150)