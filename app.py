import streamlit as st
import whisper
import numpy as np
import pickle
import chromadb
from sentence_transformers import SentenceTransformer
import tempfile
import os
from pydub import AudioSegment

# Load the subtitle embeddings (Already Generated)
with open("subtitle_embeddings.pkl", "rb") as f:
    subtitle_data = pickle.load(f)

# Initialize ChromaDB for fast search
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="subtitles")

# Load Whisper Model
whisper_model = whisper.load_model("base")

# Load Sentence Transformer Model (For Semantic Search)
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Function to Transcribe Audio (Using Whisper AI)
def transcribe_audio(audio_file):
    temp_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3").name
    with open(temp_path, "wb") as f:
        f.write(audio_file.getbuffer())

    # Convert to WAV (Whisper needs WAV)
    sound = AudioSegment.from_file(temp_path)
    wav_path = temp_path.replace(".mp3", ".wav")
    sound.export(wav_path, format="wav")

    # Transcribe using Whisper (with word timestamps)
    result = whisper_model.transcribe(wav_path, word_timestamps=True)
    os.remove(temp_path)
    os.remove(wav_path)

    # Convert output to subtitle format
    subtitles = []
    for i, segment in enumerate(result["segments"]):
        start_time = segment["start"]
        end_time = segment["end"]
        text = segment["text"]

        start_time_str = format_time(start_time)
        end_time_str = format_time(end_time)

        subtitles.append(f"{i+1}\n{start_time_str} --> {end_time_str}\n{text}\n")

    return subtitles

# Function to format time for SRT (HH:MM:SS,MS)
def format_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    milliseconds = int((seconds % 1) * 1000)
    return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"

# Function to Search Subtitles
def search_subtitles(query):
    query_embedding = embedder.encode(query).tolist()
    results = collection.query(query_embeddings=[query_embedding], n_results=5)
    return results["documents"][0] if results["documents"] else []

# Streamlit UI
st.title("🎬 Subtitle Search Engine (Cloning Shazam)")

# Upload Audio File
uploaded_audio = st.file_uploader("🔊 Upload an audio clip (MP3, WAV, etc.)", type=["mp3", "wav"])

if uploaded_audio:
    st.audio(uploaded_audio, format="audio/mp3")
    
    # Show animated status
    with st.spinner("📝 Transcribing audio... This may take some time."):
        with st.status("Processing audio...", expanded=True) as status:
            status.update(label="🔄 Converting audio format...", state="running")
            subtitle_list = transcribe_audio(uploaded_audio)
            status.update(label="✅ Transcription complete!", state="complete")

    if subtitle_list:
        st.write("📜 **Generated Subtitles:**")
        for subtitle in subtitle_list:
            st.text(subtitle)

        # Convert subtitles to SRT format for download
        subtitle_srt = "\n".join(subtitle_list)
        subtitle_filename = "generated_subtitles.srt"

        with open(subtitle_filename, "w", encoding="utf-8") as f:
            f.write(subtitle_srt)

        with open(subtitle_filename, "rb") as f:
            st.download_button(label="📥 Download Subtitles (.srt)", data=f, file_name="subtitles.srt", mime="text/plain")
    else:
        st.write("❌ No subtitles generated.")
