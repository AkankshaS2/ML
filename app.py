import streamlit as st
import numpy as np
import librosa
import sounddevice as sd
from tensorflow.keras.models import load_model

# Load model
@st.cache_resource
def load_my_model():
    return load_model("scream_model.h5")

model = load_my_model()

# Record audio
def record_audio(duration=3, sr=16000):
    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1)
    sd.wait()
    return audio.flatten()

# Extract MFCC
def extract_features(audio, sr=16000):
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    mfcc = np.mean(mfcc.T, axis=0)
    return mfcc.reshape(1, -1)

# UI
st.title("🚨 Human Distress Detection App")
st.write("Click the button to record audio and detect distress.")

duration = st.slider("Recording Duration (seconds)", 1, 10, 3)

if st.button("🎤 Record Audio"):
    with st.spinner("Listening..."):
        audio = record_audio(duration)

    st.success("Recording complete!")

    features = extract_features(audio)
    prediction = model.predict(features)[0][0]

    st.write(f"Prediction Score: {prediction:.4f}")

    if prediction > 0.5:
        st.error("🚨 DISTRESS DETECTED!")
    else:
        st.success("✅ Normal sound")