import streamlit as st
import numpy as np
import librosa
import librosa.display
import plotly.graph_objects as go
import plotly.express as px
import soundfile as sf
import sounddevice as sd
import os
import time
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import io
from scipy.io.wavfile import write

# Create audio_inputs directory if it doesn't exist
os.makedirs("audio_inputs", exist_ok=True)

# Set page config
st.set_page_config(page_title="Audio Analysis Tool", layout="wide")

# Title and description
st.title("Audio Analysis Tool")
st.markdown("This application provides various tools for audio analysis using Librosa.")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Audio Input", "Time Domain Visualization",
                                 "Sinusoidal Wave Generator", "Spectral Analysis",
                                 "STFT Analysis", "Complete Analysis",
                                 "Sound Classification"])

# Global variables for audio data
if 'audio_data' not in st.session_state:
    st.session_state.audio_data = None
    st.session_state.sample_rate = None
    st.session_state.file_path = None

# Function to load audio file
def load_audio(file_path):
    y, sr = librosa.load(file_path, sr=None)
    return y, sr

# Function to record audio
def record_audio(duration=5, sample_rate=22050):
    st.write(f"Recording for {duration} seconds...")
    audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
    sd.wait()
    return audio_data.flatten(), sample_rate

# Function to save audio
def save_audio(audio_data, sample_rate, filename):
    file_path = os.path.join("audio_inputs", filename)
    sf.write(file_path, audio_data, sample_rate)
    return file_path

# Function to generate sinusoidal wave
def generate_sine_wave(frequency, duration, sample_rate=22050):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    sine_wave = np.sin(2 * np.pi * frequency * t)
    return sine_wave, sample_rate

# Function to compute FFT
def compute_fft(y, sr):
    # Apply Hann window
    window = np.hanning(len(y))
    y_windowed = y * window

    # Compute FFT
    Y = np.fft.rfft(y_windowed)
    frequencies = np.fft.rfftfreq(len(y), d=1/sr)

    # Compute power spectrum
    power = np.abs(Y) ** 2

    return frequencies, power

# Function to compute STFT
def compute_stft(y, sr):
    D = librosa.stft(y)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    return S_db

# Function to compute Mel spectrogram
def compute_mel_spectrogram(y, sr):
    S = librosa.feature.melspectrogram(y=y, sr=sr)
    S_db = librosa.power_to_db(S, ref=np.max)
    return S_db

# Function to compute MFCC
def compute_mfcc(y, sr, n_mfcc=13):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return mfcc

# Function to extract MFCC features for classification
def extract_mfcc_features(y, sr, n_mfcc=13):
    mfcc = compute_mfcc(y, sr, n_mfcc)
    return np.mean(mfcc, axis=1)  # Average over time

# Audio Input Page
if page == "Audio Input":
    st.header("Audio Input")

    # File upload
    uploaded_file = st.file_uploader("Upload an audio file", type=["wav"])

    # Record audio
    col1, col2 = st.columns(2)
    with col1:
        record_duration = st.slider("Recording duration (seconds)", 1, 10, 5)
    with col2:
        record_button = st.button("Record Audio")

    if uploaded_file is not None:
        # Save uploaded file
        file_path = os.path.join("audio_inputs", uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Load audio data
        audio_data, sample_rate = load_audio(file_path)

        # Store in session state
        st.session_state.audio_data = audio_data
        st.session_state.sample_rate = sample_rate
        st.session_state.file_path = file_path

        # Display audio player
        st.audio(file_path)
        st.success(f"File uploaded and saved to {file_path}")

    elif record_button:
        # Record audio
        audio_data, sample_rate = record_audio(record_duration)

        # Generate filename with timestamp
        timestamp = int(time.time())
        filename = f"recorded_{timestamp}.wav"
        file_path = save_audio(audio_data, sample_rate, filename)

        # Store in session state
        st.session_state.audio_data = audio_data
        st.session_state.sample_rate = sample_rate
        st.session_state.file_path = file_path

        # Display audio player
        st.audio(file_path)
        st.success(f"Audio recorded and saved to {file_path}")

# Time Domain Visualization Page
elif page == "Time Domain Visualization":
    st.header("Time Domain Visualization")

    if st.session_state.audio_data is not None:
        y = st.session_state.audio_data
        sr = st.session_state.sample_rate

        # Create time axis
        t = np.arange(len(y)) / sr

        # Plot waveform
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=t, y=y, mode='lines', name='Amplitude'))
        fig.update_layout(
            title="Audio Waveform",
            xaxis_title="Time (s)",
            yaxis_title="Amplitude",
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)

        # Display audio information
        st.subheader("Audio Information")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Duration", f"{len(y)/sr:.2f} seconds")
        with col2:
            st.metric("Sample Rate", f"{sr} Hz")
        with col3:
            st.metric("Number of Samples", f"{len(y)}")
    else:
        st.warning("Please upload or record an audio file first.")

# Sinusoidal Wave Generator Page
elif page == "Sinusoidal Wave Generator":
    st.header("Sinusoidal Wave Generator")

    col1, col2 = st.columns(2)
    with col1:
        frequency = st.slider("Frequency (Hz)", 20, 20000, 440)
    with col2:
        duration = st.slider("Duration (seconds)", 0.1, 5.0, 2.0)

    generate_button = st.button("Generate Sine Wave")

    if generate_button:
        # Generate sine wave
        sine_wave, sample_rate = generate_sine_wave(frequency, duration)

        # Create time axis
        t = np.arange(len(sine_wave)) / sample_rate

        # Plot waveform
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=t, y=sine_wave, mode='lines', name='Amplitude'))
        fig.update_layout(
            title=f"Sine Wave ({frequency} Hz)",
            xaxis_title="Time (s)",
            yaxis_title="Amplitude",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

        # Save sine wave
        timestamp = int(time.time())
        filename = f"sine_{frequency}Hz_{timestamp}.wav"
        file_path = save_audio(sine_wave, sample_rate, filename)

        # Display audio player
        st.audio(file_path)
        st.success(f"Sine wave generated and saved to {file_path}")

        # Store in session state
        st.session_state.audio_data = sine_wave
        st.session_state.sample_rate = sample_rate
        st.session_state.file_path = file_path

# Spectral Analysis Page
elif page == "Spectral Analysis":
    st.header("Spectral Analysis")

    if st.session_state.audio_data is not None:
        y = st.session_state.audio_data
        sr = st.session_state.sample_rate

        # Compute FFT
        frequencies, power = compute_fft(y, sr)

        # Plot power spectrum
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=frequencies, y=power, mode='lines', name='Power'))
        fig.update_layout(
            title="Power Spectrum",
            xaxis_title="Frequency (Hz)",
            yaxis_title="Power",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

        # Plot power spectrum in dB
        power_db = 10 * np.log10(power + 1e-10)  # Add small value to avoid log(0)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=frequencies, y=power_db, mode='lines', name='Power (dB)'))
        fig.update_layout(
            title="Power Spectrum (dB)",
            xaxis_title="Frequency (Hz)",
            yaxis_title="Power (dB)",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Please upload or record an audio file first.")

# STFT Analysis Page
elif page == "STFT Analysis":
    st.header("Short-Time Fourier Transform (STFT) Analysis")

    if st.session_state.audio_data is not None:
        y = st.session_state.audio_data
        sr = st.session_state.sample_rate

        # Compute STFT
        S_db = compute_stft(y, sr)

        # Create figure using matplotlib for better compatibility with librosa
        fig, ax = plt.subplots(figsize=(10, 4))
        img = librosa.display.specshow(S_db, x_axis='time', y_axis='log', ax=ax, sr=sr)
        fig.colorbar(img, ax=ax, format="%+2.0f dB")
        ax.set_title("Spectrogram (dB)")

        # Convert matplotlib figure to plotly
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        buf.seek(0)
        st.image(buf)
    else:
        st.warning("Please upload or record an audio file first.")

# Complete Analysis Page
elif page == "Complete Analysis":
    st.header("Complete Audio Analysis")

    if st.session_state.audio_data is not None:
        y = st.session_state.audio_data
        sr = st.session_state.sample_rate

        # Compute STFT
        S_db = compute_stft(y, sr)

        # Compute Mel spectrogram
        mel_S_db = compute_mel_spectrogram(y, sr)

        # Compute MFCC
        mfcc = compute_mfcc(y, sr)

        # Create figures using matplotlib
        fig, axs = plt.subplots(3, 1, figsize=(10, 12))

        # Plot STFT
        img1 = librosa.display.specshow(S_db, x_axis='time', y_axis='log', ax=axs[0], sr=sr)
        fig.colorbar(img1, ax=axs[0], format="%+2.0f dB")
        axs[0].set_title("Spectrogram (dB)")

        # Plot Mel spectrogram
        img2 = librosa.display.specshow(mel_S_db, x_axis='time', y_axis='mel', ax=axs[1], sr=sr)
        fig.colorbar(img2, ax=axs[1], format="%+2.0f dB")
        axs[1].set_title("Mel Spectrogram (dB)")

        # Plot MFCC
        img3 = librosa.display.specshow(mfcc, x_axis='time', ax=axs[2], sr=sr)
        fig.colorbar(img3, ax=axs[2])
        axs[2].set_title("MFCC")

        plt.tight_layout()

        # Convert matplotlib figure to plotly
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        buf.seek(0)
        st.image(buf)
    else:
        st.warning("Please upload or record an audio file first.")

# Sound Classification Page
elif page == "Sound Classification":
    st.header("Sound Classification with MFCC")

    # Initialize session state variables if they don't exist
    if 'class_0_features' not in st.session_state:
        st.session_state.class_0_features = []
        st.session_state.class_0_files = set()

    if 'class_1_features' not in st.session_state:
        st.session_state.class_1_features = []
        st.session_state.class_1_files = set()

    st.subheader("Training Data")
    st.write("Upload audio files for training the classifier. You need at least 2 files per class.")

    # Class tabs
    class_tab_0, class_tab_1 = st.tabs(["Class 0", "Class 1"])

    # Class 0 tab
    with class_tab_0:
        st.write(f"Current Class 0 samples: {len(st.session_state.class_0_features)}")
        st.write(f"Files: {', '.join(st.session_state.class_0_files) if st.session_state.class_0_files else 'None'}")

        # File upload for Class 0
        uploaded_files_0 = st.file_uploader("Upload audio files for Class 0",
                                          type=["wav"],
                                          accept_multiple_files=True,
                                          key="upload_class_0")

        # Add files button for Class 0
        if uploaded_files_0:
            if st.button("Add files to Class 0"):
                added_count = 0
                for uploaded_file in uploaded_files_0:
                    if uploaded_file.name not in st.session_state.class_0_files:
                        # Save uploaded file
                        file_path = os.path.join("audio_inputs", uploaded_file.name)
                        with open(file_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())

                        # Load audio data
                        y, sr = load_audio(file_path)

                        # Extract MFCC features
                        features = extract_mfcc_features(y, sr)

                        # Add to Class 0 data
                        st.session_state.class_0_features.append(features)
                        st.session_state.class_0_files.add(uploaded_file.name)
                        added_count += 1

                if added_count > 0:
                    st.success(f"Added {added_count} files to Class 0")
                    st.rerun()
                else:
                    st.info("These files have already been added to Class 0")

    # Class 1 tab
    with class_tab_1:
        st.write(f"Current Class 1 samples: {len(st.session_state.class_1_features)}")
        st.write(f"Files: {', '.join(st.session_state.class_1_files) if st.session_state.class_1_files else 'None'}")

        # File upload for Class 1
        uploaded_files_1 = st.file_uploader("Upload audio files for Class 1",
                                          type=["wav"],
                                          accept_multiple_files=True,
                                          key="upload_class_1")

        # Add files button for Class 1
        if uploaded_files_1:
            if st.button("Add files to Class 1"):
                added_count = 0
                for uploaded_file in uploaded_files_1:
                    if uploaded_file.name not in st.session_state.class_1_files:
                        # Save uploaded file
                        file_path = os.path.join("audio_inputs", uploaded_file.name)
                        with open(file_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())

                        # Load audio data
                        y, sr = load_audio(file_path)

                        # Extract MFCC features
                        features = extract_mfcc_features(y, sr)

                        # Add to Class 1 data
                        st.session_state.class_1_features.append(features)
                        st.session_state.class_1_files.add(uploaded_file.name)
                        added_count += 1

                if added_count > 0:
                    st.success(f"Added {added_count} files to Class 1")
                    st.rerun()
                else:
                    st.info("These files have already been added to Class 1")

    # Display training data summary
    col1, col2 = st.columns(2)

    with col1:
        total_samples = len(st.session_state.class_0_features) + len(st.session_state.class_1_features)
        st.write(f"Total training samples: {total_samples}")
        st.write(f"Class 0: {len(st.session_state.class_0_features)} samples")
        st.write(f"Class 1: {len(st.session_state.class_1_features)} samples")

    with col2:
        # Clear training data button
        if st.button("Clear all training data"):
            st.session_state.class_0_features = []
            st.session_state.class_0_files = set()
            st.session_state.class_1_features = []
            st.session_state.class_1_files = set()
            if 'classifier' in st.session_state:
                del st.session_state.classifier
                del st.session_state.scaler
            st.success("Training data cleared")
            st.rerun()

    # Train classifier
    train_button = st.button("Train Classifier")

    min_samples_per_class = 2
    has_enough_samples = (len(st.session_state.class_0_features) >= min_samples_per_class and
                          len(st.session_state.class_1_features) >= min_samples_per_class)

    if not has_enough_samples and train_button:
        st.error(f"You need at least {min_samples_per_class} samples for each class to train the classifier.")

    if train_button and has_enough_samples:
        # Combine features from both classes
        features = st.session_state.class_0_features + st.session_state.class_1_features

        # Create labels (0 for Class 0, 1 for Class 1)
        labels = [0] * len(st.session_state.class_0_features) + [1] * len(st.session_state.class_1_features)

        # Convert to numpy arrays
        X = np.array(features)
        y = np.array(labels)

        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Train classifier
        classifier = KNeighborsClassifier(n_neighbors=3)
        classifier.fit(X_scaled, y)

        # Store classifier and scaler in session state
        st.session_state.classifier = classifier
        st.session_state.scaler = scaler

        st.success("Classifier trained successfully!")

    # Prediction
    st.subheader("Prediction")
    st.write("Upload an audio file for prediction.")

    # File upload for prediction
    prediction_file = st.file_uploader("Upload audio file for prediction", type=["wav"], key="prediction_file")

    if prediction_file is not None and 'classifier' in st.session_state:
        # Save uploaded file
        file_path = os.path.join("audio_inputs", prediction_file.name)
        with open(file_path, "wb") as f:
            f.write(prediction_file.getbuffer())

        # Load audio data
        y, sr = load_audio(file_path)

        # Extract MFCC features
        features = extract_mfcc_features(y, sr)

        # Standardize features
        X_pred = st.session_state.scaler.transform(features.reshape(1, -1))

        # Make prediction
        prediction = st.session_state.classifier.predict(X_pred)[0]

        # Display prediction
        st.write(f"Prediction: Class {prediction}")

        # Display audio player
        st.audio(file_path)
    elif prediction_file is not None and 'classifier' not in st.session_state:
        st.warning("Please train the classifier first.")
