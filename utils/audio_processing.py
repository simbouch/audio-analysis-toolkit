import numpy as np
import librosa
import librosa.display
import soundfile as sf
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

def load_audio(file_path, sr=None):
    """
    Load an audio file using librosa
    
    Parameters:
    -----------
    file_path : str
        Path to the audio file
    sr : int or None
        Target sample rate. If None, uses the native sample rate
        
    Returns:
    --------
    y : np.ndarray
        Audio time series
    sr : int
        Sample rate
    """
    y, sr = librosa.load(file_path, sr=sr)
    return y, sr

def save_audio(audio_data, sample_rate, file_path):
    """
    Save audio data to a file
    
    Parameters:
    -----------
    audio_data : np.ndarray
        Audio time series
    sample_rate : int
        Sample rate
    file_path : str
        Path to save the audio file
        
    Returns:
    --------
    file_path : str
        Path where the audio file was saved
    """
    sf.write(file_path, audio_data, sample_rate)
    return file_path

def generate_sine_wave(frequency, duration, sample_rate=22050):
    """
    Generate a sinusoidal wave
    
    Parameters:
    -----------
    frequency : float
        Frequency of the sine wave in Hz
    duration : float
        Duration of the sine wave in seconds
    sample_rate : int
        Sample rate in Hz
        
    Returns:
    --------
    sine_wave : np.ndarray
        Sine wave time series
    sample_rate : int
        Sample rate
    """
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    sine_wave = np.sin(2 * np.pi * frequency * t)
    return sine_wave, sample_rate

def compute_fft(y, sr):
    """
    Compute the Fast Fourier Transform of an audio signal
    
    Parameters:
    -----------
    y : np.ndarray
        Audio time series
    sr : int
        Sample rate
        
    Returns:
    --------
    frequencies : np.ndarray
        Frequency bins
    power : np.ndarray
        Power spectrum
    """
    # Apply Hann window
    window = np.hanning(len(y))
    y_windowed = y * window
    
    # Compute FFT
    Y = np.fft.rfft(y_windowed)
    frequencies = np.fft.rfftfreq(len(y), d=1/sr)
    
    # Compute power spectrum
    power = np.abs(Y) ** 2
    
    return frequencies, power

def compute_stft(y, sr):
    """
    Compute the Short-Time Fourier Transform of an audio signal
    
    Parameters:
    -----------
    y : np.ndarray
        Audio time series
    sr : int
        Sample rate
        
    Returns:
    --------
    S_db : np.ndarray
        STFT in dB scale
    """
    D = librosa.stft(y)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    return S_db

def compute_mel_spectrogram(y, sr):
    """
    Compute the Mel spectrogram of an audio signal
    
    Parameters:
    -----------
    y : np.ndarray
        Audio time series
    sr : int
        Sample rate
        
    Returns:
    --------
    S_db : np.ndarray
        Mel spectrogram in dB scale
    """
    S = librosa.feature.melspectrogram(y=y, sr=sr)
    S_db = librosa.power_to_db(S, ref=np.max)
    return S_db

def compute_mfcc(y, sr, n_mfcc=13):
    """
    Compute the Mel-Frequency Cepstral Coefficients of an audio signal
    
    Parameters:
    -----------
    y : np.ndarray
        Audio time series
    sr : int
        Sample rate
    n_mfcc : int
        Number of MFCCs to compute
        
    Returns:
    --------
    mfcc : np.ndarray
        MFCC features
    """
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return mfcc

def extract_mfcc_features(y, sr, n_mfcc=13):
    """
    Extract MFCC features for classification
    
    Parameters:
    -----------
    y : np.ndarray
        Audio time series
    sr : int
        Sample rate
    n_mfcc : int
        Number of MFCCs to compute
        
    Returns:
    --------
    features : np.ndarray
        MFCC features (averaged over time)
    """
    mfcc = compute_mfcc(y, sr, n_mfcc)
    return np.mean(mfcc, axis=1)  # Average over time

def train_knn_classifier(features, labels, n_neighbors=3):
    """
    Train a K-Nearest Neighbors classifier
    
    Parameters:
    -----------
    features : np.ndarray
        Feature matrix
    labels : np.ndarray
        Target labels
    n_neighbors : int
        Number of neighbors
        
    Returns:
    --------
    classifier : KNeighborsClassifier
        Trained classifier
    scaler : StandardScaler
        Feature scaler
    """
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)
    
    # Train classifier
    classifier = KNeighborsClassifier(n_neighbors=n_neighbors)
    classifier.fit(X_scaled, labels)
    
    return classifier, scaler

def predict(classifier, scaler, features):
    """
    Make a prediction using a trained classifier
    
    Parameters:
    -----------
    classifier : object
        Trained classifier
    scaler : StandardScaler
        Feature scaler
    features : np.ndarray
        Features to predict
        
    Returns:
    --------
    prediction : int
        Predicted class
    """
    # Standardize features
    X_pred = scaler.transform(features.reshape(1, -1))
    
    # Make prediction
    prediction = classifier.predict(X_pred)[0]
    
    return prediction
