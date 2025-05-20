# Audio Analysis Project

This project provides tools for audio analysis using Librosa, Streamlit, and other Python libraries. It includes features for audio visualization, sinusoidal wave creation, spectral analysis, and MFCC-based classification.

## Features

- Audio file upload and recording
- Time domain visualization of audio signals
- Sinusoidal wave generation
- Spectral analysis (FFT)
- Short-Time Fourier Transform (STFT) visualization
- Mel-Frequency Cepstral Coefficients (MFCC) extraction
- Simple sound classification

## Setup

1. Create a virtual environment:
   ```
   python -m venv venv
   ```

2. Activate the virtual environment:
   - Windows: `.\venv\Scripts\activate`
   - macOS/Linux: `source venv/bin/activate`

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Run the Streamlit app:
   ```
   streamlit run app.py
   ```

## Project Structure

- `app.py`: Main Streamlit application
- `audio_inputs/`: Directory for storing uploaded/recorded audio files
- `utils/`: Utility functions for audio processing

## Dependencies

- numpy
- librosa
- streamlit
- plotly
- soundfile
- scikit-learn
- matplotlib
- sounddevice
- pyaudio
