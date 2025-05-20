import os
import requests
import zipfile
import io
import shutil

def download_sample_sounds():
    """
    Download sample sounds for the classification task.
    This function downloads a small subset of the UrbanSound8K dataset.
    """
    # Create directory for sample sounds
    os.makedirs("audio_samples", exist_ok=True)
    
    # URLs for sample sounds (these are example URLs, replace with actual URLs)
    sample_urls = {
        "class0_dog_bark_1": "https://github.com/musikalkemist/DeepLearningForAudioWithPython/raw/master/11-%20Environmental%20Sound%20Classification%20with%20CNNs/audio/dog_bark_0.wav",
        "class0_dog_bark_2": "https://github.com/musikalkemist/DeepLearningForAudioWithPython/raw/master/11-%20Environmental%20Sound%20Classification%20with%20CNNs/audio/dog_bark_1.wav",
        "class0_dog_bark_3": "https://github.com/musikalkemist/DeepLearningForAudioWithPython/raw/master/11-%20Environmental%20Sound%20Classification%20with%20CNNs/audio/dog_bark_2.wav",
        "class1_car_horn_1": "https://github.com/musikalkemist/DeepLearningForAudioWithPython/raw/master/11-%20Environmental%20Sound%20Classification%20with%20CNNs/audio/car_horn_0.wav",
        "class1_car_horn_2": "https://github.com/musikalkemist/DeepLearningForAudioWithPython/raw/master/11-%20Environmental%20Sound%20Classification%20with%20CNNs/audio/car_horn_1.wav",
        "class1_car_horn_3": "https://github.com/musikalkemist/DeepLearningForAudioWithPython/raw/master/11-%20Environmental%20Sound%20Classification%20with%20CNNs/audio/car_horn_2.wav",
    }
    
    # Download each sample
    for name, url in sample_urls.items():
        try:
            print(f"Downloading {name}...")
            response = requests.get(url)
            if response.status_code == 200:
                file_path = os.path.join("audio_samples", f"{name}.wav")
                with open(file_path, "wb") as f:
                    f.write(response.content)
                print(f"Downloaded {file_path}")
            else:
                print(f"Failed to download {name}: HTTP {response.status_code}")
        except Exception as e:
            print(f"Error downloading {name}: {e}")
    
    print("Download completed!")

if __name__ == "__main__":
    download_sample_sounds()
