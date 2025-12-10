import numpy as np
import torch
import os
import requests
import shutil
import cv2
import uuid
import subprocess
import torchaudio
import folder_paths
from PIL import Image
from typing import List, Union


def pil2tensor(image: Union[Image.Image, List[Image.Image]]) -> torch.Tensor:
    """
    Convert PIL image(s) to tensor, matching ComfyUI's implementation.
    
    Args:
        image: Single PIL Image or list of PIL Images
        
    Returns:
        torch.Tensor: Image tensor with values normalized to [0, 1]
    """
    if isinstance(image, list):
        if len(image) == 0:
            return torch.empty(0)
        return torch.cat([pil2tensor(img) for img in image], dim=0)

    if image.mode == 'RGBA':
        image = image.convert('RGB')
    elif image.mode != 'RGB':
        image = image.convert('RGB')

    img_array = np.array(image).astype(np.float32) / 255.0

    return torch.from_numpy(img_array)[None,]


def tensor2pil(image: torch.Tensor) -> List[Image.Image]:
    """
    Convert tensor to PIL image(s), matching ComfyUI's implementation.
    
    Args:
        image: Tensor with shape [B, H, W, 3] or [H, W, 3], values in range [0, 1]
        
    Returns:
        List[Image.Image]: List of PIL Images
    """
    batch_count = image.size(0) if len(image.shape) > 3 else 1
    if batch_count > 1:
        out = []
        for i in range(batch_count):
            out.extend(tensor2pil(image[i]))
        return out

    numpy_image = np.clip(255.0 * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)

    return [Image.fromarray(numpy_image)]


class ComflyVideoAdapter:
    def __init__(self, video_path_or_url):
        if video_path_or_url.startswith('http'):
            self.is_url = True
            self.video_url = video_path_or_url
            self.video_path = None
        else:
            self.is_url = False
            self.video_path = video_path_or_url
            self.video_url = None
        
    def get_dimensions(self):
        if self.is_url:
            return 1280, 720
        else:
            try: 
                cap = cv2.VideoCapture(self.video_path)
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cap.release()
                return width, height
            except Exception as e:
                print(f"Error getting video dimensions: {str(e)}")
                return 1280, 720
            
    def save_to(self, output_path, format="auto", codec="auto", metadata=None):
        if self.is_url:
            try:
                response = requests.get(self.video_url, stream=True)
                response.raise_for_status()
                
                with open(output_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                return True
            except Exception as e:
                print(f"Error downloading video from URL: {str(e)}")
                return False
        else:
            try:
                shutil.copyfile(self.video_path, output_path)
                return True
            except Exception as e:
                print(f"Error saving video: {str(e)}")
                return False

def create_audio_object(audio_url):
    """Create an audio object compatible with ComfyUI's audio nodes"""
    if not audio_url:
        return {
            "waveform": torch.zeros((1, 1, 44100)),  
            "sample_rate": 44100
        }
        
    try:
        temp_dir = os.path.join(folder_paths.get_temp_directory(), "suno_audio")
        os.makedirs(temp_dir, exist_ok=True)
        temp_file = os.path.join(temp_dir, f"suno_{str(uuid.uuid4())[:8]}.mp3")
        
        response = requests.get(audio_url, stream=True)
        response.raise_for_status()
        
        with open(temp_file, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        try:
            waveform, sample_rate = torchaudio.load(temp_file)
            if len(waveform.shape) == 2:  
                waveform = waveform.unsqueeze(0)  
            
            return {
                "waveform": waveform,
                "sample_rate": sample_rate
            }
        except Exception as e:
            print(f"Error loading audio with torchaudio: {str(e)}")

            try:
                if hasattr(folder_paths, "get_ffmpeg_path"):
                    ffmpeg_path = folder_paths.get_ffmpeg_path()
                else:
                    ffmpeg_path = shutil.which("ffmpeg")
                
                if ffmpeg_path:
                    temp_wav = temp_file.replace(".mp3", ".wav")
                    subprocess.run([ffmpeg_path, "-y", "-i", temp_file, temp_wav], 
                                  check=True, capture_output=True)

                    waveform, sample_rate = torchaudio.load(temp_wav)
                    if len(waveform.shape) == 2:  
                        waveform = waveform.unsqueeze(0)  

                    try:
                        os.remove(temp_wav)
                    except:
                        pass
                        
                    return {
                        "waveform": waveform,
                        "sample_rate": sample_rate
                    }
                else:
                    raise Exception("ffmpeg not found, can't process audio")
            except Exception as ffmpeg_error:
                print(f"Error with ffmpeg conversion: {str(ffmpeg_error)}")

                return {
                    "waveform": torch.zeros((1, 1, 44100)),  
                    "sample_rate": 44100,
                    "url": audio_url  
                }
        
    except Exception as e:
        print(f"Error downloading or processing audio: {str(e)}")

    return {
        "waveform": torch.zeros((1, 1, 44100)),
        "sample_rate": 44100
    }