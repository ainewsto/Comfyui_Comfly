import os
import io
import tempfile
import math
import random
import torch
import wave
import requests
import time
import numpy as np
from PIL import Image
from io import BytesIO
import json
import comfy.utils
import re
import aiohttp
import asyncio
import base64
import uuid
import folder_paths
import mimetypes
import cv2
import shutil
import subprocess
import concurrent.futures
import threading

from ..utils import pil2tensor, tensor2pil, ComflyVideoAdapter
from ..comfly_config import get_config, save_config, baseurl
from comfy.comfy_types import IO


def _comfly_split_asset_ids(s):
    if not s or not str(s).strip():
        return []
    return [p.strip() for p in re.split(r"[\n,]+", str(s)) if p.strip()]


def _comfly_asset_id_to_url(s):
    """Bare id -> asset://<id>; already http(s) or asset:// left unchanged."""
    s = (s or "").strip()
    if not s:
        return None
    low = s.lower()
    if low.startswith("http://") or low.startswith("https://") or low.startswith("asset://"):
        return s
    return f"asset://{s}"


def _parse_asset_bundle_only(bundle_json):
    """Parse JSON from Asset ID Bundle (asset_id strings per slot) -> five fields for API."""
    if not bundle_json or not str(bundle_json).strip():
        return "", "", "", "", ""
    try:
        b = json.loads(bundle_json.strip())
    except Exception as e:
        print(f"[Comfly Seedance] asset_bundle JSON parse failed: {e}")
        return "", "", "", "", ""

    def jl(key):
        v = b.get(key)
        if isinstance(v, list):
            return ",".join(str(x).strip() for x in v if str(x).strip())
        if isinstance(v, str):
            return v.strip()
        return ""

    return (
        str(b.get("first_frame") or "").strip(),
        str(b.get("last_frame") or "").strip(),
        jl("ref_images"),
        jl("videos"),
        jl("audios"),
    )


def _comfy_waveform_to_wav_bytes(waveform, sample_rate):
    """
    Encode Comfy AUDIO tensor to PCM WAV bytes without torchaudio.save (avoids TorchCodec requirement).
    Accepts shapes [C, T], [B, C, T], or [T] (mono).
    """
    wf = waveform.detach().cpu().float()
    if wf.dim() == 3:
        wf = wf.squeeze(0)
    if wf.dim() == 1:
        wf = wf.unsqueeze(0)
    if wf.dim() != 2:
        raise ValueError(f"Expected waveform [C, T], got shape {tuple(wf.shape)}")
    channels, _ = wf.shape
    wf = wf.clamp(-1.0, 1.0)
    pcm = (wf.numpy() * 32767.0).astype(np.int16)
    if channels == 1:
        interleaved = pcm[0]
    else:
        interleaved = np.transpose(pcm, (1, 0)).reshape(-1)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(int(channels))
        wav.setsampwidth(2)
        wav.setframerate(int(sample_rate))
        wav.writeframes(interleaved.tobytes())
    return buf.getvalue()


def _doubao_seedance_io_file_to_bytes(media_input, bytesio_ext=".mp4", label="media"):
    """
    Resolve ComfyUI file-like media (IO.VIDEO VideoFromFile, paths, dicts) to raw bytes.
    VideoFromFile exposes data via get_stream_source() -> str path or BytesIO, not .path.
    Returns (file_bytes, filename) or (None, None).
    """
    if media_input is None:
        return None, None

    get_stream = getattr(media_input, "get_stream_source", None)
    if callable(get_stream):
        try:
            source = media_input.get_stream_source()
            if isinstance(source, str):
                source = source.strip()
                if source and os.path.isfile(source):
                    with open(source, "rb") as f:
                        return f.read(), os.path.basename(source)
                if source:
                    print(f"[Comfly Seedance] {label}: path not found on disk: {source}")
                return None, None
            if isinstance(source, BytesIO):
                source.seek(0)
                data = source.read()
                if data:
                    return data, f"reference_{label}_{abs(hash(data)) % 10**10}{bytesio_ext}"
                return None, None
            if hasattr(source, "read"):
                if hasattr(source, "seek"):
                    source.seek(0)
                data = source.read()
                if data:
                    return data, f"reference_{label}_{abs(hash(data)) % 10**10}{bytesio_ext}"
                return None, None
        except Exception as e:
            print(f"[Comfly Seedance] {label}: get_stream_source() failed: {e}")

    if isinstance(media_input, str):
        p = media_input.strip()
        if p and os.path.isfile(p):
            with open(p, "rb") as f:
                return f.read(), os.path.basename(p)
        return None, None

    if isinstance(media_input, dict):
        p = (
            media_input.get("path")
            or media_input.get("file")
            or media_input.get("file_path")
            or media_input.get("filename")
            or ""
        )
        p = str(p).strip() if p else ""
        if p and os.path.isfile(p):
            with open(p, "rb") as f:
                return f.read(), os.path.basename(p)
        return None, None

    for attr in ("path", "file_path"):
        p = getattr(media_input, attr, None)
        if p and isinstance(p, str):
            p = p.strip()
            if p and os.path.isfile(p):
                with open(p, "rb") as f:
                    return f.read(), os.path.basename(p)

    print(
        f"[Comfly Seedance] Could not read bytes for {label} from type {type(media_input).__name__}; "
        "expected file path, dict with path, or object with get_stream_source() (e.g. VideoFromFile)."
    )
    return None, None


def _doubao_seedance_video_input_to_bytes(video_input):
    """IO.VIDEO -> bytes (wrapper with video defaults)."""
    return _doubao_seedance_io_file_to_bytes(video_input, ".mp4", "video")


class Comfly_Doubao_Seedream:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "model": ("STRING", {"default": "doubao-seedream-3-0-t2i-250415"}),
                "response_format": (["url", "b64_json"], {"default": "url"}),
                "size": (["1024x1024", "864x1152", "1152x864", "1280x720", "720x1280", "832x1248", 
                         "1248x832", "1512x648", "Custom"], {"default": "1024x1024"}),
                "Custom_size": ("STRING", {"default": "1536x1024", "multiline": False}),
                "guidance_scale": ("FLOAT", {"default": 2.5, "min": 1.0, "max": 10.0, "step": 0.1}),
            },
            "optional": {
                "apikey": ("STRING", {"default": ""}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 2147483647}),
                "watermark": ("BOOLEAN", {"default": True})
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "response")
    FUNCTION = "generate_image"
    CATEGORY = "Comfly/Doubao"

    def __init__(self):
        self.api_key = get_config().get('api_key', '')
        self.timeout = 300

    def get_headers(self):
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
    
    def validate_custom_size(self, size_str):
        """Validate a custom size string to ensure it's in the correct format and within allowed range."""
        try:
            if 'x' not in size_str:
                return False, "Custom size must be in format 'widthxheight'"
            
            width, height = map(int, size_str.split('x'))

            if width < 512 or width > 2048 or height < 512 or height > 2048:
                return False, f"Custom size dimensions must be between 512 and 2048 pixels. Got {width}x{height}."
            
            return True, f"{width}x{height}"
        except ValueError:
            return False, "Custom size must contain valid integers in format 'widthxheight'"
    
    def generate_image(self, prompt, model, response_format="url", size="1024x1024", 
                       Custom_size="1536x1024", guidance_scale=2.5, apikey="", 
                       seed=-1, watermark=True):
        if apikey.strip():
            self.api_key = apikey
            config = get_config()
            config['api_key'] = apikey
            
        if not self.api_key:
            error_message = "API key not found in Comflyapi.json"
            print(error_message)
            blank_image = Image.new('RGB', (1024, 1024), color='white')
            blank_tensor = pil2tensor(blank_image)
            return (blank_tensor, error_message)
            
        try:
            pbar = comfy.utils.ProgressBar(100)
            pbar.update_absolute(10)

            final_size = size
            if size == "Custom":
                is_valid, result = self.validate_custom_size(Custom_size)
                if not is_valid:
                    error_message = result
                    print(error_message)
                    blank_image = Image.new('RGB', (1024, 1024), color='white')
                    blank_tensor = pil2tensor(blank_image)
                    return (blank_tensor, error_message)
                final_size = result
            
            payload = {
                "model": model,
                "prompt": prompt,
                "response_format": response_format,
                "size": final_size,
                "guidance_scale": guidance_scale,
                "watermark": watermark
            }
            
            if seed != -1:
                payload["seed"] = seed
            
            response = requests.post(
                f"{baseurl}/v1/images/generations",
                headers=self.get_headers(),
                json=payload,
                timeout=self.timeout
            )
            
            pbar.update_absolute(30)
            
            if response.status_code != 200:
                error_message = f"API Error: {response.status_code} - {response.text}"
                print(error_message)
                blank_image = Image.new('RGB', (1024, 1024), color='white')
                blank_tensor = pil2tensor(blank_image)
                return (blank_tensor, error_message)
                
            result = response.json()
            
            pbar.update_absolute(50)
            
            if "data" not in result or not result["data"]:
                error_message = "No image data in response"
                print(error_message)
                blank_image = Image.new('RGB', (1024, 1024), color='white')
                blank_tensor = pil2tensor(blank_image)
                return (blank_tensor, error_message)
            
            image_url = None
            image_data = None

            if response_format == "url":
                image_url = result["data"][0].get("url")
                if not image_url:
                    error_message = "No image URL in response"
                    print(error_message)
                    blank_image = Image.new('RGB', (1024, 1024), color='white')
                    blank_tensor = pil2tensor(blank_image)
                    return (blank_tensor, error_message)
                    
                try:
                    img_response = requests.get(image_url, timeout=self.timeout)
                    img_response.raise_for_status()
                    image_data = BytesIO(img_response.content)
                except Exception as e:
                    error_message = f"Error downloading image: {str(e)}"
                    print(error_message)
                    blank_image = Image.new('RGB', (1024, 1024), color='white')
                    blank_tensor = pil2tensor(blank_image)
                    return (blank_tensor, error_message)
            else:
                b64_data = result["data"][0].get("b64_json")
                if not b64_data:
                    error_message = "No base64 data in response"
                    print(error_message)
                    blank_image = Image.new('RGB', (1024, 1024), color='white')
                    blank_tensor = pil2tensor(blank_image)
                    return (blank_tensor, error_message)
                    
                image_data = BytesIO(base64.b64decode(b64_data))
            
            pbar.update_absolute(80)

            try:
                pil_image = Image.open(image_data)
                tensor_image = pil2tensor(pil_image)

                response_info = {
                    "prompt": prompt,
                    "model": model,
                    "size": final_size,
                    "guidance_scale": guidance_scale,
                    "seed": seed if seed != -1 else "auto",
                    "url": image_url if image_url else "base64 data"
                }
                
                pbar.update_absolute(100)
                return (tensor_image, json.dumps(response_info, indent=2))
                
            except Exception as e:
                error_message = f"Error processing image: {str(e)}"
                print(error_message)
                blank_image = Image.new('RGB', (1024, 1024), color='white')
                blank_tensor = pil2tensor(blank_image)
                return (blank_tensor, error_message)
                
        except Exception as e:
            error_message = f"Error generating image: {str(e)}"
            print(error_message)
            blank_image = Image.new('RGB', (1024, 1024), color='white')
            blank_tensor = pil2tensor(blank_image)
            return (blank_tensor, error_message)


class Comfly_Doubao_Seedream_4:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "model": ("STRING", {"default": "doubao-seedream-4-0-250828"}),
                "response_format": (["url", "b64_json"], {"default": "url"}),
                "resolution": (["1K", "2K", "4K"], {"default": "1K"}),
            },
            "optional": {
                "aspect_ratio": (["1:1", "4:3", "3:4", "16:9", "9:16", "2:3", "3:2", "21:9", "9:21", "Custom"], {"default": "1:1"}),
                "width": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 1}),
                "height": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 1}),
                "apikey": ("STRING", {"default": ""}),
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
                "image4": ("IMAGE",),
                "image5": ("IMAGE",),
                "sequential_image_generation": (["disabled", "auto"], {"default": "disabled"}),
                "max_images": ("INT", {"default": 1, "min": 1, "max": 15, "step": 1}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 2147483647}),
                "watermark": ("BOOLEAN", {"default": True}),
                "stream": ("BOOLEAN", {"default": False}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("image", "response", "image_url")
    FUNCTION = "generate_image"
    CATEGORY = "Comfly/Doubao"

    def __init__(self):
        self.api_key = get_config().get('api_key', '')
        self.timeout = 900
        self.size_mapping = {
            
            "1K": {
                "1:1": "1024x1024",
                "4:3": "1152x864",
                "3:4": "864x1152",
                "16:9": "1280x720",
                "9:16": "720x1280",
                "2:3": "832x1248",
                "3:2": "1248x832",
                "21:9": "1512x648",
                "9:21": "648x1512"
            },

            "2K": {
                "1:1": "2048x2048",
                "4:3": "2048x1536",
                "3:4": "1536x2048",
                "16:9": "2048x1152",
                "9:16": "1152x2048",
                "2:3": "1536x2048",
                "3:2": "2048x1536",
                "21:9": "2048x864",
                "9:21": "864x2048"
            },

            "4K": {
                "1:1": "4096x4096",
                "4:3": "4096x3072",
                "3:4": "3072x4096",
                "16:9": "4096x2304",
                "9:16": "2304x4096",
                "2:3": "3072x4096",
                "3:2": "4096x3072",
                "21:9": "4096x1728",
                "9:21": "1728x4096"
            }
        }

        self.resolution_factors = {
            "1K": 1,
            "2K": 2,
            "4K": 4
        }

    def get_headers(self):
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
    
    def image_to_base64(self, image_tensor):
        """Convert tensor to base64 string"""
        if image_tensor is None:
            return None
            
        pil_image = tensor2pil(image_tensor)[0]
        buffered = BytesIO()
        pil_image.save(buffered, format="PNG")
        image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        return f"data:image/png;base64,{image_base64}"
    
    def generate_image(self, prompt, model, response_format="url", resolution="1K", 
                  aspect_ratio="1:1", width=1024, height=1024, apikey="", 
                  image1=None, image2=None, image3=None, image4=None, image5=None, 
                  sequential_image_generation="disabled", max_images=1, seed=-1, 
                  watermark=True, stream=False):
        if apikey.strip():
            self.api_key = apikey
            config = get_config()
            config['api_key'] = apikey
            
        if not self.api_key:
            error_message = "API key not found in Comflyapi.json"
            print(error_message)
            blank_image = Image.new('RGB', (1024, 1024), color='white')
            blank_tensor = pil2tensor(blank_image)
            return (blank_tensor, error_message, "")
            
        try:
            pbar = comfy.utils.ProgressBar(100)
            pbar.update_absolute(10)

            if aspect_ratio == "Custom":

                scale_factor = self.resolution_factors.get(resolution, 1)
                scaled_width = int(width * scale_factor)
                scaled_height = int(height * scale_factor)
    
                final_size = f"{scaled_width}x{scaled_height}"
                print(f"Using custom dimensions with {resolution} scaling: {final_size}")
            else:
                if resolution in self.size_mapping and aspect_ratio in self.size_mapping[resolution]:
                    final_size = self.size_mapping[resolution][aspect_ratio]
                else:
                    final_size = "1024x1024"
                    print(f"Warning: Combination of {resolution} resolution and {aspect_ratio} aspect ratio not found. Using {final_size}.")
            
            payload = {
                "model": model,
                "prompt": prompt,
                "response_format": response_format,
                "size": final_size,
                "watermark": watermark,
                "stream": stream
            }

            if sequential_image_generation == "auto":
                payload["sequential_image_generation"] = sequential_image_generation
                payload["n"] = max_images
                
            if seed != -1:
                payload["seed"] = seed

            image_urls = []
            for img in [image1, image2, image3, image4, image5]:
                if img is not None:
                    batch_size = img.shape[0]
                    for i in range(batch_size):
                        single_image = img[i:i+1]
                        image_base64 = self.image_to_base64(single_image)
                        if image_base64:
                            image_urls.append(image_base64)
            
            if image_urls:
                payload["image"] = image_urls
            
            response = requests.post(
                f"{baseurl}/v1/images/generations",
                headers=self.get_headers(),
                json=payload,
                timeout=self.timeout
            )
            
            pbar.update_absolute(30)
            
            if response.status_code != 200:
                error_message = f"API Error: {response.status_code} - {response.text}"
                print(error_message)
                blank_image = Image.new('RGB', (1024, 1024), color='white')
                blank_tensor = pil2tensor(blank_image)
                return (blank_tensor, error_message, "")
                
            result = response.json()
            
            pbar.update_absolute(50)
            
            if "data" not in result or not result["data"]:
                error_message = "No image data in response"
                print(error_message)
                blank_image = Image.new('RGB', (1024, 1024), color='white')
                blank_tensor = pil2tensor(blank_image)
                return (blank_tensor, error_message, "")
            
            image_url = None
            image_data = None
            generated_images = []
            image_urls = []
            for item in result["data"]:
                if response_format == "url":
                    image_url = item.get("url")
                    if not image_url:
                        continue
                    
                    image_urls.append(image_url)
                    
                    try:
                        img_response = requests.get(image_url, timeout=self.timeout)
                        img_response.raise_for_status()
                        image_data = BytesIO(img_response.content)
                        
                        pil_image = Image.open(image_data)
                        tensor_image = pil2tensor(pil_image)
                        generated_images.append(tensor_image)
                    except Exception as e:
                        print(f"Error downloading image: {str(e)}")
                else:
                    b64_data = item.get("b64_json")
                    if not b64_data:
                        continue
                        
                    image_data = BytesIO(base64.b64decode(b64_data))
                    
                    pil_image = Image.open(image_data)
                    tensor_image = pil2tensor(pil_image)
                    generated_images.append(tensor_image)
            
            pbar.update_absolute(80)
            if not generated_images:
                error_message = "Failed to process any images"
                print(error_message)
                blank_image = Image.new('RGB', (1024, 1024), color='white')
                blank_tensor = pil2tensor(blank_image)
                return (blank_tensor, error_message, "")
            
            combined_tensor = torch.cat(generated_images, dim=0)
                
            response_info = {
                "prompt": prompt,
                "model": model,
                "resolution": resolution,
                "size": final_size,
                "seed": seed if seed != -1 else "auto",
                "urls": image_urls if image_urls else [],
                "sequential_image_generation": sequential_image_generation,
                "aspect_ratio": aspect_ratio
            }

            if aspect_ratio == "Custom":
                response_info["original_dimensions"] = f"{width}x{height}"
                response_info["scaled_dimensions"] = final_size
            
            if sequential_image_generation == "auto":
                response_info["max_images"] = max_images
            
            response_info["images_generated"] = len(generated_images)
            
            pbar.update_absolute(100)
            first_image_url = image_urls[0] if image_urls else ""
            return (combined_tensor, json.dumps(response_info, indent=2), first_image_url)
                
        except Exception as e:
            error_message = f"Error generating image: {str(e)}"
            print(error_message)
            blank_image = Image.new('RGB', (1024, 1024), color='white')
            blank_tensor = pil2tensor(blank_image)
            return (blank_tensor, error_message, "")


class Comfly_Doubao_Seededit:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "prompt": ("STRING", {"multiline": True}),
                "model": ("STRING", {"default": "doubao-seededit-3-0-i2i-250628"}),
                "response_format": (["url", "b64_json"], {"default": "url"}),
                "size": ("STRING", {"default": "adaptive"}),
                "guidance_scale": ("FLOAT", {"default": 5.5, "min": 1.0, "max": 10.0, "step": 0.1}),
            },
            "optional": {
                "apikey": ("STRING", {"default": ""}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 2147483647}),
                "watermark": ("BOOLEAN", {"default": True})
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "response")
    FUNCTION = "edit_image"
    CATEGORY = "Comfly/Doubao"

    def __init__(self):
        self.api_key = get_config().get('api_key', '')
        self.timeout = 300

    def get_headers(self):
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
    
    def edit_image(self, image, prompt, model, response_format="url", size="adaptive", 
                guidance_scale=5.5, apikey="", seed=-1, watermark=True):
        if apikey.strip():
            self.api_key = apikey
            config = get_config()
            config['api_key'] = apikey
            
        if not self.api_key:
            error_message = "API key not found in Comflyapi.json"
            print(error_message)
            return (image, error_message)
            
        try:
            pbar = comfy.utils.ProgressBar(100)
            pbar.update_absolute(10)

            pil_image = tensor2pil(image)[0]

            buffered = BytesIO()
            pil_image.save(buffered, format="PNG")
            img_base64 = f"data:image/png;base64,{base64.b64encode(buffered.getvalue()).decode('utf-8')}"
            
            payload = {
                "model": model,
                "prompt": prompt,
                "image": img_base64,
                "response_format": response_format,
                "size": size,
                "guidance_scale": guidance_scale,
                "watermark": watermark
            }
            
            if seed != -1:
                payload["seed"] = seed
            
            response = requests.post(
                f"{baseurl}/v1/images/generations",
                headers=self.get_headers(),
                json=payload,
                timeout=self.timeout
            )
            
            pbar.update_absolute(30)
            
            if response.status_code != 200:
                error_message = f"API Error: {response.status_code} - {response.text}"
                print(error_message)
                return (image, error_message)
                
            result = response.json()
            
            pbar.update_absolute(50)
            
            if "data" not in result or not result["data"]:
                error_message = "No image data in response"
                print(error_message)
                return (image, error_message)
            
            image_url = None
            image_data = None

            if response_format == "url":
                image_url = result["data"][0].get("url")
                if not image_url:
                    error_message = "No image URL in response"
                    print(error_message)
                    return (image, error_message)
                    
                try:
                    img_response = requests.get(image_url, timeout=self.timeout)
                    img_response.raise_for_status()
                    image_data = BytesIO(img_response.content)
                except Exception as e:
                    error_message = f"Error downloading image: {str(e)}"
                    print(error_message)
                    return (image, error_message)
            else:
                b64_data = result["data"][0].get("b64_json")
                if not b64_data:
                    error_message = "No base64 data in response"
                    print(error_message)
                    return (image, error_message)
                    
                image_data = BytesIO(base64.b64decode(b64_data))
            
            pbar.update_absolute(80)

            try:
                edited_pil_image = Image.open(image_data)
                edited_tensor = pil2tensor(edited_pil_image)

                response_info = {
                    "prompt": prompt,
                    "model": model,
                    "size": size,
                    "guidance_scale": guidance_scale,
                    "seed": seed if seed != -1 else "auto",
                    "url": image_url if image_url else "base64 data"
                }
                
                pbar.update_absolute(100)
                return (edited_tensor, json.dumps(response_info, indent=2))
                
            except Exception as e:
                error_message = f"Error processing edited image: {str(e)}"
                print(error_message)
                return (image, error_message)
                
        except Exception as e:
            error_message = f"Error editing image: {str(e)}"
            print(error_message)
            return (image, error_message)


class ComflyJimengApi:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "scale": ("FLOAT", {"default": 2.5, "min": 1.0, "max": 10.0, "step": 0.1}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 2147483647, "step": 1}),
                "width": ("INT", {"default": 1328, "min": 512, "max": 2048, "step": 8}),
                "height": ("INT", {"default": 1328, "min": 512, "max": 2048, "step": 8}),
            },
            "optional": {
                "api_key": ("STRING", {"default": ""}),
                "use_pre_llm": ("BOOLEAN", {"default": False}),
                "add_logo": ("BOOLEAN", {"default": False}),
                "logo_position": (["右下角", "左下角", "左上角", "右上角"], {"default": "右下角"}),
                "logo_language": (["中文", "英文"], {"default": "中文"}),
                "logo_text": ("STRING", {"default": "", "multiline": False}),
                "logo_opacity": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.1}),
                "image": ("IMAGE",),  
                "image_url": ("STRING", {"default": "", "multiline": False}),  
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("generated_image", "response", "image_url")
    FUNCTION = "generate_image"
    CATEGORY = "Comfly/Doubao"
    
    def __init__(self):
        super().__init__()
        self.api_key = get_config().get('api_key', '')
        self.timeout = 300

    def get_headers(self):
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
    def get_logo_position_value(self, position_str):
        position_map = {
            "右下角": 0,
            "左下角": 1,
            "左上角": 2,
            "右上角": 3
        }
        return position_map.get(position_str, 0)
        
    def get_logo_language_value(self, language_str):
        language_map = {
            "中文": 0,
            "英文": 1
        }
        return language_map.get(language_str, 0)
    
    def upload_image(self, image_tensor):
        """Upload image to the file endpoint and return the URL"""
        try:
            pil_image = tensor2pil(image_tensor)[0]

            buffered = BytesIO()
            pil_image.save(buffered, format="PNG")
            file_content = buffered.getvalue()

            files = {'file': ('image.png', file_content, 'image/png')}

            response = requests.post(
                f"{baseurl}/v1/files",
                headers={"Authorization": f"Bearer {self.api_key}"},
                files=files,
                timeout=self.timeout
            )
            
            response.raise_for_status()
            result = response.json()
            
            if 'url' in result:
                return result['url']
            else:
                print(f"Unexpected response from file upload API: {result}")
                return None
                
        except Exception as e:
            print(f"Error uploading image: {str(e)}")
            return None
    
    def generate_image(self, prompt, scale=2.5, seed=-1, width=1328, height=1328, use_pre_llm=False, 
                      add_logo=False, logo_position="右下角", logo_language="中文", 
                      logo_text="", logo_opacity=0.3, api_key="", image=None, image_url=""):
        if api_key.strip():
            self.api_key = api_key
            config = get_config()
            config['api_key'] = api_key
            
        try:
            if not self.api_key:
                error_message = "API key not found in Comflyapi.json"
                print(error_message)

                blank_image = Image.new('RGB', (width, height), color='white')
                blank_tensor = pil2tensor(blank_image)
                return (blank_tensor, error_message, "")

            pbar = comfy.utils.ProgressBar(100)
            pbar.update_absolute(10)

            uploaded_image_url = None
            if image is not None:
                pbar.update_absolute(20)
                print("Uploading image...")
                uploaded_image_url = self.upload_image(image)
                if uploaded_image_url:
                    if not prompt.strip():
                        prompt = "Generate a 3D style version of this image"
                else:
                    print("Image upload failed, proceeding without image")

            final_image_url = uploaded_image_url if uploaded_image_url else image_url

            model_name = "seedream-3.0"  
            
            position_value = self.get_logo_position_value(logo_position)
            language_value = self.get_logo_language_value(logo_language)
            
            logo_info = {
                "add_logo": add_logo,
                "position": position_value,
                "language": language_value,
                "opacity": logo_opacity
            }
 
            if logo_text:
                logo_info["logo_text_content"] = logo_text

            payload = {
                "req_key": "high_aes_general_v30l_zt2i",
                "prompt": prompt,
                "scale": scale,
                "seed": seed,
                "width": width,
                "height": height,
                "use_pre_llm": use_pre_llm,
                "return_url": True,
                "logo_info": logo_info
            }

            if final_image_url:
                combined_prompt = f"{final_image_url} {prompt}"
                
                messages = [
                    {
                        "role": "user",
                        "content": combined_prompt
                    }
                ]
                
                chat_payload = {
                    "model": model_name,
                    "messages": messages,
                    "temperature": 0.5,
                    "top_p": 1,
                    "presence_penalty": 0,
                    "max_tokens": 8192,
                    "stream": True
                }

                api_url = f"{baseurl}/v1/chat/completions"
                headers = self.get_headers()

                pbar.update_absolute(30)

                response = requests.post(
                    api_url,
                    headers=headers,
                    json=chat_payload,
                    timeout=self.timeout,
                    stream=True
                )

                full_response = ""
                for line in response.iter_lines():
                    if line:
                        line_text = line.decode('utf-8')
                        if line_text.startswith('data: '):
                            data = line_text[6:]
                            if data == '[DONE]':
                                break
                            try:
                                chunk = json.loads(data)
                                if 'choices' in chunk and chunk['choices']:
                                    delta = chunk['choices'][0].get('delta', {})
                                    if 'content' in delta:
                                        content = delta['content']
                                        full_response += content
                            except json.JSONDecodeError:
                                continue

                image_url = ""
                image_urls = self.extract_image_urls(full_response)
                if image_urls:
                    image_url = image_urls[0]
                    print(f"Found image URL in response: {image_url}")

                if image_url:
                    response_info = f"**Image Generation with {model_name}**\n\n"
                    response_info += f"Prompt: {prompt}\n"
                    response_info += f"Generated image URL: {image_url}\n\n"
                    response_info += f"Model response: {full_response}"

                    try:
                        img_response = requests.get(image_url, timeout=self.timeout)
                        img_response.raise_for_status()
                        generated_image = Image.open(BytesIO(img_response.content))
                        generated_tensor = pil2tensor(generated_image)
                        pbar.update_absolute(100)
                        return (generated_tensor, response_info, image_url)
                    except Exception as e:
                        error_message = f"Error downloading result image: {str(e)}"
                        print(error_message)
                        if image is not None:
                            return (image, response_info + f"\n\nError: {error_message}", image_url)
                        else:
                            blank_image = Image.new('RGB', (width, height), color='white')
                            blank_tensor = pil2tensor(blank_image)
                            return (blank_tensor, response_info + f"\n\nError: {error_message}", image_url)
                else:
                    error_message = "No image URL found in response"
                    print(error_message)
                    response_info = f"**Error: {error_message}**\n\nFull response: {full_response}"
                    if image is not None:
                        return (image, response_info, "")
                    else:
                        blank_image = Image.new('RGB', (width, height), color='white')
                        blank_tensor = pil2tensor(blank_image)
                        return (blank_tensor, response_info, "")
            
            else:
                api_url = f"{baseurl}/volcv/v1?Action=CVProcess&Version=2022-08-31"
                
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                response_info = f"**Jimeng Image Generation Request**\n\n"
                response_info += f"Prompt: {prompt}\n"
                response_info += f"Scale: {scale}\n"
                response_info += f"Seed: {seed}\n"
                response_info += f"Dimensions: {width}x{height}\n"
                response_info += f"Time: {timestamp}\n\n"
                
                try:
                    response = requests.post(
                        api_url,
                        headers=self.get_headers(),
                        json=payload,
                        timeout=self.timeout
                    )
                except requests.exceptions.Timeout:
                    error_message = f"API request timed out after {self.timeout} seconds"
                    print(error_message)
                    response_info += f"Error: {error_message}"
                    blank_image = Image.new('RGB', (width, height), color='white')
                    blank_tensor = pil2tensor(blank_image)
                    return (blank_tensor, response_info, "")

                if response.status_code != 200:
                    error_message = f"API Error: Status {response.status_code}\nResponse: {response.text}"
                    print(error_message)
                    response_info += f"Error: {error_message}"
                    blank_image = Image.new('RGB', (width, height), color='white')
                    blank_tensor = pil2tensor(blank_image)
                    return (blank_tensor, response_info, "")
                    
                result = response.json()
                
                pbar.update_absolute(70)

                if result.get("code") != 10000:
                    error_message = f"API Error: {result.get('message', 'Unknown error')}\nDetails: {json.dumps(result, indent=2)}"
                    print(error_message)
                    response_info += f"Error: {error_message}"
                    blank_image = Image.new('RGB', (width, height), color='white')
                    blank_tensor = pil2tensor(blank_image)
                    return (blank_tensor, response_info, "")

                image_url = ""
                if "image_urls" in result["data"] and result["data"]["image_urls"]:
                    image_url = result["data"]["image_urls"][0]
                    response_info += f"Success!\n\nImage URL: {image_url}\n\n"
                    
                    if "vlm_result" in result["data"] and result["data"]["vlm_result"]:
                        response_info += f"VLM Description: {result['data']['vlm_result']}\n"
                else:
                    error_message = "No image URL found in response"
                    print(error_message)
                    response_info += f"Error: {error_message}\nFull response: {json.dumps(result, indent=2)}"
                    blank_image = Image.new('RGB', (width, height), color='white')
                    blank_tensor = pil2tensor(blank_image)
                    return (blank_tensor, response_info, "")
                
                print(f"Found image URL: {image_url}")

                try:
                    img_response = requests.get(image_url, timeout=self.timeout)
                    img_response.raise_for_status()
                except requests.exceptions.Timeout:
                    error_message = f"Timeout while downloading result image after {self.timeout} seconds"
                    print(error_message)
                    response_info += f"Error: {error_message}"
                    blank_image = Image.new('RGB', (width, height), color='white')
                    blank_tensor = pil2tensor(blank_image)
                    return (blank_tensor, response_info, image_url)  
                except Exception as e:
                    error_message = f"Error downloading result image: {str(e)}"
                    print(error_message)
                    response_info += f"Error: {error_message}"
                    blank_image = Image.new('RGB', (width, height), color='white')
                    blank_tensor = pil2tensor(blank_image)
                    return (blank_tensor, response_info, image_url)  
                    
                generated_image = Image.open(BytesIO(img_response.content))
                
                generated_tensor = pil2tensor(generated_image)
                
                pbar.update_absolute(100)
            
                if "request_id" in result:
                    response_info += f"Request ID: {result['request_id']}\n"
                
                if "time_elapsed" in result:
                    response_info += f"Processing Time: {result['time_elapsed']}\n"
                
                return (generated_tensor, response_info, image_url)
                
        except Exception as e:
            error_message = f"Error in image generation: {str(e)}"
            print(error_message)
            blank_image = Image.new('RGB', (width, height), color='white')
            blank_tensor = pil2tensor(blank_image)
            return (blank_tensor, error_message, "")
            
    def extract_image_urls(self, response_text):
        """Extract image URLs from markdown format in response"""
        image_pattern = r'!\[.*?\]\((.*?)\)'
        matches = re.findall(image_pattern, response_text)

        if not matches:
            url_pattern = r'https?://\S+\.(?:jpg|jpeg|png|gif|webp)'
            matches = re.findall(url_pattern, response_text)
            
        return matches if matches else []


class ComflyJimengVideoApi:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "duration": ([5, 10], {"default": 5}),
                "aspect_ratio": (["1:1", "21:9", "16:9", "9:16", "4:3", "3:4"], {"default": "16:9"}),
                "cfg_scale": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 1.0, "step": 0.1}),
            },
            "optional": {
                "api_key": ("STRING", {"default": ""}),
                "image": ("IMAGE",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647})
            }
        }
    
    RETURN_TYPES = (IO.VIDEO, "STRING", "STRING", "STRING")
    RETURN_NAMES = ("video", "task_id", "response", "video_url")
    FUNCTION = "generate_video"
    CATEGORY = "Comfly/Doubao"

    def __init__(self):
        self.api_key = get_config().get('api_key', '')
        self.timeout = 300

    def get_headers(self):
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
    
    def upload_image(self, image_tensor):
        """Upload image to the file endpoint and return the URL"""
        try:
            pil_image = tensor2pil(image_tensor)[0]

            buffered = BytesIO()
            pil_image.save(buffered, format="PNG")
            file_content = buffered.getvalue()

            files = {'file': ('image.png', file_content, 'image/png')}

            response = requests.post(
                f"{baseurl}/v1/files",
                headers={"Authorization": f"Bearer {self.api_key}"},
                files=files,
                timeout=self.timeout
            )
            
            response.raise_for_status()
            result = response.json()
            
            if 'url' in result:
                return result['url']
            else:
                print(f"Unexpected response from file upload API: {result}")
                return None
                
        except Exception as e:
            print(f"Error uploading image: {str(e)}")
            return None
    
    def generate_video(self, prompt, duration, aspect_ratio, cfg_scale, api_key="", image=None, seed=0):
        if api_key.strip():
            self.api_key = api_key
            config = get_config()
            config['api_key'] = api_key
            
        if not self.api_key:
            error_response = {"code": "error", "message": "API key not found in Comflyapi.json"}
            return ("", "", json.dumps(error_response), "")
            
        try:
            pbar = comfy.utils.ProgressBar(100)
            pbar.update_absolute(10)

            payload = {
                "prompt": prompt,
                "duration": int(duration),
                "aspect_ratio": aspect_ratio,
                "cfg_scale": cfg_scale
            }

            if seed > 0:
                payload["seed"] = seed

            image_url = None
            if image is not None:
                pbar.update_absolute(20)
                image_url = self.upload_image(image)
                if image_url:
                    payload["image_url"] = image_url
                else:
                    error_message = "Failed to upload image. Please check your image and try again."
                    print(error_message)
                    return ("", "", json.dumps({"code": "error", "message": error_message}), "")

            pbar.update_absolute(30)
            response = requests.post(
                f"{baseurl}/jimeng/submit/videos",
                headers=self.get_headers(),
                json=payload,
                timeout=self.timeout
            )
            
            if response.status_code != 200:
                error_message = f"API error: {response.status_code} - {response.text}"
                print(error_message)
                return ("", "", json.dumps({"code": "error", "message": error_message}), "")
                
            result = response.json()
            
            if result.get("code") != "success":
                error_message = f"API returned error: {result.get('message', 'Unknown error')}"
                print(error_message)
                return ("", "", json.dumps({"code": "error", "message": error_message}), "")
                
            task_id = result.get("data")
            if not task_id:
                error_message = "No task ID returned from API"
                print(error_message)
                return ("", "", json.dumps({"code": "error", "message": error_message}), "")
            
            pbar.update_absolute(40)
            video_url = None
            attempts = 0
            max_attempts = 18  
            start_time = time.time()
            max_wait_time = 300 
        
            while attempts < max_attempts:
                current_time = time.time()
                elapsed_time = current_time - start_time

                if elapsed_time > max_wait_time:
                    error_message = f"Video generation timeout after {elapsed_time:.1f} seconds (max: {max_wait_time}s)"
                    print(error_message)
                    return ("", task_id, json.dumps({"code": "error", "message": error_message}), "")
                
                time.sleep(5)  
                attempts += 1
                
                try:
                    status_response = requests.get(
                        f"{baseurl}/jimeng/fetch/{task_id}",
                        headers=self.get_headers(),
                        timeout=30
                    )
                    
                    if status_response.status_code != 200:
                        continue
                        
                    status_result = status_response.json()

                    if status_result.get("code") != "success":
                        continue

                    data = status_result.get("data", {})
                    progress = data.get("progress", "0%")
                    status = data.get("status", "")

                    try:
                        if progress.endswith('%'):
                            progress_num = int(progress.rstrip('%'))
                            pbar_value = min(90, 40 + progress_num * 50 / 100)
                            pbar.update_absolute(pbar_value)
                    except (ValueError, AttributeError):
                        progress_value = min(80, 40 + (attempts * 40 // max_attempts))
                        pbar.update_absolute(progress_value)

                    if status == "SUCCESS":
                        video_url = None

                        if "video" in data:
                            video_url = data["video"]

                        elif "data" in data and isinstance(data["data"], dict):
                            nested_data = data["data"]
                            if "video" in nested_data:
                                video_url = nested_data["video"]
                            elif "videos" in nested_data and isinstance(nested_data["videos"], list) and len(nested_data["videos"]) > 0:
                                if "url" in nested_data["videos"][0]:
                                    video_url = nested_data["videos"][0]["url"]

                        elif "task_result" in data:
                            task_result = data["task_result"]
                            if "videos" in task_result and isinstance(task_result["videos"], list) and len(task_result["videos"]) > 0:
                                if "url" in task_result["videos"][0]:
                                    video_url = task_result["videos"][0]["url"]
                        
                        if video_url:
                            break
                        else:
                            continue

                    elif status == "FAILED":
                        fail_reason = data.get("fail_reason", "Unknown error")
                        error_message = f"Video generation failed: {fail_reason}"
                        print(error_message)
                        return ("", task_id, json.dumps({"code": "error", "message": error_message}), "")
                    
                    elif status in ["PENDING", "PROCESSING", "RUNNING"]:
                        continue
                    else:
                        continue
                    
                except requests.exceptions.Timeout:
                    continue
                except Exception as e:
                    continue
            
            if not video_url:
                error_message = f"Video generation timeout or failed to retrieve video URL after {attempts} attempts, elapsed time: {elapsed_time:.1f}s"
                print(error_message)
                return ("", task_id, json.dumps({"code": "error", "message": error_message}), "")

            if video_url:
                pbar.update_absolute(95)
                print(f"Video generation completed, URL: {video_url}")            
                
                video_adapter = ComflyVideoAdapter(video_url)
                return (video_adapter, task_id, json.dumps({"code": "success", "url": video_url}), video_url)
            
        except Exception as e:
            error_message = f"Error generating video: {str(e)}"
            print(error_message)
            import traceback
            traceback.print_exc()
            return ("", "", json.dumps({"code": "error", "message": error_message}), "")



class ComflySeededit:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "prompt": ("STRING", {"multiline": True}),
                "scale": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 2147483647, "step": 1}),
            },
            "optional": {
                "api_key": ("STRING", {"default": ""}),
                "add_logo": ("BOOLEAN", {"default": False}),
                "logo_position": (["右下角", "左下角", "左上角", "右上角"], {"default": "右下角"}),
                "logo_language": (["中文", "英文"], {"default": "中文"}),
                "logo_text": ("STRING", {"default": "", "multiline": False}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("edited_image", "response", "image_url")
    FUNCTION = "edit_image"
    CATEGORY = "Comfly/Doubao"
    
    def __init__(self):
        super().__init__()
        self.api_key = get_config().get('api_key', '')
        self.timeout = 300

    def get_headers(self):
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
    def get_logo_position_value(self, position_str):
        position_map = {
            "右下角": 0,
            "左下角": 1,
            "左上角": 2,
            "右上角": 3
        }
        return position_map.get(position_str, 0)
        
    def get_logo_language_value(self, language_str):
        language_map = {
            "中文": 0,
            "英文": 1
        }
        return language_map.get(language_str, 0)
    
    def edit_image(self, image, prompt, scale=0.5, seed=-1, add_logo=False, logo_position="右下角", 
                   logo_language="中文", logo_text="", api_key=""):
        if api_key.strip():
            self.api_key = api_key
            config = get_config()
            config['api_key'] = api_key
            
        try:
            if not self.api_key:
                error_message = "API key not found in Comflyapi.json"
                print(error_message)
                return (image, error_message, "")

            pil_image = tensor2pil(image)[0]

            buffered = BytesIO()
            pil_image.save(buffered, format="JPEG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

            pbar = comfy.utils.ProgressBar(100)
            pbar.update_absolute(10)
            
            position_value = self.get_logo_position_value(logo_position)
            language_value = self.get_logo_language_value(logo_language)
            
            logo_info = {
                "add_logo": add_logo,
                "position": position_value,
                "language": language_value
            }
 
            if logo_text:
                logo_info["logo_text_content"] = logo_text

            payload = {
                "req_key": "byteedit_v2.0",
                "binary_data_base64": [img_base64],
                "prompt": prompt,
                "scale": scale,
                "seed": seed,
                "return_url": True,
                "logo_info": logo_info
            }

            api_url = f"{baseurl}/volcv/v1?Action=CVProcess&Version=2022-08-31"
            
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            response_info = f"**SeedEdit Request**\n\n"
            response_info += f"Prompt: {prompt}\n"
            response_info += f"Scale: {scale}\n"
            response_info += f"Seed: {seed}\n"
            response_info += f"Time: {timestamp}\n\n"
            
            try:
                response = requests.post(
                    api_url,
                    headers=self.get_headers(),
                    json=payload,
                    timeout=self.timeout
                )
            except requests.exceptions.Timeout:
                error_message = f"API request timed out after {self.timeout} seconds"
                print(error_message)
                response_info += f"Error: {error_message}"
                return (image, response_info, "")

            if response.status_code != 200:
                error_message = f"API Error: Status {response.status_code}\nResponse: {response.text}"
                print(error_message)
                response_info += f"Error: {error_message}"
                return (image, response_info, "")
                
            result = response.json()
            
            pbar.update_absolute(70)

            if result.get("code") != 10000:
                error_message = f"API Error: {result.get('message', 'Unknown error')}\nDetails: {json.dumps(result, indent=2)}"
                print(error_message)
                response_info += f"Error: {error_message}"
                return (image, response_info, "")

            image_url = ""
            if "image_urls" in result["data"] and result["data"]["image_urls"]:
                image_url = result["data"]["image_urls"][0]
                response_info += f"Success!\n\nImage URL: {image_url}\n\n"
                
                if "vlm_result" in result["data"] and result["data"]["vlm_result"]:
                    response_info += f"VLM Description: {result['data']['vlm_result']}\n"
            else:
                error_message = "No image URL found in response"
                print(error_message)
                response_info += f"Error: {error_message}\nFull response: {json.dumps(result, indent=2)}"
                return (image, response_info, "")
            
            print(f"Found image URL: {image_url}")

            try:
                img_response = requests.get(image_url, timeout=self.timeout)
                img_response.raise_for_status()
            except requests.exceptions.Timeout:
                error_message = f"Timeout while downloading result image after {self.timeout} seconds"
                print(error_message)
                response_info += f"Error: {error_message}"
                return (image, response_info, image_url) 
            except Exception as e:
                error_message = f"Error downloading result image: {str(e)}"
                print(error_message)
                response_info += f"Error: {error_message}"
                return (image, response_info, image_url)  
                
            edited_image = Image.open(BytesIO(img_response.content))

            edited_tensor = pil2tensor(edited_image)
            
            pbar.update_absolute(100)
        
            if "request_id" in result:
                response_info += f"Request ID: {result['request_id']}\n"
            
            if "time_elapsed" in result:
                response_info += f"Processing Time: {result['time_elapsed']}\n"
            
            return (edited_tensor, response_info, image_url)
            
        except Exception as e:
            error_message = f"Error in image editing: {str(e)}"
            print(error_message)
            return (image, error_message, "")


class Comfly_Doubao_Seedance2_0:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "model": (["doubao-seedance-2-0-260128", "doubao-seedance-2-0-fast-260128"], {"default": "doubao-seedance-2-0-260128"}),
                "duration": ("INT", {"default": 5, "min": 4, "max": 15, "step": 1}),
                "ratio": (["16:9", "9:16", "1:1", "4:3", "3:4", "21:9", "9:21", "adaptive"], {"default": "16:9"}),
                "resolution": (["720p", "480p"], {"default": "720p"}),
            },
            "optional": {
                "apikey": ("STRING", {"default": ""}),
                "first_frame": ("IMAGE",),
                "last_frame": ("IMAGE",),
                "ref_image1": ("IMAGE",),
                "ref_image2": ("IMAGE",),
                "ref_image3": ("IMAGE",),
                "ref_image4": ("IMAGE",),
                "ref_image5": ("IMAGE",),
                "ref_image6": ("IMAGE",),
                "ref_image7": ("IMAGE",),
                "ref_image8": ("IMAGE",),
                "ref_image9": ("IMAGE",),
                "video1": (IO.VIDEO, {"tooltip": "Reference video input."}),
                "video2": (IO.VIDEO, {"tooltip": "Reference video input."}),
                "video3": (IO.VIDEO, {"tooltip": "Reference video input."}),
                "audio1": (IO.AUDIO, {"tooltip": "Reference audio input."}),
                "audio2": (IO.AUDIO, {"tooltip": "Reference audio input."}),
                "audio3": (IO.AUDIO, {"tooltip": "Reference audio input."}),
                "generate_audio": ("BOOLEAN", {"default": True}),
                "return_last_frame": ("BOOLEAN", {"default": False}),
                "web_search": ("BOOLEAN", {"default": False}),
                "watermark": ("BOOLEAN", {"default": False}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 2147483647}),
                "asset_bundle": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": "Wire from «Asset ID Bundle» output. JSON built from Asset Upload asset_id strings per slot.",
                    },
                ),
            }
        }

    RETURN_TYPES = (IO.VIDEO, "STRING", "STRING", "STRING", "IMAGE")
    RETURN_NAMES = ("video", "task_id", "response", "video_url", "last_frame_image")
    FUNCTION = "generate_video"
    CATEGORY = "Comfly/Doubao"

    def __init__(self):
        self.api_key = get_config().get('api_key', '')
        self.timeout = 600
        self.poll_interval = 10
        self.max_wait_time = 600

    def get_headers(self):
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

    def image_tensor_to_base64(self, image_tensor):
        if image_tensor is None:
            return None
        try:
            pil_image = tensor2pil(image_tensor)[0]
            buffered = BytesIO()
            pil_image.save(buffered, format="PNG")
            b64_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
            return f"data:image/png;base64,{b64_str}"
        except Exception as e:
            print(f"Image to base64 error: {str(e)}")
            return None

    def upload_file(self, file_content, filename, content_type):
        try:
            files = {'file': (filename, file_content, content_type)}
            response = requests.post(
                f"{baseurl}/v1/files",
                headers={"Authorization": f"Bearer {self.api_key}"},
                files=files,
                timeout=self.timeout
            )
            response.raise_for_status()
            result = response.json()
            if 'url' in result:
                return result['url']
            else:
                print(f"Unexpected response from file upload API: {result}")
                return None
        except Exception as e:
            print(f"File upload error: {str(e)}")
            return None

    def upload_image_get_url(self, image_tensor):
        """Upload IMAGE tensor to /v1/files as PNG; return HTTPS URL for assets/create."""
        if image_tensor is None:
            return None
        try:
            pil_image = tensor2pil(image_tensor)[0]
            buf = BytesIO()
            pil_image.save(buf, format="PNG")
            data = buf.getvalue()
            fn = f"upload_{abs(hash(data)) % 10**10}.png"
            url = self.upload_file(data, fn, "image/png")
            if url:
                print(f"Image uploaded successfully: {url}")
            return url
        except Exception as e:
            print(f"Image upload error: {str(e)}")
            return None

    def upload_video_get_url(self, video_input):
        if video_input is None:
            return None
        try:
            file_content, filename = _doubao_seedance_video_input_to_bytes(video_input)
            if not file_content:
                return None
            if not filename:
                filename = f"reference_video_{abs(hash(file_content)) % 10**10}.mp4"

            mime_type, _ = mimetypes.guess_type(filename)
            if not mime_type:
                mime_type = "video/mp4"

            url = self.upload_file(file_content, filename, mime_type)
            if url:
                print(f"Video uploaded successfully: {url}")
            return url
        except Exception as e:
            print(f"Video upload error: {str(e)}")
            return None

    def upload_audio_get_url(self, audio_input):
        """
        POST /v1/files: prefer reading an on-disk / stream file as raw bytes (no re-encode).
        Only when the input is pure Comfy AUDIO {waveform, sample_rate} with no file path,
        encode to WAV via _comfy_waveform_to_wav_bytes.
        """
        if audio_input is None:
            return None
        try:
            # 1) Path, dict.path, get_stream_source(), etc. — upload original bytes
            file_content, filename = _doubao_seedance_io_file_to_bytes(audio_input, ".wav", "audio")
            if file_content:
                if not filename:
                    filename = f"reference_audio_{abs(hash(file_content)) % 10**10}.wav"
                mime_type, _ = mimetypes.guess_type(filename)
                if not mime_type:
                    mime_type = "audio/wav"
                url = self.upload_file(file_content, filename, mime_type)
                if url:
                    print(f"Audio uploaded successfully (from file/stream): {url}")
                return url

            # 2) Standard Comfy AUDIO: waveform tensor only — must encode to WAV bytes
            if isinstance(audio_input, dict) and audio_input.get("waveform") is not None:
                waveform = audio_input["waveform"]
                if torch.is_tensor(waveform):
                    sample_rate = int(audio_input.get("sample_rate", 44100))
                    if waveform.dim() == 3:
                        waveform = waveform.squeeze(0)
                    if waveform.dim() == 1:
                        waveform = waveform.unsqueeze(0)
                    file_content = _comfy_waveform_to_wav_bytes(waveform, sample_rate)
                    url = self.upload_file(file_content, "audio.wav", "audio/wav")
                    if url:
                        print(f"Audio uploaded successfully (from waveform): {url}")
                    return url

            return None
        except Exception as e:
            print(f"Audio upload error: {str(e)}")
            return None

    def download_image_from_url(self, url):
        try:
            img_response = requests.get(url, timeout=60)
            img_response.raise_for_status()
            pil_image = Image.open(BytesIO(img_response.content))
            return pil2tensor(pil_image)
        except Exception as e:
            print(f"Error downloading last frame image: {str(e)}")
            return None

    def generate_video(self, prompt, model, duration, ratio, resolution,
                       apikey="",
                       first_frame=None, last_frame=None,
                       ref_image1=None, ref_image2=None, ref_image3=None,
                       ref_image4=None, ref_image5=None, ref_image6=None,
                       ref_image7=None, ref_image8=None, ref_image9=None,
                       video1=None, video2=None, video3=None,
                       audio1=None, audio2=None, audio3=None,
                       generate_audio=True, return_last_frame=False,
                       web_search=False,
                       watermark=False, seed=-1,
                       asset_bundle=""):

        blank_image = Image.new('RGB', (1, 1), color='black')
        blank_tensor = pil2tensor(blank_image)

        if apikey and apikey.strip():
            self.api_key = apikey.strip()
            config = get_config()
            config['api_key'] = self.api_key

        if not self.api_key:
            return ("", "", json.dumps({"error": "API key not found."}), "", blank_tensor)

        try:
            pbar = comfy.utils.ProgressBar(100)
            pbar.update_absolute(5)

            asset_id_first_frame, asset_id_last_frame, asset_ids_ref_images, asset_ids_ref_videos, asset_ids_ref_audios = _parse_asset_bundle_only(
                asset_bundle
            )

            content = []
            content.append({"type": "text", "text": prompt})

            frame_count = 0
            has_first_tensor = first_frame is not None
            has_last_tensor = last_frame is not None

            if has_first_tensor:
                b64 = self.image_tensor_to_base64(first_frame)
                if b64:
                    entry = {"type": "image_url", "image_url": {"url": b64}}
                    if has_last_tensor:
                        entry["role"] = "first_frame"
                    content.append(entry)
                    frame_count += 1

            if not has_first_tensor:
                u = _comfly_asset_id_to_url(asset_id_first_frame)
                if u:
                    content.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": u},
                            "role": "first_frame",
                        }
                    )
                    frame_count += 1

            has_first_effective = has_first_tensor or bool(_comfly_asset_id_to_url(asset_id_first_frame))

            if has_last_tensor:
                if not has_first_effective:
                    print("Warning: last_frame without first_frame, skipping.")
                else:
                    b64 = self.image_tensor_to_base64(last_frame)
                    if b64:
                        content.append({
                            "type": "image_url",
                            "image_url": {"url": b64},
                            "role": "last_frame"
                        })
                        frame_count += 1

            if not has_last_tensor and has_first_effective:
                u = _comfly_asset_id_to_url(asset_id_last_frame)
                if u:
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": u},
                        "role": "last_frame"
                    })
                    frame_count += 1

            ref_images = [ref_image1, ref_image2, ref_image3, ref_image4, ref_image5,
                          ref_image6, ref_image7, ref_image8, ref_image9]
            ref_count = 0
            for img in ref_images:
                if img is not None:
                    b64 = self.image_tensor_to_base64(img)
                    if b64:
                        content.append({
                            "type": "image_url",
                            "image_url": {"url": b64},
                            "role": "reference_image"
                        })
                        ref_count += 1

            for aid in _comfly_split_asset_ids(asset_ids_ref_images):
                u = _comfly_asset_id_to_url(aid)
                if u:
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": u},
                        "role": "reference_image"
                    })
                    ref_count += 1

            pbar.update_absolute(15)

            video_inputs = [video1, video2, video3]
            video_count = 0
            for vid in video_inputs:
                url = self.upload_video_get_url(vid)
                if url:
                    content.append({
                        "type": "video_url",
                        "video_url": {"url": url},
                        "role": "reference_video"
                    })
                    video_count += 1

            for aid in _comfly_split_asset_ids(asset_ids_ref_videos):
                u = _comfly_asset_id_to_url(aid)
                if u:
                    content.append({
                        "type": "video_url",
                        "video_url": {"url": u},
                        "role": "reference_video"
                    })
                    video_count += 1

            pbar.update_absolute(25)

            audio_inputs = [audio1, audio2, audio3]
            audio_count = 0
            for aud in audio_inputs:
                url = self.upload_audio_get_url(aud)
                if url:
                    content.append({
                        "type": "audio_url",
                        "audio_url": {"url": url},
                        "role": "reference_audio"
                    })
                    audio_count += 1

            for aid in _comfly_split_asset_ids(asset_ids_ref_audios):
                u = _comfly_asset_id_to_url(aid)
                if u:
                    content.append({
                        "type": "audio_url",
                        "audio_url": {"url": u},
                        "role": "reference_audio"
                    })
                    audio_count += 1

            pbar.update_absolute(30)

            payload = {
                "model": model,
                "content": content,
                "duration": int(duration),
                "ratio": ratio,
                "resolution": resolution,
                "generate_audio": generate_audio,
                "return_last_frame": return_last_frame,
                "watermark": watermark
            }

            if web_search:
                payload["tools"] = [{"type": "web_search"}]
            if seed != -1:
                payload["seed"] = seed

            print(f"Seedance 2.0: model={model}, duration={duration}s, ratio={ratio}, resolution={resolution}")
            print(f"Content: text:1, frames:{frame_count}, ref_images:{ref_count}, videos:{video_count}, audios:{audio_count}")

            response = requests.post(
                f"{baseurl}/seedance/v3/contents/generations/tasks",
                headers=self.get_headers(),
                json=payload,
                timeout=self.timeout
            )

            pbar.update_absolute(35)

            if response.status_code != 200:
                error_message = f"API Error: {response.status_code} - {response.text}"
                print(error_message)
                return ("", "", json.dumps({"error": error_message}), "", blank_tensor)

            result = response.json()
            task_id = result.get("id", "") or result.get("task_id", "")
            if not task_id:
                return ("", "", json.dumps({"error": f"No task ID. Response: {json.dumps(result)}"}), "", blank_tensor)

            print(f"Task ID: {task_id}")
            pbar.update_absolute(40)
            start_time = time.time()
            video_url = None
            last_frame_url = None
            final_status_data = None

            while True:
                elapsed = time.time() - start_time
                if elapsed > self.max_wait_time:
                    return ("", task_id, json.dumps({"error": f"Timeout {elapsed:.1f}s", "task_id": task_id}), "", blank_tensor)

                time.sleep(self.poll_interval)

                try:
                    status_response = requests.get(
                        f"{baseurl}/seedance/v3/contents/generations/tasks/{task_id}",
                        headers={"Authorization": f"Bearer {self.api_key}"},
                        timeout=30
                    )
                    if status_response.status_code != 200:
                        continue

                    status_data = status_response.json()
                    final_status_data = status_data

                    raw_status = status_data.get("status", "")
                    status = raw_status.lower()
                    if status == "success":
                        status = "succeeded"
                    elif status in ("fail", "failure"):
                        status = "failed"

                    progress_str = status_data.get("progress", "")
                    progress = min(90, 40 + int((elapsed / self.max_wait_time) * 50))
                    pbar.update_absolute(progress)

                    if status == "succeeded":
                        # Root { "content": { "video_url": "..." }, "status": "succeeded" } — common shape
                        root_content = status_data.get("content")
                        if isinstance(root_content, dict):
                            video_url = root_content.get("video_url") or root_content.get("videoUrl")

                        # Nested: data.content.video_url
                        data = status_data.get("data")
                        if isinstance(data, dict):
                            data_content = data.get("content")
                            if isinstance(data_content, dict):
                                if not video_url:
                                    video_url = data_content.get("video_url") or data_content.get("videoUrl")
                                if video_url:
                                    print("Found video in data.content.video_url")
                            if not video_url:
                                video_url = data.get("video_url") or data.get("videoUrl")

                        if not video_url:
                            results = status_data.get("results", [])
                            if isinstance(results, list):
                                for r in results:
                                    if isinstance(r, dict):
                                        r_url = r.get("url", "")
                                        r_type = r.get("outputType", "")
                                        if r_type in ("mp4", "video") or r_url.endswith(".mp4"):
                                            video_url = r_url
                                            break
                                        elif r_url and not video_url:
                                            video_url = r_url

                        if not video_url:
                            content_list = status_data.get("content")
                            if isinstance(content_list, list):
                                for item in content_list:
                                    if not isinstance(item, dict):
                                        continue
                                    item_type = item.get("type", "")
                                    item_role = item.get("role", "")
                                    if item_type == "video_url":
                                        vu = item.get("video_url")
                                        if isinstance(vu, dict):
                                            video_url = vu.get("url", "")
                                        elif isinstance(vu, str):
                                            video_url = vu
                                        if video_url:
                                            break
                                    if item_type == "image_url" and item_role == "last_frame":
                                        iu = item.get("image_url")
                                        if isinstance(iu, dict):
                                            last_frame_url = iu.get("url", "")
                                        elif isinstance(iu, str):
                                            last_frame_url = iu

                        if not video_url:
                            video_url = status_data.get("video_url") or status_data.get("videoUrl")

                        if not last_frame_url and return_last_frame:
                            last_frame_url = (
                                status_data.get("last_frame_url")
                                or status_data.get("lastFrameUrl")
                                or status_data.get("last_frame_image_url")
                            )
                            if not last_frame_url:
                                lf = status_data.get("last_frame") or status_data.get("lastFrame")
                                if isinstance(lf, dict):
                                    last_frame_url = lf.get("url", "")
                                elif isinstance(lf, str):
                                    last_frame_url = lf

                        if video_url:
                            print(f"Video ready: {video_url}")
                            break
                        else:
                            print(f"Succeeded but no video URL found: {json.dumps(status_data, indent=2)}")
                            return ("", task_id, json.dumps(status_data, indent=2), "", blank_tensor)

                    elif status == "failed":
                        fail_reason = status_data.get("fail_reason", "") or status_data.get("failReason", "")
                        print(f"Task failed: {fail_reason}")
                        return ("", task_id, json.dumps(status_data, indent=2), "", blank_tensor)

                except requests.exceptions.Timeout:
                    continue
                except Exception as e:
                    continue

            if video_url:
                pbar.update_absolute(95)

                last_frame_tensor = blank_tensor
                if return_last_frame and last_frame_url:
                    downloaded_frame = self.download_image_from_url(last_frame_url)
                    if downloaded_frame is not None:
                        last_frame_tensor = downloaded_frame

                response_info = {
                    "task_id": task_id,
                    "model": model,
                    "status": "succeeded",
                    "video_url": video_url,
                    "duration": duration,
                    "ratio": ratio,
                    "resolution": resolution,
                    "generate_audio": generate_audio,
                    "return_last_frame": return_last_frame,
                    "seed": seed if seed != -1 else "auto",
                    "first_frame": has_first_effective,
                    "last_frame_input": has_first_effective
                    and (has_last_tensor or bool(_comfly_asset_id_to_url(asset_id_last_frame))),
                    "reference_images": ref_count,
                    "reference_videos": video_count,
                    "reference_audios": audio_count,
                }
                if last_frame_url:
                    response_info["last_frame_image_url"] = last_frame_url
                if final_status_data and isinstance(final_status_data, dict):
                    data = final_status_data.get("data")
                    if isinstance(data, dict):
                        if "duration" in data:
                            response_info["actual_duration"] = data["duration"]
                        if "usage" in data:
                            response_info["usage"] = data["usage"]
                    if "usage" in final_status_data:
                        response_info["usage"] = final_status_data["usage"]
                    if "duration" in final_status_data and "actual_duration" not in response_info:
                        response_info["actual_duration"] = final_status_data["duration"]

                # Prefer comfy_api VideoFromFile so Save Video (IO.VIDEO) can preview/save reliably
                video_out = ComflyVideoAdapter(video_url)
                try:
                    from comfy_api.latest import VideoFromFile as CFVideoFromFile

                    fd, tmp_path = tempfile.mkstemp(suffix=".mp4", prefix="comfly_seedance_")
                    os.close(fd)
                    if video_out.save_to(tmp_path):
                        video_out = CFVideoFromFile(tmp_path)
                except Exception as e:
                    print(f"[Comfly Seedance] Using ComflyVideoAdapter (VideoFromFile unavailable): {e}")

                pbar.update_absolute(100)
                return (video_out, task_id, json.dumps(response_info, indent=2), video_url, last_frame_tensor)
            else:
                return ("", task_id, json.dumps({"error": "No video URL"}), "", blank_tensor)

        except Exception as e:
            error_message = f"Seedance 2.0 error: {str(e)}"
            print(error_message)
            import traceback
            traceback.print_exc()
            return ("", "", json.dumps({"error": error_message}), "", blank_tensor)
        


class Comfly_Doubao_Seedance2_0_AssetIdBundle:
    """
    Collect asset_id strings from «Comfly Doubao Seedance 2.0 Asset Upload» (one slot per wire),
    same layout as Seedance 2.0, into one JSON for the main node's asset_bundle input.
    """

    @classmethod
    def INPUT_TYPES(cls):
        opt = {
            "apikey": ("STRING", {"default": ""}),
            "first_frame": ("STRING", {"default": "", "tooltip": "asset_id from Asset Upload"}),
            "last_frame": ("STRING", {"default": "", "tooltip": "asset_id from Asset Upload"}),
        }
        for i in range(1, 10):
            opt[f"ref_image{i}"] = ("STRING", {"default": "", "tooltip": "asset_id from Asset Upload"})
        for i in range(1, 4):
            opt[f"video{i}"] = ("STRING", {"default": "", "tooltip": "asset_id from Asset Upload"})
        for i in range(1, 4):
            opt[f"audio{i}"] = ("STRING", {"default": "", "tooltip": "asset_id from Asset Upload"})
        return {"required": {}, "optional": opt}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("asset_bundle",)
    FUNCTION = "bundle"
    CATEGORY = "Comfly/Doubao"

    def bundle(
        self,
        apikey="",
        first_frame="",
        last_frame="",
        ref_image1="",
        ref_image2="",
        ref_image3="",
        ref_image4="",
        ref_image5="",
        ref_image6="",
        ref_image7="",
        ref_image8="",
        ref_image9="",
        video1="",
        video2="",
        video3="",
        audio1="",
        audio2="",
        audio3="",
    ):
        def sid(x):
            return (x or "").strip() if x is not None else ""

        ff = sid(first_frame)
        lf = sid(last_frame)
        ref_images = []
        for i in range(1, 10):
            t = sid(locals().get(f"ref_image{i}"))
            if t:
                ref_images.append(t)
        videos = []
        for i in range(1, 4):
            t = sid(locals().get(f"video{i}"))
            if t:
                videos.append(t)
        audios = []
        for i in range(1, 4):
            t = sid(locals().get(f"audio{i}"))
            if t:
                audios.append(t)

        payload = {
            "first_frame": ff,
            "last_frame": lf,
            "ref_images": ref_images,
            "videos": videos,
            "audios": audios,
        }
        return (json.dumps(payload, ensure_ascii=False),)


class Comfly_Doubao_Seedance2_0_Asset:
    """
    Create Seedance asset from Comfy IMAGE / VIDEO / AUDIO.
    Image → /v1/files (HTTPS URL); video/audio → same upload path as Seedance 2.0. assetType inferred; no url widget.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "apikey": ("STRING", {"default": ""}),
                "name": ("STRING", {"default": "", "multiline": False}),
                "image": ("IMAGE",),
                "video": (IO.VIDEO, {"tooltip": "Reference video; upload same as Seedance 2.0."}),
                "audio": (IO.AUDIO, {"tooltip": "Reference audio; upload same as Seedance 2.0."}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("asset_id", "status", "response")
    FUNCTION = "upload_asset"
    CATEGORY = "Comfly/Doubao"

    def __init__(self):
        self.api_key = get_config().get('api_key', '')
        self.timeout = 120
        self.poll_interval = 3
        self.max_wait_time = 300

    def get_headers(self):
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

    def query_asset_status(self, asset_id):
        payload = {"assetId": asset_id}
        response = requests.post(
            f"{baseurl}/seedance/v3/assets/query",
            headers=self.get_headers(),
            json=payload,
            timeout=self.timeout
        )
        response.raise_for_status()
        return response.json()

    def upload_asset(self, apikey="", name="", image=None, video=None, audio=None):
        if apikey and str(apikey).strip():
            self.api_key = str(apikey).strip()
            config = get_config()
            config['api_key'] = self.api_key

        if not self.api_key:
            error_message = "API key not found."
            print(error_message)
            return ("", "", json.dumps({"error": error_message}))

        seed = Comfly_Doubao_Seedance2_0()
        seed.api_key = self.api_key
        seed.timeout = self.timeout

        asset_type = None
        media_url = None
        wired = []
        if image is not None:
            wired.append("image")
        if video is not None:
            wired.append("video")
        if audio is not None:
            wired.append("audio")
        if len(wired) > 1:
            print(f"[Comfly Seedance Asset] Multiple media inputs connected ({', '.join(wired)}); using priority: image > video > audio.")

        if image is not None:
            asset_type = "Image"
            media_url = seed.upload_image_get_url(image)
        elif video is not None:
            asset_type = "Video"
            media_url = seed.upload_video_get_url(video)
        elif audio is not None:
            asset_type = "Audio"
            media_url = seed.upload_audio_get_url(audio)
        else:
            err = "Connect image, video, or audio."
            print(err)
            return ("", "", json.dumps({"error": err}))

        if not media_url:
            err = "Could not obtain HTTPS URL for asset (upload failed or empty media)."
            print(err)
            return ("", "", json.dumps({"error": err}))

        display_name = (name or "").strip() or f"asset_{uuid.uuid4().hex[:12]}"

        try:
            pbar = comfy.utils.ProgressBar(100)
            pbar.update_absolute(5)

            payload = {
                "url": media_url,
                "assetType": asset_type,
                "name": display_name
            }

            response = requests.post(
                f"{baseurl}/seedance/v3/assets/create",
                headers=self.get_headers(),
                json=payload,
                timeout=self.timeout
            )

            if response.status_code != 200:
                error_message = f"API Error: {response.status_code} - {response.text}"
                print(error_message)
                return ("", "", json.dumps({"error": error_message}))

            result = response.json()

            if result.get("code") != 0:
                error_message = result.get("msg", "Unknown error")
                print(f"Asset upload failed: {error_message}")
                return ("", "", json.dumps(result, indent=2))

            data = result.get("data", {})
            asset_id = data.get("assetId", "")
            status = data.get("status", "")

            print(f"Asset created: id={asset_id}, status={status}")
            pbar.update_absolute(20)

            if status == "Active":
                pbar.update_absolute(100)
                return (asset_id, status, json.dumps(result, indent=2))

            start_time = time.time()
            last_query_result = result

            while True:
                elapsed = time.time() - start_time
                if elapsed > self.max_wait_time:
                    error_message = f"Asset processing timeout after {elapsed:.1f}s. Last status: {status}"
                    print(error_message)
                    return ("", status, json.dumps({"error": error_message, "asset_id": asset_id, "last_status": status}))

                time.sleep(self.poll_interval)
                progress = min(90, 20 + int((elapsed / self.max_wait_time) * 70))
                pbar.update_absolute(progress)

                try:
                    query_result = self.query_asset_status(asset_id)
                    last_query_result = query_result

                    if query_result.get("code") != 0:
                        print(f"Query error: {query_result.get('msg', 'Unknown')}, retrying...")
                        continue

                    query_data = query_result.get("data", {})
                    status = query_data.get("status", "")
                    print(f"Asset {asset_id}: status={status} ({elapsed:.1f}s)")

                    if status == "Active":
                        pbar.update_absolute(100)
                        print(f"Asset ready: {asset_id}")
                        return (asset_id, status, json.dumps(query_result, indent=2))

                    if status in ("Failed", "Error", "Deleted"):
                        error_message = f"Asset processing failed with status: {status}"
                        print(error_message)
                        return ("", status, json.dumps(query_result, indent=2))

                except requests.exceptions.Timeout:
                    print("Query timeout, retrying...")
                    continue
                except Exception as e:
                    print(f"Query error: {str(e)}, retrying...")
                    continue

        except Exception as e:
            error_message = f"Asset upload error: {str(e)}"
            print(error_message)
            return ("", "", json.dumps({"error": error_message}))