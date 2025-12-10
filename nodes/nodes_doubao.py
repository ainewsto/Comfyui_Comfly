import os
import io
import math
import random
import torch
import torchaudio
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
