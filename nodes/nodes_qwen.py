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

from ..utils import pil2tensor, tensor2pil
from ..comfly_config import get_config, save_config, baseurl
from comfy.comfy_types import IO


class Comfly_qwen_image:
    
    """
    A node that generates images using Qwen AI service
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "size": (["512x512", "1024x1024", "768x1024", "576x1024", "1024x768", "1024x576", "Custom"], {"default": "1024x768"}),
                "Custom_size": ("STRING", {"default": "Enter custom size (e.g. 1280x720)", "multiline": False}),
                "model": (["qwen-image"], {"default": "qwen-image"}),
                "num_images": ([1, 2, 3, 4], {"default": 1}),
            },
            "optional": {
                "api_key": ("STRING", {"default": ""}),
                "num_inference_steps": ("INT", {"default": 30, "min": 2, "max": 50, "step": 1}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "guidance_scale": ("FLOAT", {"default": 2.5, "min": 0, "max": 20, "step": 0.5}),
                "enable_safety_checker": ("BOOLEAN", {"default": True}),
                "negative_prompt": ("STRING", {"default": "", "multiline": True}),
                "output_format": (["jpeg", "png"], {"default": "png"}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("image", "response", "image_url")
    FUNCTION = "generate_image"
    CATEGORY = "Comfly/Qwen"
       
    def __init__(self):
        self.api_key = get_config().get('api_key', '')
        self.timeout = 300

    def get_headers(self):
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
 
    def generate_image(self, prompt, size, Custom_size, model, num_images=1,
                       api_key="", num_inference_steps=30, seed=0, guidance_scale=2.5, 
                       enable_safety_checker=True, negative_prompt="", output_format="png"):
        if api_key.strip():
            self.api_key = api_key
            config = get_config()
            config['api_key'] = api_key
            
        try:
            if not self.api_key:
                error_message = "API key not found in Comflyapi.json"
                print(error_message)
                blank_image = Image.new('RGB', (1024, 1024), color='white')
                blank_tensor = pil2tensor(blank_image)
                return (blank_tensor, error_message, "")
                
            pbar = comfy.utils.ProgressBar(100)
            pbar.update_absolute(10)
            
            actual_size = Custom_size if size == "Custom" else size

            if size == "Custom" and (Custom_size == "Enter custom size (e.g. 1280x720)" or "x" not in Custom_size):
                error_message = "Please enter a valid custom size in the format 'widthxheight' (e.g. 1280x720)"
                print(error_message)
                blank_image = Image.new('RGB', (1024, 1024), color='white')
                blank_tensor = pil2tensor(blank_image)
                return (blank_tensor, error_message, "")

            payload = {
                "prompt": prompt,
                "size": actual_size,  
                "model": model,
                "n": num_images,  
            }

            if num_inference_steps != 30:
                payload["num_inference_steps"] = num_inference_steps
                
            if seed != 0:
                payload["seed"] = seed
                
            if guidance_scale != 2.5:
                payload["guidance_scale"] = guidance_scale
                
            if not enable_safety_checker:
                payload["enable_safety_checker"] = enable_safety_checker
                
            if negative_prompt.strip():
                payload["negative_prompt"] = negative_prompt
                
            if output_format != "png":
                payload["output_format"] = output_format
            
            pbar.update_absolute(30)
            
            response = requests.post(
                f"{baseurl}/v1/images/generations", 
                headers=self.get_headers(),
                json=payload,
                timeout=self.timeout
            )
            
            pbar.update_absolute(50)
            
            if response.status_code != 200:
                error_message = f"API Error: {response.status_code} - {response.text}"
                print(error_message)
                blank_image = Image.new('RGB', (1024, 1024), color='white')
                blank_tensor = pil2tensor(blank_image)
                return (blank_tensor, error_message, "")
                
            result = response.json()

            timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            response_info = f"**Qwen Image Generation ({timestamp})**\n\n"
            response_info += f"Prompt: {prompt}\n"
            response_info += f"Model: {model}\n"
            response_info += f"Size: {actual_size}\n"
            response_info += f"Number of Images: {num_images}\n"
            response_info += f"Seed: {seed}\n\n"
            
            generated_images = []
            image_urls = []
            
            if "data" in result and result["data"]:
                for i, item in enumerate(result["data"]):
                    pbar.update_absolute(50 + (i+1) * 40 // len(result["data"]))
                    
                    if "b64_json" in item:
                        image_data = base64.b64decode(item["b64_json"])
                        generated_image = Image.open(BytesIO(image_data))
                        generated_tensor = pil2tensor(generated_image)
                        generated_images.append(generated_tensor)
                    elif "url" in item:
                        image_url = item["url"]
                        image_urls.append(image_url)
                        try:
                            img_response = requests.get(image_url, timeout=self.timeout)
                            img_response.raise_for_status()
                            generated_image = Image.open(BytesIO(img_response.content))
                            generated_tensor = pil2tensor(generated_image)
                            generated_images.append(generated_tensor)
                        except Exception as e:
                            print(f"Error downloading image from URL: {str(e)}")
            else:
                error_message = "No generated images in response"
                print(error_message)
                response_info += f"Error: {error_message}\n"
                blank_image = Image.new('RGB', (1024, 1024), color='white')
                blank_tensor = pil2tensor(blank_image)
                return (blank_tensor, response_info, "")
                
            if generated_images:
                combined_tensor = torch.cat(generated_images, dim=0)
                
                pbar.update_absolute(100)
                return (combined_tensor, response_info, image_urls[0] if image_urls else "")
            else:
                error_message = "No images were successfully processed"
                print(error_message)
                response_info += f"Error: {error_message}\n"
                blank_image = Image.new('RGB', (1024, 1024), color='white')
                blank_tensor = pil2tensor(blank_image)
                return (blank_tensor, response_info, "")
                
        except Exception as e:
            error_message = f"Error in image generation: {str(e)}"
            print(error_message)
            blank_image = Image.new('RGB', (1024, 1024), color='white')
            blank_tensor = pil2tensor(blank_image)
            return (blank_tensor, error_message, "")


class Comfly_qwen_image_edit:
    
    """
    A node that edits images using Qwen AI service
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "image": ("IMAGE",),
                "size": (["512x512", "1024x1024", "768x1024", "576x1024", "1024x768", "1024x576", "Custom"], {"default": "1024x768"}),
                "Custom_size": ("STRING", {"default": "Enter custom size (e.g. 1280x720)", "multiline": False}),
                "model": (["qwen-image-edit"], {"default": "qwen-image-edit"}),
            },
            "optional": {
                "apikey": ("STRING", {"default": ""}),
                "num_inference_steps": ("INT", {"default": 30, "min": 2, "max": 50, "step": 1}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "guidance_scale": ("FLOAT", {"default": 4.0, "min": 0, "max": 20, "step": 0.5}),
                "enable_safety_checker": ("BOOLEAN", {"default": True}),
                "negative_prompt": ("STRING", {"default": "", "multiline": True}),
                "output_format": (["jpeg", "png"], {"default": "png"}),
                "num_images": ("INT", {"default": 1, "min": 1, "max": 4}),
                "acceleration": (["none", "regular", "high"], {"default": "none"}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("image", "response", "image_url")
    FUNCTION = "edit_image"
    CATEGORY = "Comfly/Qwen"
       
    def __init__(self):
        self.api_key = get_config().get('api_key', '')
        self.timeout = 300

    def get_headers(self):
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
 
    def edit_image(self, prompt, image, size, Custom_size, model,
                  apikey="", num_inference_steps=30, seed=0, guidance_scale=4.0, 
                  enable_safety_checker=True, negative_prompt="", output_format="png",
                  num_images=1, acceleration="none"):
        if apikey.strip():
            self.api_key = apikey
            config = get_config()
            config['api_key'] = apikey
            
        try:
            if not self.api_key:
                error_message = "API key not found in Comflyapi.json"
                print(error_message)
                return (image, error_message, "")
                
            pbar = comfy.utils.ProgressBar(100)
            pbar.update_absolute(10)
            
            actual_size = Custom_size if size == "Custom" else size

            if size == "Custom" and (Custom_size == "Enter custom size (e.g. 1280x720)" or "x" not in Custom_size):
                error_message = "Please enter a valid custom size in the format 'widthxheight' (e.g. 1280x720)"
                print(error_message)
                return (image, error_message, "")

            pil_image = tensor2pil(image)[0]

            buffered = BytesIO()
            pil_image.save(buffered, format="PNG")
            buffered.seek(0) 

            files = {
                'image': ('image.png', buffered, 'image/png')
            }
            
            data = {
                "prompt": prompt,
                "size": actual_size,  
                "model": model,
                "n": str(num_images),
            }

            if num_inference_steps != 30:
                data["num_inference_steps"] = str(num_inference_steps)
                
            if seed != 0:
                data["seed"] = str(seed)
                
            if guidance_scale != 4.0:
                data["guidance_scale"] = str(guidance_scale)
                
            if not enable_safety_checker:
                data["enable_safety_checker"] = str(enable_safety_checker).lower()
                
            if negative_prompt.strip():
                data["negative_prompt"] = negative_prompt
                
            if output_format != "png":
                data["output_format"] = output_format
                
            if acceleration != "none":
                data["acceleration"] = acceleration
            
            pbar.update_absolute(30)

            headers = {"Authorization": f"Bearer {self.api_key}"}
            
            response = requests.post(
                f"{baseurl}/v1/images/edits", 
                headers=headers,
                files=files,
                data=data,
                timeout=self.timeout
            )
            
            pbar.update_absolute(50)
            
            if response.status_code != 200:
                error_message = f"API Error: {response.status_code} - {response.text}"
                print(error_message)
                return (image, error_message, "")
                
            result = response.json()

            timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            response_info = f"**Qwen Image Edit ({timestamp})**\n\n"
            response_info += f"Prompt: {prompt}\n"
            response_info += f"Model: {model}\n"
            response_info += f"Size: {actual_size}\n"
            response_info += f"Number of Images: {num_images}\n"
            response_info += f"Acceleration: {acceleration}\n"
            response_info += f"Seed: {seed}\n\n"
            
            edited_images = []
            image_urls = []
            
            if "data" in result and result["data"]:
                for i, item in enumerate(result["data"]):
                    pbar.update_absolute(50 + (i+1) * 40 // len(result["data"]))
                    
                    if "b64_json" in item:
                        image_data = base64.b64decode(item["b64_json"])
                        edited_image = Image.open(BytesIO(image_data))
                        edited_tensor = pil2tensor(edited_image)
                        edited_images.append(edited_tensor)
                    elif "url" in item:
                        image_url = item["url"]
                        image_urls.append(image_url)
                        try:
                            img_response = requests.get(image_url, timeout=self.timeout)
                            img_response.raise_for_status()
                            edited_image = Image.open(BytesIO(img_response.content))
                            edited_tensor = pil2tensor(edited_image)
                            edited_images.append(edited_tensor)
                        except Exception as e:
                            print(f"Error downloading image from URL: {str(e)}")
            else:
                error_message = "No edited images in response"
                print(error_message)
                response_info += f"Error: {error_message}\n"
                return (image, response_info, "")
                
            if edited_images:
                combined_tensor = torch.cat(edited_images, dim=0)
                
                pbar.update_absolute(100)
                return (combined_tensor, response_info, image_urls[0] if image_urls else "")
            else:
                error_message = "No images were successfully processed"
                print(error_message)
                response_info += f"Error: {error_message}\n"
                return (image, response_info, "")
                
        except Exception as e:
            error_message = f"Error in image editing: {str(e)}"
            print(error_message)
            return (image, error_message, "")


class Comfly_Z_image_turbo:
    """
    Comfly Z Image Turbo node
    Generates images using Z Image Turbo API
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "model": (["z-image-turbo"], {"default": "z-image-turbo"}),
                "size": (["512x512", "768x768", "1024x1024", "1280x720", "720x1280", "1536x1024", "1024x1536", "Custom"], {"default": "1024x1024"}),
                "output_format": (["jpeg", "png", "webp"], {"default": "jpeg"}),
            },
            "optional": {
                "custom_size": ("STRING", {"default": "1024x1024", "placeholder": "Enter custom size (e.g. 1280x720)"}),
                "apikey": ("STRING", {"default": ""}),
                "guidance_scale": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 20.0, "step": 0.04}),
                "num_inference_steps": ("INT", {"default": 8, "min": 1, "max": 50, "step": 1}),
                "output_quality": ("INT", {"default": 80, "min": 0, "max": 100, "step": 1}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("image", "image_url", "response")
    FUNCTION = "generate_image"
    CATEGORY = "Comfly/Qwen"

    def __init__(self):
        self.api_key = get_config().get('api_key', '')
        self.timeout = 300

    def get_headers(self):
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
    
    def generate_image(self, prompt, model="z-image-turbo", size="1024x1024", output_format="jpg",
                      custom_size="1024x1024", apikey="", guidance_scale=0.0, num_inference_steps=8,
                      output_quality=80, seed=0):
        if apikey.strip():
            self.api_key = apikey
            config = get_config()
            config['api_key'] = apikey
            
        if not self.api_key:
            error_message = "API key not found in Comflyapi.json"
            print(error_message)
            blank_image = Image.new('RGB', (1024, 1024), color='white')
            blank_tensor = pil2tensor(blank_image)
            return (blank_tensor, "", error_message)
            
        try:
            pbar = comfy.utils.ProgressBar(100)
            pbar.update_absolute(10)

            actual_size = custom_size if size == "Custom" else size

            if size == "Custom":
                if "x" not in custom_size:
                    error_message = "Custom size must be in format 'widthxheight' (e.g. 1280x720)"
                    print(error_message)
                    blank_image = Image.new('RGB', (1024, 1024), color='white')
                    blank_tensor = pil2tensor(blank_image)
                    return (blank_tensor, "", error_message)
                
                try:
                    width, height = map(int, custom_size.split('x'))
                    if width < 64 or width > 2048 or height < 64 or height > 2048:
                        error_message = "Width and height must be between 64 and 2048"
                        print(error_message)
                        blank_image = Image.new('RGB', (1024, 1024), color='white')
                        blank_tensor = pil2tensor(blank_image)
                        return (blank_tensor, "", error_message)
                except ValueError:
                    error_message = "Invalid custom size format. Use 'widthxheight' (e.g. 1280x720)"
                    print(error_message)
                    blank_image = Image.new('RGB', (1024, 1024), color='white')
                    blank_tensor = pil2tensor(blank_image)
                    return (blank_tensor, "", error_message)

            try:
                width, height = map(int, actual_size.split('x'))
            except:
                width, height = 1024, 1024

            payload = {
                "prompt": prompt,
                "model": model,
                "size": actual_size,
                "output_format": output_format,
                "guidance_scale": guidance_scale,
                "num_inference_steps": num_inference_steps,
                "output_quality": output_quality
            }
            
            if seed > 0:
                payload["seed"] = seed
            
            pbar.update_absolute(30)

            response = requests.post(
                f"{baseurl}/v1/images/generations",
                headers=self.get_headers(),
                json=payload,
                timeout=self.timeout
            )
            
            pbar.update_absolute(50)
            
            if response.status_code != 200:
                error_message = f"API Error: {response.status_code} - {response.text}"
                print(error_message)
                blank_image = Image.new('RGB', (width, height), color='white')
                blank_tensor = pil2tensor(blank_image)
                return (blank_tensor, "", error_message)
                
            result = response.json()

            timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            response_info = f"**Z Image Turbo Generation ({timestamp})**\n\n"
            response_info += f"Prompt: {prompt}\n"
            response_info += f"Model: {model}\n"
            response_info += f"Size: {actual_size}\n"
            response_info += f"Output Format: {output_format}\n"
            response_info += f"Guidance Scale: {guidance_scale}\n"
            response_info += f"Steps: {num_inference_steps}\n"
            response_info += f"Output Quality: {output_quality}\n"
            response_info += f"Seed: {seed if seed > 0 else 'auto'}\n\n"
            
            image_url = ""
            generated_image = None
            
            if "data" in result and result["data"]:
                item = result["data"][0]
                pbar.update_absolute(70)
                
                if "b64_json" in item:
                    image_data = base64.b64decode(item["b64_json"])
                    generated_image = Image.open(BytesIO(image_data))
                elif "url" in item:
                    image_url = item["url"]
                    response_info += f"Image URL: {image_url}\n"
                    try:
                        img_response = requests.get(image_url, timeout=self.timeout)
                        img_response.raise_for_status()
                        generated_image = Image.open(BytesIO(img_response.content))
                    except Exception as e:
                        print(f"Error downloading image from URL: {str(e)}")
                        response_info += f"Error: {str(e)}\n"
            else:
                error_message = "No image data in response"
                print(error_message)
                response_info += f"Error: {error_message}\n"
                blank_image = Image.new('RGB', (width, height), color='white')
                blank_tensor = pil2tensor(blank_image)
                return (blank_tensor, "", response_info)
            
            pbar.update_absolute(90)
            
            if generated_image:
                generated_tensor = pil2tensor(generated_image)
                pbar.update_absolute(100)
                return (generated_tensor, image_url, response_info)
            else:
                error_message = "Failed to process image"
                print(error_message)
                response_info += f"Error: {error_message}\n"
                blank_image = Image.new('RGB', (width, height), color='white')
                blank_tensor = pil2tensor(blank_image)
                return (blank_tensor, "", response_info)
                
        except Exception as e:
            error_message = f"Error in image generation: {str(e)}"
            print(error_message)
            import traceback
            traceback.print_exc()
            blank_image = Image.new('RGB', (1024, 1024), color='white')
            blank_tensor = pil2tensor(blank_image)
            return (blank_tensor, "", error_message)