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
         
        
class Comfly_Flux_Kontext_Edit:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
            },
            "optional": {
                "image": ("IMAGE",),
                "model": (["flux-kontext-dev", "flux-kontext-pro", "flux-kontext-max"], {"default": "flux-kontext-pro"}),
                "apikey": ("STRING", {"default": ""}),
                "aspect_ratio": (["21:9", "16:9", "4:3", "3:2", "1:1", "2:3", "3:4", "9:16", "9:21"], 
                         {"default": "1:1"}),
                "num_of_images": ("INT", {"default": 1, "min": 1, "max": 4}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 2147483647})
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "image_url")
    FUNCTION = "generate_image"
    CATEGORY = "Comfly/Flux"

    def __init__(self):
        self.api_key = get_config().get('api_key', '')
        self.timeout = 300

    def get_headers(self):
        return {
            "Authorization": f"Bearer {self.api_key}"
        }
    
    def generate_image(self, prompt, image=None, model="flux-kontext-pro", 
                  apikey="", aspect_ratio="1:1", num_of_images=1,
                  seed=-1):
        if apikey.strip():
            self.api_key = apikey
            config = get_config()
            config['api_key'] = apikey
            
        if not self.api_key:
            error_message = "API key not found in Comflyapi.json"
            print(error_message)

            if image is None:
                blank_image = Image.new('RGB', (1024, 1024), color='white')
                blank_tensor = pil2tensor(blank_image)
                return (blank_tensor, "")
            return (image, "")
        
        try:
            pbar = comfy.utils.ProgressBar(100)
            pbar.update_absolute(10)

            if image is not None:
                pil_image = tensor2pil(image)[0]

                buffered = BytesIO()
                pil_image.save(buffered, format="PNG")
                buffered.seek(0)

                files = {
                    'image': ('image.png', buffered, 'image/png')
                }
                
                data = {
                    'prompt': prompt,
                    'model': model
                }

                if aspect_ratio != "Default":
                    data["aspect_ratio"] = aspect_ratio

                if seed != -1:
                    data["seed"] = str(seed)
                    
                if num_of_images > 1:
                    data["n"] = str(num_of_images)

                pbar.update_absolute(30)
                response = requests.post(
                    f"{baseurl}/v1/images/edits",
                    headers=self.get_headers(),
                    data=data,
                    files=files,
                    timeout=self.timeout
                )
            else:
                payload = {
                    "prompt": prompt,
                    "model": model,
                    "n": num_of_images
                }
                
                if aspect_ratio != "Default":
                    payload["aspect_ratio"] = aspect_ratio
                
                if seed != -1:
                    payload["seed"] = seed
                    
                headers = self.get_headers()
                headers["Content-Type"] = "application/json"
                
                response = requests.post(
                    f"{baseurl}/v1/images/generations",
                    headers=headers,
                    json=payload,
                    timeout=self.timeout
                )
            
            pbar.update_absolute(50)
            
            if response.status_code != 200:
                error_message = f"API Error: {response.status_code} - {response.text}"
                print(error_message)
                if image is None:
                    blank_image = Image.new('RGB', (1024, 1024), color='white')
                    blank_tensor = pil2tensor(blank_image)
                    return (blank_tensor, "")
                return (image, "")
                
            result = response.json()

            if not result.get("data") or not result["data"]:
                error_message = "No image data in response"
                print(error_message)
                if image is None:
                    blank_image = Image.new('RGB', (1024, 1024), color='white')
                    blank_tensor = pil2tensor(blank_image)
                    return (blank_tensor, "")
                return (image, "")

            generated_tensors = []
            image_urls = []
            
            for i, item in enumerate(result["data"]):
                pbar.update_absolute(50 + (i+1) * 40 // len(result["data"]))
                
                if "url" in item:
                    image_url = item["url"]
                    image_urls.append(image_url)
                    
                    try:
                        img_response = requests.get(image_url, timeout=self.timeout)
                        img_response.raise_for_status()
                        generated_image = Image.open(BytesIO(img_response.content))
                        generated_tensor = pil2tensor(generated_image)
                        generated_tensors.append(generated_tensor)
                    except Exception as e:
                        print(f"Error downloading image from URL: {str(e)}")
                        
                elif "b64_json" in item:
                    image_data = base64.b64decode(item["b64_json"])
                    generated_image = Image.open(BytesIO(image_data))
                    generated_tensor = pil2tensor(generated_image)
                    generated_tensors.append(generated_tensor)
            
            pbar.update_absolute(100)
            
            if generated_tensors:
                combined_tensor = torch.cat(generated_tensors, dim=0)
                return (combined_tensor, "\n".join(image_urls))
            else:
                error_message = "Failed to process any images"
                print(error_message)
                if image is None:
                    blank_image = Image.new('RGB', (1024, 1024), color='white')
                    blank_tensor = pil2tensor(blank_image)
                    return (blank_tensor, "")
                return (image, "")
            
        except Exception as e:
            error_message = f"Error in image generation: {str(e)}"
            print(error_message)
            if image is None:
                blank_image = Image.new('RGB', (1024, 1024), color='white')
                blank_tensor = pil2tensor(blank_image)
                return (blank_tensor, "")
            return (image, "")



class Comfly_Flux_Kontext_bfl:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "model": (["flux-kontext-pro", "flux-kontext-max"], {"default": "flux-kontext-pro"}),
            },
            "optional": {
                "api_key": ("STRING", {"default": ""}),
                "input_image": ("IMAGE",),
                "seed": ("INT", {"default": -1, "min": -1, "max": 2147483647}),
                "aspect_ratio": (["21:9", "16:9", "4:3", "3:2", "1:1", "2:3", "3:4", "9:16", "9:21"], 
                         {"default": "1:1"}),
                "output_format": (["png", "jpeg"], {"default": "png"}),
                "prompt_upsampling": ("BOOLEAN", {"default": False}),
                "safety_tolerance": ("INT", {"default": 2, "min": 0, "max": 6})
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("image", "image_url", "response")
    FUNCTION = "generate_image"
    CATEGORY = "Comfly/Flux"
    OUTPUT_NODE = True

    def __init__(self):
        self.api_key = get_config().get('api_key', '')
        self.timeout = 300

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
        return base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    def generate_image(self, prompt, model="flux-kontext-pro", input_image=None, 
                      seed=-1, aspect_ratio="1:1", output_format="png", 
                      prompt_upsampling=False, safety_tolerance=2, api_key=""):
        if api_key.strip():
            self.api_key = api_key
            config = get_config()
            config['api_key'] = api_key

        if input_image is not None:
            default_tensor = input_image  
        else:
            blank_image = Image.new('RGB', (512, 512), color='white')
            default_tensor = pil2tensor(blank_image)
            
        if not self.api_key:
            error_response = {"status": "failed", "message": "API key not found"}
            return (default_tensor, "", json.dumps(error_response))
            
        pbar = comfy.utils.ProgressBar(100)
        pbar.update_absolute(10)

        api_endpoint = f"{baseurl}/bfl/v1/{model}"
        
        try:
            payload = {
                "prompt": prompt,
                "output_format": output_format,
                "prompt_upsampling": prompt_upsampling,
                "safety_tolerance": safety_tolerance
            }

            if input_image is not None:
                input_image_base64 = self.image_to_base64(input_image)
                if input_image_base64:
                    payload["input_image"] = input_image_base64
            
            if seed != -1:
                payload["seed"] = seed
                
            if aspect_ratio:
                payload["aspect_ratio"] = aspect_ratio

            response = requests.post(
                api_endpoint,
                headers=self.get_headers(),
                json=payload,
                timeout=self.timeout
            )
            
            pbar.update_absolute(30)
            
            if response.status_code != 200:
                error_message = f"API Error: {response.status_code} - {response.text}"
                print(error_message)
                return (default_tensor, "", json.dumps({"status": "failed", "message": error_message}))
                
            result = response.json()
            
            if "id" not in result or "polling_url" not in result:
                error_message = "Invalid response format from API"
                print(error_message)
                return (default_tensor, "", json.dumps({"status": "failed", "message": error_message}))
                
            task_id = result["id"]
            polling_url = result["polling_url"]

            pbar.update_absolute(40)

            max_attempts = 60  
            attempts = 0
            image_url = ""
            
            while attempts < max_attempts:
                time.sleep(10)
                attempts += 1
                
                try:
                    result_response = requests.get(
                        f"{baseurl}/bfl/v1/get_result?id={task_id}",
                        headers=self.get_headers(),
                        timeout=self.timeout
                    )
                    
                    if result_response.status_code != 200:
                        continue
                        
                    result_data = result_response.json()
                    status = result_data.get("status")
                    
                    if status == "Ready":
                        if "result" in result_data and "sample" in result_data["result"]:
                            image_url = result_data["result"]["sample"]
                            break

                    progress = min(80, 40 + (attempts * 40 // max_attempts))
                    pbar.update_absolute(progress)
                        
                except Exception as e:
                    print(f"Error checking generation status: {str(e)}")
            
            if not image_url:
                error_message = "Failed to retrieve generated image URL after multiple attempts"
                print(error_message)
                return (default_tensor, "", json.dumps({"status": "failed", "message": error_message}))

            pbar.update_absolute(90)
            
            try:
                img_response = requests.get(image_url, timeout=self.timeout)
                img_response.raise_for_status()
                
                generated_image = Image.open(BytesIO(img_response.content))
                generated_tensor = pil2tensor(generated_image)
                
                pbar.update_absolute(100)
                
                result_info = {
                    "status": "success",
                    "task_id": task_id,
                    "prompt": prompt,
                    "model": model,
                    "seed": seed if seed > 0 else "random",
                    "aspect_ratio": aspect_ratio
                }
                
                return (generated_tensor, image_url, json.dumps(result_info))
                
            except Exception as e:
                error_message = f"Error downloading generated image: {str(e)}"
                print(error_message)
                return (default_tensor, image_url, json.dumps({"status": "partial_success", "message": error_message, "image_url": image_url}))
            
        except Exception as e:
            error_message = f"Error in image generation: {str(e)}"
            print(error_message)
            import traceback
            traceback.print_exc()
            return (default_tensor, "", json.dumps({"status": "failed", "message": error_message}))


class Comfly_Flux_2_Pro:
    """
    Comfly Flux 2 Pro node
    Generates images using the Flux 2 Pro API with support for multiple input images.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
            },
            "optional": {
                "api_key": ("STRING", {"default": ""}),
                "input_image": ("IMAGE",),
                "input_image_2": ("IMAGE",),
                "input_image_3": ("IMAGE",),
                "input_image_4": ("IMAGE",),
                "input_image_5": ("IMAGE",),
                "input_image_6": ("IMAGE",),
                "input_image_7": ("IMAGE",),
                "input_image_8": ("IMAGE",),
                "seed": ("INT", {"default": -1, "min": -1, "max": 2147483647}),
                "width": ("INT", {"default": 1024, "min": 64, "max": 2048, "step": 8}),
                "height": ("INT", {"default": 1024, "min": 64, "max": 2048, "step": 8}),
                "safety_tolerance": ("INT", {"default": 2, "min": 0, "max": 5}),
                "output_format": (["jpeg", "png"], {"default": "png"}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("image", "image_url", "response")
    FUNCTION = "generate_image"
    CATEGORY = "Comfly/Flux"

    def __init__(self):
        self.api_key = get_config().get('api_key', '')
        self.timeout = 600

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
        return base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    def generate_image(self, prompt, api_key="", input_image=None, input_image_2=None,
                      input_image_3=None, input_image_4=None, input_image_5=None,
                      input_image_6=None, input_image_7=None, input_image_8=None,
                      seed=-1, width=1024, height=1024, safety_tolerance=2, 
                      output_format="png"):
        
        if api_key.strip():
            self.api_key = api_key
            config = get_config()
            config['api_key'] = api_key

        blank_image = Image.new('RGB', (width, height), color='white')
        default_tensor = pil2tensor(blank_image)
            
        if not self.api_key:
            error_response = {"status": "failed", "message": "API key not found"}
            return (default_tensor, "", json.dumps(error_response))
            
        pbar = comfy.utils.ProgressBar(100)
        pbar.update_absolute(10)
        
        try:
            payload = {
                "prompt": prompt,
                "safety_tolerance": safety_tolerance,
                "output_format": output_format
            }

            if width > 0:
                payload["width"] = width
            if height > 0:
                payload["height"] = height

            if seed != -1:
                payload["seed"] = seed

            image_inputs = [
                ("input_image", input_image),
                ("input_image_2", input_image_2),
                ("input_image_3", input_image_3),
                ("input_image_4", input_image_4),
                ("input_image_5", input_image_5),
                ("input_image_6", input_image_6),
                ("input_image_7", input_image_7),
                ("input_image_8", input_image_8),
            ]
            
            for field_name, img in image_inputs:
                if img is not None:
                    img_base64 = self.image_to_base64(img)
                    if img_base64:
                        payload[field_name] = img_base64
            
            pbar.update_absolute(20)

            response = requests.post(
                f"{baseurl}/bfl/v1/flux-2-pro",
                headers=self.get_headers(),
                json=payload,
                timeout=self.timeout
            )
            
            pbar.update_absolute(30)
            
            if response.status_code != 200:
                error_message = f"API Error: {response.status_code} - {response.text}"
                print(error_message)
                return (default_tensor, "", json.dumps({"status": "failed", "message": error_message}))
                
            result = response.json()
            
            if "id" not in result:
                error_message = "No task ID in response"
                print(error_message)
                return (default_tensor, "", json.dumps({"status": "failed", "message": error_message}))
                
            task_id = result["id"]
            polling_url = result.get("polling_url", "")
            
            pbar.update_absolute(40)

            max_attempts = 120
            attempts = 0
            image_url = ""
            
            while attempts < max_attempts:
                time.sleep(5)
                attempts += 1
                
                try:
                    result_response = requests.get(
                        f"{baseurl}/bfl/v1/get_result?id={task_id}",
                        headers=self.get_headers(),
                        timeout=self.timeout
                    )
                    
                    if result_response.status_code != 200:
                        continue
                        
                    result_data = result_response.json()
                    status = result_data.get("status", "")

                    progress = min(90, 40 + (attempts * 50 // max_attempts))
                    pbar.update_absolute(progress)
                    
                    if status == "Ready":
                        if "result" in result_data and "sample" in result_data["result"]:
                            image_url = result_data["result"]["sample"]
                            break
                    elif status in ["Failed", "Error"]:
                        error_message = f"Task failed: {result_data.get('details', 'Unknown error')}"
                        print(error_message)
                        return (default_tensor, "", json.dumps({"status": "failed", "message": error_message}))
                        
                except Exception as e:
                    print(f"Error checking generation status: {str(e)}")
            
            if not image_url:
                error_message = "Failed to retrieve generated image URL after multiple attempts"
                print(error_message)
                return (default_tensor, "", json.dumps({"status": "failed", "message": error_message}))
            
            pbar.update_absolute(90)

            try:
                img_response = requests.get(image_url, timeout=self.timeout)
                img_response.raise_for_status()
                
                generated_image = Image.open(BytesIO(img_response.content))
                generated_tensor = pil2tensor(generated_image)
                
                pbar.update_absolute(100)
                
                result_info = {
                    "status": "success",
                    "task_id": task_id,
                    "prompt": prompt,
                    "seed": result_data.get("result", {}).get("seed", seed),
                    "width": width,
                    "height": height,
                    "image_url": image_url
                }
                
                return (generated_tensor, image_url, json.dumps(result_info))
                
            except Exception as e:
                error_message = f"Error downloading generated image: {str(e)}"
                print(error_message)
                return (default_tensor, image_url, json.dumps({"status": "partial_success", "message": error_message, "image_url": image_url}))
            
        except Exception as e:
            error_message = f"Error in image generation: {str(e)}"
            print(error_message)
            import traceback
            traceback.print_exc()
            return (default_tensor, "", json.dumps({"status": "failed", "message": error_message}))


class Comfly_Flux_2_Flex:
    """
    Comfly Flux 2 Flex node
    Generates images using the Flux 2 Flex API with support for multiple input images,
    prompt upsampling, guidance, and steps parameters.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
            },
            "optional": {
                "api_key": ("STRING", {"default": ""}),
                "input_image": ("IMAGE",),
                "input_image_2": ("IMAGE",),
                "input_image_3": ("IMAGE",),
                "input_image_4": ("IMAGE",),
                "input_image_5": ("IMAGE",),
                "input_image_6": ("IMAGE",),
                "input_image_7": ("IMAGE",),
                "input_image_8": ("IMAGE",),
                "prompt_upsampling": ("BOOLEAN", {"default": True}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 2147483647}),
                "width": ("INT", {"default": 1024, "min": 64, "max": 2048, "step": 8}),
                "height": ("INT", {"default": 1024, "min": 64, "max": 2048, "step": 8}),
                "guidance": ("FLOAT", {"default": 5.0, "min": 1.5, "max": 10.0, "step": 0.1}),
                "steps": ("INT", {"default": 50, "min": 1, "max": 50}),
                "safety_tolerance": ("INT", {"default": 2, "min": 0, "max": 5}),
                "output_format": (["jpeg", "png"], {"default": "png"}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("image", "image_url", "response")
    FUNCTION = "generate_image"
    CATEGORY = "Comfly/Flux"

    def __init__(self):
        self.api_key = get_config().get('api_key', '')
        self.timeout = 600

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
        return base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    def generate_image(self, prompt, api_key="", input_image=None, input_image_2=None,
                      input_image_3=None, input_image_4=None, input_image_5=None,
                      input_image_6=None, input_image_7=None, input_image_8=None,
                      prompt_upsampling=True, seed=-1, width=1024, height=1024,
                      guidance=5.0, steps=50, safety_tolerance=2, output_format="png"):
        
        if api_key.strip():
            self.api_key = api_key
            config = get_config()
            config['api_key'] = api_key

        blank_image = Image.new('RGB', (width, height), color='white')
        default_tensor = pil2tensor(blank_image)
            
        if not self.api_key:
            error_response = {"status": "failed", "message": "API key not found"}
            return (default_tensor, "", json.dumps(error_response))
            
        pbar = comfy.utils.ProgressBar(100)
        pbar.update_absolute(10)
        
        try:
            payload = {
                "prompt": prompt,
                "prompt_upsampling": prompt_upsampling,
                "guidance": guidance,
                "steps": steps,
                "safety_tolerance": safety_tolerance,
                "output_format": output_format
            }

            if width > 0:
                payload["width"] = width
            if height > 0:
                payload["height"] = height

            if seed != -1:
                payload["seed"] = seed

            image_inputs = [
                ("input_image", input_image),
                ("input_image_2", input_image_2),
                ("input_image_3", input_image_3),
                ("input_image_4", input_image_4),
                ("input_image_5", input_image_5),
                ("input_image_6", input_image_6),
                ("input_image_7", input_image_7),
                ("input_image_8", input_image_8),
            ]
            
            for field_name, img in image_inputs:
                if img is not None:
                    img_base64 = self.image_to_base64(img)
                    if img_base64:
                        payload[field_name] = img_base64
            
            pbar.update_absolute(20)

            response = requests.post(
                f"{baseurl}/bfl/v1/flux-2-flex",
                headers=self.get_headers(),
                json=payload,
                timeout=self.timeout
            )
            
            pbar.update_absolute(30)
            
            if response.status_code != 200:
                error_message = f"API Error: {response.status_code} - {response.text}"
                print(error_message)
                return (default_tensor, "", json.dumps({"status": "failed", "message": error_message}))
                
            result = response.json()
            
            if "id" not in result:
                error_message = "No task ID in response"
                print(error_message)
                return (default_tensor, "", json.dumps({"status": "failed", "message": error_message}))
                
            task_id = result["id"]
            polling_url = result.get("polling_url", "")
            
            pbar.update_absolute(40)

            max_attempts = 120
            attempts = 0
            image_url = ""
            
            while attempts < max_attempts:
                time.sleep(5)
                attempts += 1
                
                try:
                    result_response = requests.get(
                        f"{baseurl}/bfl/v1/get_result?id={task_id}",
                        headers=self.get_headers(),
                        timeout=self.timeout
                    )
                    
                    if result_response.status_code != 200:
                        continue
                        
                    result_data = result_response.json()
                    status = result_data.get("status", "")

                    progress = min(90, 40 + (attempts * 50 // max_attempts))
                    pbar.update_absolute(progress)
                    
                    if status == "Ready":
                        if "result" in result_data and "sample" in result_data["result"]:
                            image_url = result_data["result"]["sample"]
                            break
                    elif status in ["Failed", "Error"]:
                        error_message = f"Task failed: {result_data.get('details', 'Unknown error')}"
                        print(error_message)
                        return (default_tensor, "", json.dumps({"status": "failed", "message": error_message}))
                        
                except Exception as e:
                    print(f"Error checking generation status: {str(e)}")
            
            if not image_url:
                error_message = "Failed to retrieve generated image URL after multiple attempts"
                print(error_message)
                return (default_tensor, "", json.dumps({"status": "failed", "message": error_message}))
            
            pbar.update_absolute(90)

            try:
                img_response = requests.get(image_url, timeout=self.timeout)
                img_response.raise_for_status()
                
                generated_image = Image.open(BytesIO(img_response.content))
                generated_tensor = pil2tensor(generated_image)
                
                pbar.update_absolute(100)
                
                result_info = {
                    "status": "success",
                    "task_id": task_id,
                    "prompt": prompt,
                    "seed": result_data.get("result", {}).get("seed", seed),
                    "width": width,
                    "height": height,
                    "guidance": guidance,
                    "steps": steps,
                    "prompt_upsampling": prompt_upsampling,
                    "image_url": image_url
                }
                
                return (generated_tensor, image_url, json.dumps(result_info))
                
            except Exception as e:
                error_message = f"Error downloading generated image: {str(e)}"
                print(error_message)
                return (default_tensor, image_url, json.dumps({"status": "partial_success", "message": error_message, "image_url": image_url}))
            
        except Exception as e:
            error_message = f"Error in image generation: {str(e)}"
            print(error_message)
            import traceback
            traceback.print_exc()
            return (default_tensor, "", json.dumps({"status": "failed", "message": error_message}))