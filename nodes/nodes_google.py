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


class Comfly_Googel_Veo3:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "model": (["veo3", "veo3-fast", "veo3-pro", "veo3-fast-frames", "veo3-pro-frames", "veo3.1", "veo3.1-pro", "veo3.1-components"], {"default": "veo3"}),
                "enhance_prompt": ("BOOLEAN", {"default": False}),
                "aspect_ratio": (["16:9", "9:16"], {"default": "16:9"}),
            },
            "optional": {
                "apikey": ("STRING", {"default": ""}),
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647}),
                "enable_upsample": ("BOOLEAN", {"default": False}),
            }
        }
    
    RETURN_TYPES = (IO.VIDEO, "STRING", "STRING")
    RETURN_NAMES = ("video", "video_url", "response")
    FUNCTION = "generate_video"
    CATEGORY = "Comfly/Google"

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
    
    def generate_video(self, prompt, model="veo3", enhance_prompt=False, aspect_ratio="16:9", apikey="", 
                      image1=None, image2=None, image3=None, seed=0, enable_upsample=False):
        
        if apikey.strip():
            self.api_key = apikey
            
        if not self.api_key:
            error_response = {"code": "error", "message": "API key not found in Comflyapi.json"}
            return ("", "", json.dumps(error_response))
            
        try:
            pbar = comfy.utils.ProgressBar(100)
            pbar.update_absolute(10)

            has_images = any(img is not None for img in [image1, image2, image3])
 
            payload = {
                "prompt": prompt,
                "model": model,
                "enhance_prompt": enhance_prompt
            }
 
            if seed > 0:
                payload["seed"] = seed

            if model in ["veo3", "veo3-fast", "veo3-pro", "veo3.1", "veo3.1-pro", "veo3.1-components"] and aspect_ratio:
                payload["aspect_ratio"] = aspect_ratio

            if model in ["veo3", "veo3-fast", "veo3-pro", "veo3.1", "veo3.1-pro", "veo3.1-components"] and enable_upsample:
                payload["enable_upsample"] = enable_upsample

            if has_images:
                images_base64 = []
                for img in [image1, image2, image3]:
                    if img is not None:
                        batch_size = img.shape[0]
                        for i in range(batch_size):
                            single_image = img[i:i+1]
                            image_base64 = self.image_to_base64(single_image)
                            if image_base64:
                                images_base64.append(f"data:image/png;base64,{image_base64}")
                
                if images_base64:
                    payload["images"] = images_base64

            response = requests.post(
                f"{baseurl}/google/v1/models/veo/videos",
                headers=self.get_headers(),
                json=payload,
                timeout=self.timeout
            )
            
            if response.status_code != 200:
                error_message = f"API Error: {response.status_code} - {response.text}"
                print(error_message)
                return ("", "", json.dumps({"code": "error", "message": error_message}))
                
            result = response.json()
            
            if result.get("code") != "success":
                error_message = f"API returned error: {result.get('message', 'Unknown error')}"
                print(error_message)
                return ("", "", json.dumps({"code": "error", "message": error_message}))
                
            task_id = result.get("data")
            if not task_id:
                error_message = "No task ID returned from API"
                print(error_message)
                return ("", "", json.dumps({"code": "error", "message": error_message}))
            
            pbar.update_absolute(30)

            max_attempts = 150 
            attempts = 0
            video_url = None
            
            while attempts < max_attempts:
                time.sleep(2) 
                attempts += 1
                
                try:
                    status_response = requests.get(
                        f"{baseurl}/google/v1/tasks/{task_id}",
                        headers=self.get_headers(),
                        timeout=self.timeout
                    )
                    
                    if status_response.status_code != 200:
                        continue
                        
                    status_result = status_response.json()
                    
                    if status_result.get("code") != "success":
                        continue
                    
                    data = status_result.get("data", {})
                    status = data.get("status", "")
                    progress = data.get("progress", "0%")
                    
                    try:
                        if progress.endswith('%'):
                            progress_num = int(progress.rstrip('%'))
                            pbar_value = min(90, 30 + progress_num * 60 / 100)
                            pbar.update_absolute(pbar_value)
                    except (ValueError, AttributeError):
                        progress_value = min(80, 30 + (attempts * 50 // max_attempts))
                        pbar.update_absolute(progress_value)
                    
                    if status == "SUCCESS":
                        if "data" in data and "video_url" in data["data"]:
                            video_url = data["data"]["video_url"]
                            break
                    elif status == "FAILURE":
                        fail_reason = data.get("fail_reason", "Unknown error")
                        error_message = f"Video generation failed: {fail_reason}"
                        print(error_message)
                        return ("", "", json.dumps({"code": "error", "message": error_message}))
                        
                except Exception as e:
                    print(f"Error checking generation status: {str(e)}")
            
            if not video_url:
                error_message = "Failed to retrieve video URL after multiple attempts"
                print(error_message)
                return ("", "", json.dumps({"code": "error", "message": error_message}))
            
            if video_url:
                pbar.update_absolute(95)
                
                response_data = {
                    "code": "success",
                    "task_id": task_id,
                    "prompt": prompt,
                    "model": model,
                    "enhance_prompt": enhance_prompt,
                    "aspect_ratio": aspect_ratio if model in ["veo3", "veo3-fast", "veo3-pro"] else "default",
                    "enable_upsample": enable_upsample if model in ["veo3", "veo3-fast", "veo3-pro"] else False,
                    "video_url": video_url,
                    "images_count": len([img for img in [image1, image2, image3] if img is not None])
                }
                
                video_adapter = ComflyVideoAdapter(video_url)
                return (video_adapter, video_url, json.dumps(response_data))
            
        except Exception as e:
            error_message = f"Error generating video: {str(e)}"
            print(error_message)
            return ("", "", json.dumps({"code": "error", "message": error_message}))


class Comfly_nano_banana:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True}),
                "model": (["nano-banana-2","gemini-3-pro-image-preview", "gemini-2.5-flash-image", "nano-banana", "nano-banana-hd", "gemini-2.5-flash-image-preview"], {"default": "nano-banana"}),
            },
            "optional": {
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
                "image4": ("IMAGE",),
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "top_p": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 1.0, "step": 0.05}),
                "apikey": ("STRING", {"default": ""}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647}),
                "max_tokens": ("INT", {"default": 32768, "min": 1, "max": 32768})
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("image", "response", "image_url")
    FUNCTION = "process"
    CATEGORY = "Comfly/Google"

    def __init__(self):
        self.api_key = get_config().get('api_key', '')
        self.timeout = 300

    def get_headers(self):
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

    def image_to_base64(self, image_tensor):
        """Convert tensor to base64 string with data URI prefix"""
        if image_tensor is None:
            return None
            
        pil_image = tensor2pil(image_tensor)[0]
        buffered = BytesIO()
        pil_image.save(buffered, format="PNG")
        base64_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
        return base64_str

    def send_request_streaming(self, payload):
        """Send a streaming request to the API"""
        full_response = ""
        session = requests.Session()
        
        try:
            response = session.post(
                f"{baseurl}/v1/chat/completions",
                headers=self.get_headers(),
                json=payload, 
                stream=True,
                timeout=self.timeout
            )
            response.raise_for_status()

            for line in response.iter_lines():
                if line:
                    line_text = line.decode('utf-8').strip()
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
            
            return full_response
            
        except requests.exceptions.Timeout:
            raise TimeoutError(f"API request timed out after {self.timeout} seconds")
        except Exception as e:
            raise Exception(f"Error in streaming response: {str(e)}")

    def process(self, text, model="gemini-2.5-flash-image-preview", 
                image1=None, image2=None, image3=None, image4=None,
                temperature=1.0, top_p=0.95, apikey="", seed=0, max_tokens=32768):

        if apikey.strip():
            self.api_key = apikey

        default_image = None
        for img in [image1, image2, image3, image4]:
            if img is not None:
                default_image = img
                break

        if default_image is None:
            blank_image = Image.new('RGB', (512, 512), color='white')
            default_image = pil2tensor(blank_image)

        try:
            if not self.api_key:
                return (default_image, "API key not provided. Please set your API key.", "")

            pbar = comfy.utils.ProgressBar(100)
            pbar.update_absolute(10)

            content = [{"type": "text", "text": text}]

            images_added = 0
            for idx, img in enumerate([image1, image2, image3, image4], 1):
                if img is not None:
                    batch_size = img.shape[0]
                    print(f"Processing image{idx} with {batch_size} batch size")
                    
                    for i in range(batch_size):
                        single_image = img[i:i+1]
                        image_base64 = self.image_to_base64(single_image)
                        if image_base64:
                            content.append({
                                "type": "image_url", 
                                "image_url": {"url": f"data:image/png;base64,{image_base64}"}
                            })
                            images_added += 1

            print(f"Total of {images_added} images added to the request")

            messages = [{
                "role": "user",
                "content": content
            }]

            payload = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "top_p": top_p,
                "max_tokens": max_tokens,
                "stream": True 
            }

            if seed > 0:
                payload["seed"] = seed

            pbar.update_absolute(30)

            try:
                response_text = self.send_request_streaming(payload)
                pbar.update_absolute(70)
            except Exception as e:
                error_message = f"API Error: {str(e)}"
                print(error_message)
                return (default_image, error_message, "")

            base64_pattern = r'data:image\/[^;]+;base64,([A-Za-z0-9+/=]+)'
            base64_matches = re.findall(base64_pattern, response_text)
            
            if base64_matches:
                try:
                    image_data = base64.b64decode(base64_matches[0])
                    generated_image = Image.open(BytesIO(image_data))
                    generated_tensor = pil2tensor(generated_image)
                    
                    pbar.update_absolute(100)
                    return (generated_tensor, response_text, f"data:image/png;base64,{base64_matches[0]}")
                except Exception as e:
                    print(f"Error processing base64 image data: {str(e)}")

            image_pattern = r'!\[.*?\]\((.*?)\)'
            matches = re.findall(image_pattern, response_text)
            
            if not matches:
                url_pattern = r'https?://\S+\.(?:jpg|jpeg|png|gif|webp)'
                matches = re.findall(url_pattern, response_text)
            
            if not matches:
                all_urls_pattern = r'https?://\S+'
                matches = re.findall(all_urls_pattern, response_text)
                
            if matches:
                image_url = matches[0]
                try:
                    img_response = requests.get(image_url, timeout=self.timeout)
                    img_response.raise_for_status()
                    
                    generated_image = Image.open(BytesIO(img_response.content))
                    generated_tensor = pil2tensor(generated_image)
                    
                    pbar.update_absolute(100)
                    return (generated_tensor, response_text, image_url)
                except Exception as e:
                    print(f"Error downloading image: {str(e)}")
                    return (default_image, f"{response_text}\n\nError downloading image: {str(e)}", image_url)
            else:
                pbar.update_absolute(100)
                return (default_image, response_text, "")
                
        except Exception as e:
            error_message = f"Error processing request: {str(e)}"
            print(error_message)
            return (default_image, error_message, "")


class Comfly_nano_banana_edit:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "mode": (["text2img", "img2img"], {"default": "text2img"}),
                "model": (["nano-banana", "nano-banana-hd"], {"default": "nano-banana"}),
                "aspect_ratio": (["16:9", "4:3", "4:5", "3:2", "1:1", "2:3", "3:4", "5:4", "9:16", "21:9"], {"default": "1:1"}),
            },
            "optional": {
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
                "image4": ("IMAGE",),
                "apikey": ("STRING", {"default": ""}),
                "response_format": (["url", "b64_json"], {"default": "url"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647}),
                "webhook": ("STRING", {"default": ""})
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "response")
    FUNCTION = "generate_image"
    CATEGORY = "Comfly/Google"

    def __init__(self):
        self.api_key = get_config().get('api_key', '')
        self.timeout = 600

    def get_headers(self):
        return {
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
    
    def _download_image_content(self, url):
        """Helper to download image bytes"""
        try:
            resp = requests.get(url, timeout=self.timeout)
            resp.raise_for_status()
            return resp.content, url, None
        except Exception as e:
            return None, url, str(e)

    def generate_image(self, prompt, mode="text2img", model="nano-banana", aspect_ratio="1:1", 
                      image1=None, image2=None, image3=None, image4=None,
                      apikey="", response_format="url", seed=0, webhook=""):
        if apikey.strip():
            self.api_key = apikey
            
        if not self.api_key:
            error_message = "API key not found in Comflyapi.json"
            print(error_message)
            blank_image = Image.new('RGB', (1024, 1024), color='white')
            blank_tensor = pil2tensor(blank_image)
            return (blank_tensor, error_message)
            
        try:
            pbar = comfy.utils.ProgressBar(100)
            pbar.update_absolute(10)

            final_prompt = prompt

            query_params = {"async": "true"}
            if webhook:
                query_params["webhook"] = webhook
            
            if mode == "text2img":
                headers = self.get_headers()
                headers["Content-Type"] = "application/json"
                
                payload = {
                    "prompt": final_prompt,
                    "model": model,
                    "aspect_ratio": aspect_ratio  
                }
                    
                if response_format:
                    payload["response_format"] = response_format

                if seed > 0:
                    payload["seed"] = seed

                response = requests.post(
                    f"{baseurl}/v1/images/generations",
                    headers=headers,
                    json=payload,
                    params=query_params,
                    timeout=self.timeout
                )
            else:
                headers = self.get_headers()
                
                files = []
                for img in [image1, image2, image3, image4]:
                    if img is not None:
                        pil_img = tensor2pil(img)[0]
                        buffered = BytesIO()
                        pil_img.save(buffered, format="PNG")
                        buffered.seek(0)
                        files.append(('image', ('image.png', buffered, 'image/png')))
                
                data = {
                    "prompt": final_prompt,
                    "model": model,
                    "aspect_ratio": aspect_ratio 
                }
                
                if response_format:
                    data["response_format"] = response_format

                if seed > 0:
                    data["seed"] = str(seed)

                response = requests.post(
                    f"{baseurl}/v1/images/edits",
                    headers=headers,
                    data=data,
                    files=files,
                    params=query_params,
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

            if "task_id" not in result:
                error_message = "No task_id in response"
                print(error_message)
                blank_image = Image.new('RGB', (1024, 1024), color='white')
                blank_tensor = pil2tensor(blank_image)
                return (blank_tensor, error_message)
            
            task_id = result["task_id"]
            print(f"[Comfly_nano_banana_edit] Task submitted successfully. Task ID: {task_id}")
            
            pbar.update_absolute(40)

            max_retries = 150 
            retry_count = 0
            
            while retry_count < max_retries:
                time.sleep(2)
                retry_count += 1
                
                try:
                    status_response = requests.get(
                        f"{baseurl}/v1/images/tasks/{task_id}",
                        headers=self.get_headers(),
                        timeout=self.timeout
                    )
                    
                    if status_response.status_code != 200:
                        print(f"[Comfly_nano_banana_edit] Status check failed with code: {status_response.status_code}")
                        continue
                        
                    status_result = status_response.json()

                    outer_data = status_result.get("data", {})
                    status = outer_data.get("status", "")
                    progress = outer_data.get("progress", "0%")
                   
                    pbar.update_absolute(40 + min(50, retry_count * 50 // max_retries))
                    
                    if status == "SUCCESS":
                        inner_data = outer_data.get("data", {})
                        image_list = inner_data.get("data", [])
                        
                        if not image_list:
                            error_message = "No image data in successful response"
                            print(f"[Comfly_nano_banana_edit] {error_message}")
                            blank_image = Image.new('RGB', (1024, 1024), color='white')
                            blank_tensor = pil2tensor(blank_image)
                            return (blank_tensor, error_message)
                        
                        generated_tensors = []
                        response_info = f"Generated {len(image_list)} images using {model}\n"
                        response_info += f"Aspect ratio: {aspect_ratio}\n"
                        response_info += f"Task ID: {task_id}\n"

                        if seed > 0:
                            response_info += f"Seed: {seed}\n"
                        
                        print(f"\n[Comfly_nano_banana_edit] ========== Generated Image URLs ==========")

                        urls_to_download = []
                        b64_items = []

                        for i, item in enumerate(image_list):
                            if "b64_json" in item and item["b64_json"]:
                                b64_items.append((i, item["b64_json"]))
                            elif "url" in item and item["url"]:
                                urls_to_download.append((i, item["url"]))

                        results_map = {}
                        
                        for idx, b64_data in b64_items:
                            try:
                                image_data_bytes = base64.b64decode(b64_data)
                                generated_image = Image.open(BytesIO(image_data_bytes))
                                generated_tensor = pil2tensor(generated_image)
                                results_map[idx] = generated_tensor
                                response_info += f"Image {idx+1}: Base64 data\n"
                                print(f"[Comfly_nano_banana_edit] Image {idx+1}: Base64 encoded data processed")
                            except Exception as e:
                                print(f"Error processing base64 image {idx+1}: {e}")

                        if urls_to_download:
                            print(f"[Comfly_nano_banana_edit] Downloading {len(urls_to_download)} images in parallel...")
                            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                                future_to_idx = {executor.submit(self._download_image_content, url): (idx, url) for idx, url in urls_to_download}
                                
                                for future in concurrent.futures.as_completed(future_to_idx):
                                    idx, url = future_to_idx[future]
                                    try:
                                        content, _, err = future.result()
                                        if err:
                                            print(f"[Comfly_nano_banana_edit] Error downloading image {idx+1}: {err}")
                                            continue
                                            
                                        generated_image = Image.open(BytesIO(content))
                                        generated_tensor = pil2tensor(generated_image)
                                        results_map[idx] = generated_tensor
                                        response_info += f"Image {idx+1}: {url}\n"
                                        print(f"[Comfly_nano_banana_edit] Image {idx+1} downloaded successfully")
                                    except Exception as exc:
                                        print(f"[Comfly_nano_banana_edit] Exception processing image {idx+1}: {exc}")

                        for i in range(len(image_list)):
                            if i in results_map:
                                generated_tensors.append(results_map[i])

                        print(f"[Comfly_nano_banana_edit] ==========================================\n")
                        
                        pbar.update_absolute(100)
                        
                        if generated_tensors:
                            combined_tensor = torch.cat(generated_tensors, dim=0)
                            return (combined_tensor, response_info)
                        else:
                            error_message = "Failed to process any images"
                            print(error_message)
                            blank_image = Image.new('RGB', (1024, 1024), color='white')
                            blank_tensor = pil2tensor(blank_image)
                            return (blank_tensor, error_message)
                    
                    elif status == "FAILURE":
                        fail_reason = outer_data.get("fail_reason", "Unknown error")
                        error_message = f"Task failed: {fail_reason}"
                        print(f"[Comfly_nano_banana_edit] {error_message}")
                        blank_image = Image.new('RGB', (1024, 1024), color='white')
                        blank_tensor = pil2tensor(blank_image)
                        return (blank_tensor, error_message)
                        
                except Exception as e:
                    print(f"[Comfly_nano_banana_edit] Error checking task status: {str(e)}")
                    continue
            
            error_message = f"Task timed out after {max_retries} retries"
            print(f"[Comfly_nano_banana_edit] {error_message}")
            blank_image = Image.new('RGB', (1024, 1024), color='white')
            blank_tensor = pil2tensor(blank_image)
            return (blank_tensor, error_message)
            
        except Exception as e:
            error_message = f"Error in image generation: {str(e)}"
            print(f"[Comfly_nano_banana_edit] {error_message}")
            blank_image = Image.new('RGB', (1024, 1024), color='white')
            blank_tensor = pil2tensor(blank_image)
            return (blank_tensor, error_message)


class Comfly_nano_banana2_edit:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "mode": (["text2img", "img2img"], {"default": "text2img"}),
                "model": (["nano-banana-2", "nano-banana-2-2k", "nano-banana-2-4k"], {"default": "nano-banana-2"}),
                "aspect_ratio": (["auto", "16:9", "4:3", "4:5", "3:2", "1:1", "2:3", "3:4", "5:4", "9:16", "21:9"], {"default": "auto"}),
                "image_size": (["1K", "2K", "4K"], {"default": "2K"}),
            },
            "optional": {
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
                "image4": ("IMAGE",),
                "image5": ("IMAGE",),
                "image6": ("IMAGE",),
                "image7": ("IMAGE",),
                "image8": ("IMAGE",),
                "image9": ("IMAGE",),
                "image10": ("IMAGE",),
                "image11": ("IMAGE",),
                "image12": ("IMAGE",),
                "image13": ("IMAGE",),
                "image14": ("IMAGE",),
                "apikey": ("STRING", {"default": ""}),
                "response_format": (["url", "b64_json"], {"default": "url"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647}),
                "webhook": ("STRING", {"default": ""})
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("image", "response", "image_url")
    FUNCTION = "generate_image"
    CATEGORY = "Comfly/Google"

    def __init__(self):
        self.api_key = get_config().get('api_key', '')
        self.timeout = 600

    def get_headers(self):
        return {
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
    
    def _download_image_content(self, url):
        """Helper to download image bytes"""
        try:
            resp = requests.get(url, timeout=self.timeout)
            resp.raise_for_status()
            return resp.content, url, None
        except Exception as e:
            return None, url, str(e)
    
    def generate_image(self, prompt, mode="text2img", model="nano-banana-2", aspect_ratio="auto", 
                      image_size="2K", image1=None, image2=None, image3=None, image4=None,
                      image5=None, image6=None, image7=None, image8=None, image9=None, 
                      image10=None, image11=None, image12=None, image13=None, image14=None,
                      apikey="", response_format="url", seed=0, webhook=""):

        if apikey.strip():
            self.api_key = apikey
            
        if not self.api_key:
            error_message = "API key not found in Comflyapi.json"
            print(error_message)
            blank_image = Image.new('RGB', (1024, 1024), color='white')
            blank_tensor = pil2tensor(blank_image)
            return (blank_tensor, error_message, "")
            
        try:
            pbar = comfy.utils.ProgressBar(100)
            pbar.update_absolute(10)

            final_prompt = prompt

            query_params = {"async": "true"}
            if webhook:
                query_params["webhook"] = webhook
            
            image_count = 0
            
            if mode == "text2img":
                headers = self.get_headers()
                headers["Content-Type"] = "application/json"
                
                payload = {
                    "prompt": final_prompt,
                    "model": model,
                    "aspect_ratio": aspect_ratio
                }

                if model == "nano-banana-2":
                    payload["image_size"] = image_size
                    
                if response_format:
                    payload["response_format"] = response_format

                if seed > 0:
                    payload["seed"] = seed

                response = requests.post(
                    f"{baseurl}/v1/images/generations",
                    headers=headers,
                    json=payload,
                    params=query_params,
                    timeout=self.timeout
                )
            else:
                headers = self.get_headers()

                all_images = [image1, image2, image3, image4, image5, image6, image7, 
                             image8, image9, image10, image11, image12, image13, image14]
                
                files = []
                for img in all_images:
                    if img is not None:
                        pil_img = tensor2pil(img)[0]
                        buffered = BytesIO()
                        pil_img.save(buffered, format="PNG")
                        buffered.seek(0)
                        files.append(('image', (f'image_{image_count}.png', buffered, 'image/png')))
                        image_count += 1
                
                print(f"[Comfly_nano_banana2_edit] Processing {image_count} input images")
                
                data = {
                    "prompt": final_prompt,
                    "model": model,
                    "aspect_ratio": aspect_ratio
                }
                
                if model == "nano-banana-2":
                    data["image_size"] = image_size
                
                if response_format:
                    data["response_format"] = response_format

                if seed > 0:
                    data["seed"] = str(seed)

                response = requests.post(
                    f"{baseurl}/v1/images/edits",
                    headers=headers,
                    data=data,
                    files=files,
                    params=query_params,
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

            if "task_id" not in result:
                error_message = "No task_id in response"
                print(error_message)
                blank_image = Image.new('RGB', (1024, 1024), color='white')
                blank_tensor = pil2tensor(blank_image)
                return (blank_tensor, error_message, "")
            
            task_id = result["task_id"]
            print(f"[Comfly_nano_banana2_edit] Task submitted successfully. Task ID: {task_id}")
            
            pbar.update_absolute(40)

            max_retries = 150 
            retry_count = 0
            
            while retry_count < max_retries:
                time.sleep(2) 
                retry_count += 1
                
                try:
                    status_response = requests.get(
                        f"{baseurl}/v1/images/tasks/{task_id}",
                        headers=self.get_headers(),
                        timeout=self.timeout
                    )
                    
                    if status_response.status_code != 200:
                        print(f"[Comfly_nano_banana2_edit] Status check failed with code: {status_response.status_code}")
                        continue
                        
                    status_result = status_response.json()

                    outer_data = status_result.get("data", {})
                    status = outer_data.get("status", "")
                    progress = outer_data.get("progress", "0%")
                    
                    pbar.update_absolute(40 + min(50, retry_count * 50 // max_retries))
                    
                    if status == "SUCCESS":
                        inner_data = outer_data.get("data", {})
                        image_list = inner_data.get("data", [])
                        
                        if not image_list:
                            error_message = "No image data in successful response"
                            print(f"[Comfly_nano_banana2_edit] {error_message}")
                            blank_image = Image.new('RGB', (1024, 1024), color='white')
                            blank_tensor = pil2tensor(blank_image)
                            return (blank_tensor, error_message, "")
                        
                        generated_tensors = []
                        first_image_url = "" 
                        
                        response_info = f"Generated {len(image_list)} images using {model}\n"

                        if model == "nano-banana-2":
                            response_info += f"Image size: {image_size}\n"
                        
                        response_info += f"Aspect ratio: {aspect_ratio}\n"
                        response_info += f"Task ID: {task_id}\n"
                        
                        if mode == "img2img":
                            response_info += f"Input images: {image_count}\n"

                        if seed > 0:
                            response_info += f"Seed: {seed}\n"
                        
                        print(f"\n[Comfly_nano_banana2_edit] ========== Generated Image URLs ==========")

                        urls_to_download = []
                        b64_items = []
                        results_map = {}

                        for i, item in enumerate(image_list):
                            if "b64_json" in item and item["b64_json"]:
                                b64_items.append((i, item["b64_json"]))
                            elif "url" in item and item["url"]:
                                urls_to_download.append((i, item["url"]))
                                if i == 0: first_image_url = item["url"]

                        for idx, b64_data in b64_items:
                            try:
                                image_data_bytes = base64.b64decode(b64_data)
                                generated_image = Image.open(BytesIO(image_data_bytes))
                                generated_tensor = pil2tensor(generated_image)
                                results_map[idx] = generated_tensor
                                response_info += f"Image {idx+1}: Base64 data\n"
                                print(f"[Comfly_nano_banana2_edit] Image {idx+1}: Base64 encoded data processed")
                            except Exception as e:
                                print(f"Error processing base64 image {idx+1}: {e}")

                        if urls_to_download:
                            print(f"[Comfly_nano_banana2_edit] Downloading {len(urls_to_download)} images in parallel...")
                            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                                future_to_idx = {executor.submit(self._download_image_content, url): (idx, url) for idx, url in urls_to_download}
                                
                                for future in concurrent.futures.as_completed(future_to_idx):
                                    idx, url = future_to_idx[future]
                                    try:
                                        content, _, err = future.result()
                                        if err:
                                            print(f"[Comfly_nano_banana2_edit] Error downloading image {idx+1}: {err}")
                                            continue
                                            
                                        generated_image = Image.open(BytesIO(content))
                                        generated_tensor = pil2tensor(generated_image)
                                        results_map[idx] = generated_tensor
                                        response_info += f"Image {idx+1}: {url}\n"
                                        print(f"[Comfly_nano_banana2_edit] Image {idx+1} downloaded successfully")
                                    except Exception as exc:
                                        print(f"[Comfly_nano_banana2_edit] Exception processing image {idx+1}: {exc}")

                        for i in range(len(image_list)):
                            if i in results_map:
                                generated_tensors.append(results_map[i])

                        print(f"[Comfly_nano_banana2_edit] ==========================================\n")
                        
                        pbar.update_absolute(100)
                        
                        if generated_tensors:
                            combined_tensor = torch.cat(generated_tensors, dim=0)
                            return (combined_tensor, response_info, first_image_url)
                        else:
                            error_message = "Failed to process any images"
                            print(error_message)
                            blank_image = Image.new('RGB', (1024, 1024), color='white')
                            blank_tensor = pil2tensor(blank_image)
                            return (blank_tensor, error_message, "")
                    
                    elif status == "FAILURE":
                        fail_reason = outer_data.get("fail_reason", "Unknown error")
                        error_message = f"Task failed: {fail_reason}"
                        print(f"[Comfly_nano_banana2_edit] {error_message}")
                        blank_image = Image.new('RGB', (1024, 1024), color='white')
                        blank_tensor = pil2tensor(blank_image)
                        return (blank_tensor, error_message, "")
                        
                except Exception as e:
                    print(f"[Comfly_nano_banana2_edit] Error checking task status: {str(e)}")
                    continue
            
            error_message = f"Task timed out after {max_retries} retries"
            print(f"[Comfly_nano_banana2_edit] {error_message}")
            blank_image = Image.new('RGB', (1024, 1024), color='white')
            blank_tensor = pil2tensor(blank_image)
            return (blank_tensor, error_message, "")
            
        except Exception as e:
            error_message = f"Error in image generation: {str(e)}"
            print(f"[Comfly_nano_banana2_edit] {error_message}")
            blank_image = Image.new('RGB', (1024, 1024), color='white')
            blank_tensor = pil2tensor(blank_image)
            return (blank_tensor, error_message, "")


class ComflyGeminiTextOnly:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "model": ("STRING", {"default": "gemini-2.5-pro"}),
            },
            "optional": {
                "image": ("IMAGE",),
                "video": ("VIDEO",),
                "api_key": ("STRING", {"default": ""}),
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "top_p": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 1.0, "step": 0.01}),
                "max_tokens": ("INT", {"default": 4096, "min": 1, "max": 8192}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("response",)
    FUNCTION = "generate_text"
    CATEGORY = "Comfly-v2/Google"

    def __init__(self):
        self.api_key = get_config().get('api_key', '')
        self.timeout = 120

    def get_headers(self):
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

    def tensor_to_base64(self, tensor):
        if tensor is None:
            return None
        if tensor.dtype != torch.uint8:
            tensor = (tensor * 255).clamp(0, 255).byte()
        tensor = tensor.cpu()
        if tensor.shape[-1] == 3:
            img = Image.fromarray(tensor.numpy(), 'RGB')
        else:
            img = Image.fromarray(tensor.numpy(), 'RGBA')
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    def generate_text(self, prompt, model, temperature, top_p, max_tokens, seed, image=None, video=None, api_key=""):
        if api_key.strip():
            self.api_key = api_key
            config = get_config()
            config['api_key'] = api_key
            save_config(config)

        if not self.api_key:
            return ("API key not found in Comflyapi.json",)

        try:
            content = [{"type": "text", "text": prompt}]

            if video is not None:
                video_url = getattr(video, 'video_url', None)
                if video_url:
                    content.append({
                        "type": "video_url",
                        "video_url": {"url": video_url}
                    })
            elif image is not None:
                if len(image.shape) == 4:
                    image = image[0]
                img_b64 = self.tensor_to_base64(image)
                if img_b64:
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{img_b64}"}
                    })

            messages = [{"role": "user", "content": content}]

            payload = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "top_p": top_p,
                "max_tokens": max_tokens,
                "seed": seed if seed > 0 else None
            }

            response = requests.post(
                f"{baseurl}/v1/chat/completions",
                headers=self.get_headers(),
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            result = response.json()
            text = result["choices"][0]["message"]["content"]
            return (text,)

        except Exception as e:
            return (f"Error: {str(e)}",)