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


class Comfly_kling_text2video:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "model_name": (["kling-v2-1-master", "kling-v2-master", "kling-v1-6", "kling-v1-5", "kling-v1"], {"default": "kling-v1-6"}),
                "imagination": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
                "aspect_ratio": (["1:1", "16:9", "9:16"], {"default": "1:1"}),
                "mode": (["std", "pro"], {"default": "std"}),
                "duration": (["5", "10"], {"default": "5"}),
                "num_videos": ("INT", {"default": 1, "min": 1, "max": 4}),
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647})
            },
            "optional": {
                "api_key": ("STRING", {"default": ""}),
                "camera": (["none", "horizontal", "vertical", "zoom", "vertical_shake", "horizontal_shake", 
                          "rotate", "master_down_zoom", "master_zoom_up", "master_right_rotate_zoom", 
                          "master_left_rotate_zoom"], {"default": "none"}),
                "camera_value": ("FLOAT", {"default": 0, "min": -10, "max": 10, "step": 0.1})
            }
        }

    RETURN_TYPES = (IO.VIDEO, "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("video", "video_url", "task_id", "video_id", "response")
    FUNCTION = "generate_video"
    CATEGORY = "Comfly/Comfly_kling"

    def __init__(self):
        super().__init__()
        self.api_key = get_config().get('api_key', '')
        self.timeout = 300
        self.model_capabilities = {
            "kling-v1": {
                "std": {
                    "5": {"video": True, "camera": True},
                    "10": {"video": True, "camera": False}
                },
                "pro": {
                    "5": {"video": True, "camera": False},
                    "10": {"video": True, "camera": False}
                }
            },
            "kling-v1-5": {
                "std": {
                    "5": {"video": False, "camera": False},
                    "10": {"video": False, "camera": False}
                },
                "pro": {
                    "5": {"video": False, "camera": True},
                    "10": {"video": False, "camera": False}
                }
            },
            "kling-v1-6": {
                "std": {
                    "5": {"video": True, "camera": False},
                    "10": {"video": True, "camera": False}
                },
                "pro": {
                    "5": {"video": False, "camera": False},
                    "10": {"video": False, "camera": False}
                }
            },
            "kling-v2-master": {
                "std": {
                    "5": {"video": False, "camera": False},
                    "10": {"video": False, "camera": False}
                },
                "pro": {
                    "5": {"video": False, "camera": False},
                    "10": {"video": False, "camera": False}
                }
            }
        }

    def get_headers(self):
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
    def get_camera_json(self, camera, camera_value=0):
        camera_mappings = {
            "none": {"type":"empty","horizontal":0,"vertical":0,"zoom":0,"tilt":0,"pan":0,"roll":0},
            "horizontal": {"type":"horizontal","horizontal":camera_value,"vertical":0,"zoom":0,"tilt":0,"pan":0,"roll":0}, 
            "vertical": {"type":"vertical","horizontal":0,"vertical":camera_value,"zoom":0,"tilt":0,"pan":0,"roll":0},
            "zoom": {"type":"zoom","horizontal":0,"vertical":0,"zoom":camera_value,"tilt":0,"pan":0,"roll":0},
            "vertical_shake": {"type":"vertical_shake","horizontal":0,"vertical":camera_value,"zoom":0.5,"tilt":0,"pan":0,"roll":0},
            "horizontal_shake": {"type":"horizontal_shake","horizontal":camera_value,"vertical":0,"zoom":0.5,"tilt":0,"pan":0,"roll":0}, 
            "rotate": {"type":"rotate","horizontal":0,"vertical":0,"zoom":0,"tilt":0,"pan":camera_value,"roll":0},
            "master_down_zoom": {"type":"zoom","horizontal":0,"vertical":0,"zoom":camera_value,"tilt":camera_value,"pan":0,"roll":0},
            "master_zoom_up": {"type":"zoom","horizontal":0.2,"vertical":0,"zoom":camera_value,"tilt":0,"pan":0,"roll":0},
            "master_right_rotate_zoom": {"type":"rotate","horizontal":0,"vertical":0,"zoom":camera_value,"tilt":0,"pan":0,"roll":camera_value},
            "master_left_rotate_zoom": {"type":"rotate","horizontal":0,"vertical":0,"zoom":camera_value,"tilt":0,"pan":camera_value,"roll":0},
        }
        return json.dumps(camera_mappings.get(camera, camera_mappings["none"]))

    def generate_video(self, prompt, model_name, imagination, aspect_ratio, mode, duration, num_videos, 
                  negative_prompt="", camera="none", camera_value=0, seed=0, api_key=""):
        if api_key.strip():
            self.api_key = api_key
            config = get_config()
            config['api_key'] = api_key
            
        if not self.api_key:
            error_response = {"task_status": "failed", "task_status_msg": "API key not found in Comflyapi.json"}
            return ("", "", "", "", json.dumps(error_response))
            
        camera_json = {}
        if model_name == "kling-v1":  
            camera_json = self.get_camera_json(camera, camera_value)
        else:
            camera_json = self.get_camera_json("none", 0)
            
        payload = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "image": "",
            "image_tail": "",
            "aspect_ratio": aspect_ratio,
            "duration": duration,
            "model_name": model_name,
            "imagination": imagination,
            "num_videos": num_videos,
            "camera_json": camera_json,
            "seed": seed
        }

        if model_name != "kling-v2-master":
            payload["mode"] = mode
        else:
            print("Note: kling-v2-master model doesn't use mode parameter")
            
        try:
            pbar = comfy.utils.ProgressBar(100)  
            response = requests.post(
                f"{baseurl}/kling/v1/videos/text2video",
                headers=self.get_headers(),
                json=payload
            )
            response.raise_for_status()
            result = response.json()
            if result["code"] != 0:
                error_response = {"task_status": "failed", "task_status_msg": f"API Error: {result['message']}"}
                return ("", "", "", "", json.dumps(error_response))
                
            task_id = result["data"]["task_id"]
            pbar.update_absolute(5)  
            
            last_status = {}
            while True:
                time.sleep(2)
                status_response = requests.get(
                    f"{baseurl}/kling/v1/videos/text2video/{task_id}",
                    headers=self.get_headers()
                )
                status_response.raise_for_status()
                status_result = status_response.json()
                last_status = status_result["data"]
                
                progress = status_result["data"].get("progress", 0)
                pbar.update_absolute(progress)
                
                if status_result["data"]["task_status"] == "succeed":
                    pbar.update_absolute(100) 
                    video_url = status_result["data"]["task_result"]["videos"][0]["url"]
                    video_id = status_result["data"]["task_result"]["videos"][0]["id"]

                    response_data = {
                        "task_status": "succeed",
                        "task_status_msg": "Video generated successfully",
                        "progress": 100,
                        "video_url": video_url  
                    }
                    
                    video_adapter = ComflyVideoAdapter(video_url)
                    return (video_adapter, video_url, task_id, video_id, json.dumps(response_data))
                
                elif status_result["data"]["task_status"] == "failed":
                    error_msg = status_result["data"].get("task_status_msg", "Unknown error")
                    error_response = {
                        "task_status": "failed", 
                        "task_status_msg": error_msg,
                    }
                    print(f"Task failed: {error_msg}")
                    return ("", "", task_id, "", json.dumps(error_response))
        except Exception as e:
            error_response = {"task_status": "failed", "task_status_msg": f"Error generating video: {str(e)}"}
            print(f"Error generating video: {str(e)}")
            return ("", "", "", "", json.dumps(error_response))



class Comfly_kling_image2video:
    @classmethod 
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "prompt": ("STRING", {"multiline": True}),
                "model_name": (["kling-v2-1", "kling-v2-1-master", "kling-v2-master", "kling-v1-6", "kling-v1-5", "kling-v1"], {"default": "kling-v1-6"}),
                "imagination": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
                "aspect_ratio": (["1:1", "16:9", "9:16"], {"default": "1:1"}),
                "mode": (["std", "pro"], {"default": "std"}),
                "duration": (["5", "10"], {"default": "5"}),
                "num_videos": ("INT", {"default": 1, "min": 1, "max": 4}),
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647})
            },
            "optional": {
                "image_tail": ("IMAGE",),
                "api_key": ("STRING", {"default": ""}),
                "camera": (["none", "horizontal", "vertical", "zoom", "vertical_shake", "horizontal_shake", 
                          "rotate", "master_down_zoom", "master_zoom_up", "master_right_rotate_zoom", 
                          "master_left_rotate_zoom"], {"default": "none"}),
                "camera_value": ("FLOAT", {"default": 0, "min": -10, "max": 10, "step": 0.1}),      
            }
        }

    RETURN_TYPES = (IO.VIDEO, "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("video", "video_url", "task_id", "video_id", "response")
    FUNCTION = "generate_video"
    CATEGORY = "Comfly/Comfly_kling"

    def __init__(self):
        self.api_key = get_config().get('api_key', '')
        self.timeout = 300
        self.model_compatibility = {
            "kling-v1": {
                "std": {"5": True, "10": True},  
                "pro": {"5": True, "10": True}
            },
            "kling-v1-5": {
                "std": {"5": False, "10": False},  
                "pro": {"5": True, "10": True}
            },
            "kling-v1-6": {
                "std": {"5": False, "10": False},  
                "pro": {"5": True, "10": True}
            },
            "kling-v2-master": {
                "std": {"5": False, "10": False},
                "pro": {"5": False, "10": False}
            },
            "kling-v2-1": {
                "std": {"5": False, "10": False},  
                "pro": {"5": True, "10": True}  
            }
        }

    def get_headers(self):
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
    def check_tail_image_compatibility(self, model_name, mode, duration):
        try:
            if model_name == "kling-v2-1":
                return mode == "pro"
            
            return self.model_compatibility.get(model_name, {}).get(mode, {}).get(duration, False)
        except:
            return False

    def generate_video(self, image, prompt, model_name, imagination, aspect_ratio, mode, duration, 
                  num_videos, negative_prompt="", camera="none", camera_value=0, seed=0, image_tail=None, api_key=""):
        if api_key.strip():
            self.api_key = api_key
            config = get_config()
            config['api_key'] = api_key
            
        if not self.api_key:
            error_response = {"task_status": "failed", "task_status_msg": "API key not found in Comflyapi.json"}
            return ("", "", "", "", json.dumps(error_response))

        has_tail_image = image_tail is not None

        if has_tail_image:
            check_mode = "std" if model_name == "kling-v2-master" else mode
            tail_compatible = self.check_tail_image_compatibility(model_name, check_mode, duration)
            if not tail_compatible:
                warning_message = f"Warning: model/mode/duration({model_name}/{mode if model_name != 'kling-v2-master' else 'N/A'}/{duration}) does not support using both image and image_tail."
                print(warning_message)

                if model_name == "kling-v1-5" or model_name == "kling-v1-6":
                    if mode == "std":
                        suggestion = "\nSuggestion: Try switching to 'pro' mode which supports tail images."
                        warning_message += suggestion
                
                error_response = {
                    "task_status": "failed", 
                    "task_status_msg": warning_message
                }
                return ("", "", "", "", json.dumps(error_response))
        
        camera_json = {}
        if model_name in ["kling-v1-5", "kling-v1-6"] and mode == "pro" and camera != "none": 
            camera_json = self.get_camera_json(camera, camera_value)
        else:
            camera_json = self.get_camera_json("none", 0)
            
        try:
            pil_image = tensor2pil(image)[0]
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            buffered = BytesIO()
            pil_image.save(buffered, format="JPEG", quality=95)
            image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

            image_tail_base64 = ""
            if has_tail_image:
                pil_tail = tensor2pil(image_tail)[0]
                if pil_tail.mode != 'RGB':
                    pil_tail = pil_tail.convert('RGB')
                tail_buffered = BytesIO()
                pil_tail.save(tail_buffered, format="JPEG", quality=95)
                image_tail_base64 = base64.b64encode(tail_buffered.getvalue()).decode('utf-8')

            payload = {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "image": image_base64,
                "image_tail": image_tail_base64,
                "aspect_ratio": aspect_ratio,
                "duration": duration,
                "model_name": model_name,
                "imagination": imagination,  
                "num_videos": num_videos,
                "camera_json": camera_json if isinstance(camera_json, str) else json.dumps(camera_json),
                "seed": seed
            }
            
            if model_name != "kling-v2-master":
                payload["mode"] = mode

            print(f"Sending request with parameters: model={model_name}, duration={duration}, aspect_ratio={aspect_ratio}")
            if model_name != "kling-v2-master":
                print(f"Mode: {mode}")
            else:
                print("Note: kling-v2-master model doesn't use mode parameter")
                
            print(f"Image base64 length: {len(image_base64)}")
            if has_tail_image:
                print(f"Image tail base64 length: {len(image_tail_base64)}")
            
            pbar = comfy.utils.ProgressBar(100)
            pbar.update_absolute(5)  
            response = requests.post(
                f"{baseurl}/kling/v1/videos/image2video",
                headers=self.get_headers(),
                json=payload,
                timeout=self.timeout
            )

            print(f"Response status: {response.status_code}")
            if response.status_code != 200:
                error_message = f"Error: {response.status_code} {response.reason} - {response.text}"
                print(error_message)
                error_response = {"task_status": "failed", "task_status_msg": error_message}
                return ("", "", "", "", json.dumps(error_response))
            
            result = response.json()
            if result["code"] != 0:
                error_response = {"task_status": "failed", "task_status_msg": f"API Error: {result['message']}"}
                return ("", "", "", "", json.dumps(error_response))
                
            task_id = result["data"]["task_id"]
            pbar.update_absolute(10) 
            
            last_status = {}
            while True:
                time.sleep(2)
                status_response = requests.get(
                    f"{baseurl}/kling/v1/videos/image2video/{task_id}",
                    headers=self.get_headers(),
                    timeout=self.timeout
                )
                status_response.raise_for_status()
                status_result = status_response.json()
                last_status = status_result["data"]

                progress = 0
                if status_result["data"]["task_status"] == "processing":
                    progress = status_result["data"].get("progress", 50)
                elif status_result["data"]["task_status"] == "succeed":
                    progress = 100
                
                pbar.update_absolute(progress)
                
                if status_result["data"]["task_status"] == "succeed":
                    pbar.update_absolute(100) 
                    video_url = status_result["data"]["task_result"]["videos"][0]["url"]
                    video_id = status_result["data"]["task_result"]["videos"][0]["id"]
                    
                    response_data = {
                        "task_status": "succeed",
                        "task_status_msg": "Video generated successfully",
                        "progress": 100,
                        "video_url": video_url
                    }
                    
                    video_adapter = ComflyVideoAdapter(video_url)
                    return (video_adapter, video_url, task_id, video_id, json.dumps(response_data))
                
                elif status_result["data"]["task_status"] == "failed":
                    error_msg = status_result["data"].get("task_status_msg", "Unknown error")
                    error_response = {
                        "task_status": "failed", 
                        "task_status_msg": error_msg,
                    }
                    print(f"Task failed: {error_msg}")
                    return ("", "", task_id, "", json.dumps(error_response))
        except Exception as e:
            error_response = {"task_status": "failed", "task_status_msg": f"Error generating video: {str(e)}"}
            print(f"Error generating video: {str(e)}")
            return ("", "", "", "", json.dumps(error_response))

    def get_camera_json(self, camera, camera_value=0):
        camera_mappings = {
            "none": {"type":"empty","horizontal":0,"vertical":0,"zoom":0,"tilt":0,"pan":0,"roll":0},
            "horizontal": {"type":"horizontal","horizontal":camera_value,"vertical":0,"zoom":0,"tilt":0,"pan":0,"roll":0}, 
            "vertical": {"type":"vertical","horizontal":0,"vertical":camera_value,"zoom":0,"tilt":0,"pan":0,"roll":0},
            "zoom": {"type":"zoom","horizontal":0,"vertical":0,"zoom":camera_value,"tilt":0,"pan":0,"roll":0},
            "vertical_shake": {"type":"vertical_shake","horizontal":0,"vertical":camera_value,"zoom":0.5,"tilt":0,"pan":0,"roll":0},
            "horizontal_shake": {"type":"horizontal_shake","horizontal":camera_value,"vertical":0,"zoom":0.5,"tilt":0,"pan":0,"roll":0}, 
            "rotate": {"type":"rotate","horizontal":0,"vertical":0,"zoom":0,"tilt":0,"pan":camera_value,"roll":0},
            "master_down_zoom": {"type":"zoom","horizontal":0,"vertical":0,"zoom":camera_value,"tilt":camera_value,"pan":0,"roll":0},
            "master_zoom_up": {"type":"zoom","horizontal":0.2,"vertical":0,"zoom":camera_value,"tilt":0,"pan":0,"roll":0},
            "master_right_rotate_zoom": {"type":"rotate","horizontal":0,"vertical":0,"zoom":camera_value,"tilt":0,"pan":0,"roll":camera_value},
            "master_left_rotate_zoom": {"type":"rotate","horizontal":0,"vertical":0,"zoom":camera_value,"tilt":0,"pan":camera_value,"roll":0},
        }
        return json.dumps(camera_mappings.get(camera, camera_mappings["none"]))


class Comfly_kling_multi_image2video:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "model_name": (["kling-v1-6"], {"default": "kling-v1-6"}),
                "mode": (["std", "pro"], {"default": "std"}),
                "duration": (["5", "10"], {"default": "5"}),
                "aspect_ratio": (["16:9", "9:16", "1:1"], {"default": "16:9"}),
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
            },
            "optional": {
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
                "image4": ("IMAGE",),
                "api_key": ("STRING", {"default": ""}),
                "max_retries": ("INT", {"default": 10, "min": 1, "max": 30}),
                "initial_timeout": ("INT", {"default": 600, "min": 30, "max": 900}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647})
            }
        }

    RETURN_TYPES = (IO.VIDEO, "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("video", "video_url", "task_id", "video_id", "response")
    FUNCTION = "generate_video"
    CATEGORY = "Comfly/Comfly_kling"

    def __init__(self):
        self.api_key = get_config().get('api_key', '')
        self.timeout = 300
        self.session = requests.Session()
        retry_strategy = requests.packages.urllib3.util.retry.Retry(
            total=5,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "POST"]
        )
        adapter = requests.adapters.HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

    def get_headers(self):
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
    
    def image_to_base64(self, image_tensor):
        """Convert tensor to base64 string without data URI prefix"""
        if image_tensor is None:
            return None
            
        try:
            pil_image = tensor2pil(image_tensor)[0]
            buffered = BytesIO()
            pil_image.save(buffered, format="PNG")
            return base64.b64encode(buffered.getvalue()).decode('utf-8')
        except Exception as e:
            print(f"Error converting image to base64: {str(e)}")
            return None
    
    def make_request_with_retry(self, method, url, **kwargs):
        max_retries = kwargs.pop('max_retries', 10)
        initial_timeout = kwargs.pop('initial_timeout', self.timeout)
        
        for attempt in range(1, max_retries + 1):
            current_timeout = min(initial_timeout * (2 ** (attempt - 1)), 900)  
            
            try:
                kwargs['timeout'] = current_timeout
                
                if method.lower() == 'get':
                    response = self.session.get(url, **kwargs)
                else:
                    response = self.session.post(url, **kwargs)
                
                response.raise_for_status()
                return response
            
            except requests.exceptions.Timeout as e:
                if attempt == max_retries:
                    raise
                wait_time = min(2 ** (attempt - 1), 60)  
                time.sleep(wait_time)
            
            except requests.exceptions.ConnectionError as e:
                if attempt == max_retries:
                    raise
                wait_time = min(2 ** (attempt - 1), 60)
                time.sleep(wait_time)
            
            except requests.exceptions.HTTPError as e:
                if e.response.status_code in (400, 401, 403):
                    print(f"Client error: {str(e)}")
                    raise
                if attempt == max_retries:
                    raise
                wait_time = min(2 ** (attempt - 1), 60)
                time.sleep(wait_time)
            
            except Exception as e:
                if attempt == max_retries:
                    raise
                wait_time = min(2 ** (attempt - 1), 60)
                time.sleep(wait_time)
    
    def poll_task_status(self, task_id, max_attempts=100, initial_interval=2, max_interval=60, headers=None, pbar=None):
        attempt = 0
        interval = initial_interval
        last_status = None
        
        while attempt < max_attempts:
            attempt += 1
            
            try:
                status_response = self.make_request_with_retry(
                    'get',
                    f"{baseurl}/kling/v1/videos/multi-image2video/{task_id}",
                    headers=headers
                )
                
                status_result = status_response.json()
                
                if status_result["code"] != 0:
                    print(f"API returned error code: {status_result['code']} - {status_result['message']}")
                    if status_result["code"] in (400, 401, 403): 
                        return {"task_status": "failed", "task_status_msg": status_result["message"]}
                    time.sleep(interval)
                    interval = min(interval * 1.5, max_interval)
                    continue
                
                last_status = status_result["data"]

                if pbar:
                    progress = 0
                    if last_status["task_status"] == "processing":
                        progress = 50
                    elif last_status["task_status"] == "succeed":
                        progress = 100
                    pbar.update_absolute(progress)

                if last_status["task_status"] == "succeed":
                    return last_status
                elif last_status["task_status"] == "failed":
                    return last_status

                if last_status["task_status"] == "processing":
                    interval = min(interval * 1.2, max_interval) 
                else:
                    interval = max(interval * 0.8, initial_interval)  
                    
                time.sleep(interval)
                
            except Exception as e:
                time.sleep(interval)
                interval = min(interval * 2, max_interval)

        if last_status:
            return last_status
        else:
            return {"task_status": "failed", "task_status_msg": "Maximum polling attempts reached without getting a valid status"}
    
    def generate_video(self, prompt, model_name, mode, duration, aspect_ratio, negative_prompt="", 
                  image1=None, image2=None, image3=None, image4=None, api_key="", 
                  max_retries=10, initial_timeout=300, seed=0):
        if api_key.strip():
            self.api_key = api_key
            config = get_config()
            config['api_key'] = api_key
            
        if not self.api_key:
            error_response = {"task_status": "failed", "task_status_msg": "API key not found in Comflyapi.json"}
            return ("", "", "", "", json.dumps(error_response))
            
        pbar = comfy.utils.ProgressBar(100)
        pbar.update_absolute(10)
        
        try:
            image_list = []
            for idx, img in enumerate([image1, image2, image3, image4], 1):
                if img is not None:
                    base64_str = self.image_to_base64(img)
                    if base64_str:
                        image_list.append({"image": base64_str})
                    
            if not image_list:
                error_response = {"task_status": "failed", "task_status_msg": "No valid images provided"}
                return ("", "", "", "", json.dumps(error_response))

            payload = {
                "model_name": model_name,
                "image_list": image_list,
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "mode": mode,
                "duration": duration,
                "aspect_ratio": aspect_ratio
            }
            
            if seed > 0:
                payload["seed"] = seed

            headers = self.get_headers()
            pbar.update_absolute(30)
            
            print("Submitting multi-image video generation request...")
            response = self.make_request_with_retry(
                'post',
                f"{baseurl}/kling/v1/videos/multi-image2video",
                headers=headers,
                json=payload,
                max_retries=max_retries,
                initial_timeout=initial_timeout
            )
            
            result = response.json()
            
            if result["code"] != 0:
                error_message = f"API Error: {result['message']}"
                print(error_message)
                error_response = {"task_status": "failed", "task_status_msg": error_message}
                return ("", "", "", "", json.dumps(error_response))
                
            task_id = result["data"]["task_id"]
            pbar.update_absolute(40)
            print(f"Multi-image video generation task submitted. Task ID: {task_id}")

            print("Waiting for video generation to complete...")
            last_status = self.poll_task_status(
                task_id, 
                max_attempts=100,
                initial_interval=2, 
                max_interval=60,
                headers=headers,
                pbar=pbar
            )
            
            if last_status["task_status"] == "succeed":
                pbar.update_absolute(100)
                video_url = last_status["task_result"]["videos"][0]["url"]
                video_id = last_status["task_result"]["videos"][0]["id"]
                
                response_data = {
                    "task_status": "succeed",
                    "task_status_msg": "Video generated successfully",
                    "progress": 100,
                    "video_url": video_url
                }
                
                video_adapter = ComflyVideoAdapter(video_url)
                return (video_adapter, video_url, task_id, video_id, json.dumps(response_data))
            
            elif last_status["task_status"] == "failed":
                error_msg = last_status.get("task_status_msg", "Unknown error")
                error_response = {
                    "task_status": "failed", 
                    "task_status_msg": error_msg,
                }
                print(f"Task failed: {error_msg}")
                return ("", "", task_id, "", json.dumps(error_response))
            
            else:
                error_msg = f"Unexpected task status: {last_status.get('task_status', 'unknown')}"
                error_response = {
                    "task_status": "failed", 
                    "task_status_msg": error_msg,
                }
                print(f"Task error: {error_msg}")
                return ("", "", task_id, "", json.dumps(error_response))
        
        except Exception as e:
            import traceback
            traceback.print_exc()
            error_response = {"task_status": "failed", "task_status_msg": f"Error generating video: {str(e)}"}
            print(f"Error generating video: {str(e)}")
            return ("", "", "", "", json.dumps(error_response))


class Comfly_video_extend:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_id": ("STRING", {"default": "", "multiline": False, "forceInput": True}),
                "prompt": ("STRING", {"default": "", "multiline": True}),
            },
            "optional": {
                "api_key": ("STRING", {"default": ""})
            }
        }

    RETURN_TYPES = (IO.VIDEO, "STRING", "STRING")
    RETURN_NAMES = ("video", "video_id", "response")
    FUNCTION = "extend_video"
    CATEGORY = "Comfly/Comfly_kling"

    def __init__(self):
        self.api_key = get_config().get('api_key', '')
        self.timeout = 300

    def extend_video(self, video_id, prompt="", api_key=""):
        if api_key.strip():
            self.api_key = api_key
            config = get_config()
            config['api_key'] = api_key
            
        if not self.api_key:
            error_response = {"task_status": "failed", "task_status_msg": "API key not found in Comflyapi.json"}
            return ("", "", json.dumps(error_response))
            
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        payload = {
            "video_id": video_id,
            "prompt": prompt
        }
        try:
            response = requests.post(
                f"{baseurl}/kling/v1/videos/video-extend",
                headers=headers,
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            result = response.json()
            if result["code"] != 0:
                error_response = {"task_status": "failed", "task_status_msg": f"API Error: {result['message']}"}
                return ("", "", json.dumps(error_response))
                
            task_id = result["data"]["task_id"]
            pbar = comfy.utils.ProgressBar(100)
            
            last_status = {}
            while True:
                time.sleep(2)
                status_response = requests.get(
                    f"{baseurl}/kling/v1/videos/video-extend/{task_id}",
                    headers=headers,
                    timeout=self.timeout
                )
                status_response.raise_for_status()
                status_result = status_response.json()
                last_status = status_result["data"]
                
                progress = 0
                if status_result["data"]["task_status"] == "processing":
                    progress = 50
                elif status_result["data"]["task_status"] == "succeed":
                    progress = 100
                pbar.update_absolute(progress)
                
                if status_result["data"]["task_status"] == "succeed":
                    video_url = status_result["data"]["task_result"]["videos"][0]["url"]
                    new_video_id = status_result["data"]["task_result"]["videos"][0]["id"]
                    
                    response_data = {
                        "task_status": "succeed",
                        "task_status_msg": "Video extended successfully",
                        "progress": 100,
                        "video_url": video_url
                    }
                    
                    video_adapter = ComflyVideoAdapter(video_url)
                    return (video_adapter, new_video_id, json.dumps(response_data))
                
                elif status_result["data"]["task_status"] == "failed":
                    error_msg = status_result["data"].get("task_status_msg", "Unknown error")
                    error_response = {
                        "task_status": "failed", 
                        "task_status_msg": error_msg,
                    }
                    print(f"Task failed: {error_msg}")
                    return ("", "", json.dumps(error_response))
                    
        except Exception as e:
            error_response = {"task_status": "failed", "task_status_msg": f"Error extending video: {str(e)}"}
            print(f"Error extending video: {str(e)}")
            return ("", "", json.dumps(error_response))


class Comfly_lip_sync:
    @classmethod
    def INPUT_TYPES(cls):
        cls.zh_voices = [
            ["阳光少年", "genshin_vindi2"],
            ["懂事小弟", "zhinen_xuesheng"],
            ["运动少年", "tiyuxi_xuedi"],
            ["青春少女", "ai_shatang"],
            ["温柔小妹", "genshin_klee2"],
            ["元气少女", "genshin_kirara"],
            ["阳光男生", "ai_kaiya"],
            ["幽默小哥", "tiexin_nanyou"],
            ["文艺小哥", "ai_chenjiahao_712"],
            ["甜美邻家", "girlfriend_1_speech02"],
            ["温柔姐姐", "chat1_female_new-3"],
            ["职场女青", "girlfriend_2_speech02"],
            ["活泼男童", "cartoon-boy-07"],
            ["俏皮女童", "cartoon-girl-01"],
            ["稳重老爸", "ai_huangyaoshi_712"],
            ["温柔妈妈", "you_pingjing"],
            ["严肃上司", "ai_laoguowang_712"],
            ["优雅贵妇", "chengshu_jiejie"],
            ["慈祥爷爷", "zhuxi_speech02"],
            ["唠叨爷爷", "uk_oldman3"],
            ["唠叨奶奶", "laopopo_speech02"],
            ["和蔼奶奶", "heainainai_speech02"],
            ["东北老铁", "dongbeilaotie_speech02"],
            ["重庆小伙", "chongqingxiaohuo_speech02"],
            ["四川妹子", "chuanmeizi_speech02"],
            ["潮汕大叔", "chaoshandashu_speech02"],
            ["台湾男生", "ai_taiwan_man2_speech02"],
            ["西安掌柜", "xianzhanggui_speech02"],
            ["天津姐姐", "tianjinjiejie_speech02"],
            ["新闻播报男", "diyinnansang_DB_CN_M_04-v2"],
            ["译制片男", "yizhipiannan-v1"],
            ["元气少女", "guanxiaofang-v2"],
            ["撒娇女友", "tianmeixuemei-v1"],
            ["刀片烟嗓", "daopianyansang-v1"],
            ["乖巧正太", "mengwa-v1"]
        ]
        
        cls.en_voices = [
            ["Sunny", "genshin_vindi2"],
            ["Sage", "zhinen_xuesheng"],
            ["Ace", "AOT"],
            ["Blossom", "ai_shatang"],
            ["Peppy", "genshin_klee2"],
            ["Dove", "genshin_kirara"],
            ["Shine", "ai_kaiya"],
            ["Anchor", "oversea_male1"],
            ["Lyric", "ai_chenjiahao_712"],
            ["Melody", "girlfriend_4_speech02"],
            ["Tender", "chat1_female_new-3"],
            ["Siren", "chat_0407_5-1"],
            ["Zippy", "cartoon-boy-07"],
            ["Bud", "uk_boy1"],
            ["Sprite", "cartoon-girl-01"],
            ["Candy", "PeppaPig_platform"],
            ["Beacon", "ai_huangzhong_712"],
            ["Rock", "ai_huangyaoshi_712"],
            ["Titan", "ai_laoguowang_712"],
            ["Grace", "chengshu_jiejie"],
            ["Helen", "you_pingjing"],
            ["Lore", "calm_story1"],
            ["Crag", "uk_man2"],
            ["Prattle", "laopopo_speech02"],
            ["Hearth", "heainainai_speech02"],
            ["The Reader", "reader_en_m-v1"],
            ["Commercial Lady", "commercial_lady_en_f-v1"]
        ]
        
        return {
            "required": {
                "video_id": ("STRING", {"default": "", "multiline": False, "forceInput": True}),
                "task_id": ("STRING", {"default": "", "multiline": False, "forceInput": True}),
                "mode": (["text2video", "audio2video"], {"default": "text2video"}),
                "text": ("STRING", {"multiline": True, "default": ""}),
                "voice_language": (["zh", "en"], {"default": "zh"}),
                "zh_voice": ([name for name, _ in cls.zh_voices], {"default": cls.zh_voices[0][0]}),
                "en_voice": ([name for name, _ in cls.en_voices], {"default": cls.en_voices[0][0]}),
                "voice_speed": ("FLOAT", {"default": 1.0, "min": 0.8, "max": 2.0, "step": 0.1}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647})
            },
            "optional": {
                "api_key": ("STRING", {"default": ""}),
                "video_url": ("STRING", {"default": ""}),
                "audio_type": (["file", "url"], {"default": "file"}),
                "audio_file": ("STRING", {"default": ""}),
                "audio_url": ("STRING", {"default": ""})
            }
        }

    RETURN_TYPES = (IO.VIDEO, "STRING", "STRING", "STRING")
    RETURN_NAMES = ("video", "video_url", "task_id", "response")
    FUNCTION = "process_lip_sync"
    CATEGORY = "Comfly/Comfly_kling"

    def __init__(self):
        self.api_key = get_config().get('api_key', '')
        self.timeout = 300
        self.zh_voice_map = {name: voice_id for name, voice_id in self.__class__.zh_voices}
        self.en_voice_map = {name: voice_id for name, voice_id in self.__class__.en_voices}
        
    def process_lip_sync(self, video_id, task_id, mode, text, voice_language, zh_voice, en_voice, voice_speed, seed=0,
                    video_url="", audio_type="file", audio_file="", audio_url="", api_key=""):
    
        if api_key.strip():
            self.api_key = api_key
            config = get_config()
            config['api_key'] = api_key
            
        if not self.api_key:
            error_response = {"task_status": "failed", "task_status_msg": "API key not found in Comflyapi.json"}
            return ("", "", "", json.dumps(error_response))
        
        if voice_language == "zh":
            voice_id = self.zh_voice_map.get(zh_voice, "")
        else:
            voice_id = self.en_voice_map.get(en_voice, "")
                
        headers = {
            "Content-Type": "application/json", 
            "Authorization": f"Bearer {self.api_key}"
        }
        payload = {
            "input": {
                "task_id": task_id,
                "mode": mode,
                "video_id": video_id if video_id else None,
                "video_url": video_url if video_url else None,
                "text": text if mode == "text2video" else None,
                "voice_id": voice_id if mode == "text2video" else None,
                "voice_language": voice_language if mode == "text2video" else None,
                "voice_speed": voice_speed if mode == "text2video" else None,
                "audio_type": audio_type if mode == "audio2video" else None,
                "audio_file": audio_file if mode == "audio2video" and audio_type == "file" else None,
                "audio_url": audio_url if mode == "audio2video" and audio_type == "url" else None,
                "seed": seed
            }
        }
        try:
            pbar = comfy.utils.ProgressBar(100)
            pbar.update_absolute(5)  
            response = requests.post(
                f"{baseurl}/kling/v1/videos/lip-sync",
                headers=headers,
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            result = response.json()
            if result["code"] != 0:
                error_response = {"task_status": "failed", "task_status_msg": f"API Error: {result['message']}"}
                return ("", "", "", json.dumps(error_response))
                    
            task_id = result["data"]["task_id"]
            pbar.update_absolute(10) 
                
            last_status = {}
            while True:
                time.sleep(2)
                status_response = requests.get(
                    f"{baseurl}/kling/v1/videos/lip-sync/{task_id}",
                    headers=headers,
                    timeout=self.timeout
                )
                status_response.raise_for_status()
                status_result = status_response.json()
                last_status = status_result["data"]
                    
                if status_result["data"]["task_status"] == "processing":
                    progress = status_result["data"].get("progress", 50)  
                    pbar.update_absolute(progress)
                elif status_result["data"]["task_status"] == "succeed":
                    pbar.update_absolute(100)  
                    video_url = status_result["data"]["task_result"]["videos"][0]["url"]

                    response_data = {
                        "task_status": "succeed",
                        "task_status_msg": "Lip sync completed successfully",
                        "progress": 100,
                        "video_url": video_url
                    }
                    
                    video_adapter = ComflyVideoAdapter(video_url)
                    return (video_adapter, video_url, task_id, json.dumps(response_data))
                        
                elif status_result["data"]["task_status"] == "failed":
                    error_msg = status_result["data"].get("task_status_msg", "Unknown error")
                    error_response = {
                        "task_status": "failed", 
                        "task_status_msg": error_msg,
                    }
                    print(f"Task failed: {error_msg}")
                    return ("", "", task_id, json.dumps(error_response))
                        
        except Exception as e:
            error_response = {"task_status": "failed", "task_status_msg": f"Error in lip sync process: {str(e)}"}
            print(f"Error in lip sync process: {str(e)}")
            return ("", "", "", json.dumps(error_response))