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

from ..utils import pil2tensor, tensor2pil, ComflyVideoAdapter, create_audio_object
from ..comfly_config import get_config, save_config, baseurl
from comfy.comfy_types import IO


class Comfly_MiniMax_video:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "model": (["MiniMax-Hailuo-02", "T2V-01", "T2V-01-Director", "I2V-01-Director", "I2V-01-live", "I2V-01", "S2V-01"], {"default": "MiniMax-Hailuo-02"}),
                "duration": (["6", "10"], {"default": "6"}),
                "resolution": (["720P","768P", "1080P"], {"default": "768P"}),
            },
            "optional": {
                "api_key": ("STRING", {"default": ""}),
                "prompt_optimizer": ("BOOLEAN", {"default": True}),
                "fast_pretreatment": ("BOOLEAN", {"default": False}),
                "first_frame_image": ("IMAGE",),
                "last_frame_image": ("IMAGE",),
                "subject_reference": ("IMAGE",),  
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647}),
            }
        }
    
    RETURN_TYPES = (IO.VIDEO, "STRING", "STRING")
    RETURN_NAMES = ("video", "task_id", "response")
    FUNCTION = "generate_video"
    CATEGORY = "Comfly/MiniMax"

    def __init__(self):
        self.api_key = get_config().get('api_key', '')
        self.timeout = 600
        self.api_endpoint = f"{baseurl}/minimax/v1/video_generation"
        self.query_endpoint = f"{baseurl}/minimax/v1/query/video_generation"

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
    
    def generate_video(self, prompt, model="MiniMax-Hailuo-02", duration="6", resolution="768P", 
               prompt_optimizer=True, fast_pretreatment=False, first_frame_image=None, last_frame_image=None,
               subject_reference=None, api_key="", seed=0):
        if api_key.strip():
            self.api_key = api_key
            config = get_config()
            config['api_key'] = api_key
            
        if not self.api_key:
            error_response = {"status": "error", "message": "API key not provided or not found in config"}
            return (None, "", json.dumps(error_response))
            
        try:
            pbar = comfy.utils.ProgressBar(100)
            pbar.update_absolute(10)

            payload = {
                "model": model,
                "prompt": prompt,
                "duration": int(duration),
                "resolution": resolution,
                "prompt_optimizer": prompt_optimizer
            }

            if seed > 0:
                payload["seed"] = seed

            if model == "MiniMax-Hailuo-02":
                payload["fast_pretreatment"] = fast_pretreatment

                if fast_pretreatment and last_frame_image is not None:
                    print("Note: fast_pretreatment is disabled when last_frame_image is provided")
                    payload["fast_pretreatment"] = False

            if model in ["T2V-01", "T2V-01-Director"]:
                if first_frame_image is not None or last_frame_image is not None:
                    print(f"Warning: Model {model} only supports text-to-video. Image inputs will be ignored.")
                
            elif model in ["I2V-01-Director", "I2V-01-live", "I2V-01"]:
                if first_frame_image is None:
                    print(f"Warning: Model {model} requires first_frame_image for image-to-video generation.")
                if last_frame_image is not None:
                    print(f"Warning: Model {model} doesn't support last_frame_image. It will be ignored.")

            if first_frame_image is not None and model != "T2V-01" and model != "T2V-01-Director":
                image_base64 = self.image_to_base64(first_frame_image)
                if image_base64:
                    payload["first_frame_image"] = f"data:image/png;base64,{image_base64}"

            if last_frame_image is not None and model == "MiniMax-Hailuo-02":
                image_base64 = self.image_to_base64(last_frame_image)
                if image_base64:
                    payload["last_frame_image"] = f"data:image/png;base64,{image_base64}"

            if model == "S2V-01" and subject_reference is not None:
                image_base64 = self.image_to_base64(subject_reference)
                if image_base64:
                    payload["subject_reference"] = {
                        "type": "character",
                        "image": [f"data:image/png;base64,{image_base64}"]
                    }
            
            response = requests.post(
                self.api_endpoint,
                headers=self.get_headers(),
                json=payload,
                timeout=self.timeout
            )
            
            pbar.update_absolute(30)
            
            if response.status_code != 200:
                error_message = f"API Error: {response.status_code} - {response.text}"
                print(error_message)
                return (None, "", json.dumps({"status": "error", "message": error_message}))
                
            result = response.json()
            
            if "base_resp" not in result or result["base_resp"]["status_code"] != 0:
                error_message = f"API returned error: {result.get('base_resp', {}).get('status_msg', 'Unknown error')}"
                print(error_message)
                return (None, "", json.dumps({"status": "error", "message": error_message}))
                
            task_id = result.get("task_id")
            if not task_id:
                error_message = "No task ID returned from API"
                print(error_message)
                return (None, "", json.dumps({"status": "error", "message": error_message}))
            
            pbar.update_absolute(40)
            print(f"Video generation task submitted. Task ID: {task_id}")

            max_attempts = 120  
            attempts = 0
            file_id = None
            video_url = None
            
            while attempts < max_attempts:
                time.sleep(10)  
                attempts += 1
                
                try:
                    status_response = requests.get(
                        f"{self.query_endpoint}?task_id={task_id}",
                        headers=self.get_headers(),
                        timeout=self.timeout
                    )
                    
                    if status_response.status_code != 200:
                        print(f"Error checking status: {status_response.status_code} - {status_response.text}")
                        continue
                        
                    status_result = status_response.json()
                    
                    if "base_resp" not in status_result or status_result["base_resp"]["status_code"] != 0:
                        print(f"Error in status response: {status_result.get('base_resp', {}).get('status_msg', 'Unknown error')}")
                        continue
                    
                    status = status_result.get("status", "")
                    
                    progress_value = min(80, 40 + (attempts * 40 // max_attempts))
                    pbar.update_absolute(progress_value)
                    
                    if status == "Success":
                        file_id = status_result.get("file_id")
                        if file_id:
                            video_retrieval_url = f"{baseurl}/minimax/v1/files/retrieve?file_id={file_id}"
                            file_response = requests.get(
                                video_retrieval_url,
                                headers=self.get_headers(),
                                timeout=self.timeout
                            )
                            
                            if file_response.status_code == 200:
                                file_data = file_response.json()
                                if "file" in file_data and "download_url" in file_data["file"]:
                                    video_url = file_data["file"]["download_url"]
                                    break
                                else:
                                    video_url = f"{baseurl}/minimax/v1/file?file_id={file_id}"
                                    break
                            else:
                                video_url = f"{baseurl}/minimax/v1/file?file_id={file_id}"
                                break
                    elif status == "Failed":
                        error_message = f"Video generation failed: {status_result.get('base_resp', {}).get('status_msg', 'Unknown error')}"
                        print(error_message)
                        return (None, task_id, json.dumps({"status": "error", "message": error_message}))
                    
                except Exception as e:
                    print(f"Error checking generation status: {str(e)}")
            
            if not file_id:
                error_message = "Failed to retrieve file_id after multiple attempts"
                print(error_message)
                return (None, task_id, json.dumps({"status": "error", "message": error_message}))
                
            if not video_url:
                video_url = f"{baseurl}/minimax/v1/file?file_id={file_id}"
            
            pbar.update_absolute(90)
            print(f"Video generation completed. URL: {video_url}")
            
            video_adapter = ComflyVideoAdapter(video_url)
            
            response_data = {
                "status": "success",
                "task_id": task_id,
                "file_id": file_id,
                "video_url": video_url,
                "width": status_result.get("video_width", 0),
                "height": status_result.get("video_height", 0)
            }
            
            pbar.update_absolute(100)
            return (video_adapter, task_id, json.dumps(response_data))
            
        except Exception as e:
            error_message = f"Error generating video: {str(e)}"
            print(error_message)
            import traceback
            traceback.print_exc()
            return (None, "", json.dumps({"status": "error", "message": error_message}))