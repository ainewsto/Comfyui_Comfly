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

from ..utils import create_audio_object
from ..comfly_config import get_config, save_config, baseurl
from comfy.comfy_types import IO


class Comfly_suno_description:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "title": ("STRING", {"default": ""}),
                "description_prompt": ("STRING", {"multiline": True}),
                "version": (["v3.0", "v3.5", "v4", "v4.5", "v4.5+", "v5"], {"default": "v4.5"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647}),
                "make_instrumental": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "apikey": ("STRING", {"default": ""})
            }
        }
    
    RETURN_TYPES = ("AUDIO", "AUDIO", "STRING", "STRING", "STRING", "STRING", "STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("audio1", "audio2", "audio_url1", "audio_url2", "prompt", "task_id", "response", "clip_id1", "clip_id2", "tags", "title")
    FUNCTION = "generate_music"
    CATEGORY = "Comfly/Suno"

    def __init__(self):
        self.api_key = get_config().get('api_key', '')
        self.timeout = 300
        
    def get_headers(self):
        return {
            "Content-Type": "application/json",
            "Authorization": "Bearer " + self.api_key
        }
    
    def generate_music(self, title, description_prompt, version="v4.5", seed=0, make_instrumental=False, apikey=""):
        if apikey.strip():
            self.api_key = apikey
            config = get_config()
            config['api_key'] = apikey
            
        if not self.api_key:
            error_message = "API key not found in Comflyapi.json"
            print(error_message)
            empty_audio = create_audio_object("")
            return (empty_audio, empty_audio, "", "", "", "", error_message, "", "", "", "")
        
        mv_mapping = {
            "v3.0": "chirp-v3.0",
            "v3.5": "chirp-v3.5", 
            "v4": "chirp-v4",
            "v4.5": "chirp-auk",
            "v4.5+": "chirp-bluejay",
            "v5": "chirp-crow"
        }
        
        mv = mv_mapping.get(version, "chirp-auk")
            
        try:
            payload = {
                "gpt_description_prompt": description_prompt,
                "make_instrumental": make_instrumental,
                "mv": mv,
                "prompt": ""
            }
            if seed > 0:
                payload["seed"] = seed
            
            pbar = comfy.utils.ProgressBar(100)
            pbar.update_absolute(10)
           
            response = requests.post(
                f"{baseurl}/suno/generate/description-mode",
                headers=self.get_headers(),
                json=payload,
                timeout=self.timeout
            )
            
            pbar.update_absolute(20)
           
            if response.status_code != 200:
                error_message = f"API Error: {response.status_code} - {response.text}"
                print(error_message)
                empty_audio = create_audio_object("")
                return (empty_audio, empty_audio, "", "", "", "", error_message, "", "", "", "")
                
            result = response.json()
           
            if "id" not in result:
                error_message = "No task ID in response"
                print(error_message)
                empty_audio = create_audio_object("")
                return (empty_audio, empty_audio, "", "", "", "", error_message, "", "", "", "")
                
            task_id = result.get("id")
            
            if "clips" not in result or len(result["clips"]) < 2:
                error_message = "Expected at least 2 clips in the response"
                print(error_message)
                empty_audio = create_audio_object("")
                return (empty_audio, empty_audio, "", "", "", task_id, error_message, "", "", "", "")
                
            clip_ids = [clip["id"] for clip in result["clips"]]
            if len(clip_ids) < 2:
                error_message = "Expected at least 2 clip IDs"
                print(error_message)
                empty_audio = create_audio_object("")
                return (empty_audio, empty_audio, "", "", "", task_id, error_message, "", "", "", "")
                
            pbar.update_absolute(30)
            max_attempts = 30
            attempts = 0
            final_clips = []
            generated_prompt = ""
            extracted_tags = ""
            generated_title = ""  
           
            while attempts < max_attempts and len(final_clips) < 2:
                time.sleep(5)
                attempts += 1
                
                try:
                    clip_response = requests.get(
                        f"{baseurl}/suno/feed/{','.join(clip_ids)}",
                        headers=self.get_headers(),
                        timeout=self.timeout
                    )
                    
                    if clip_response.status_code != 200:
                        continue
                        
                    clips_data = clip_response.json()
                   
                    progress = min(80, 30 + (attempts * 50 // max_attempts))
                    pbar.update_absolute(progress)
                    complete_clips = [
                        clip for clip in clips_data 
                        if clip.get("status") == "complete" and (clip.get("audio_url") or clip.get("state") == "succeeded")
                    ]
                    for clip in complete_clips:
                        if clip.get("id") in clip_ids and clip not in final_clips:
                            final_clips.append(clip)
                            if not generated_prompt and "prompt" in clip:
                                generated_prompt = clip["prompt"]
                            if not extracted_tags and "tags" in clip:
                                extracted_tags = clip["tags"]
                            if not generated_title and "title" in clip and clip["title"]:
                                generated_title = clip["title"]
                    
                    if len(final_clips) >= 2:
                        break
                        
                except Exception as e:
                    print(f"[Description Debug] Error checking clip status: {str(e)}")
            
            if len(final_clips) < 2:
                error_message = f"Only received {len(final_clips)} complete clips after {max_attempts} attempts"
                print(error_message)
                
                if not final_clips:
                    empty_audio = create_audio_object("")
                    return (empty_audio, empty_audio, "", "", "", task_id, error_message, "", "", "", "")
                
            final_title = generated_title if generated_title else title
            for clip in final_clips:
                if "title" not in clip or not clip["title"]:
                    clip["title"] = final_title
                    
            audio_urls = []
            clip_id_values = []
            
            for clip in final_clips[:2]:  
                audio_url = ""
                if "audio_url" in clip and clip["audio_url"]:
                    audio_url = clip["audio_url"]
                elif "cdn1.suno.ai" in str(clip):
                    match = re.search(r'https://cdn1\.suno\.ai/[^"\']+\.mp3', str(clip))
                    if match:
                        audio_url = match.group(0)
                
                if audio_url:
                    print(f"Found audio URL: {audio_url}")
                    audio_urls.append(audio_url)
                else:
                    print(f"No audio URL found in clip")
                    audio_urls.append("")
                    
                clip_id_value = clip.get("id", "")
                if clip_id_value:
                    clip_id_values.append(clip_id_value)
                else:
                    clip_id_values.append("")
                
            while len(audio_urls) < 2:
                audio_urls.append("")
                
            while len(clip_id_values) < 2:
                clip_id_values.append("")
            audio_objects = [create_audio_object(url) for url in audio_urls[:2]]
            while len(audio_objects) < 2:
                audio_objects.append(create_audio_object(""))
            
            pbar.update_absolute(100)
            
            response_info = {
                "status": "success",
                "prompt": generated_prompt,
                "title": final_title, 
                "version": version,
                "seed": seed if seed > 0 else "auto",
                "make_instrumental": make_instrumental,
                "clips_generated": len(final_clips),
                "tags": extracted_tags
            }
            return (
                audio_objects[0],
                audio_objects[1],
                audio_urls[0],
                audio_urls[1],
                generated_prompt,
                task_id,
                json.dumps(response_info),
                clip_id_values[0],
                clip_id_values[1],
                extracted_tags,
                final_title  
            )
                
        except Exception as e:
            error_message = f"Error generating music: {str(e)}"
            print(error_message)
            import traceback
            traceback.print_exc()
            empty_audio = create_audio_object("")
            return (empty_audio, empty_audio, "", "", "", "", error_message, "", "", "", "")


class Comfly_suno_lyrics:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647}),
            },
            "optional": {
                "apikey": ("STRING", {"default": ""})
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("lyrics", "response", "title", "tags")
    FUNCTION = "generate_lyrics"
    CATEGORY = "Comfly/Suno"

    def __init__(self):
        self.api_key = get_config().get('api_key', '')
        self.timeout = 300
        
    def get_headers(self):
        return {
            "Content-Type": "application/json",
            "Authorization": "Bearer " + self.api_key
        }
        
    def generate_lyrics(self, prompt, seed=0, apikey=""):
        if apikey.strip():
            self.api_key = apikey
            config = get_config()
            config['api_key'] = apikey
            
        if not self.api_key:
            error_message = "API key not found in Comflyapi.json"
            print(error_message)
            return ("", json.dumps({"status": "error", "message": error_message}), "", "")
            
        try:
            payload = {"prompt": prompt}
            
            if seed > 0:
                payload["seed"] = seed

            response = requests.post(
                f"{baseurl}/suno/generate/lyrics/",
                headers=self.get_headers(),
                json=payload,
                timeout=self.timeout
            )
            
            if response.status_code != 200:
                error_message = f"API Error: {response.status_code} - {response.text}"
                print(error_message)
                return ("", json.dumps({"status": "error", "message": error_message}), "", "")
                
            result = response.json()
            
            if "id" not in result:
                error_message = "No task ID in response"
                print(error_message)
                return ("", json.dumps({"status": "error", "message": error_message}), "", "")
                
            task_id = result.get("id")

            max_attempts = 30
            attempts = 0
            lyrics_text = ""
            generated_title = ""  
            tags = ""
            
            while attempts < max_attempts:
                time.sleep(2)
                attempts += 1
                
                try:
                    lyrics_response = requests.get(
                        f"{baseurl}/suno/lyrics/{task_id}",
                        headers=self.get_headers(),
                        timeout=self.timeout
                    )
                    
                    if lyrics_response.status_code != 200:
                        continue
                        
                    lyrics_data = lyrics_response.json()
                    
                    if lyrics_data.get("status") == "complete" or lyrics_data.get("status") == "succeed":
                        lyrics_text = lyrics_data.get("text", "")
                        generated_title = lyrics_data.get("title", "")  
                        tags = lyrics_data.get("tags", "")
                        break
                        
                except Exception as e:
                    print(f"Error checking lyrics status: {str(e)}")
            
            if not lyrics_text:
                error_message = f"Failed to generate lyrics after {max_attempts} attempts"
                print(error_message)
                return ("", json.dumps({"status": "error", "message": error_message}), "", "")
            
            success_response = {
                "status": "success",
                "title": generated_title,  
                "prompt": prompt,
                "seed": seed if seed > 0 else "auto",
                "tags": tags
            }
            
            return (lyrics_text, json.dumps(success_response), generated_title, tags)  
                
        except Exception as e:
            error_message = f"Error generating lyrics: {str(e)}"
            print(error_message)
            return ("", json.dumps({"status": "error", "message": error_message}), "", "")



class Comfly_suno_custom:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "title": ("STRING", {"default": ""}),
                "version": (["v3.0", "v3.5", "v4", "v4.5", "v4.5+", "v5"], {"default": "v4.5"}),
                "prompt": ("STRING", {"multiline": True}), 
                "tags": ("STRING", {"default": ""}),  
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647}),
            },
            "optional": {
                "apikey": ("STRING", {"default": ""})
            }
        }
    
    RETURN_TYPES = ("AUDIO", "AUDIO", "STRING", "STRING", "STRING", "STRING", 
                   "STRING", "STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("audio1", "audio2", "audio_url1", "audio_url2", "task_id", "response",
                   "clip_id1", "clip_id2", "image_large_url1", "image_large_url2", "tags", "title")
    FUNCTION = "generate_music"
    CATEGORY = "Comfly/Suno"

    def __init__(self):
        self.api_key = get_config().get('api_key', '')
        self.timeout = 300

    def get_headers(self):
        return {
            "Content-Type": "application/json",
            "Authorization": "Bearer " + self.api_key
        }
    
    def generate_music(self, title, version="v4.5", prompt="", tags="", seed=0, apikey=""):
        if apikey.strip():
            self.api_key = apikey
            config = get_config()
            config['api_key'] = apikey
            
        if not self.api_key:
            error_message = "API key not found in Comflyapi.json"
            print(error_message)
            empty_audio = create_audio_object("")
            return (empty_audio, empty_audio, "", "", "", error_message, 
                "", "", "", "", "", "")
        
        mv_mapping = {
            "v3.0": "chirp-v3.0",
            "v3.5": "chirp-v3.5", 
            "v4": "chirp-v4",
            "v4.5": "chirp-auk",
            "v4.5+": "chirp-bluejay",
            "v5": "chirp-crow"
        }
        
        mv = mv_mapping.get(version, "chirp-auk")
            
        try:
            payload = {
                "prompt": prompt,
                "tags": tags,
                "mv": mv,
                "title": title  
            }
            if seed > 0:
                payload["seed"] = seed
            
            pbar = comfy.utils.ProgressBar(100)
            pbar.update_absolute(10)
           
            response = requests.post(
                f"{baseurl}/suno/generate",
                headers=self.get_headers(),
                json=payload,
                timeout=self.timeout
            )
            
            pbar.update_absolute(20)
            
            if response.status_code != 200:
                error_message = f"API Error: {response.status_code} - {response.text}"
                print(error_message)
                empty_audio = create_audio_object("")
                return (empty_audio, empty_audio, "", "", "", error_message, 
                    "", "", "", "", "", "")
                
            result = response.json()
            
            if "id" not in result:
                error_message = "No task ID in response"
                print(error_message)
                empty_audio = create_audio_object("")
                return (empty_audio, empty_audio, "", "", "", error_message, 
                    "", "", "", "", "", "")
                
            task_id = result.get("id")
            
            if "clips" not in result or len(result["clips"]) < 2:
                error_message = "Expected at least 2 clips in the response"
                print(error_message)
                empty_audio = create_audio_object("")
                return (empty_audio, empty_audio, "", "", task_id, error_message, 
                    "", "", "", "", "", "")
                
            clip_ids = [clip["id"] for clip in result["clips"]]
            if len(clip_ids) < 2:
                error_message = "Expected at least 2 clip IDs"
                print(error_message)
                empty_audio = create_audio_object("")
                return (empty_audio, empty_audio, "", "", task_id, error_message, 
                    "", "", "", "", "", "")
                
            pbar.update_absolute(30)
            max_attempts = 30
            attempts = 0
            final_clips = []
            final_tags = tags  
            generated_title = ""  
            
            while attempts < max_attempts and len(final_clips) < 2:
                time.sleep(5)
                attempts += 1
                
                try:
                    clip_response = requests.get(
                        f"{baseurl}/suno/feed/{','.join(clip_ids)}",
                        headers=self.get_headers(),
                        timeout=self.timeout
                    )
                    
                    if clip_response.status_code != 200:
                        continue
                        
                    clips_data = clip_response.json()
                    
                    progress = min(80, 30 + (attempts * 50 // max_attempts))
                    pbar.update_absolute(progress)
                    complete_clips = [
                        clip for clip in clips_data 
                        if clip.get("status") == "complete" and (clip.get("audio_url") or clip.get("state") == "succeeded")
                    ]
                    
                    for clip in complete_clips:
                        if clip.get("id") in clip_ids and clip not in final_clips:
                            final_clips.append(clip)
                            if "tags" in clip and clip["tags"]:
                                final_tags = clip["tags"]
                            if "title" in clip and clip["title"]:
                                generated_title = clip["title"]
                    
                    if len(final_clips) >= 2:
                        break
                        
                except Exception as e:
                    print(f"[Custom Debug] Error checking clip status: {str(e)}")
            
            if len(final_clips) < 2:
                error_message = f"Only received {len(final_clips)} complete clips after {max_attempts} attempts"
                print(error_message)
                
                if not final_clips:
                    empty_audio = create_audio_object("")
                    return (empty_audio, empty_audio, "", "", task_id, error_message, 
                        "", "", "", "", "", "")
            final_title = generated_title if generated_title else title
                    
            audio_urls = []
            clip_id_values = []
            image_large_urls = []
            
            for clip in final_clips[:2]:
                audio_url = ""
                if "audio_url" in clip and clip["audio_url"]:
                    audio_url = clip["audio_url"]
                elif "cdn1.suno.ai" in str(clip):
                    match = re.search(r'https://cdn1\.suno\.ai/[^"\']+\.mp3', str(clip))
                    if match:
                        audio_url = match.group(0)
                
                audio_urls.append(audio_url if audio_url else "")
                clip_id = clip.get("clip_id", clip.get("id", ""))
                clip_id_values.append(clip_id)
                image_large_url = clip.get("image_large_url", "")
                image_large_urls.append(image_large_url)
            while len(audio_urls) < 2:
                audio_urls.append("")
                
            while len(clip_id_values) < 2:
                clip_id_values.append("")
                
            while len(image_large_urls) < 2:
                image_large_urls.append("")
            audio_objects = [create_audio_object(url) for url in audio_urls[:2]]
            while len(audio_objects) < 2:
                audio_objects.append(create_audio_object(""))
            
            pbar.update_absolute(100)
            
            response_info = {
                "status": "success",
                "title": final_title,
                "version": version,
                "seed": seed if seed > 0 else "auto",
                "clips_generated": len(final_clips),
                "tags": final_tags
            }
            return (
                audio_objects[0],  
                audio_objects[1],  
                audio_urls[0],     
                audio_urls[1],     
                task_id,
                json.dumps(response_info),
                clip_id_values[0],
                clip_id_values[1],
                image_large_urls[0],
                image_large_urls[1],
                final_tags,
                final_title  
            )
                
        except Exception as e:
            error_message = f"Error generating music: {str(e)}"
            print(error_message)
            import traceback
            traceback.print_exc()
            empty_audio = create_audio_object("")
            return (empty_audio, empty_audio, "", "", "", error_message, "", "", "", "", "", "")
        

class Comfly_suno_upload:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
            },
            "optional": {
                "api_key": ("STRING", {"default": ""}),
                "upload_filename": ("STRING", {"default": "audio.mp3"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647}),
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("clip_id", "title", "tags", "lyrics", "response")
    FUNCTION = "upload_audio"
    CATEGORY = "Comfly/Suno"

    def __init__(self):
        self.api_key = get_config().get('api_key', '')
        self.timeout = 300

    def get_headers(self):
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

    def upload_audio(self, audio, api_key="", upload_filename="audio.mp3", seed=0):
        if api_key.strip():
            self.api_key = api_key
            config = get_config()
            config['api_key'] = api_key
            
        if not self.api_key:
            error_message = "API key not found in Comflyapi.json"
            print(error_message)
            return ("", "", "", "", error_message)
            
        try:
            pbar = comfy.utils.ProgressBar(100)
            pbar.update_absolute(10)
 
            extension = upload_filename.split('.')[-1] if '.' in upload_filename else "mp3"
            payload = {"extension": extension}

            if seed > 0:
                payload["seed"] = seed
            
            response = requests.post(
                f"{baseurl}/suno/uploads/audio",
                headers=self.get_headers(),
                json=payload,
                timeout=self.timeout
            )
            
            if response.status_code != 200:
                error_message = f"Failed to get upload URL: {response.status_code} - {response.text}"
                print(error_message)
                return ("", "", "", "", error_message)
                
            upload_data = response.json()
            upload_id = upload_data["id"]
            upload_url = upload_data["url"]
            fields = upload_data["fields"]
            
            pbar.update_absolute(30)

            waveform = audio["waveform"]
            sample_rate = audio["sample_rate"]

            if len(waveform.shape) == 3:
                waveform = waveform.squeeze(0)

            temp_file = io.BytesIO()
            torchaudio.save(temp_file, waveform, sample_rate, format="mp3")
            temp_file.seek(0)
            audio_data = temp_file.read()

            files = {
                'Content-Type': ('', fields['Content-Type']),
                'key': ('', fields['key']),
                'AWSAccessKeyId': ('', fields['AWSAccessKeyId']),
                'policy': ('', fields['policy']),
                'signature': ('', fields['signature']),
                'file': (upload_filename, audio_data, 'audio/mpeg')
            }
            
            upload_response = requests.post(upload_url, files=files, timeout=self.timeout)
            
            if upload_response.status_code != 204:
                error_message = f"Failed to upload audio: {upload_response.status_code}"
                print(error_message)
                return ("", "", "", "", error_message)
                
            pbar.update_absolute(50)

            finish_payload = {
                "upload_type": "file_upload",
                "upload_filename": upload_filename
            }
            
            finish_response = requests.post(
                f"{baseurl}/suno/uploads/audio/{upload_id}/upload-finish",
                headers=self.get_headers(),
                json=finish_payload,
                timeout=self.timeout
            )
            
            if finish_response.status_code != 200:
                error_message = f"Failed to finish upload: {finish_response.status_code}"
                print(error_message)
                return ("", "", "", "", error_message)
                
            pbar.update_absolute(60)

            max_attempts = 20
            attempts = 0
            clip_id = ""
            title = ""
            tags = ""
            lyrics = ""
            
            while attempts < max_attempts:
                time.sleep(2)
                attempts += 1
                
                status_response = requests.get(
                    f"{baseurl}/suno/uploads/audio/{upload_id}",
                    headers=self.get_headers(),
                    timeout=self.timeout
                )
                
                if status_response.status_code != 200:
                    continue
                    
                status_data = status_response.json()
                status = status_data.get("status", "")
                
                pbar.update_absolute(60 + (attempts * 20 // max_attempts))
                
                if status == "complete":
                    init_response = requests.post(
                        f"{baseurl}/suno/uploads/audio/{upload_id}/initialize-clip",
                        headers=self.get_headers(),
                        json={},
                        timeout=self.timeout
                    )
                    
                    if init_response.status_code == 200:
                        init_data = init_response.json()
                        clip_id = init_data.get("clip_id", "")

                        if clip_id:
                            try:
                                clip_detail_response = requests.get(
                                    f"{baseurl}/suno/feed/{clip_id}",
                                    headers=self.get_headers(),
                                    timeout=self.timeout
                                )
                                
                                if clip_detail_response.status_code == 200:
                                    clip_details = clip_detail_response.json()
                                    if isinstance(clip_details, list) and len(clip_details) > 0:
                                        clip_info = clip_details[0]
                                    else:
                                        clip_info = clip_details
                                    
                                    title = clip_info.get("title", "")
                                    tags = clip_info.get("tags", "")
                                    lyrics = clip_info.get("metadata", {}).get("prompt", "") or clip_info.get("prompt", "")
                                    
                            except Exception as e:
                                print(f"Error fetching clip details: {str(e)}")
                        
                        pbar.update_absolute(100)
                        
                        response_info = {
                            "status": "success",
                            "upload_id": upload_id,
                            "clip_id": clip_id,
                            "title": title,
                            "tags": tags,
                            "lyrics": lyrics,
                            "seed": seed if seed > 0 else "auto",
                            "upload_filename": upload_filename
                        }
                        
                        print(f"Audio uploaded successfully. Clip ID: {clip_id}")
                        return (clip_id, title, tags, lyrics, json.dumps(response_info))
                    else:
                        error_message = f"Failed to initialize clip: {init_response.status_code}"
                        print(error_message)
                        return ("", "", "", "", error_message)
                        
                elif status in ["failed", "error"]:
                    error_message = f"Upload failed with status: {status}"
                    print(error_message)
                    return ("", "", "", "", error_message)
            
            error_message = "Upload timeout - status check exceeded maximum attempts"
            print(error_message)
            return ("", "", "", "", error_message)
            
        except Exception as e:
            error_message = f"Error uploading audio: {str(e)}"
            print(error_message)
            return ("", "", "", "", error_message)


class Comfly_suno_upload_extend:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip_id": ("STRING", {"forceInput": True}),
                "prompt": ("STRING", {"multiline": True}),
                "tags": ("STRING", {"default": ""}),
                "title": ("STRING", {"default": ""}),
                "continue_at": ("INT", {"default": 28, "min": 0, "max": 120}),
                "version": (["v3.0", "v3.5", "v4", "v4.5", "v4.5+", "v5"], {"default": "v5"}),
            },
            "optional": {
                "api_key": ("STRING", {"default": ""}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647}),
            }
        }
    
    RETURN_TYPES = ("AUDIO", "AUDIO", "STRING", "STRING", "STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("audio1", "audio2", "audio_url1", "audio_url2", "task_id", "response", "clip_id1", "clip_id2", "duration")
    FUNCTION = "extend_audio"
    CATEGORY = "Comfly/Suno"

    def __init__(self):
        self.api_key = get_config().get('api_key', '')
        self.timeout = 300

    def get_headers(self):
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

    def extend_audio(self, clip_id, prompt, tags="", title="", continue_at=28, version="v5", api_key="", seed=0):
        if api_key.strip():
            self.api_key = api_key
            config = get_config()
            config['api_key'] = api_key
            
        if not self.api_key:
            error_message = "API key not found in Comflyapi.json"
            print(error_message)
            empty_audio = create_audio_object("")
            return (empty_audio, empty_audio, "", "", "", error_message, "", "", "")

        mv_mapping = {
            "v3.0": "chirp-v3.0",
            "v3.5": "chirp-v3.5", 
            "v4": "chirp-v4",
            "v4.5": "chirp-auk",
            "v4.5+": "chirp-bluejay",
            "v5": "chirp-crow"
        }
        
        mv = mv_mapping.get(version, "chirp-crow")
            
        try:
            pbar = comfy.utils.ProgressBar(100)
            pbar.update_absolute(10)

            payload = {
                "continue_at": continue_at,
                "continue_clip_id": clip_id,
                "mv": mv,
                "prompt": prompt,
                "tags": tags,
                "task": "upload_extend",
                "title": title
            }
            
            if seed > 0:
                payload["seed"] = seed

            response = requests.post(
                f"{baseurl}/suno/generate",
                headers=self.get_headers(),
                json=payload,
                timeout=self.timeout
            )
            
            pbar.update_absolute(20)
            
            if response.status_code != 200:
                error_message = f"API Error: {response.status_code} - {response.text}"
                print(error_message)
                empty_audio = create_audio_object("")
                return (empty_audio, empty_audio, "", "", "", error_message, "", "", "")
                
            result = response.json()
            
            if "id" not in result:
                error_message = "No task ID in response"
                print(error_message)
                empty_audio = create_audio_object("")
                return (empty_audio, empty_audio, "", "", "", error_message, "", "", "")
                
            task_id = result.get("id")
            
            if "clips" not in result or len(result["clips"]) < 2:
                error_message = "Expected at least 2 clips in the response"
                print(error_message)
                empty_audio = create_audio_object("")
                return (empty_audio, empty_audio, "", "", task_id, error_message, "", "", "")
                
            clip_ids = [clip["id"] for clip in result["clips"]]
            
            pbar.update_absolute(30)

            max_attempts = 40
            attempts = 0
            final_clips = []
            
            while attempts < max_attempts and len(final_clips) < 2:
                time.sleep(5)
                attempts += 1
                
                try:
                    clip_response = requests.get(
                        f"{baseurl}/suno/feed/{','.join(clip_ids)}",
                        headers=self.get_headers(),
                        timeout=self.timeout
                    )
                    
                    if clip_response.status_code != 200:
                        continue
                        
                    clips_data = clip_response.json()
                   
                    progress = min(80, 30 + (attempts * 50 // max_attempts))
                    pbar.update_absolute(progress)
                    
                    complete_clips = [
                        clip for clip in clips_data 
                        if clip.get("status") == "complete" and (
                            clip.get("audio_url") or 
                            clip.get("state") == "succeeded"
                        )
                    ]
                    
                    for clip in complete_clips:
                        if clip.get("id") in clip_ids and clip not in final_clips:
                            final_clips.append(clip)
                    
                    if len(final_clips) >= 2:
                        break
                        
                except Exception as e:
                    print(f"Error checking clip status: {str(e)}")
            
            if len(final_clips) < 2:
                error_message = f"Only received {len(final_clips)} complete clips after {max_attempts} attempts"
                print(error_message)
                
                if not final_clips:
                    empty_audio = create_audio_object("")
                    return (empty_audio, empty_audio, "", "", task_id, error_message, "", "", "")

            audio_urls = []
            clip_id_values = []
            durations = []
            
            for clip in final_clips[:2]:
                audio_url = ""
                if "audio_url" in clip and clip["audio_url"]:
                    audio_url = clip["audio_url"]
                    print(f"Found audio URL: {audio_url}")
                elif "cdn1.suno.ai" in str(clip):
                    match = re.search(r'https://cdn1\.suno\.ai/[^"\']+\.mp3', str(clip))
                    if match:
                        audio_url = match.group(0)
                
                audio_urls.append(audio_url if audio_url else "")

                clip_id_value = clip.get("clip_id", clip.get("id", ""))
                clip_id_values.append(clip_id_value)

                duration = clip.get("duration", clip.get("metadata", {}).get("duration", 0))
                durations.append(str(duration))
                
            while len(audio_urls) < 2:
                audio_urls.append("")
                
            while len(clip_id_values) < 2:
                clip_id_values.append("")
                
            while len(durations) < 2:
                durations.append("0")

            audio_objects = [create_audio_object(url) for url in audio_urls[:2]]
            while len(audio_objects) < 2:
                audio_objects.append(create_audio_object(""))
            
            pbar.update_absolute(100)
            
            response_info = {
                "status": "success",
                "original_clip_id": clip_id,
                "continue_at": continue_at,
                "extended_clips": len(final_clips),
                "version": version,
                "title": title,
                "tags": tags
            }

            duration_info = durations[0] if durations[0] != "0" else (durations[1] if len(durations) > 1 else "0")
            
            return (
                audio_objects[0],  
                audio_objects[1],  
                audio_urls[0],     
                audio_urls[1],     
                task_id,
                json.dumps(response_info),
                clip_id_values[0],
                clip_id_values[1],
                duration_info
            )
                
        except Exception as e:
            error_message = f"Error extending audio: {str(e)}"
            print(error_message)
            import traceback
            traceback.print_exc()
            empty_audio = create_audio_object("")
            return (empty_audio, empty_audio, "", "", "", error_message, "", "", "")


class Comfly_suno_cover:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "cover_clip_id": ("STRING", {"forceInput": True}),
                "prompt": ("STRING", {"multiline": True}),
                "title": ("STRING", {"default": ""}),
                "tags": ("STRING", {"default": ""}),
                "version": (["v3.0", "v3.5", "v4", "v4.5", "v4.5+", "v5"], {"default": "v5"}),
                "make_instrumental": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "api_key": ("STRING", {"default": ""}),
                "negative_tags": ("STRING", {"default": ""}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647}),
            }
        }
    
    RETURN_TYPES = ("AUDIO", "AUDIO", "STRING", "STRING", "STRING", "STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("audio1", "audio2", "audio_url1", "audio_url2", "task_id", "response", "clip_id1", "clip_id2", "image_url1", "image_url2")
    FUNCTION = "generate_cover"
    CATEGORY = "Comfly/Suno"

    def __init__(self):
        self.api_key = get_config().get('api_key', '')
        self.timeout = 300

    def get_headers(self):
        return {
            "accept": "*/*",
            "Authorization": f"Bearer {self.api_key}",
            "content-type": "application/json"
        }
    
    def generate_cover(self, cover_clip_id, prompt, title="", tags="", version="v5", 
                    make_instrumental=False, api_key="", negative_tags="", seed=0):
        if api_key.strip():
            self.api_key = api_key
            config = get_config()
            config['api_key'] = api_key
            
        if not self.api_key:
            error_message = "API key not found in Comflyapi.json"
            print(error_message)
            empty_audio = create_audio_object("")
            return (empty_audio, empty_audio, "", "", "", error_message, "", "", "", "")
        mv_mapping = {
            "v3.0": "chirp-v3.0",
            "v3.5": "chirp-v3.5", 
            "v4": "chirp-v4-tau",
            "v4.5": "chirp-auk",
            "v4.5+": "chirp-bluejay",
            "v5": "chirp-crow"
        }
        
        mv = mv_mapping.get(version, "chirp-v4-tau")
            
        try:
            pbar = comfy.utils.ProgressBar(100)
            pbar.update_absolute(10)
            payload = {
                "prompt": prompt,
                "generation_type": "TEXT",
                "tags": tags,
                "negative_tags": negative_tags,
                "mv": mv,
                "title": title,
                "continue_clip_id": None,
                "continue_at": None,
                "continued_aligned_prompt": None,
                "infill_start_s": None,
                "infill_end_s": None,
                "task": "cover",
                "make_instrumental": make_instrumental,
                "cover_clip_id": cover_clip_id
            }
            
            if seed > 0:
                payload["seed"] = seed
           
            response = requests.post(
                f"{baseurl}/suno/submit/music",
                headers=self.get_headers(),
                json=payload,
                timeout=self.timeout
            )
            
            pbar.update_absolute(30)
            
            if response.status_code != 200:
                error_message = f"API Error: {response.status_code} - {response.text}"
                print(error_message)
                empty_audio = create_audio_object("")
                return (empty_audio, empty_audio, "", "", "", error_message, "", "", "", "")
                
            result = response.json()
            clips = []
            task_id = ""
            
            if isinstance(result, dict):
                if result.get("code") == "success" and "data" in result:
                    task_id = result["data"]
                    if isinstance(task_id, str):
                        clips = self.wait_for_task_completion(task_id, pbar)
                    elif isinstance(task_id, list):
                        clips = task_id
                    elif isinstance(task_id, dict) and "clips" in task_id:
                        clips = task_id["clips"]
                    else:
                        clips = []
                elif "status" in result and result.get("status") == "SUCCESS" and "data" in result:
                    clips = result["data"]
                    task_id = result.get("task_id", "")
                    print(f"Found completed response with {len(clips)} clips")
                    pbar.update_absolute(80)
                elif "clips" in result:
                    clips = result["clips"]
                    task_id = result.get("id", result.get("task_id", ""))
                elif "data" in result and isinstance(result["data"], list):
                    clips = result["data"]
                    task_id = result.get("id", result.get("task_id", ""))
                elif "id" in result:
                    task_id = result["id"]
                    clips = []
                    clips = self.wait_for_task_completion(task_id, pbar)
                else:
                    task_id = str(result)[:50]
                    clips = []
            elif isinstance(result, list):
                clips = result
                task_id = "direct_response"
            else:
                error_message = f"Unexpected response format: {result}"
                print(error_message)
                empty_audio = create_audio_object("")
                return (empty_audio, empty_audio, "", "", "", error_message, "", "", "", "")
            
            if len(clips) == 0:
                error_message = f"No clips found in response. Task ID: {task_id}"
                print(error_message)
                empty_audio = create_audio_object("")
                return (empty_audio, empty_audio, "", "", task_id, error_message, "", "", "", "")
            if len(clips) < 2:
                while len(clips) < 2:
                    clips.append(clips[0] if clips else {})
            audio_urls = []
            clip_id_values = []
            image_urls = []
            
            for i, clip in enumerate(clips[:2]):
               
                audio_url = ""
                if "audio_url" in clip and clip["audio_url"]:
                    audio_url = clip["audio_url"]
                    print(f"Found audio_url: {audio_url}")
                
                audio_urls.append(audio_url)
                
                clip_id_value = clip.get("clip_id", clip.get("id", ""))
                clip_id_values.append(clip_id_value)
                image_url = clip.get("image_url", clip.get("image_large_url", ""))
                image_urls.append(image_url)
            while len(audio_urls) < 2:
                audio_urls.append("")
            while len(clip_id_values) < 2:
                clip_id_values.append("")
            while len(image_urls) < 2:
                image_urls.append("")
            pbar.update_absolute(90)
            audio_objects = [create_audio_object(url) for url in audio_urls[:2]]
            while len(audio_objects) < 2:
                audio_objects.append(create_audio_object(""))
            
            pbar.update_absolute(100)
            
            response_info = {
                "status": "success",
                "cover_clip_id": cover_clip_id,
                "version": version,
                "title": title,
                "tags": tags,
                "make_instrumental": make_instrumental,
                "clips_generated": len(clips),
                "audio_urls": audio_urls,
                "clip_ids": clip_id_values
            }

            return (
                audio_objects[0],  
                audio_objects[1],  
                audio_urls[0],     
                audio_urls[1],     
                task_id,
                json.dumps(response_info),
                clip_id_values[0],
                clip_id_values[1],
                image_urls[0],
                image_urls[1]
            )
                
        except Exception as e:
            error_message = f"Error generating cover: {str(e)}"
            print(error_message)
            import traceback
            traceback.print_exc()
            empty_audio = create_audio_object("")
            return (empty_audio, empty_audio, "", "", "", error_message, "", "", "", "")
        
    def wait_for_task_completion(self, task_id, pbar):
        max_attempts = 50
        attempts = 0
        
        while attempts < max_attempts:
            time.sleep(3)
            attempts += 1
            
            try:
                status_response = requests.get(
                    f"{baseurl}/suno/task/{task_id}",
                    headers=self.get_headers(),
                    timeout=self.timeout
                )
                
                if status_response.status_code == 200:
                    status_result = status_response.json()

                    if status_result.get("status") == "SUCCESS" and "data" in status_result:
                        clips = status_result["data"]
                        if isinstance(clips, list) and len(clips) > 0:
                            return clips
                    elif status_result.get("status") in ["FAILED", "ERROR"]:
                        break

                feed_response = requests.get(
                    f"{baseurl}/suno/feed/{task_id}",
                    headers=self.get_headers(),
                    timeout=self.timeout
                )
                
                if feed_response.status_code == 200:
                    feed_result = feed_response.json()
                    if isinstance(feed_result, list):
                        complete_clips = [
                            clip for clip in feed_result 
                            if clip.get("status") == "complete" and (
                                clip.get("audio_url") or clip.get("state") == "succeeded"
                            )
                        ]
                        if len(complete_clips) >= 1:
                            return complete_clips
                
                progress = min(85, 35 + (attempts * 50 // max_attempts))
                pbar.update_absolute(progress)
                
            except Exception as e:
                print(f"Error checking task status: {str(e)}")
        
        print(f"Task {task_id} did not complete within {max_attempts} attempts")
        return []