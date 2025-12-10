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


class Comfly_vidu_img2video:

    VOICE_OPTIONS = {
        "中文(普通话)": [
            "male-qn-qingse", "male-qn-jingying", "male-qn-badao", "male-qn-daxuesheng",
            "female-shaonv", "female-yujie", "female-chengshu", "female-tianmei",
            "male-qn-qingse-jingpin", "male-qn-jingying-jingpin", "male-qn-badao-jingpin", 
            "male-qn-daxuesheng-jingpin", "female-shaonv-jingpin", "female-yujie-jingpin", 
            "female-chengshu-jingpin", "female-tianmei-jingpin",
            "clever_boy", "cute_boy", "lovely_girl", "cartoon_pig",
            "bingjiao_didi", "junlang_nanyou", "chunzhen_xuedi", "lengdan_xiongzhang", "badao_shaoye",
            "tianxin_xiaoling", "qiaopi_mengmei", "wumei_yujie", "diadia_xuemei", "danya_xuejie",
            "Chinese (Mandarin)_Reliable_Executive", "Chinese (Mandarin)_News_Anchor", 
            "Chinese (Mandarin)_Mature_Woman", "Chinese (Mandarin)_Unrestrained_Young_Man",
            "Arrogant_Miss", "Robot_Armor", "Chinese (Mandarin)_Kind-hearted_Antie",
            "Chinese (Mandarin)_HK_Flight_Attendant", "Chinese (Mandarin)_Humorous_Elder",
            "Chinese (Mandarin)_Gentleman", "Chinese (Mandarin)_Warm_Bestie",
            "Chinese (Mandarin)_Male_Announcer", "Chinese (Mandarin)_Sweet_Lady",
            "Chinese (Mandarin)_Southern_Young_Man", "Chinese (Mandarin)_Wise_Women",
            "Chinese (Mandarin)_Gentle_Youth", "Chinese (Mandarin)_Warm_Girl",
            "Chinese (Mandarin)_Kind-hearted_Elder", "Chinese (Mandarin)_Cute_Spirit",
            "Chinese (Mandarin)_Radio_Host", "Chinese (Mandarin)_Lyrical_Voice",
            "Chinese (Mandarin)_Straightforward_Boy", "Chinese (Mandarin)_Sincere_Adult",
            "Chinese (Mandarin)_Gentle_Senior", "Chinese (Mandarin)_Stubborn_Friend",
            "Chinese (Mandarin)_Crisp_Girl", "Chinese (Mandarin)_Pure-hearted_Boy",
            "Chinese (Mandarin)_Soft_Girl"
        ],
        "中文(粤语)": [
            "Cantonese_ProfessionalHost（F)", "Cantonese_GentleLady",
            "Cantonese_ProfessionalHost（M)", "Cantonese_PlayfulMan",
            "Cantonese_CuteGirl", "Cantonese_KindWoman"
        ],
        "English": [
            "Santa_Claus", "Grinch", "Rudolph", "Arnold", "Charming_Santa", "Charming_Lady",
            "Sweet_Girl", "Cute_Elf", "Attractive_Girl", "Serene_Woman",
            "English_Trustworthy_Man", "English_Graceful_Lady", "English_Aussie_Bloke",
            "English_Whispering_girl", "English_Diligent_Man", "English_Gentle-voiced_man"
        ],
        "日本語": [
            "Japanese_IntellectualSenior", "Japanese_DecisivePrincess", "Japanese_LoyalKnight",
            "Japanese_DominantMan", "Japanese_SeriousCommander", "Japanese_ColdQueen",
            "Japanese_DependableWoman", "Japanese_GentleButler", "Japanese_KindLady",
            "Japanese_CalmLady", "Japanese_OptimisticYouth", "Japanese_GenerousIzakayaOwner",
            "Japanese_SportyStudent", "Japanese_InnocentBoy", "Japanese_GracefulMaiden"
        ],
        "한국어": [
            "Korean_SweetGirl", "Korean_CheerfulBoyfriend", "Korean_EnchantingSister",
            "Korean_ShyGirl", "Korean_ReliableSister", "Korean_StrictBoss", "Korean_SassyGirl",
            "Korean_ChildhoodFriendGirl", "Korean_PlayboyCharmer", "Korean_ElegantPrincess",
            "Korean_BraveFemaleWarrior", "Korean_BraveYouth", "Korean_CalmLady",
            "Korean_EnthusiasticTeen", "Korean_SoothingLady", "Korean_IntellectualSenior",
            "Korean_LonelyWarrior", "Korean_MatureLady", "Korean_InnocentBoy",
            "Korean_CharmingSister", "Korean_AthleticStudent", "Korean_BraveAdventurer",
            "Korean_CalmGentleman", "Korean_WiseElf", "Korean_CheerfulCoolJunior",
            "Korean_DecisiveQueen", "Korean_ColdYoungMan", "Korean_MysteriousGirl",
            "Korean_QuirkyGirl", "Korean_ConsiderateSenior", "Korean_CheerfulLittleSister",
            "Korean_DominantMan", "Korean_AirheadedGirl", "Korean_ReliableYouth",
            "Korean_FriendlyBigSister", "Korean_GentleBoss", "Korean_ColdGirl",
            "Korean_HaughtyLady", "Korean_CharmingElderSister", "Korean_IntellectualMan",
            "Korean_CaringWoman", "Korean_WiseTeacher", "Korean_ConfidentBoss",
            "Korean_AthleticGirl", "Korean_PossessiveMan", "Korean_GentleWoman",
            "Korean_CockyGuy", "Korean_ThoughtfulWoman", "Korean_OptimisticYouth"
        ],
        "Español": [
            "Spanish_SereneWoman", "Spanish_MaturePartner", "Spanish_CaptivatingStoryteller",
            "Spanish_Narrator", "Spanish_WiseScholar", "Spanish_Kind-heartedGirl",
            "Spanish_DeterminedManager", "Spanish_BossyLeader", "Spanish_ReservedYoungMan",
            "Spanish_ConfidentWoman", "Spanish_ThoughtfulMan", "Spanish_Strong-WilledBoy",
            "Spanish_SophisticatedLady", "Spanish_RationalMan", "Spanish_AnimeCharacter",
            "Spanish_Deep-tonedMan", "Spanish_Fussyhostess", "Spanish_SincereTeen",
            "Spanish_FrankLady", "Spanish_Comedian", "Spanish_Debator", "Spanish_ToughBoss",
            "Spanish_Wiselady", "Spanish_Steadymentor", "Spanish_Jovialman",
            "Spanish_SantaClaus", "Spanish_Rudolph", "Spanish_Intonategirl", "Spanish_Arnold",
            "Spanish_Ghost", "Spanish_HumorousElder", "Spanish_EnergeticBoy",
            "Spanish_WhimsicalGirl", "Spanish_StrictBoss", "Spanish_ReliableMan",
            "Spanish_SereneElder", "Spanish_AngryMan", "Spanish_AssertiveQueen",
            "Spanish_CaringGirlfriend", "Spanish_PowerfulSoldier", "Spanish_PassionateWarrior",
            "Spanish_ChattyGirl", "Spanish_RomanticHusband", "Spanish_CompellingGirl",
            "Spanish_PowerfulVeteran", "Spanish_SensibleManager", "Spanish_ThoughtfulLady"
        ],
        "Português": [
            "Portuguese_SentimentalLady", "Portuguese_BossyLeader", "Portuguese_Wiselady",
            "Portuguese_Strong-WilledBoy", "Portuguese_Deep-VoicedGentleman", "Portuguese_UpsetGirl",
            "Portuguese_PassionateWarrior", "Portuguese_AnimeCharacter", "Portuguese_ConfidentWoman",
            "Portuguese_AngryMan", "Portuguese_CaptivatingStoryteller", "Portuguese_Godfather",
            "Portuguese_ReservedYoungMan", "Portuguese_SmartYoungGirl", "Portuguese_Kind-heartedGirl",
            "Portuguese_Pompouslady", "Portuguese_Grinch", "Portuguese_Debator",
            "Portuguese_SweetGirl", "Portuguese_AttractiveGirl", "Portuguese_ThoughtfulMan",
            "Portuguese_PlayfulGirl", "Portuguese_GorgeousLady", "Portuguese_LovelyLady",
            "Portuguese_SereneWoman", "Portuguese_SadTeen", "Portuguese_MaturePartner",
            "Portuguese_Comedian", "Portuguese_NaughtySchoolgirl", "Portuguese_Narrator",
            "Portuguese_ToughBoss", "Portuguese_Fussyhostess", "Portuguese_Dramatist",
            "Portuguese_Steadymentor", "Portuguese_Jovialman", "Portuguese_CharmingQueen",
            "Portuguese_SantaClaus", "Portuguese_Rudolph", "Portuguese_Arnold",
            "Portuguese_CharmingSanta", "Portuguese_CharmingLady", "Portuguese_Ghost",
            "Portuguese_HumorousElder", "Portuguese_CalmLeader", "Portuguese_GentleTeacher",
            "Portuguese_EnergeticBoy", "Portuguese_ReliableMan", "Portuguese_SereneElder",
            "Portuguese_GrimReaper", "Portuguese_AssertiveQueen", "Portuguese_WhimsicalGirl",
            "Portuguese_StressedLady", "Portuguese_FriendlyNeighbor", "Portuguese_CaringGirlfriend",
            "Portuguese_PowerfulSoldier", "Portuguese_FascinatingBoy", "Portuguese_RomanticHusband",
            "Portuguese_StrictBoss", "Portuguese_InspiringLady", "Portuguese_PlayfulSpirit",
            "Portuguese_ElegantGirl", "Portuguese_CompellingGirl", "Portuguese_PowerfulVeteran",
            "Portuguese_SensibleManager", "Portuguese_ThoughtfulLady", "Portuguese_TheatricalActor",
            "Portuguese_FragileBoy", "Portuguese_ChattyGirl", "Portuguese_Conscientiousinstructor",
            "Portuguese_RationalMan", "Portuguese_WiseScholar", "Portuguese_FrankLady",
            "Portuguese_DeterminedManager"
        ],
        "Français": [
            "French_Male_Speech_New", "French_Female_News Anchor", "French_CasualMan",
            "French_MovieLeadFemale", "French_FemaleAnchor", "French_MaleNarrator"
        ],
        "Bahasa Indonesia": [
            "Indonesian_SweetGirl", "Indonesian_ReservedYoungMan", "Indonesian_CharmingGirl",
            "Indonesian_CalmWoman", "Indonesian_ConfidentWoman", "Indonesian_CaringMan",
            "Indonesian_BossyLeader", "Indonesian_DeterminedBoy", "Indonesian_GentleGirl"
        ],
        "Deutsch": [
            "German_FriendlyMan", "German_SweetLady", "German_PlayfulMan"
        ],
        "Русский": [
            "Russian_HandsomeChildhoodFriend", "Russian_BrightHeroine", "Russian_AmbitiousWoman",
            "Russian_ReliableMan", "Russian_CrazyQueen", "Russian_PessimisticGirl",
            "Russian_AttractiveGuy", "Russian_Bad-temperedBoy"
        ],
        "Italiano": [
            "Italian_BraveHeroine", "Italian_Narrator", "Italian_WanderingSorcerer",
            "Italian_DiligentLeader"
        ],
        "العربية": [
            "Arabic_CalmWoman", "Arabic_FriendlyGuy"
        ],
        "Türkçe": [
            "Turkish_CalmWoman", "Turkish_Trustworthyman"
        ],
        "Українська": [
            "Ukrainian_CalmWoman", "Ukrainian_WiseScholar"
        ],
        "Nederlands": [
            "Dutch_kindhearted_girl", "Dutch_bossy_leader"
        ],
        "Tiếng Việt": [
            "Vietnamese_kindhearted_girl"
        ]
    }

    @classmethod
    def INPUT_TYPES(cls):
        all_voices = []
        for lang, voices in cls.VOICE_OPTIONS.items():
            all_voices.extend(voices)
        
        return {
            "required": {
                "image": ("IMAGE",),
                "model": (["viduq2-pro", "viduq2-turbo", "viduq1", "viduq1-classic", "vidu2.0", "vidu1.5"], 
                         {"default": "viduq2-pro"}),
            },
            "optional": {
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "api_key": ("STRING", {"default": ""}),
                "audio": ("BOOLEAN", {"default": False}),
                "voice_language": (list(cls.VOICE_OPTIONS.keys()), {"default": "中文(普通话)"}),
                "voice_id": (all_voices, {"default": "male-qn-jingying"}),
                "is_rec": ("BOOLEAN", {"default": False}),
                "duration": ("INT", {"default": 5, "min": 1, "max": 10}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647}),
                "resolution": (["540p", "720p", "1080p"], {"default": "720p"}),
                "movement_amplitude": (["auto", "small", "medium", "large"], {"default": "auto"}),
                "bgm": ("BOOLEAN", {"default": False}),
                "off_peak": ("BOOLEAN", {"default": False}),
                "watermark": ("BOOLEAN", {"default": False}),
                "wm_position": ([1, 2, 3, 4], {"default": 3}),
            }
        }
    
    RETURN_TYPES = (IO.VIDEO, "STRING", "STRING", "STRING")
    RETURN_NAMES = ("video", "video_url", "task_id", "response")
    FUNCTION = "generate_video"
    CATEGORY = "Comfly/Vidu"

    def __init__(self):
        self.api_key = get_config().get('api_key', '')
        self.timeout = 900

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
        return f"data:image/png;base64,{base64_str}"
    
    def generate_video(self, image, model="viduq2-pro", prompt="", api_key="", 
                      audio=False, voice_language="中文(普通话)", voice_id="male-qn-jingying", 
                      is_rec=False, duration=5, seed=0, resolution="720p", 
                      movement_amplitude="auto", bgm=False, off_peak=False, 
                      watermark=False, wm_position=3):
        
        if api_key.strip():
            self.api_key = api_key
            config = get_config()
            config['api_key'] = api_key
            
        if not self.api_key:
            error_response = {"status": "error", "message": "API key not provided or not found in config"}
            empty_video = ComflyVideoAdapter("")
            return (empty_video, "", "", json.dumps(error_response))
            
        try:
            pbar = comfy.utils.ProgressBar(100)
            pbar.update_absolute(10)

            image_base64 = self.image_to_base64(image)
            if not image_base64:
                error_message = "Failed to convert image to base64"
                print(error_message)
                empty_video = ComflyVideoAdapter("")
                return (empty_video, "", "", json.dumps({"status": "error", "message": error_message}))

            payload = {
                "model": model,
                "images": [image_base64],  
                "duration": duration,
                "seed": seed if seed > 0 else 0,
                "resolution": resolution,
                "movement_amplitude": movement_amplitude,
                "off_peak": off_peak
            }

            if prompt.strip():
                payload["prompt"] = prompt

            if audio:
                payload["audio"] = True
                payload["voice_id"] = voice_id
            
            if is_rec:
                payload["is_rec"] = True
            
            if bgm:
                payload["bgm"] = True
            
            if watermark:
                payload["watermark"] = True
                payload["wm_position"] = wm_position

            pbar.update_absolute(20)

            response = requests.post(
                f"{baseurl}/vidu/v2/img2video",
                headers=self.get_headers(),
                json=payload,
                timeout=self.timeout
            )
            
            pbar.update_absolute(30)
            
            if response.status_code != 200:
                error_message = f"API Error: {response.status_code} - {response.text}"
                print(error_message)
                empty_video = ComflyVideoAdapter("")
                return (empty_video, "", "", json.dumps({"status": "error", "message": error_message}))
                
            result = response.json()
            
            if "task_id" not in result:
                error_message = f"No task_id in response: {result}"
                print(error_message)
                empty_video = ComflyVideoAdapter("")
                return (empty_video, "", "", json.dumps({"status": "error", "message": error_message}))
                
            task_id = result.get("task_id")
            
            pbar.update_absolute(40)

            max_attempts = 120
            attempts = 0
            video_url = None
            
            while attempts < max_attempts:
                time.sleep(10)
                attempts += 1
                
                try:
                    status_response = requests.get(
                        f"{baseurl}/vidu/v2/tasks/{task_id}/creations",
                        headers=self.get_headers(),
                        timeout=self.timeout
                    )
                    
                    if status_response.status_code != 200:
                        print(f"Status check failed: {status_response.status_code} - {status_response.text}")
                        continue
                        
                    status_result = status_response.json()
                    
                    state = status_result.get("state", "")

                    progress_value = min(90, 40 + (attempts * 50 // max_attempts))
                    pbar.update_absolute(progress_value)
                    
                    if state == "success":
                        creations = status_result.get("creations", [])
                        if creations and len(creations) > 0:
                            video_url = creations[0].get("url", "")
                            if video_url:
                                print(f"Video URL found: {video_url}")
                                break
                    elif state == "failed":
                        err_code = status_result.get("err_code", "Unknown error")
                        error_message = f"Video generation failed: {err_code}"
                        print(error_message)
                        empty_video = ComflyVideoAdapter("")
                        return (empty_video, "", task_id, json.dumps({"status": "error", "message": error_message}))
                        
                except Exception as e:
                    print(f"Error checking generation status (attempt {attempts}): {str(e)}")
            
            if not video_url:
                error_message = f"Failed to retrieve video URL after {max_attempts} attempts"
                print(error_message)
                empty_video = ComflyVideoAdapter("")
                return (empty_video, "", task_id, json.dumps({"status": "error", "message": error_message}))
            
            pbar.update_absolute(95)
            print(f"Video generation completed. URL: {video_url}")

            video_adapter = ComflyVideoAdapter(video_url)
            
            response_data = {
                "status": "success",
                "task_id": task_id,
                "video_url": video_url,
                "model": model,
                "duration": duration,
                "resolution": resolution,
                "seed": result.get("seed", seed),
                "voice_language": voice_language if audio else "N/A",
                "voice_id": voice_id if audio else "N/A"
            }
            
            pbar.update_absolute(100)
            return (video_adapter, video_url, task_id, json.dumps(response_data))
            
        except Exception as e:
            error_message = f"Error generating video: {str(e)}"
            print(error_message)
            import traceback
            traceback.print_exc()
            empty_video = ComflyVideoAdapter("")
            return (empty_video, "", "", json.dumps({"status": "error", "message": error_message}))


class Comfly_vidu_text2video:
    """
    Comfly Vidu Text to Video node
    Generates videos from text prompts using Vidu API
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "model": (["viduq2", "viduq1", "vidu1.5"], {"default": "viduq2"}),
            },
            "optional": {
                "api_key": ("STRING", {"default": ""}),
                "style": (["general", "anime"], {"default": "general"}),
                "duration": ("INT", {"default": 5, "min": 1, "max": 10}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647}),
                "aspect_ratio": (["16:9", "9:16", "3:4", "4:3", "1:1"], {"default": "16:9"}),
                "resolution": (["360p", "540p", "720p", "1080p"], {"default": "720p"}),
                "movement_amplitude": (["auto", "small", "medium", "large"], {"default": "auto"}),
                "bgm": ("BOOLEAN", {"default": False}),
                "off_peak": ("BOOLEAN", {"default": False}),
                "watermark": ("BOOLEAN", {"default": False}),
                "wm_position": ([1, 2, 3, 4], {"default": 3}),
            }
        }
    
    RETURN_TYPES = (IO.VIDEO, "STRING", "STRING", "STRING")
    RETURN_NAMES = ("video", "video_url", "task_id", "response")
    FUNCTION = "generate_video"
    CATEGORY = "Comfly/Vidu"

    def __init__(self):
        self.api_key = get_config().get('api_key', '')
        self.timeout = 900

    def get_headers(self):
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
    
    def generate_video(self, prompt, model="viduq2", api_key="", style="general",
                      duration=5, seed=0, aspect_ratio="16:9", resolution="720p",
                      movement_amplitude="auto", bgm=False, off_peak=False,
                      watermark=False, wm_position=3):
        
        if api_key.strip():
            self.api_key = api_key
            config = get_config()
            config['api_key'] = api_key
            
        if not self.api_key:
            error_response = {"status": "error", "message": "API key not provided or not found in config"}
            empty_video = ComflyVideoAdapter("")
            return (empty_video, "", "", json.dumps(error_response))
            
        try:
            pbar = comfy.utils.ProgressBar(100)
            pbar.update_absolute(10)

            payload = {
                "model": model,
                "prompt": prompt,
                "duration": duration,
                "seed": seed if seed > 0 else 0,
                "aspect_ratio": aspect_ratio,
                "resolution": resolution,
                "movement_amplitude": movement_amplitude,
                "off_peak": off_peak
            }

            if model != "viduq2":
                payload["style"] = style
            
            if bgm:
                payload["bgm"] = True
            
            if watermark:
                payload["watermark"] = True
                payload["wm_position"] = wm_position

            pbar.update_absolute(20)

            response = requests.post(
                f"{baseurl}/vidu/v2/text2video",
                headers=self.get_headers(),
                json=payload,
                timeout=self.timeout
            )
            
            pbar.update_absolute(30)
            
            if response.status_code != 200:
                error_message = f"API Error: {response.status_code} - {response.text}"
                print(error_message)
                empty_video = ComflyVideoAdapter("")
                return (empty_video, "", "", json.dumps({"status": "error", "message": error_message}))
                
            result = response.json()
            
            if "task_id" not in result:
                error_message = f"No task_id in response: {result}"
                print(error_message)
                empty_video = ComflyVideoAdapter("")
                return (empty_video, "", "", json.dumps({"status": "error", "message": error_message}))
                
            task_id = result.get("task_id")
            
            pbar.update_absolute(40)

            max_attempts = 120
            attempts = 0
            video_url = None
            
            while attempts < max_attempts:
                time.sleep(10)
                attempts += 1
                
                try:
                    status_response = requests.get(
                        f"{baseurl}/vidu/v2/tasks/{task_id}/creations",
                        headers=self.get_headers(),
                        timeout=self.timeout
                    )
                    
                    if status_response.status_code != 200:
                        print(f"Status check failed: {status_response.status_code} - {status_response.text}")
                        continue
                        
                    status_result = status_response.json()
                    
                    state = status_result.get("state", "")

                    progress_value = min(90, 40 + (attempts * 50 // max_attempts))
                    pbar.update_absolute(progress_value)
                    
                    if state == "success":
                        creations = status_result.get("creations", [])
                        if creations and len(creations) > 0:
                            video_url = creations[0].get("url", "")
                            if video_url:
                                print(f"Video URL found: {video_url}")
                                break
                    elif state == "failed":
                        err_code = status_result.get("err_code", "Unknown error")
                        error_message = f"Video generation failed: {err_code}"
                        print(error_message)
                        empty_video = ComflyVideoAdapter("")
                        return (empty_video, "", task_id, json.dumps({"status": "error", "message": error_message}))
                        
                except Exception as e:
                    print(f"Error checking generation status (attempt {attempts}): {str(e)}")
            
            if not video_url:
                error_message = f"Failed to retrieve video URL after {max_attempts} attempts"
                print(error_message)
                empty_video = ComflyVideoAdapter("")
                return (empty_video, "", task_id, json.dumps({"status": "error", "message": error_message}))
            
            pbar.update_absolute(95)
            print(f"Video generation completed. URL: {video_url}")

            video_adapter = ComflyVideoAdapter(video_url)
            
            response_data = {
                "status": "success",
                "task_id": task_id,
                "video_url": video_url,
                "model": model,
                "duration": duration,
                "resolution": resolution,
                "aspect_ratio": aspect_ratio,
                "seed": result.get("seed", seed)
            }
            
            pbar.update_absolute(100)
            return (video_adapter, video_url, task_id, json.dumps(response_data))
            
        except Exception as e:
            error_message = f"Error generating video: {str(e)}"
            print(error_message)
            import traceback
            traceback.print_exc()
            empty_video = ComflyVideoAdapter("")
            return (empty_video, "", "", json.dumps({"status": "error", "message": error_message}))


class Comfly_vidu_ref2video:
    """
    Comfly Vidu Reference to Video node
    Generates videos from reference images with optional audio
    """
    
    VOICE_OPTIONS = Comfly_vidu_img2video.VOICE_OPTIONS
    
    @classmethod
    def INPUT_TYPES(cls):
        all_voices = [""]
        for lang, voices in cls.VOICE_OPTIONS.items():
            all_voices.extend(voices)
        
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "model": (["viduq2", "viduq1", "vidu2.0", "vidu1.5"], {"default": "viduq2"}),
            },
            "optional": {
                "api_key": ("STRING", {"default": ""}),
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
                "image4": ("IMAGE",),
                "image5": ("IMAGE",),
                "image6": ("IMAGE",),
                "image7": ("IMAGE",),
                "audio": ("BOOLEAN", {"default": False}),
                "subject1_id": ("STRING", {"default": "1"}),
                "subject1_voice_id": (all_voices, {"default": ""}),
                "subject2_id": ("STRING", {"default": "2"}),
                "subject2_voice_id": (all_voices, {"default": ""}),
                "subject3_id": ("STRING", {"default": "3"}),
                "subject3_voice_id": (all_voices, {"default": ""}),
                "duration": ("INT", {"default": 5, "min": 1, "max": 10}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647}),
                "aspect_ratio": (["16:9", "9:16", "4:3", "3:4", "1:1"], {"default": "16:9"}),
                "resolution": (["540p", "720p", "1080p"], {"default": "720p"}),
                "movement_amplitude": (["auto", "small", "medium", "large"], {"default": "auto"}),
                "bgm": ("BOOLEAN", {"default": False}),
                "off_peak": ("BOOLEAN", {"default": False}),
                "watermark": ("BOOLEAN", {"default": False}),
                "wm_position": ([1, 2, 3, 4], {"default": 3}),
            }
        }
    
    RETURN_TYPES = (IO.VIDEO, "STRING", "STRING", "STRING")
    RETURN_NAMES = ("video", "video_url", "task_id", "response")
    FUNCTION = "generate_video"
    CATEGORY = "Comfly/Vidu"

    def __init__(self):
        self.api_key = get_config().get('api_key', '')
        self.timeout = 900

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
        return f"data:image/png;base64,{base64_str}"
    
    def generate_video(self, prompt, model="viduq2", api_key="",
                      image1=None, image2=None, image3=None, image4=None,
                      image5=None, image6=None, image7=None,
                      audio=False, subject1_id="1", subject1_voice_id="",
                      subject2_id="2", subject2_voice_id="",
                      subject3_id="3", subject3_voice_id="",
                      duration=5, seed=0, aspect_ratio="16:9", resolution="720p",
                      movement_amplitude="auto", bgm=False, off_peak=False,
                      watermark=False, wm_position=3):
        
        if api_key.strip():
            self.api_key = api_key
            config = get_config()
            config['api_key'] = api_key
            
        if not self.api_key:
            error_response = {"status": "error", "message": "API key not provided or not found in config"}
            empty_video = ComflyVideoAdapter("")
            return (empty_video, "", "", json.dumps(error_response))
            
        try:
            pbar = comfy.utils.ProgressBar(100)
            pbar.update_absolute(10)

            all_images = [image1, image2, image3, image4, image5, image6, image7]
            image_base64_list = []
            
            for img in all_images:
                if img is not None:
                    img_base64 = self.image_to_base64(img)
                    if img_base64:
                        image_base64_list.append(img_base64)
            
            if not image_base64_list:
                error_message = "No images provided. At least one image is required."
                print(error_message)
                empty_video = ComflyVideoAdapter("")
                return (empty_video, "", "", json.dumps({"status": "error", "message": error_message}))

            payload = {
                "model": model,
                "prompt": prompt,
                "duration": duration,
                "seed": seed if seed > 0 else 0,
                "aspect_ratio": aspect_ratio,
                "resolution": resolution,
                "movement_amplitude": movement_amplitude,
                "off_peak": off_peak
            }

            if audio:
                subjects = []
 
                subject_images = [[], [], []]
                for i, img_b64 in enumerate(image_base64_list):
                    subject_idx = min(i // 3, 2)  
                    if len(subject_images[subject_idx]) < 3:
                        subject_images[subject_idx].append(img_b64)

                subject_configs = [
                    (subject1_id, subject1_voice_id, subject_images[0]),
                    (subject2_id, subject2_voice_id, subject_images[1]),
                    (subject3_id, subject3_voice_id, subject_images[2])
                ]
                
                for subj_id, voice_id, images in subject_configs:
                    if images:
                        subject = {
                            "id": subj_id,
                            "images": images,
                            "voice_id": voice_id if voice_id else ""
                        }
                        subjects.append(subject)
                
                if subjects:
                    payload["subjects"] = subjects
                    payload["audio"] = True
            else:
                payload["images"] = image_base64_list
                if bgm:
                    payload["bgm"] = True
            
            if watermark:
                payload["watermark"] = True
                payload["wm_position"] = wm_position

            pbar.update_absolute(20)

            response = requests.post(
                f"{baseurl}/vidu/v2/reference2video",
                headers=self.get_headers(),
                json=payload,
                timeout=self.timeout
            )
            
            pbar.update_absolute(30)
            
            if response.status_code != 200:
                error_message = f"API Error: {response.status_code} - {response.text}"
                print(error_message)
                empty_video = ComflyVideoAdapter("")
                return (empty_video, "", "", json.dumps({"status": "error", "message": error_message}))
                
            result = response.json()
            
            if "task_id" not in result:
                error_message = f"No task_id in response: {result}"
                print(error_message)
                empty_video = ComflyVideoAdapter("")
                return (empty_video, "", "", json.dumps({"status": "error", "message": error_message}))
                
            task_id = result.get("task_id")
            
            pbar.update_absolute(40)

            max_attempts = 120
            attempts = 0
            video_url = None
            
            while attempts < max_attempts:
                time.sleep(10)
                attempts += 1
                
                try:
                    status_response = requests.get(
                        f"{baseurl}/vidu/v2/tasks/{task_id}/creations",
                        headers=self.get_headers(),
                        timeout=self.timeout
                    )
                    
                    if status_response.status_code != 200:
                        print(f"Status check failed: {status_response.status_code} - {status_response.text}")
                        continue
                        
                    status_result = status_response.json()
                    
                    state = status_result.get("state", "")

                    progress_value = min(90, 40 + (attempts * 50 // max_attempts))
                    pbar.update_absolute(progress_value)
                    
                    if state == "success":
                        creations = status_result.get("creations", [])
                        if creations and len(creations) > 0:
                            video_url = creations[0].get("url", "")
                            if video_url:
                                print(f"Video URL found: {video_url}")
                                break
                    elif state == "failed":
                        err_code = status_result.get("err_code", "Unknown error")
                        error_message = f"Video generation failed: {err_code}"
                        print(error_message)
                        empty_video = ComflyVideoAdapter("")
                        return (empty_video, "", task_id, json.dumps({"status": "error", "message": error_message}))
                        
                except Exception as e:
                    print(f"Error checking generation status (attempt {attempts}): {str(e)}")
            
            if not video_url:
                error_message = f"Failed to retrieve video URL after {max_attempts} attempts"
                print(error_message)
                empty_video = ComflyVideoAdapter("")
                return (empty_video, "", task_id, json.dumps({"status": "error", "message": error_message}))
            
            pbar.update_absolute(95)
            print(f"Video generation completed. URL: {video_url}")

            video_adapter = ComflyVideoAdapter(video_url)
            
            response_data = {
                "status": "success",
                "task_id": task_id,
                "video_url": video_url,
                "model": model,
                "duration": duration,
                "resolution": resolution,
                "aspect_ratio": aspect_ratio,
                "audio": audio,
                "images_count": len(image_base64_list),
                "seed": result.get("seed", seed)
            }
            
            pbar.update_absolute(100)
            return (video_adapter, video_url, task_id, json.dumps(response_data))
            
        except Exception as e:
            error_message = f"Error generating video: {str(e)}"
            print(error_message)
            import traceback
            traceback.print_exc()
            empty_video = ComflyVideoAdapter("")
            return (empty_video, "", "", json.dumps({"status": "error", "message": error_message}))


class Comfly_vidu_start_end2video:
    """
    Comfly Vidu Start-End Frame to Video node
    Generates videos from start and end frame images
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "start_image": ("IMAGE",),
                "end_image": ("IMAGE",),
                "model": (["viduq2-pro", "viduq2-turbo", "viduq1", "viduq1-classic", "vidu2.0", "vidu1.5"], 
                         {"default": "viduq2-pro"}),
            },
            "optional": {
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "api_key": ("STRING", {"default": ""}),
                "is_rec": ("BOOLEAN", {"default": False}),
                "duration": ("INT", {"default": 5, "min": 1, "max": 8}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647}),
                "resolution": (["360p", "540p", "720p", "1080p"], {"default": "720p"}),
                "movement_amplitude": (["auto", "small", "medium", "large"], {"default": "auto"}),
                "bgm": ("BOOLEAN", {"default": False}),
                "off_peak": ("BOOLEAN", {"default": False}),
                "watermark": ("BOOLEAN", {"default": False}),
                "wm_position": ([1, 2, 3, 4], {"default": 3}),
            }
        }
    
    RETURN_TYPES = (IO.VIDEO, "STRING", "STRING", "STRING")
    RETURN_NAMES = ("video", "video_url", "task_id", "response")
    FUNCTION = "generate_video"
    CATEGORY = "Comfly/Vidu"

    def __init__(self):
        self.api_key = get_config().get('api_key', '')
        self.timeout = 900

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
        return f"data:image/png;base64,{base64_str}"
    
    def generate_video(self, start_image, end_image, model="viduq2-pro", prompt="", api_key="",
                      is_rec=False, duration=5, seed=0, resolution="720p",
                      movement_amplitude="auto", bgm=False, off_peak=False,
                      watermark=False, wm_position=3):
        
        if api_key.strip():
            self.api_key = api_key
            config = get_config()
            config['api_key'] = api_key
            
        if not self.api_key:
            error_response = {"status": "error", "message": "API key not provided or not found in config"}
            empty_video = ComflyVideoAdapter("")
            return (empty_video, "", "", json.dumps(error_response))
            
        try:
            pbar = comfy.utils.ProgressBar(100)
            pbar.update_absolute(10)

            start_image_base64 = self.image_to_base64(start_image)
            end_image_base64 = self.image_to_base64(end_image)
            
            if not start_image_base64 or not end_image_base64:
                error_message = "Failed to convert start or end image to base64"
                print(error_message)
                empty_video = ComflyVideoAdapter("")
                return (empty_video, "", "", json.dumps({"status": "error", "message": error_message}))

            payload = {
                "model": model,
                "images": [start_image_base64, end_image_base64],
                "duration": duration,
                "seed": seed if seed > 0 else 0,
                "resolution": resolution,
                "movement_amplitude": movement_amplitude,
                "off_peak": off_peak
            }

            if prompt.strip() and not is_rec:
                payload["prompt"] = prompt
            
            if is_rec:
                payload["is_rec"] = True
            
            if bgm:
                payload["bgm"] = True
            
            if watermark:
                payload["watermark"] = True
                payload["wm_position"] = wm_position

            pbar.update_absolute(20)

            response = requests.post(
                f"{baseurl}/vidu/v2/start-end2video",
                headers=self.get_headers(),
                json=payload,
                timeout=self.timeout
            )
            
            pbar.update_absolute(30)
            
            if response.status_code != 200:
                error_message = f"API Error: {response.status_code} - {response.text}"
                print(error_message)
                empty_video = ComflyVideoAdapter("")
                return (empty_video, "", "", json.dumps({"status": "error", "message": error_message}))
                
            result = response.json()
            
            if "task_id" not in result:
                error_message = f"No task_id in response: {result}"
                print(error_message)
                empty_video = ComflyVideoAdapter("")
                return (empty_video, "", "", json.dumps({"status": "error", "message": error_message}))
                
            task_id = result.get("task_id")
            
            pbar.update_absolute(40)

            max_attempts = 120
            attempts = 0
            video_url = None
            
            while attempts < max_attempts:
                time.sleep(10)
                attempts += 1
                
                try:
                    status_response = requests.get(
                        f"{baseurl}/vidu/v2/tasks/{task_id}/creations",
                        headers=self.get_headers(),
                        timeout=self.timeout
                    )
                    
                    if status_response.status_code != 200:
                        print(f"Status check failed: {status_response.status_code} - {status_response.text}")
                        continue
                        
                    status_result = status_response.json()
                    
                    state = status_result.get("state", "")

                    progress_value = min(90, 40 + (attempts * 50 // max_attempts))
                    pbar.update_absolute(progress_value)
                    
                    if state == "success":
                        creations = status_result.get("creations", [])
                        if creations and len(creations) > 0:
                            video_url = creations[0].get("url", "")
                            if video_url:
                                print(f"Video URL found: {video_url}")
                                break
                    elif state == "failed":
                        err_code = status_result.get("err_code", "Unknown error")
                        error_message = f"Video generation failed: {err_code}"
                        print(error_message)
                        empty_video = ComflyVideoAdapter("")
                        return (empty_video, "", task_id, json.dumps({"status": "error", "message": error_message}))
                        
                except Exception as e:
                    print(f"Error checking generation status (attempt {attempts}): {str(e)}")
            
            if not video_url:
                error_message = f"Failed to retrieve video URL after {max_attempts} attempts"
                print(error_message)
                empty_video = ComflyVideoAdapter("")
                return (empty_video, "", task_id, json.dumps({"status": "error", "message": error_message}))
            
            pbar.update_absolute(95)
            print(f"Video generation completed. URL: {video_url}")

            video_adapter = ComflyVideoAdapter(video_url)
            
            response_data = {
                "status": "success",
                "task_id": task_id,
                "video_url": video_url,
                "model": model,
                "duration": duration,
                "resolution": resolution,
                "is_rec": is_rec,
                "seed": result.get("seed", seed)
            }
            
            pbar.update_absolute(100)
            return (video_adapter, video_url, task_id, json.dumps(response_data))
            
        except Exception as e:
            error_message = f"Error generating video: {str(e)}"
            print(error_message)
            import traceback
            traceback.print_exc()
            empty_video = ComflyVideoAdapter("")
            return (empty_video, "", "", json.dumps({"status": "error", "message": error_message}))
