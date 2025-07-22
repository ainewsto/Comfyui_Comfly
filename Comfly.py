import os
import io
import math
import random
import torch
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
from .utils import pil2tensor, tensor2pil
from comfy.utils import common_upscale
from comfy.comfy_types import IO



def get_config():
    try:
        config_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'Comflyapi.json')
        with open(config_path, 'r') as f:  
            config = json.load(f)
        return config
    except:
        return {}

def save_config(config):
    config_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'Comflyapi.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)


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



############################# Midjourney ###########################

class ComflyBaseNode:
    def __init__(self):
        self.midjourney_api_url = {
            "turbo mode": "https://ai.comfly.chat/mj-turbo",
            "fast mode": "https://ai.comfly.chat/mj-fast",
            "relax mode": "https://ai.comfly.chat/mj-relax"
        }
        self.api_key = get_config().get('api_key', '') 
        self.speed = "fast mode"
        self.timeout = 800

    def set_speed(self, speed):
        self.speed = speed

    def get_headers(self):
        return {
            "Content-Type": "application/json",
            "Authorization": "Bearer " + self.api_key
        }

    def generateUUID(self):
        return str(uuid.uuid4())

    async def midjourney_submit_action(self, action, taskId, index, custom_id):
        headers = self.get_headers()
        payload = {
            "customId": custom_id,
            "taskId": taskId
        }
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(f"{self.midjourney_api_url[self.speed]}/mj/submit/action", headers=headers, json=payload, timeout=self.timeout) as response:
                    if response.status == 200:
                        try:
                            data = await response.json()

                            if data.get("status") == "FAILURE":
                                fail_reason = data.get("fail_reason", "Unknown failure reason")
                                error_message = f"Action submission failed: {fail_reason}"
                                print(error_message)
                                raise Exception(error_message)
                                
                            return data
                        except aiohttp.client_exceptions.ContentTypeError:
                            text_response = await response.text()
                            
                            try:
                                import json
                                data = json.loads(text_response)

                                if data.get("status") == "FAILURE":
                                    fail_reason = data.get("fail_reason", "Unknown failure reason")
                                    error_message = f"Action submission failed: {fail_reason}"
                                    print(error_message)
                                    raise Exception(error_message)
                                    
                                return data
                            except json.JSONDecodeError:
                                if text_response and len(text_response) < 100:
                                    return {"result": text_response.strip()}
                                raise Exception(f"Invalid response format: {text_response}")
                    else:
                        error_message = f"Error submitting Midjourney action: {response.status}"
                        try:
                            error_details = await response.text()
                            error_message += f" - {error_details}"
                        except:
                            pass
                        raise Exception(error_message)
        except asyncio.TimeoutError:
            error_message = f"Timeout error: Request to submit action timed out after {self.timeout} seconds"
            raise Exception(error_message)
        except Exception as e:
            if "Action submission failed" in str(e):
                raise 
            print(f"Exception in midjourney_submit_action: {str(e)}")
            raise e


    def extract_taskId(self, U, action, index):
        pattern = fr'"customId": "MJ::JOB::{action}::{index}::(.*?)"'
        match = re.search(pattern, U)
        if match:
            return match.group(1)
        else:
            raise ValueError(f"Invalid custom ID format in U: {U}")

    async def midjourney_submit_imagine_task(self, prompt, ar, no, c, s, iw, tile, r, video, sw, cw, sv, seed):
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer " + self.api_key
        }
        payload = {
            "base64Array": [],
            "instanceId": "",
            "modes": [],
            "notifyHook": "",
            "prompt": prompt,
            "remix": True,
            "state": "",
            "ar": ar,
            "no": no,
            "c": c,
            "s": s,
            "iw": iw,
            "tile": tile,
            "r": r,
            "video": video,
            "sw": sw,
            "cw": cw,
            "sv": sv,
            "seed": seed
        }
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(f"{self.midjourney_api_url[self.speed]}/mj/submit/imagine", headers=headers, json=payload, timeout=self.timeout) as response:
                    if response.status == 200:
                        try:
                            data = await response.json()

                            if data.get("status") == "FAILURE":
                                fail_reason = data.get("fail_reason", "Unknown failure reason")
                                error_message = f"Midjourney task failed: {fail_reason}"
                                print(error_message)
                                raise Exception(error_message)
                                
                            return data["result"]
                        except aiohttp.client_exceptions.ContentTypeError:
                            text_response = await response.text()
                            
                            try:
                                import json
                                data = json.loads(text_response)
 
                                if data.get("status") == "FAILURE":
                                    fail_reason = data.get("fail_reason", "Unknown failure reason")
                                    error_message = f"Midjourney task failed: {fail_reason}"
                                    print(error_message)
                                    raise Exception(error_message)
                                    
                                return data["result"]
                            except (json.JSONDecodeError, KeyError):
                                if text_response and len(text_response) < 100:
                                    return text_response.strip()
                                raise Exception(f"Server returned invalid response: {text_response}")
                    else:
                        error_message = f"Error submitting Midjourney task: {response.status}"
                        try:
                            error_details = await response.text()
                            error_message += f" - {error_details}"
                        except:
                            pass
                        raise Exception(error_message)
        except asyncio.TimeoutError:
            error_message = f"Timeout error: Request to submit imagine task timed out after {self.timeout} seconds"
            raise Exception(error_message)
        except Exception as e:
            if "Midjourney task failed" in str(e):
                raise  
            print(f"Exception in midjourney_submit_imagine_task: {str(e)}")
            raise e


    async def midjourney_fetch_task_result(self, taskId):
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer " + self.api_key
        }
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.midjourney_api_url[self.speed]}/mj/task/{taskId}/fetch", headers=headers, timeout=self.timeout) as response:
                    if response.status == 200:
                        try:
                            data = await response.json()
                            return data
                        except aiohttp.client_exceptions.ContentTypeError:
                            text_response = await response.text()
                            try:
                                import json
                                data = json.loads(text_response)
                                return data
                            except json.JSONDecodeError:
                                if text_response and len(text_response) < 100:
                                    return {"status": "SUCCESS", "progress": "100%", "imageUrl": text_response.strip()}
                                raise Exception(f"Server returned invalid response: {text_response}")
                    else:
                        error_message = f"Error fetching Midjourney task result: {response.status}"
                        try:
                            error_details = await response.text()
                            error_message += f" - {error_details}"
                        except:
                            pass
                        raise Exception(error_message)
        except asyncio.TimeoutError:
            error_message = f"Timeout error: Request to fetch task result timed out after {self.timeout} seconds"
            raise Exception(error_message)



class Comfly_upload(ComflyBaseNode):
    """
    Comfly_upload node
    Uploads an image to Midjourney and returns the URL link.
    Inputs:
        image (IMAGE): Input image to be uploaded.
    Outputs:
        url (STRING): URL link of the uploaded image.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "api_key": ("STRING", {"default": ""}),
            },
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("url",)
    FUNCTION = "upload_image"
    CATEGORY = "Comfly/Midjourney"

    async def upload_image_to_midjourney(self, image):

        image = tensor2pil(image)[0]
        buffered = BytesIO()
        image_format = "PNG" 
        image.save(buffered, format=image_format)
        image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

        payload = {
            "base64Array": [f"data:image/{image_format.lower()};base64,{image_base64}"],
            "instanceId": "",
            "notifyHook": ""
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(f"{self.midjourney_api_url[self.speed]}/mj/submit/upload-discord-images", headers=self.get_headers(), json=payload, timeout=self.timeout) as response:
                    if response.status == 200:
                        try:
                            data = await response.json()

                            if data.get("status") == "FAILURE":
                                fail_reason = data.get("fail_reason", "Unknown failure reason")
                                error_message = f"Image upload failed: {fail_reason}"
                                print(error_message)
                                raise Exception(error_message)
                                
                            if "result" in data and data["result"]:
                                return data["result"][0]
                            else:
                                error_message = f"Unexpected response from Midjourney API: {data}"
                                raise Exception(error_message)
                        except aiohttp.client_exceptions.ContentTypeError:
                            text_response = await response.text()
                            try:
                                import json
                                data = json.loads(text_response)
                                if data.get("status") == "FAILURE":
                                    fail_reason = data.get("fail_reason", "Unknown failure reason")
                                    error_message = f"Image upload failed: {fail_reason}"
                                    print(error_message)
                                    raise Exception(error_message)
                                    
                                if "result" in data and data["result"]:
                                    return data["result"][0]
                            except (json.JSONDecodeError, KeyError):
                                if text_response and len(text_response) < 100:
                                    return text_response.strip()
                                raise Exception(f"Invalid response format: {text_response}")
                    else:
                        error_message = f"Error uploading image to Midjourney: {response.status}"
                        try:
                            error_details = await response.text()
                            error_message += f" - {error_details}"
                        except:
                            pass
                        raise Exception(error_message)
                        
        except asyncio.TimeoutError:
            error_message = f"Timeout error: Request to upload image timed out after {self.timeout} seconds"
            raise Exception(error_message)


        
    def upload_image(self, image, api_key=""):
        if api_key.strip():
            self.api_key = api_key
            config = get_config()
            config['api_key'] = api_key
            save_config(config)
            
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            url = loop.run_until_complete(self.upload_image_to_midjourney(image))
            loop.close()
            return (url,)
        except Exception as e:
            raise e
        

class Comfly_Mj(ComflyBaseNode):
    """
    Comfly_Mj node
    Processes text or image inputs using Midjourney AI model and returns the processed results.
    Inputs:
        text (STRING, optional): Input text.
        api_key (STRING): API key for Midjourney.
        model_version (STRING): Selected Midjourney model version (v 6.1, v 6.0, v 5.2, v 5.1, niji 6, niji 5, niji 4).
        speed (STRING): Selected speed mode (turbo mode, fast mode, relax mode).
        ar (STRING): Aspect ratio.
        no (STRING): Number of images.
        c (STRING): Chaos value (0-100).
        s (STRING): Stylize value (0-1000).
        iw (STRING): Image weight (0-2).
        tile (BOOL): Enable/disable tile.
        r (STRING): Repeat value (1-40).
        video (BOOL): Enable/disable video.
        sw (STRING): Style weight (0-1000).
        cw (STRING): Color weight (0-100).
        sv (STRING): Style variation (1-4).
        seed (INT): Random seed.
        cref (STRING): Creative reference.
        oref (STRING): Object reference.
        sref (STRING): Style reference.
        positive (STRING): Additional positive prompt to be appended to the main prompt.
    Outputs:
        image_url (STRING): URL of the processed image.
        text (STRING): Processed text output.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True}),
                "speed": (["turbo mode", "fast mode", "relax mode"], {"default": "fast mode"}), 
            },
            "optional": {
                "text_en": ("STRING", {"multiline": True, "default": ""}),
                "api_key": ("STRING", {"default": ""}),  
                "ar": ("STRING", {"default": "1:1"}),
                "model_version": (["v 7", "v 6.1", "v 6.0", "v 5.2", "v 5.1", "niji 6", "niji 5", "niji 4"], {"default": "v 6.1"}),
                "no": ("STRING", {"default": "", "forceInput": True}),
                "c": ("INT", {"default": 0, "min": 0, "max": 100, "forceInput": True}),
                "s": ("INT", {"default": 0, "min": 0, "max": 1000, "forceInput": True}),
                "iw": ("FLOAT", {"default": 0, "min": 0, "max": 2, "forceInput": True}),
                "r": ("INT", {"default": 1, "min": 1, "max": 40, "forceInput": True}),
                "sw": ("INT", {"default": 0, "min": 0, "max": 1000, "forceInput": True}),
                "cw": ("INT", {"default": 0, "min": 0, "max": 100, "forceInput": True}),
                "sv": (["1", "2", "3", "4"], {"default": "1", "forceInput": True}),
                "oref": ("STRING", {"default": "none", "forceInput": True}),
                "cref": ("STRING", {"default": "none", "forceInput": True}),
                "sref": ("STRING", {"default": "none", "forceInput": True}),
                "positive": ("STRING", {"default": "", "forceInput": True}),                
                "video": ("BOOLEAN", {"default": False}),
                "tile": ("BOOLEAN", {"default": False}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING", "STRING")  
    RETURN_NAMES = ("image", "text", "taskId")    
    OUTPUT_NODE = True

    FUNCTION = "process_input"

    CATEGORY = "Comfly/Midjourney"

    def __init__(self):
        super().__init__()
        self.image = None
        self.text = ""

    def process_input(self, speed, text, text_en="", image=None, model_version=None, ar=None, no=None, c=None, s=None, iw=None, r=None, sw=None, cw=None, sv=None, video=False, tile=False, seed=0, cref="none", oref="none", sref="none", positive="", api_key=""):
        if api_key.strip():
            self.api_key = api_key
            config = get_config()
            config['api_key'] = api_key
            save_config(config)
            
        self.image = image
        self.speed = speed

        prompt = text_en if text_en else text

        if positive:
            prompt += f" {positive}"

        if model_version:
            prompt += f" --{model_version}"
        if ar:
            prompt += f" --ar {ar}"
        if no:
            prompt += f" --no {no}"
        if c:
            prompt += f" --c {c}"
        if s:
            prompt += f" --s {s}"
        if iw:
            prompt += f" --iw {iw}"
        if r:
            prompt += f" --r {r}"
        if sw:
            prompt += f" --sw {sw}"
        if cw:
            prompt += f" --cw {cw}"
        if sv:
            prompt += f" --sv {sv}"
        if video:
            prompt += " --video"
        if tile:
            prompt += " --tile"
        if oref != "none":
            prompt += f" --oref {oref}"    
        if cref != "none":
            prompt += f" --cref {cref}"
        if sref != "none":
            prompt += f" --sref {sref}"

        self.text = prompt

        if self.image is not None:
            image_url, text = self.process_image()
        elif self.text:
            pbar = comfy.utils.ProgressBar(10)
            image_url, text, taskId = self.process_text(pbar, ar, no, c, s, iw, tile, r, video, sw, cw, sv, seed)
            
            response = requests.get(image_url)
            image = Image.open(BytesIO(response.content))
            tensor_image = pil2tensor(image)
            return tensor_image, text, taskId
        else:
            raise ValueError("Either image or text input must be provided for Midjourney model.")
        
        return image_url, text, taskId

    async def process_text_midjourney(self, text, pbar, ar, no, c, s, iw, tile, r, video, sw, cw, sv, seed):
        try:
            taskId = await self.midjourney_submit_imagine_task(text, ar, no, c, s, iw, tile, r, video, sw, cw, sv, seed)
            print(f"Task ID: {taskId}")
            
            task_result = None
            while True:
                await asyncio.sleep(1)
                try:
                    task_result = await self.midjourney_fetch_task_result(taskId)
 
                    if task_result.get("status") == "FAILURE":
                        fail_reason = task_result.get("fail_reason", "Unknown failure reason")
                        error_message = f"Midjourney task failed: {fail_reason}"
                        print(error_message)  
                        raise Exception(error_message)  

                    if task_result.get("status") == "SUCCESS":
                        break
                        
                    progress = task_result.get("progress", 0)
                    try:
                        progress_int = int(progress[:-1])
                    except (ValueError, TypeError):
                        progress_int = 0
                    pbar.update_absolute(progress_int)
                except Exception as e:
                    if "Midjourney task failed" in str(e):
                        raise  
                    print(f"Error fetching task result: {str(e)}")
                    await asyncio.sleep(2)
                    continue

            image_url = task_result.get("imageUrl", "")
            prompt = task_result.get("prompt", text)

            U1 = self.generate_custom_id(task_result.get("id", taskId), "upsample", 1)
            U2 = self.generate_custom_id(task_result.get("id", taskId), "upsample", 2)
            U3 = self.generate_custom_id(task_result.get("id", taskId), "upsample", 3)
            U4 = self.generate_custom_id(task_result.get("id", taskId), "upsample", 4)

            return image_url, prompt, U1, U2, U3, U4, taskId 
        except Exception as e:
            print(f"Error in process_text_midjourney: {str(e)}")
            raise e

        
    def process_text(self, pbar, ar, no, c, s, iw, tile, r, video, sw, cw, sv, seed):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        image_url, text, U1, U2, U3, U4, taskId = loop.run_until_complete(self.process_text_midjourney(self.text, pbar, ar, no, c, s, iw, tile, r, video, sw, cw, sv, seed))
        loop.close()
        
        U = json.dumps({"U1": U1, "U2": U2, "U3": U3, "U4": U4})

        self.taskId = taskId
    
        return image_url, text, self.taskId

    def generate_custom_id(self, taskId, action, index):
        uuid = self.generateUUID()
        if action in ["upsample_v6_2x_subtle", "upsample_v6_2x_creative", "pan_left", "pan_right", "pan_up", "pan_down"]:
            return f"MJ::JOB::{action}::{index}::{uuid}::SOLO"
        elif action == "Outpaint::50":
            return f"MJ::Outpaint::50::{index}::{uuid}::SOLO"
        else:
            return f"MJ::JOB::{action}::{index}::{uuid}"
    
    
class Comfly_Mju(ComflyBaseNode):
    class MidjourneyError(Exception):
        pass
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "taskId": ("STRING", {"default": "", "forceInput": True}),
                "U1": ("BOOLEAN", {"default": False}),
                "U2": ("BOOLEAN", {"default": False}),
                "U3": ("BOOLEAN", {"default": False}),
                "U4": ("BOOLEAN", {"default": False})
            },
            "optional": {
                "api_key": ("STRING", {"default": ""})
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")  
    RETURN_NAMES = ("image", "taskId")  
    FUNCTION = "run"
    CATEGORY = "Comfly/Midjourney"

    def run(self, taskId, U1=False, U2=False, U3=False, U4=False, api_key=""):
        if api_key.strip():
            self.api_key = api_key
            config = get_config()
            config['api_key'] = api_key
            save_config(config)
            
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            results = loop.run_until_complete(self.process_input(taskId, U1, U2, U3, U4))
        finally:
            loop.close()
        return results

    async def process_input(self, taskId, U1=False, U2=False, U3=False, U4=False):
        try:
            if not any([U1, U2, U3, U4]):
                raise self.MidjourneyError("NO_ACTION_SELECTED")

            action = None
            index = None
            if U1:
                action = "upsample"
                index = 1
            elif U2:
                action = "upsample"
                index = 2
            elif U3:
                action = "upsample"
                index = 3
            elif U4:
                action = "upsample"
                index = 4

            task_result = await self.midjourney_fetch_task_result(taskId)

            if task_result.get("status") == "FAILURE":
                fail_reason = task_result.get("fail_reason", "Unknown failure reason")
                error_message = f"Original task failed: {fail_reason}"
                print(error_message)
                raise self.MidjourneyError(error_message)

            message_hash = task_result["properties"]["messageHash"]
            custom_id = self.generate_custom_id(action, index, message_hash)

            response = await self.midjourney_submit_action(action, taskId, index, custom_id)

            if isinstance(response, str) and response.startswith("Error"):
                raise self.MidjourneyError(response)

            if "result" not in response:
                raise self.MidjourneyError(f"Unexpected response from Midjourney API: {response}")

            new_task_id = response["result"]
            while True:
                await asyncio.sleep(1)
                task_result = await self.midjourney_fetch_task_result(new_task_id)

                if task_result.get("status") == "FAILURE":
                    fail_reason = task_result.get("fail_reason", "Unknown failure reason")
                    error_message = f"Task failed: {fail_reason}"
                    print(error_message)
                    raise self.MidjourneyError(error_message)

                if task_result.get("status") == "SUCCESS":
                    break

            if task_result.get("code") == 5 and task_result.get("description") == "task_no_found":
                raise self.MidjourneyError(f"Task not found for taskId: {new_task_id}")

            response = requests.get(task_result["imageUrl"])
            image = Image.open(BytesIO(response.content))
            tensor_image = pil2tensor(image)
            return tensor_image, new_task_id 

        except self.MidjourneyError as e:
            print(f"Midjourney API error: {str(e)}")
            raise e
        except Exception as e:
            print(f"An error occurred while processing input: {str(e)}")
            raise e

        

    def generate_custom_id(self, action, index, message_hash):
        if action == "zoom":
            return f"MJ::CustomZoom::{message_hash}"
        elif action in ["upsample_v6_2x_subtle", "upsample_v6_2x_creative", "upsample_v5_2x", "upsample_v5_4x",
                   "pan_left", "pan_right", "pan_up", "pan_down", "reroll"]:
            return f"MJ::JOB::{action}::{index}::{message_hash}::SOLO"
        elif action.startswith("Outpaint"):
            return f"MJ::{action}::{index}::{message_hash}::SOLO"
        elif action == "Inpaint":
            return f"MJ::{action}::1::{index}::{message_hash}::SOLO"
        elif action == "CustomZoom":
            return f"MJ::CustomZoom::{index}::{message_hash}"
        else:
            return f"MJ::JOB::{action}::{index}::{message_hash}"

    def generateUUID(self):
        return str(uuid.uuid4()) 

    async def process_custom_id(self, custom_id):
        if custom_id:
            try:
                action, index, uuid = self.parse_custom_id(custom_id)
                taskId = self.extract_taskId(custom_id)

                
                response = await self.midjourney_submit_action(action, taskId, index, custom_id)
                taskId = response["result"]

                
                task_result = None
                while not task_result or task_result.get("status") != "SUCCESS":
                    await asyncio.sleep(1)
                    task_result = await self.midjourney_fetch_task_result(taskId)
                
                image_url = task_result["imageUrl"]
                return image_url
                
            except Exception as e:
                print(f"Error processing custom_id: {custom_id}. Error: {str(e)}")
                raise ValueError(f"Error processing image from custom ID: {custom_id}. Error: {str(e)}")
        else:
            print("Custom ID is empty, returning empty string")
            return ""

    def parse_custom_id(self, custom_id):
        parts = custom_id.split("::")
        action = parts[2]
        index = int(parts[3])
        uuid = parts[4]
        return action, index, uuid

    def extract_custom_id(self, U, action, index):
        pattern = fr"MJ::JOB::{action}::{index}::(\w+)"
        match = re.search(pattern, U)
        if match:
            return f"MJ::JOB::{action}::{index}::{match.group(1)}"
        else:
            return None

    def extract_taskId(self, U, action, index):
        pattern = fr'"customId": "MJ::JOB::{action}::{index}::(.*?)"'
        match = re.search(pattern, U)
        if match:
            return match.group(1)
        else:
            raise ValueError(f"Invalid custom ID format in U: {U}")

    async def midjourney_submit_action(self, action, taskId, index, custom_id):
        headers = self.get_headers()
        payload = {
            "customId": custom_id,
            "taskId": taskId
        }
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(f"{self.midjourney_api_url[self.speed]}/mj/submit/action", headers=headers, json=payload, timeout=self.timeout) as response:
                    if response.status == 200:
                        try:
                            data = await response.json()

                            if data.get("status") == "FAILURE":
                                fail_reason = data.get("fail_reason", "Unknown failure reason")
                                error_message = f"Action submission failed: {fail_reason}"
                                print(error_message)
                                raise Exception(error_message)
                                
                            return data
                        except aiohttp.client_exceptions.ContentTypeError:
                            text_response = await response.text()
                            
                            try:
                                import json
                                data = json.loads(text_response)
 
                                if data.get("status") == "FAILURE":
                                    fail_reason = data.get("fail_reason", "Unknown failure reason")
                                    error_message = f"Action submission failed: {fail_reason}"
                                    print(error_message)
                                    raise Exception(error_message)
                                    
                                return data
                            except json.JSONDecodeError:
                                if text_response and len(text_response) < 100:
                                    return {"result": text_response.strip()}
                                raise Exception(f"Invalid response format: {text_response}")
                    else:
                        error_message = f"Error submitting Midjourney action: {response.status}"
                        try:
                            error_details = await response.text()
                            error_message += f" - {error_details}"
                        except:
                            pass
                        raise Exception(error_message)
        except asyncio.TimeoutError:
            error_message = f"Timeout error: Request to submit action timed out after {self.timeout} seconds"
            raise Exception(error_message)
        except Exception as e:
            if "Action submission failed" in str(e):
                raise 
            print(f"Exception in midjourney_submit_action: {str(e)}")
            raise e


    async def midjourney_fetch_task_result(self, taskId):
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer " + self.api_key
        }
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.midjourney_api_url[self.speed]}/mj/task/{taskId}/fetch", headers=headers, timeout=self.timeout) as response:
                    if response.status == 200:
                        try:
                            data = await response.json()

                            return data
                        except aiohttp.client_exceptions.ContentTypeError:
                            text_response = await response.text()
                            try:
                                import json
                                data = json.loads(text_response)
                                return data
                            except json.JSONDecodeError:
                                if text_response and len(text_response) < 100:
                                    return {"status": "SUCCESS", "progress": "100%", "imageUrl": text_response.strip()}
                                raise Exception(f"Server returned invalid response: {text_response}")
                    else:
                        error_message = f"Error fetching Midjourney task result: {response.status}"
                        try:
                            error_details = await response.text()
                            error_message += f" - {error_details}"
                        except:
                            pass
                        raise Exception(error_message)
        except asyncio.TimeoutError:
            error_message = f"Timeout error: Request to fetch task result timed out after {self.timeout} seconds"
            raise Exception(error_message)

                

    async def midjourney_submit_imagine_task(self, prompt, ar, no, c, s, iw, tile, r, video, sw, cw, sv, seed):
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer " + self.api_key
        }
        payload = {
            "base64Array": [],
            "instanceId": "",
            "modes": [],
            "notifyHook": "",
            "prompt": prompt,
            "remix": True,
            "state": "",
            "ar": ar,
            "no": no,
            "c": c,
            "s": s,
            "iw": iw,
            "tile": tile,
            "r": r,
            "video": video,
            "sw": sw,
            "cw": cw,
            "sv": sv,
            "seed": seed
        }
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(f"{self.midjourney_api_url[self.speed]}/mj/submit/imagine", headers=headers, json=payload, timeout=self.timeout) as response:
                    if response.status == 200:
                        try:
                            data = await response.json()

                            if data.get("status") == "FAILURE":
                                fail_reason = data.get("fail_reason", "Unknown failure reason")
                                error_message = f"Midjourney task failed: {fail_reason}"
                                print(error_message)
                                raise Exception(error_message)
                                
                            return data["result"]
                        except aiohttp.client_exceptions.ContentTypeError:
                            text_response = await response.text()
                            
                            try:
                                import json
                                data = json.loads(text_response)

                                if data.get("status") == "FAILURE":
                                    fail_reason = data.get("fail_reason", "Unknown failure reason")
                                    error_message = f"Midjourney task failed: {fail_reason}"
                                    print(error_message)
                                    raise Exception(error_message)
                                    
                                return data["result"]
                            except (json.JSONDecodeError, KeyError):
                                if text_response and len(text_response) < 100:
                                    return text_response.strip()
                                raise Exception(f"Server returned invalid response: {text_response}")
                    else:
                        error_message = f"Error submitting Midjourney task: {response.status}"
                        try:
                            error_details = await response.text()
                            error_message += f" - {error_details}"
                        except:
                            pass
                        raise Exception(error_message)
        except asyncio.TimeoutError:
            error_message = f"Timeout error: Request to submit imagine task timed out after {self.timeout} seconds"
            raise Exception(error_message)
        except Exception as e:
            if "Midjourney task failed" in str(e):
                raise  
            print(f"Exception in midjourney_submit_imagine_task: {str(e)}")
            raise e


    async def midjourney_fetch_task_result(self, taskId):
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer " + self.api_key
        }
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.midjourney_api_url[self.speed]}/mj/task/{taskId}/fetch", headers=headers, timeout=self.timeout) as response:
                    if response.status == 200:
                        try:
                            data = await response.json()

                            return data
                        except aiohttp.client_exceptions.ContentTypeError:
                            text_response = await response.text()
                            try:
                                import json
                                data = json.loads(text_response)
                                return data
                            except json.JSONDecodeError:
                                if text_response and len(text_response) < 100:
                                    return {"status": "SUCCESS", "progress": "100%", "imageUrl": text_response.strip()}
                                raise Exception(f"Server returned invalid response: {text_response}")
                    else:
                        error_message = f"Error fetching Midjourney task result: {response.status}"
                        try:
                            error_details = await response.text()
                            error_message += f" - {error_details}"
                        except:
                            pass
                        raise Exception(error_message)
        except asyncio.TimeoutError:
            error_message = f"Timeout error: Request to fetch task result timed out after {self.timeout} seconds"
            raise Exception(error_message)


class Comfly_Mjv(ComflyBaseNode):
    class MidjourneyError(Exception):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "taskId": ("STRING", {"default": "", "forceInput": True}),
            },
            "optional": {
                "api_key": ("STRING", {"default": ""}),
                "upsample_v6_2x_subtle": ("BOOLEAN", {"default": False}),
                "upsample_v6_2x_creative": ("BOOLEAN", {"default": False}),
                "costume_zoom": ("BOOLEAN", {"default": False}),
                "zoom": ("FLOAT", {"default": 1.0, "min": 1.0, "max": 2.0, "step": 0.1}),
                "pan_left": ("BOOLEAN", {"default": False}),
                "pan_right": ("BOOLEAN", {"default": False}),
                "pan_up": ("BOOLEAN", {"default": False}),
                "pan_down": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("IMAGE",) 
    RETURN_NAMES = ("image",) 
    FUNCTION = "run"
    CATEGORY = "Comfly/Midjourney"

    async def submit(self, action, taskId, index, extraParams=None):
        headers = self.get_headers()
        payload = {
            "taskId": taskId,
            "action": action  
        }
        if extraParams:
            payload.update(extraParams)

        async with aiohttp.ClientSession() as session:
            async with session.post(f"{self.midjourney_api_url[self.speed]}/mj/submit/action", headers=headers, json=payload) as response: 
                if response.status == 200:
                    try:
                        data = await response.json()
                        return data
                    except aiohttp.client_exceptions.ContentTypeError:
                        text_response = await response.text()
                        print(f"API returned non-JSON response: {text_response}")
                        try:
                            import json
                            data = json.loads(text_response)
                            return data
                        except json.JSONDecodeError:
                            if text_response and len(text_response) < 100:
                                return {"result": text_response.strip()}
                            raise Exception(f"Invalid response format: {text_response}")
                else:
                    error_message = f"Error submitting Midjourney action: {response.status}"
                    print(error_message)
                    try:
                        error_details = await response.text()
                        error_message += f" - {error_details}"
                    except:
                        pass
                    raise Exception(error_message)

    def run(self, taskId, upsample_v6_2x_subtle=False, upsample_v6_2x_creative=False, costume_zoom=False, zoom="", pan_left=False, pan_right=False, pan_up=False, pan_down=False, api_key=""):
        if api_key.strip():
            self.api_key = api_key
            config = get_config()
            config['api_key'] = api_key
            save_config(config)
       
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            image_url = loop.run_until_complete(self.process_input(taskId, upsample_v6_2x_subtle, upsample_v6_2x_creative, costume_zoom, zoom, pan_left, pan_right, pan_up, pan_down))
        finally:
            loop.close()
        
        return image_url

    async def process_input(self, taskId, upsample_v6_2x_subtle=False, upsample_v6_2x_creative=False, costume_zoom=False, zoom="", pan_left=False, pan_right=False, pan_up=False, pan_down=False):
        image_url = ""

        if taskId:
            try:
                task_result = await self.midjourney_fetch_task_result(taskId)

                if task_result.get("status") == "FAILURE":
                    fail_reason = task_result.get("fail_reason", "Unknown failure reason")
                    error_message = f"Original task failed: {fail_reason}"
                    print(error_message)
                    raise Exception(error_message)
                    
                message_hash = task_result["properties"]["messageHash"]
                prompt = task_result["prompt"]
                text_en = prompt

                if costume_zoom and zoom:
                    zoom_param = f" --zoom {zoom}"
                    task_result = await self.midjourney_fetch_task_result(taskId)
                    prompt = task_result["properties"]["finalPrompt"] + zoom_param

                    custom_id = self.generate_custom_id("zoom", 1, task_result["properties"]["messageHash"])
                    response = await self.submit("ACTION", taskId, 1, {"customId": custom_id})

                    response = await self.submit_modal(prompt, response["result"])

                    image_url = await self.process_task(response["result"])

                elif upsample_v6_2x_subtle:
                    custom_id = self.generate_custom_id("upsample_v6_2x_subtle", 1, message_hash)
                    response = await self.submit("upsample_v6_2x_subtle", taskId, 1, {"customId": custom_id})
                    image_url = await self.process_task(response["result"])
                elif upsample_v6_2x_creative:
                    custom_id = self.generate_custom_id("upsample_v6_2x_creative", 1, message_hash)
                    response = await self.submit("upsample_v6_2x_creative", taskId, 1, {"customId": custom_id})
                    image_url = await self.process_task(response["result"])
                elif pan_left:
                    custom_id = self.generate_custom_id("pan_left", 1, message_hash)
                    response = await self.submit("pan_left", taskId, 1, {"customId": custom_id})
                    image_url = await self.process_task(response["result"])
                elif pan_right:
                    custom_id = self.generate_custom_id("pan_right", 1, message_hash)
                    response = await self.submit("pan_right", taskId, 1, {"customId": custom_id})
                    image_url = await self.process_task(response["result"])
                elif pan_up:
                    custom_id = self.generate_custom_id("pan_up", 1, message_hash)
                    response = await self.submit("pan_up", taskId, 1, {"customId": custom_id})
                    image_url = await self.process_task(response["result"])
                elif pan_down:
                    custom_id = self.generate_custom_id("pan_down", 1, message_hash)
                    response = await self.submit("pan_down", taskId, 1, {"customId": custom_id})
                    image_url = await self.process_task(response["result"])
                    
                if image_url:
                    response = requests.get(image_url)
                    image = Image.open(BytesIO(response.content))
                    tensor_image = pil2tensor(image)
                    return (tensor_image,)
                else:
                    return (None,)

            except Exception as e:
                error_message = f"Error processing action: {str(e)}"
                print(error_message)
                return (error_message,)
        else:
            print("No taskId provided.")
            return ("No taskId provided.",)

        return (image_url,)


    
    async def submit_modal(self, prompt, taskId):
        headers = self.get_headers()
        payload = {
            "prompt": prompt,
            "taskId": taskId
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(f"{self.midjourney_api_url[self.speed]}/mj/submit/modal", headers=headers, json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    return data
                else:
                    error_message = f"Error submitting Midjourney modal: {response.status}"
                    print(error_message)
                    raise Exception(error_message)
    
    def generate_custom_id(self, action, index, message_hash):
        if action == "zoom":
            return f"MJ::CustomZoom::{message_hash}"
        elif action in ["upsample_v6_2x_subtle", "upsample_v6_2x_creative", "upsample_v5_2x", "upsample_v5_4x",
                   "pan_left", "pan_right", "pan_up", "pan_down", "reroll"]:
            return f"MJ::JOB::{action}::{index}::{message_hash}::SOLO"
        elif action.startswith("Outpaint"):
            return f"MJ::{action}::{index}::{message_hash}::SOLO"
        elif action == "Inpaint":
            return f"MJ::{action}::1::{index}::{message_hash}::SOLO"
        elif action == "CustomZoom":
            return f"MJ::CustomZoom::{index}::{message_hash}"
        else:
            return f"MJ::JOB::{action}::{index}::{message_hash}"

    async def process_task(self, taskId):
        while True:
            await asyncio.sleep(1)
            try:
                task_result = await self.midjourney_fetch_task_result(taskId)

                if task_result.get("status") == "FAILURE":
                    fail_reason = task_result.get("fail_reason", "Unknown failure reason")
                    error_message = f"Task failed: {fail_reason}"
                    print(error_message)
                    raise Exception(error_message)

                if task_result.get("status") == "SUCCESS":
                    return task_result["imageUrl"]
                    
            except Exception as e:
                if "Task failed:" in str(e):
                    raise  
                error_message = f"Error fetching task result: {str(e)}"
                print(error_message)
                raise Exception(error_message)


    async def midjourney_fetch_task_result(self, taskId):
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer " + self.api_key
        }
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.midjourney_api_url[self.speed]}/mj/task/{taskId}/fetch", headers=headers, timeout=self.timeout) as response:
                    if response.status == 200:
                        try:
                            data = await response.json()

                            return data
                        except aiohttp.client_exceptions.ContentTypeError:
                            text_response = await response.text()
                            try:
                                import json
                                data = json.loads(text_response)
                                return data
                            except json.JSONDecodeError:
                                if text_response and len(text_response) < 100:
                                    return {"status": "SUCCESS", "progress": "100%", "imageUrl": text_response.strip()}
                                raise Exception(f"Server returned invalid response: {text_response}")
                    else:
                        error_message = f"Error fetching Midjourney task result: {response.status}"
                        try:
                            error_details = await response.text()
                            error_message += f" - {error_details}"
                        except:
                            pass
                        raise Exception(error_message)
        except asyncio.TimeoutError:
            error_message = f"Timeout error: Request to fetch task result timed out after {self.timeout} seconds"
            raise Exception(error_message)


class Comfly_mjstyle:
    styleAll = {}

    def __init__(self):
        dir = os.path.join(os.path.dirname(__file__), "docs", "mjstyle")
        if not os.path.exists(dir):
            os.mkdir(dir)
        for root, dirs, files in os.walk(dir):
            for file in files:
                if file.endswith(".json"):
                    with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        for d in data:
                            Comfly_mjstyle.styleAll[d['name']] = d

    @classmethod
    def INPUT_TYPES(cls):
        dir = os.path.join(os.path.dirname(__file__), "docs", "mjstyle")
        files_name = []
        for root, dirs, files in os.walk(dir):
            for file in files:
                if file.endswith(".json"):
                    files_name.append(file.split(".")[0])

        return {
            "required": {
                "styles_type": (files_name,),
            },
            "optional": {
                "positive": ("STRING", {"forceInput": True}),
                "negative": ("STRING", {"forceInput": True}),
            },
            "hidden": {"extra_pnginfo": "EXTRA_PNGINFO", "unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("STRING", "STRING",)
    RETURN_NAMES = ("style_positive", "style_negative",)

    FUNCTION = "run"

    OUTPUT_NODE = True

    CATEGORY = "Comfly/Midjourney"

    def replace_repeat(self, prompt):
        prompt = prompt.replace("，", ",")
        arr = prompt.split(",")
        if len(arr) != len(set(arr)):
            all_weight_prompt = re.findall(re.compile(r'[(](.*?)[)]', re.S), prompt)
            if len(all_weight_prompt) > 0:
                return prompt
            else:
                arr = [item.strip() for item in arr]
                arr = list(set(arr))
                return ", ".join(arr)
        else:
            return prompt

    def run(self, extra_pnginfo, unique_id, styles_type, positive="", negative=""):
        values = []
        for node in extra_pnginfo["workflow"]["nodes"]:
            if node["id"] == int(unique_id):
                values = node["properties"]["values"]
                break

        style_positive = ""
        style_negative = ""
        has_prompt = False

        for val in values:
            if "prompt" in Comfly_mjstyle.styleAll[val]:
                if "{positive}" in Comfly_mjstyle.styleAll[val]["prompt"] and not has_prompt:
                    style_positive = Comfly_mjstyle.styleAll[val]["prompt"].format(positive=positive)
                    has_prompt = True
                else:
                    style_positive += ", " + Comfly_mjstyle.styleAll[val]["prompt"].replace(", {positive}", "").replace("{positive}", "")
            if "negative_prompt" in Comfly_mjstyle.styleAll[val]:
                style_negative += ', ' + Comfly_mjstyle.styleAll[val]['negative_prompt'] if negative else Comfly_mjstyle.styleAll[val]['negative_prompt']

        style_positive = self.replace_repeat(style_positive) if style_positive else ""
        style_negative = self.replace_repeat(style_negative) if style_negative else ""

        return (style_positive, style_negative)


class Comfly_mj_video(ComflyBaseNode):
    """
    Comfly_mj_video node
    Generates videos using Midjourney's video API based on text prompts or text+image combination.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "motion": (["Low", "high"], {"default": "Low"}),
            },
            "optional": {
                "api_key": ("STRING", {"default": ""}),
                "image": ("IMAGE",),
                "notify_hook": ("STRING", {"default": ""}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647}),
            }
        }
    
    RETURN_TYPES = (IO.VIDEO, IO.VIDEO, IO.VIDEO, IO.VIDEO, "STRING", "STRING")
    RETURN_NAMES = ("video1", "video2", "video3", "video4", "task_id", "response")
    FUNCTION = "generate_video"
    CATEGORY = "Comfly/Midjourney"

    def __init__(self):
        super().__init__()
        self.api_key = get_config().get('api_key', '')
        self.timeout = 600  

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

    def extract_video_urls(self, response_data):
        video_urls = []
 
        if "video_urls" in response_data and response_data["video_urls"]:
            print(f"Found video_urls field: {response_data['video_urls']}")
            try:
                if isinstance(response_data["video_urls"], str):
                    video_urls_json = json.loads(response_data["video_urls"])
                    print(f"Parsed video_urls JSON: {json.dumps(video_urls_json, indent=2)}")
                    
                    if isinstance(video_urls_json, list):
                        urls = [item["url"] for item in video_urls_json if "url" in item]
                        if urls:
                            video_urls = urls
                            print(f"Extracted {len(urls)} URLs from video_urls list")
                        else:
                            print("No 'url' keys found in video_urls items")
                    else:
                        print(f"video_urls_json is not a list: {type(video_urls_json)}")
                elif isinstance(response_data["video_urls"], list):
                    urls = [item["url"] for item in response_data["video_urls"] if "url" in item]
                    if urls:
                        video_urls = urls
                        print(f"Extracted {len(urls)} URLs directly from video_urls list")
                    else:
                        print("No 'url' keys found in video_urls items")
                else:
                    print(f"video_urls is neither string nor list: {type(response_data['video_urls'])}")
            except json.JSONDecodeError as e:
                print(f"JSONDecodeError when parsing video_urls: {str(e)}")

                if isinstance(response_data["video_urls"], str) and response_data["video_urls"].startswith("http"):
                    video_urls = [response_data["video_urls"]]
                    print(f"Using video_urls as a direct URL: {response_data['video_urls']}")
            except Exception as e:
                print(f"Unexpected error when processing video_urls: {str(e)}")

        if not video_urls:
            if "videoUrl" in response_data and response_data["videoUrl"]:
                video_urls = [response_data["videoUrl"]]
                print(f"Using videoUrl field: {response_data['videoUrl']}")
            elif "video_url" in response_data and response_data["video_url"]:
                video_urls = [response_data["video_url"]]
                print(f"Using video_url field: {response_data['video_url']}")

            if "properties" in response_data and response_data["properties"]:
                try:
                    if isinstance(response_data["properties"], str):
                        props = json.loads(response_data["properties"])
                        if "messageHash" in props:
                            msg_hash = props["messageHash"]
                            print(f"Found messageHash in properties: {msg_hash}")
                except Exception as e:
                    print(f"Error processing properties field: {str(e)}")
        
        print(f"Final extracted video URLs: {video_urls}")
        return video_urls

    async def submit_video_request(self, payload):
        """Submit video generation request to Midjourney API"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://ai.comfly.chat/mj/submit/video", 
                    headers=self.get_headers(), 
                    json=payload, 
                    timeout=self.timeout
                ) as response:
                    if response.status == 200:
                        try:
                            data = await response.json()
                            if data.get("code") != 1:  
                                error_message = f"Video generation request failed: {data.get('description', 'Unknown error')}"
                                print(error_message)
                                raise Exception(error_message)
                            return data
                        except aiohttp.client_exceptions.ContentTypeError:
                            text_response = await response.text()
                            try:
                                import json
                                data = json.loads(text_response)
                                if data.get("code") != 1:
                                    error_message = f"Video generation request failed: {data.get('description', 'Unknown error')}"
                                    print(error_message)
                                    raise Exception(error_message)
                                return data
                            except json.JSONDecodeError:
                                raise Exception(f"Invalid response format: {text_response}")
                    else:
                        error_message = f"Error submitting video request: {response.status}"
                        try:
                            error_details = await response.text()
                            error_message += f" - {error_details}"
                        except:
                            pass
                        raise Exception(error_message)
        except Exception as e:
            print(f"Error in submit_video_request: {str(e)}")
            raise e

    async def fetch_video_result(self, task_id):
        """Fetch video generation result"""
        max_attempts = 120  
        attempts = 0
        
        while attempts < max_attempts:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        f"https://ai.comfly.chat/mj/task/{task_id}/fetch", 
                        headers=self.get_headers(), 
                        timeout=self.timeout
                    ) as response:
                        if response.status == 200:
                            try:
                                data = await response.json()
                                if data.get("status") == "FAILURE":
                                    fail_reason = data.get("fail_reason", "Unknown failure reason")
                                    error_message = f"Video generation failed: {fail_reason}"
                                    print(error_message)
                                    raise Exception(error_message)
                                
                                if data.get("status") == "SUCCESS":
                                    return data

                                await asyncio.sleep(5)
                                attempts += 1
                                
                            except aiohttp.client_exceptions.ContentTypeError:
                                text_response = await response.text()
                                try:
                                    data = json.loads(text_response)
                                    
                                    if data.get("status") == "FAILURE":
                                        fail_reason = data.get("fail_reason", "Unknown failure reason")
                                        raise Exception(f"Video generation failed: {fail_reason}")
                                    
                                    if data.get("status") == "SUCCESS":
                                        return data
                                        
                                except json.JSONDecodeError:
                                    await asyncio.sleep(5)
                                    attempts += 1
                        else:
                            error_message = f"Error fetching video result: {response.status}"
                            try:
                                error_details = await response.text()
                                error_message += f" - {error_details}"
                            except:
                                pass
                            print(error_message)
                            await asyncio.sleep(5)
                            attempts += 1
            except Exception as e:
                if "Video generation failed" in str(e):
                    raise
                print(f"Error in fetch attempt {attempts}: {str(e)}")
                await asyncio.sleep(5)
                attempts += 1
                
        raise Exception(f"Timeout waiting for video generation after {max_attempts} attempts")

    def generate_video(self, prompt, motion="Low", api_key="", image=None, notify_hook="", seed=0):
        if api_key.strip():
            self.api_key = api_key
            config = get_config()
            config['api_key'] = api_key
            save_config(config)
            
        if not self.api_key:
            error_response = json.dumps({
                "status": "error",
                "message": "API key not provided. Please set your API key."
            })
            empty_adapter = ComflyVideoAdapter("")
            return (empty_adapter, empty_adapter, empty_adapter, empty_adapter, "", error_response)
            
        pbar = comfy.utils.ProgressBar(100)
        pbar.update_absolute(10)
        
        try:
            payload = {
                "prompt": prompt,
                "motion": motion,
                "notifyHook": notify_hook
            }

            if seed > 0:
                payload["seed"] = seed
 
            if image is not None:
                pbar.update_absolute(20)
                print("Processing input image...")
                image_base64 = self.image_to_base64(image)
                if image_base64:
                    payload["image"] = image_base64
                else:
                    print("Warning: Failed to convert image to base64")

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(self.submit_video_request(payload))
                
                if "result" not in result:
                    error_message = f"No task ID in response: {result}"
                    print(error_message)
                    error_response = json.dumps({
                        "status": "error", 
                        "message": error_message
                    })
                    empty_adapter = ComflyVideoAdapter("")
                    return (empty_adapter, empty_adapter, empty_adapter, empty_adapter, "", error_response)
                    
                task_id = result["result"]
                print(f"Video generation task submitted successfully. Task ID: {task_id}")
                
                pbar.update_absolute(40)
                print("Waiting for video generation to complete...")

                response_data = loop.run_until_complete(self.fetch_video_result(task_id))
                
                print(f"Video generation completed. Processing results...")
                pbar.update_absolute(90)

                video_urls = self.extract_video_urls(response_data)
            
                if not video_urls:
                    if isinstance(response_data, dict) and "video_urls" in response_data:
                        video_urls_data = response_data["video_urls"]
                        print(f"Trying alternative parsing for video_urls: {video_urls_data}")
                        
                        if isinstance(video_urls_data, str) and "[{" in video_urls_data:
                            try:
                                cleaned_data = video_urls_data.replace("&quot;", '"')
                                parsed_data = json.loads(cleaned_data)
                                if isinstance(parsed_data, list):
                                    video_urls = [item["url"] for item in parsed_data if "url" in item]
                                    print(f"Successfully extracted {len(video_urls)} URLs using alternative method")
                            except Exception as e:
                                print(f"Alternative parsing failed: {str(e)}")
                
                if not video_urls:
                    response_str = json.dumps(response_data)
                    video_url_pattern = r'https?://\S+\.mp4'
                    found_urls = re.findall(video_url_pattern, response_str)
                    if found_urls:
                        video_urls = found_urls
                        print(f"Found {len(found_urls)} video URLs using regex: {found_urls}")
    
                if not video_urls:
                    error_message = "No video URLs found in response"
                    print(error_message)
                    error_response = json.dumps({
                        "status": "error",
                        "message": error_message,
                        "task_id": task_id,
                        "response_data": response_data
                    })
                    empty_adapter = ComflyVideoAdapter("")
                    return (empty_adapter, empty_adapter, empty_adapter, empty_adapter, task_id, error_response)

                video_adapters = []
                for url in video_urls[:4]:
                    video_adapters.append(ComflyVideoAdapter(url))

                while len(video_adapters) < 4:
                    video_adapters.append(ComflyVideoAdapter(""))

                success_response = json.dumps({
                    "status": "success",
                    "message": f"Generated {len(video_urls)} videos successfully",
                    "task_id": task_id,
                    "video_urls": video_urls
                })

                pbar.update_absolute(100)
                return (video_adapters[0], video_adapters[1], video_adapters[2], video_adapters[3], task_id, success_response)
                
            finally:
                loop.close()
                
        except Exception as e:
            error_message = f"Error in video generation: {str(e)}"
            print(error_message)
            error_response = json.dumps({
                "status": "error",
                "message": error_message
            })

            empty_adapter = ComflyVideoAdapter("")
            return (empty_adapter, empty_adapter, empty_adapter, empty_adapter, "", error_response)



############################# Kling ###########################

class Comfly_kling_text2video:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "model_name": (["kling-v2-master", "kling-v1-6", "kling-v1-5", "kling-v1"], {"default": "kling-v1-6"}),
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
            save_config(config)
            
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
                "https://ai.comfly.chat/kling/v1/videos/text2video",
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
                    f"https://ai.comfly.chat/kling/v1/videos/text2video/{task_id}",
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
                "model_name": (["kling-v2-master", "kling-v1-6", "kling-v1-5", "kling-v1"], {"default": "kling-v1-6"}),
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
            }
        }

    def get_headers(self):
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
    def check_tail_image_compatibility(self, model_name, mode, duration):
        try:
            return self.model_compatibility.get(model_name, {}).get(mode, {}).get(duration, False)
        except:
            return False

    def generate_video(self, image, prompt, model_name, imagination, aspect_ratio, mode, duration, 
                  num_videos, negative_prompt="", camera="none", camera_value=0, seed=0, image_tail=None, api_key=""):
        if api_key.strip():
            self.api_key = api_key
            config = get_config()
            config['api_key'] = api_key
            save_config(config)
            
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
                "https://ai.comfly.chat/kling/v1/videos/image2video",
                headers=self.get_headers(),
                json=payload,
                timeout=self.timeout
            )
            
            # Log the response status
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
                    f"https://ai.comfly.chat/kling/v1/videos/image2video/{task_id}",
                    headers=self.get_headers(),
                    timeout=self.timeout
                )
                status_response.raise_for_status()
                status_result = status_response.json()
                last_status = status_result["data"]
                
                # Update progress based on the returned status
                progress = 0
                if status_result["data"]["task_status"] == "processing":
                    # Estimate progress if not provided explicitly
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
            save_config(config)
            
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
                "https://ai.comfly.chat/kling/v1/videos/video-extend",
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
                    f"https://ai.comfly.chat/kling/v1/videos/video-extend/{task_id}",
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
            save_config(config)
            
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
                "https://ai.comfly.chat/kling/v1/videos/lip-sync",
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
                    f"https://ai.comfly.chat/kling/v1/videos/lip-sync/{task_id}",
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
   


############################# Gemini ###########################

class ComflyGeminiAPI:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "model": ("STRING", {"default": "gemini-2.0-flash-exp-image", "placeholder": "Enter model name"}),
                "resolution": (
                    [
                        "512x512", 
                        "768x768", 
                        "1024x1024", 
                        "1280x1280", 
                        "1536x1536", 
                        "2048x2048",
                        "object_image size",
                        "subject_image size",
                        "scene_image size"
                    ], 
                    {"default": "1024x1024"}
                ),
                "num_images": ("INT", {"default": 1, "min": 1, "max": 4, "step": 1}),
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "top_p": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 1.0, "step": 0.01}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647}),
                "timeout": ("INT", {"default": 120, "min": 10, "max": 600, "step": 10}),
            },
            "optional": {
                "api_key": ("STRING", {"default": ""}),
                "object_image": ("IMAGE",),  
                "subject_image": ("IMAGE",),
                "scene_image": ("IMAGE",),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("generated_images", "response", "image_url")
    FUNCTION = "process"
    CATEGORY = "Comfly/Comfly_Gemini"

    def __init__(self):
        self.api_key = get_config().get('api_key', '')
        self.timeout = 120 

    def get_headers(self):
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

    def image_to_base64(self, image):
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    def extract_image_urls(self, response_text):
        # Extract URLs from markdown image format: ![description](url)
        image_pattern = r'!\[.*?\]\((.*?)\)'
        matches = re.findall(image_pattern, response_text)
        
        # If no markdown format found, extract raw URLs that look like image links
        if not matches:
            url_pattern = r'https?://\S+\.(?:jpg|jpeg|png|gif|webp)'
            matches = re.findall(url_pattern, response_text)
            
        return matches if matches else []

    def resize_to_target_size(self, image, target_size):
        """Resize image to target size while preserving aspect ratio with padding"""
        # Convert PIL image to target size
        img_width, img_height = image.size
        target_width, target_height = target_size
        
        # Calculate the scaling factor to fit the image within the target size
        width_ratio = target_width / img_width
        height_ratio = target_height / img_height
        scale = min(width_ratio, height_ratio)
        
        # Calculate new dimensions
        new_width = int(img_width * scale)
        new_height = int(img_height * scale)
        
        # Resize the image
        resized_img = image.resize((new_width, new_height), Image.LANCZOS)
        
        # Create a new blank image with the target size
        new_img = Image.new("RGB", (target_width, target_height), (255, 255, 255))
        
        # Calculate position to paste the resized image
        paste_x = (target_width - new_width) // 2
        paste_y = (target_height - new_height) // 2
        
        # Paste the resized image
        new_img.paste(resized_img, (paste_x, paste_y))
        
        return new_img

    def parse_resolution(self, resolution_str):
        """Parse resolution string (e.g., '1024x1024') to width and height"""
        width, height = map(int, resolution_str.split('x'))
        return (width, height)

    def process(self, prompt, model, resolution, num_images, temperature, top_p, seed, timeout=120, 
                object_image=None, subject_image=None, scene_image=None, api_key=""):
        # Update API key if provided
        if api_key.strip():
            self.api_key = api_key
            config = get_config()
            config['api_key'] = api_key
            save_config(config)
            
        # Set the timeout value from user input
        self.timeout = timeout
        
        try:
            
            # Get current timestamp for formatting
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            
            # Get target size based on resolution or input images
            target_size = None
            
            # Determine if we should use an input image's size
            if resolution == "object_image size" and object_image is not None:
                pil_image = tensor2pil(object_image)[0]
                target_size = pil_image.size
            elif resolution == "subject_image size" and subject_image is not None:
                pil_image = tensor2pil(subject_image)[0]
                target_size = pil_image.size
            elif resolution == "scene_image size" and scene_image is not None:
                pil_image = tensor2pil(scene_image)[0]
                target_size = pil_image.size
            else:
                # Use the specified resolution
                target_size = self.parse_resolution(resolution)
            
            # Check if we have image inputs
            has_images = object_image is not None or subject_image is not None or scene_image is not None
            
            # Prepare message content
            content = []
            
            # Build different prompts and content based on input type
            if has_images:
                # When we have image inputs, use the original prompt without enhancements
                content.append({"type": "text", "text": prompt})
                
                # Prepare descriptions for each image type
                image_descriptions = {
                    "object_image": "an object or item",
                    "subject_image": "a subject or character",
                    "scene_image": "a scene or environment"
                }
                
                # Add available images to content
                for image_var, image_tensor in [("object_image", object_image), 
                                             ("subject_image", subject_image), 
                                             ("scene_image", scene_image)]:
                    if image_tensor is not None:
                        # Convert tensor to PIL image
                        pil_image = tensor2pil(image_tensor)[0]
                        image_base64 = self.image_to_base64(pil_image)
                        content.append({
                            "type": "image_url", 
                            "image_url": {"url": f"data:image/png;base64,{image_base64}"}
                        })
            else:
                # When we only have text input, add enhanced prompt
                # Get dimensions string from target_size
                dimensions = f"{target_size[0]}x{target_size[1]}"
                aspect_ratio = "1:1" if target_size[0] == target_size[1] else f"{target_size[0]}:{target_size[1]}"
                
                if num_images == 1:
                    enhanced_prompt = f"Generate a high-quality, detailed image with dimensions {dimensions} and aspect ratio {aspect_ratio}. Based on this description: {prompt}"
                else:
                    enhanced_prompt = f"Generate {num_images} DIFFERENT high-quality images with VARIED content, each with unique and distinct visual elements, all having the exact same dimensions of {dimensions} and aspect ratio {aspect_ratio}. Important: make sure each image has different content but maintains the same technical dimensions. Based on this description: {prompt}"
                
                content.append({"type": "text", "text": enhanced_prompt})
            
            # Create messages
            messages = [{
                "role": "user",
                "content": content
            }]
            
            # Create API payload
            payload = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "top_p": top_p,
                "seed": seed if seed > 0 else None,
                "max_tokens": 8192
            }
            
            # Make API request with progress bar
            pbar = comfy.utils.ProgressBar(100)
            pbar.update_absolute(10)
            
            # Make API request with timeout
            try:
                response = requests.post(
                    "https://ai.comfly.chat/v1/chat/completions",
                    headers=self.get_headers(),
                    json=payload,
                    timeout=self.timeout
                )
                response.raise_for_status()
                result = response.json()
            except requests.exceptions.Timeout:
                raise TimeoutError(f"API request timed out after {self.timeout} seconds")
            except requests.exceptions.RequestException as e:
                raise Exception(f"API request failed: {str(e)}")
            
            pbar.update_absolute(40)
            
            # Extract response text
            response_text = result["choices"][0]["message"]["content"]
            
            # Format the response
            formatted_response = f"**User prompt**: {prompt}\n\n**Response** ({timestamp}):\n{response_text}"
            
            # Check if response contains image URLs
            image_urls = self.extract_image_urls(response_text)
            
            if image_urls:
                try:
                    # Process all images from URLs
                    images = []
                    first_image_url = ""  # Store the first image URL
                    
                    for i, url in enumerate(image_urls):
                        pbar.update_absolute(40 + (i+1) * 50 // len(image_urls))
                        
                        if i == 0:
                            first_image_url = url  # Save the first URL
                        
                        try:
                            img_response = requests.get(url, timeout=self.timeout)
                            img_response.raise_for_status()
                            pil_image = Image.open(BytesIO(img_response.content))
                            
                            # Resize the image to the target size if necessary
                            resized_image = self.resize_to_target_size(pil_image, target_size)
                            
                            # Convert to tensor
                            img_tensor = pil2tensor(resized_image)
                            images.append(img_tensor)
                            
                        except Exception as img_error:
                            print(f"Error processing image URL {i+1}: {str(img_error)}")
                            # Continue to next image if there's an error with this one
                            continue
                    
                    if images:
                        # If all images are the same size, we can use torch.cat
                        try:
                            combined_tensor = torch.cat(images, dim=0)
                        except RuntimeError:
                            # If images are different sizes, we'll need to handle them individually
                            print("Warning: Images have different sizes, returning first image")
                            combined_tensor = images[0] if images else None
                            
                        pbar.update_absolute(100)
                        return (combined_tensor, formatted_response, first_image_url)
                    else:
                        # If no images were successfully processed
                        raise Exception("No images could be processed successfully")
                    
                except Exception as e:
                    print(f"Error processing image URLs: {str(e)}")
            
            # Return appropriate response if no image URLs were found
            pbar.update_absolute(100)
            
            # Determine which image to return in case of no output images
            reference_image = None
            if object_image is not None:
                reference_image = object_image
            elif subject_image is not None:
                reference_image = subject_image
            elif scene_image is not None:
                reference_image = scene_image
                
            if reference_image is not None:
                # If any input image was provided, return the first available one with the text response
                return (reference_image, formatted_response, "")
            else:
                # Create a default blank image with the target size
                default_image = Image.new('RGB', target_size, color='white')
                default_tensor = pil2tensor(default_image)
                return (default_tensor, formatted_response, "")
            
        except TimeoutError as e:
            error_message = f"API timeout error: {str(e)}"
            print(error_message)
            return self.handle_error(object_image, subject_image, scene_image, error_message, resolution)
            
        except Exception as e:
            error_message = f"Error calling Gemini API: {str(e)}"
            print(error_message)
            return self.handle_error(object_image, subject_image, scene_image, error_message, resolution)
    
    def handle_error(self, object_image, subject_image, scene_image, error_message, resolution="1024x1024"):
        """Handle errors with appropriate image output"""
        # Return the first available image if any
        if object_image is not None:
            return (object_image, error_message, "")
        elif subject_image is not None:
            return (subject_image, error_message, "")
        elif scene_image is not None:
            return (scene_image, error_message, "")
        else:
            # Create an error image with the specified resolution
            # Handle custom resolution options
            if resolution in ["object_image size", "subject_image size", "scene_image size"]:
                target_size = (1024, 1024)  # Default if custom option selected but no image provided
            else:
                target_size = self.parse_resolution(resolution)
                
            default_image = Image.new('RGB', target_size, color='white')
            default_tensor = pil2tensor(default_image)
            return (default_tensor, error_message, "")



############################# Doubao ###########################

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
                "https://ai.comfly.chat/v1/files",
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
            save_config(config)
            
        try:
            if not self.api_key:
                error_message = "API key not found in Comflyapi.json"
                print(error_message)

                blank_image = Image.new('RGB', (width, height), color='white')
                blank_tensor = pil2tensor(blank_image)
                return (blank_tensor, error_message, "")
                
            # Initialize progress bar
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
            
            # Modified prompt if using image
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

                api_url = "https://ai.comfly.chat/v1/chat/completions"
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
                api_url = "https://ai.comfly.chat/volcv/v1?Action=CVProcess&Version=2022-08-31"
                
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
                "https://ai.comfly.chat/v1/files",
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
            save_config(config)
            
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
                "https://ai.comfly.chat/jimeng/submit/videos",
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
                        f"https://ai.comfly.chat/jimeng/fetch/{task_id}",
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
            save_config(config)
            
        try:
            if not self.api_key:
                error_message = "API key not found in Comflyapi.json"
                print(error_message)
                return (image, error_message, "")
                
            # Convert tensor to PIL image
            pil_image = tensor2pil(image)[0]
            
            # Convert image to base64
            buffered = BytesIO()
            pil_image.save(buffered, format="JPEG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
            
            # Initialize progress bar
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
            
            # Prepare the API request
            payload = {
                "req_key": "byteedit_v2.0",
                "binary_data_base64": [img_base64],
                "prompt": prompt,
                "scale": scale,
                "seed": seed,
                "return_url": True,
                "logo_info": logo_info
            }
            
            # Call the API
            api_url = "https://ai.comfly.chat/volcv/v1?Action=CVProcess&Version=2022-08-31"
            
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
            
            # Check for status code
            if response.status_code != 200:
                error_message = f"API Error: Status {response.status_code}\nResponse: {response.text}"
                print(error_message)
                response_info += f"Error: {error_message}"
                return (image, response_info, "")
                
            result = response.json()
            
            pbar.update_absolute(70)
            
            # Check for API errors
            if result.get("code") != 10000:
                error_message = f"API Error: {result.get('message', 'Unknown error')}\nDetails: {json.dumps(result, indent=2)}"
                print(error_message)
                response_info += f"Error: {error_message}"
                return (image, response_info, "")
            
            # Get the result image URL
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
            
            # Download the image
            try:
                img_response = requests.get(image_url, timeout=self.timeout)
                img_response.raise_for_status()
            except requests.exceptions.Timeout:
                error_message = f"Timeout while downloading result image after {self.timeout} seconds"
                print(error_message)
                response_info += f"Error: {error_message}"
                return (image, response_info, image_url)  # Return the URL even though download failed
            except Exception as e:
                error_message = f"Error downloading result image: {str(e)}"
                print(error_message)
                response_info += f"Error: {error_message}"
                return (image, response_info, image_url)  # Return the URL even though download failed
                
            edited_image = Image.open(BytesIO(img_response.content))
            
            # Convert back to tensor
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
            # Return original image on error with error message
            return (image, error_message, "")



############################# Chatgpt ###########################

# reference: OpenAIGPTImage1 node from comfyui node
def downscale_input(image):
    samples = image.movedim(-1,1)

    total = int(1536 * 1024)
    scale_by = math.sqrt(total / (samples.shape[3] * samples.shape[2]))
    if scale_by >= 1:
        return image
    width = round(samples.shape[3] * scale_by)
    height = round(samples.shape[2] * scale_by)

    s = common_upscale(samples, width, height, "lanczos", "disabled")
    s = s.movedim(1,-1)
    return s


class Comfly_gpt_image_1_edit:

    _last_edited_image = None
    _conversation_history = []
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "prompt": ("STRING", {"multiline": True}),
            },
            "optional": {
                "mask": ("MASK",),
                "api_key": ("STRING", {"default": ""}),
                "model": ("STRING", {"default": "gpt-image-1"}),
                "n": ("INT", {"default": 1, "min": 1, "max": 10}),
                "quality": (["auto", "high", "medium", "low"], {"default": "auto"}),
                "size": (["auto", "1024x1024", "1536x1024", "1024x1536"], {"default": "auto"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "clear_chats": ("BOOLEAN", {"default": True}),
                "background": (["auto", "transparent", "opaque"], {"default": "auto"}),
                "output_compression": ("INT", {"default": 100, "min": 0, "max": 100}),
                "output_format": (["png", "jpeg", "webp"], {"default": "png"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("edited_image", "response", "chats")
    FUNCTION = "edit_image"
    CATEGORY = "Comfly/Chatgpt"

    def __init__(self):
        self.api_key = get_config().get('api_key', '')
        self.timeout = 300

    def get_headers(self):
        return {
            "Authorization": f"Bearer {self.api_key}"
        }
    
    def format_conversation_history(self):
        """Format the conversation history for display"""
        if not Comfly_gpt_image_1_edit._conversation_history:
            return ""
        formatted_history = ""
        for entry in Comfly_gpt_image_1_edit._conversation_history:
            formatted_history += f"**User**: {entry['user']}\n\n"
            formatted_history += f"**AI**: {entry['ai']}\n\n"
            formatted_history += "---\n\n"
        return formatted_history.strip()
    
    def edit_image(self, image, prompt, model="gpt-image-1", n=1, quality="auto", 
              seed=0, mask=None, api_key="", size="auto", clear_chats=True,
              background="auto", output_compression=100, output_format="png"):
        if api_key.strip():
            self.api_key = api_key
            config = get_config()
            config['api_key'] = api_key
            save_config(config)
 
        original_image = image
        original_batch_size = image.shape[0]
        use_saved_image = False

        if not clear_chats and Comfly_gpt_image_1_edit._last_edited_image is not None:
            if original_batch_size > 1:
                last_batch_size = Comfly_gpt_image_1_edit._last_edited_image.shape[0]
                last_image_first = Comfly_gpt_image_1_edit._last_edited_image[0:1]
                if last_image_first.shape[1:] == original_image.shape[1:]:
                    image = torch.cat([last_image_first, original_image[1:]], dim=0)
                    use_saved_image = True
            else:

                image = Comfly_gpt_image_1_edit._last_edited_image
                use_saved_image = True

        if clear_chats:
            Comfly_gpt_image_1_edit._conversation_history = []

            
        try:
            if not self.api_key:
                error_message = "API key not found in Comflyapi.json"
                print(error_message)
                return (original_image, error_message, self.format_conversation_history())
          
            pbar = comfy.utils.ProgressBar(100)
            pbar.update_absolute(10)
            
            files = {}
 
            if image is not None:
                batch_size = image.shape[0]
                for i in range(batch_size):
                    single_image = image[i:i+1]
                    scaled_image = downscale_input(single_image).squeeze()
                    
                    image_np = (scaled_image.numpy() * 255).astype(np.uint8)
                    img = Image.fromarray(image_np)
                    img_byte_arr = io.BytesIO()
                    img.save(img_byte_arr, format='PNG')
                    img_byte_arr.seek(0)
                    
                    if batch_size == 1:
                        files['image'] = ('image.png', img_byte_arr, 'image/png')
                    else:
                        if 'image[]' not in files:
                            files['image[]'] = []
                        files['image[]'].append(('image_{}.png'.format(i), img_byte_arr, 'image/png'))
            
            if mask is not None:
                if image.shape[0] != 1:
                    raise Exception("Cannot use a mask with multiple images")
                if image is None:
                    raise Exception("Cannot use a mask without an input image")
                if mask.shape[1:] != image.shape[1:-1]:
                    raise Exception("Mask and Image must be the same size")
                
                batch, height, width = mask.shape
                rgba_mask = torch.zeros(height, width, 4, device="cpu")
                rgba_mask[:,:,3] = (1-mask.squeeze().cpu())
                scaled_mask = downscale_input(rgba_mask.unsqueeze(0)).squeeze()
                mask_np = (scaled_mask.numpy() * 255).astype(np.uint8)
                mask_img = Image.fromarray(mask_np)
                mask_byte_arr = io.BytesIO()
                mask_img.save(mask_byte_arr, format='PNG')
                mask_byte_arr.seek(0)
                files['mask'] = ('mask.png', mask_byte_arr, 'image/png')

            if 'image[]' in files:

                data = {
                    'prompt': prompt,
                    'model': model,
                    'n': str(n),
                    'quality': quality
                }
                
                if size != "auto":
                    data['size'] = size
                    
                if background != "auto":
                    data['background'] = background
                    
                if output_compression != 100:
                    data['output_compression'] = str(output_compression)
                    
                if output_format != "png":
                    data['output_format'] = output_format

                image_files = []
                for file_tuple in files['image[]']:
                    image_files.append(('image', file_tuple))

                if 'mask' in files:
                    image_files.append(('mask', files['mask']))

                response = requests.post(
                    "https://ai.comfly.chat/v1/images/edits",
                    headers=self.get_headers(),
                    data=data,
                    files=image_files,
                    timeout=self.timeout
                )
            else:
                data = {
                    'prompt': prompt,
                    'model': model,
                    'n': str(n),
                    'quality': quality
                }
                
                if size != "auto":
                    data['size'] = size
                    
                if background != "auto":
                    data['background'] = background
                    
                if output_compression != 100:
                    data['output_compression'] = str(output_compression)
                    
                if output_format != "png":
                    data['output_format'] = output_format

                request_files = []

                if 'image' in files:
                    request_files.append(('image', files['image']))

                if 'mask' in files:
                    request_files.append(('mask', files['mask']))

                response = requests.post(
                    "https://ai.comfly.chat/v1/images/edits",
                    headers=self.get_headers(),
                    data=data,
                    files=request_files,
                    timeout=self.timeout
                )

            pbar.update_absolute(50)

            if response.status_code != 200:
                error_message = f"API Error: {response.status_code} - {response.text}"
                print(error_message)
                return (original_image, error_message, self.format_conversation_history())
            result = response.json()
            
            if "data" not in result or not result["data"]:
                error_message = "No image data in response"
                print(error_message)
                return (original_image, error_message, self.format_conversation_history())

            edited_images = []
            image_urls = []

            for item in result["data"]:
                if "b64_json" in item:
                    image_data = base64.b64decode(item["b64_json"])
                    edited_image = Image.open(BytesIO(image_data))
                    edited_tensor = pil2tensor(edited_image)
                    edited_images.append(edited_tensor)
                elif "url" in item:
                    image_urls.append(item["url"])
                    try:
                        img_response = requests.get(item["url"])
                        if img_response.status_code == 200:
                            edited_image = Image.open(BytesIO(img_response.content))
                            edited_tensor = pil2tensor(edited_image)
                            edited_images.append(edited_tensor)
                    except Exception as e:
                        print(f"Error downloading image from URL: {str(e)}")

            pbar.update_absolute(90)

            if edited_images:
                combined_tensor = torch.cat(edited_images, dim=0)
                response_info = f"Successfully edited {len(edited_images)} image(s)\n"
                response_info += f"Prompt: {prompt}\n"
                response_info += f"Model: {model}\n"
                response_info += f"Quality: {quality}\n"
                
                if use_saved_image:
                    response_info += "[Using previous edited image as input]\n"
                    
                if size != "auto":
                    response_info += f"Size: {size}\n"
                    
                if background != "auto":
                    response_info += f"Background: {background}\n"
                    
                if output_compression != 100:
                    response_info += f"Output Compression: {output_compression}%\n"
                    
                if output_format != "png":
                    response_info += f"Output Format: {output_format}\n"

                Comfly_gpt_image_1_edit._conversation_history.append({
                    "user": f"Edit image with prompt: {prompt}",
                    "ai": f"Generated edited image with {model}"
                })
 
                Comfly_gpt_image_1_edit._last_edited_image = combined_tensor
                
                pbar.update_absolute(100)
                return (combined_tensor, response_info, self.format_conversation_history())
            else:
                error_message = "No edited images in response"
                print(error_message)
                return (original_image, error_message, self.format_conversation_history())
            
        except Exception as e:
            error_message = f"Error in image editing: {str(e)}"
            import traceback
            print(traceback.format_exc())  
            print(error_message)
            return (original_image, error_message, self.format_conversation_history())

        

class Comfly_gpt_image_1:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
            },
            "optional": {
                "api_key": ("STRING", {"default": ""}),
                "model": ("STRING", {"default": "gpt-image-1"}),
                "n": ("INT", {"default": 1, "min": 1, "max": 10}),
                "quality": (["auto", "high", "medium", "low"], {"default": "auto"}),
                "size": (["auto", "1024x1024", "1536x1024", "1024x1536"], {"default": "auto"}),
                "background": (["auto", "transparent", "opaque"], {"default": "auto"}),
                "output_format": (["png", "jpeg", "webp"], {"default": "png"}),
                "moderation": (["auto", "low"], {"default": "auto"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("generated_image", "response")
    FUNCTION = "generate_image"
    CATEGORY = "Comfly/Chatgpt"

    def __init__(self):
        self.api_key = get_config().get('api_key', '')
        self.timeout = 300

    def get_headers(self):
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
    
    def generate_image(self, prompt, model="gpt-image-1", n=1, quality="auto", 
                size="auto", background="auto", output_format="png", 
                moderation="auto", seed=0, api_key=""):
        
        if api_key.strip():
            self.api_key = api_key
            config = get_config()
            config['api_key'] = api_key
            save_config(config)
            
        try:
            if not self.api_key:
                error_message = "API key not found in Comflyapi.json"
                print(error_message)
                blank_image = Image.new('RGB', (1024, 1024), color='white')
                blank_tensor = pil2tensor(blank_image)
                return (blank_tensor, error_message)
            pbar = comfy.utils.ProgressBar(100)
            pbar.update_absolute(10)
            payload = {
                "prompt": prompt,
                "model": model,
                "n": n,
                "quality": quality,
                "background": background,
                "output_format": output_format,
                "moderation": moderation,
            }
            
            # Only include size if it's not "auto"
            if size != "auto":
                payload["size"] = size
            
            response = requests.post(
                "https://ai.comfly.chat/v1/images/generations",
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
                return (blank_tensor, error_message)
                
            # Parse the response
            result = response.json()
            
            # Format the response information
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            response_info = f"**GPT-image-1 Generation ({timestamp})**\n\n"
            response_info += f"Prompt: {prompt}\n"
            response_info += f"Model: {model}\n"
            response_info += f"Quality: {quality}\n"
            if size != "auto":
                response_info += f"Size: {size}\n"
            response_info += f"Background: {background}\n"
            response_info += f"Seed: {seed} (Note: Seed not used by API)\n\n"
            
            # Process the generated images
            generated_images = []
            image_urls = []
            
            if "data" in result and result["data"]:
                for i, item in enumerate(result["data"]):
                    pbar.update_absolute(50 + (i+1) * 50 // len(result["data"]))
                    
                    if "b64_json" in item:
                        # Decode base64 image
                        image_data = base64.b64decode(item["b64_json"])
                        generated_image = Image.open(BytesIO(image_data))
                        generated_tensor = pil2tensor(generated_image)
                        generated_images.append(generated_tensor)
                    elif "url" in item:
                        image_urls.append(item["url"])
                        # Download and process the image from URL
                        try:
                            img_response = requests.get(item["url"])
                            if img_response.status_code == 200:
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
                return (blank_tensor, response_info)
                
            # Add usage information to the response if available
            if "usage" in result:
                response_info += "Usage Information:\n"
                if "total_tokens" in result["usage"]:
                    response_info += f"Total Tokens: {result['usage']['total_tokens']}\n"
                if "input_tokens" in result["usage"]:
                    response_info += f"Input Tokens: {result['usage']['input_tokens']}\n"
                if "output_tokens" in result["usage"]:
                    response_info += f"Output Tokens: {result['usage']['output_tokens']}\n"
                
                # Add detailed token usage if available
                if "input_tokens_details" in result["usage"]:
                    response_info += "Input Token Details:\n"
                    details = result["usage"]["input_tokens_details"]
                    if "text_tokens" in details:
                        response_info += f"  Text Tokens: {details['text_tokens']}\n"
                    if "image_tokens" in details:
                        response_info += f"  Image Tokens: {details['image_tokens']}\n"
            
            if generated_images:
                # Combine all generated images into a single tensor
                combined_tensor = torch.cat(generated_images, dim=0)
                
                pbar.update_absolute(100)
                return (combined_tensor, response_info)
            else:
                error_message = "No images were successfully processed"
                print(error_message)
                response_info += f"Error: {error_message}\n"
                blank_image = Image.new('RGB', (1024, 1024), color='white')
                blank_tensor = pil2tensor(blank_image)
                return (blank_tensor, response_info)
                
        except Exception as e:
            error_message = f"Error in image generation: {str(e)}"
            print(error_message)
            blank_image = Image.new('RGB', (1024, 1024), color='white')
            blank_tensor = pil2tensor(blank_image)
            return (blank_tensor, error_message)



class ComflyChatGPTApi:

    _last_generated_image_urls = ""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "model": ("STRING", {"default": "gpt-image-1", "multiline": False}),
            },
            "optional": {
                "api_key": ("STRING", {"default": ""}),
                "files": ("FILES",), 
                "image_url": ("STRING", {"multiline": False, "default": ""}),
                "images": ("IMAGE", {"default": None}),  
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.01}),
                "max_tokens": ("INT", {"default": 4096, "min": 1, "max": 16384, "step": 1}),
                "top_p": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "frequency_penalty": ("FLOAT", {"default": -2.0, "min": -2.0, "max": 2.0, "step": 0.01}),
                "presence_penalty": ("FLOAT", {"default": 0.0, "min": -2.0, "max": 2.0, "step": 0.01}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 2147483647}),
                "image_download_timeout": ("INT", {"default": 600, "min": 300, "max": 1200, "step": 10}),
                "clear_chats": ("BOOLEAN", {"default": True}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("images", "response", "image_urls", "chats")
    FUNCTION = "process"
    CATEGORY = "Comfly/Chatgpt"
    
    def __init__(self):
        self.api_key = get_config().get('api_key', '')
        self.timeout = 800
        self.image_download_timeout = 600
        self.api_endpoint = "https://ai.comfly.chat/v1/chat/completions"
        self.conversation_history = []

    def get_headers(self):
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
    def image_to_base64(self, image):
        """Convert PIL image to base64 string"""
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    def file_to_base64(self, file_path):
        """Convert file to base64 string and return appropriate MIME type"""
        try:
            with open(file_path, "rb") as file:
                file_content = file.read()
                encoded_content = base64.b64encode(file_content).decode('utf-8')
                # Get MIME type
                mime_type, _ = mimetypes.guess_type(file_path)
                if not mime_type:
                    # Default to binary if MIME type can't be determined
                    mime_type = "application/octet-stream"
                return encoded_content, mime_type
        except Exception as e:
            print(f"Error encoding file: {str(e)}")
            return None, None

    def extract_image_urls(self, response_text):
        """Extract image URLs from markdown format in response"""
      
        image_pattern = r'!\[.*?\]\((.*?)\)'
        matches = re.findall(image_pattern, response_text)
      
        if not matches:
            url_pattern = r'https?://\S+\.(?:jpg|jpeg|png|gif|webp)'
            matches = re.findall(url_pattern, response_text)
        
        if not matches:
            all_urls_pattern = r'https?://\S+'
            matches = re.findall(all_urls_pattern, response_text)
        return matches if matches else []

    def download_image(self, url, timeout=30):
        """Download image from URL and convert to tensor with improved error handling"""
        try:
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.9',
                'Referer': 'https://comfyui.com/'
            }
           
            response = requests.get(url, headers=headers, timeout=timeout)
            response.raise_for_status()
          
            content_type = response.headers.get('Content-Type', '')
            if not content_type.startswith('image/'):
                print(f"Warning: URL doesn't point to an image. Content-Type: {content_type}")
               
            image = Image.open(BytesIO(response.content))
            return pil2tensor(image)
        except requests.exceptions.Timeout:
            print(f"Timeout error downloading image from {url} (timeout: {timeout}s)")
            return None
        except requests.exceptions.SSLError as e:
            print(f"SSL Error downloading image from {url}: {str(e)}")
            return None
        except requests.exceptions.ConnectionError:
            print(f"Connection error downloading image from {url}")
            return None
        except requests.exceptions.RequestException as e:
            print(f"Request error downloading image from {url}: {str(e)}")
            return None
        except Exception as e:
            print(f"Error downloading image from {url}: {str(e)}")
            return None

    def format_conversation_history(self):
        """Format the conversation history for display"""
        if not self.conversation_history:
            return ""
        formatted_history = ""
        for entry in self.conversation_history:
            formatted_history += f"**User**: {entry['user']}\n\n"
            formatted_history += f"**AI**: {entry['ai']}\n\n"
            formatted_history += "---\n\n"
        return formatted_history.strip()

    def process(self, prompt, model, clear_chats=True, files=None, image_url="", images=None, temperature=0.7, 
               max_tokens=4096, top_p=1.0, frequency_penalty=0.0, presence_penalty=0.0, seed=-1,
               image_download_timeout=100, api_key=""):
        if api_key.strip():
            self.api_key = api_key
            config = get_config()
            config['api_key'] = api_key
            save_config(config)
        try:
          
            self.image_download_timeout = image_download_timeout
          
            if clear_chats:
                self.conversation_history = []
                
            if not self.api_key:
                error_message = "API key not found in Comflyapi.json"
                print(error_message)
               
                blank_img = Image.new('RGB', (512, 512), color='white')
                return (pil2tensor(blank_img), error_message, "", self.format_conversation_history()) 
            
            pbar = comfy.utils.ProgressBar(100)
            pbar.update_absolute(10)
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
           
            if seed < 0:
                seed = random.randint(0, 2147483647)
                print(f"Using random seed: {seed}")
           
            content = []
            
            content.append({"type": "text", "text": prompt})
            
           
            if not clear_chats and ComflyChatGPTApi._last_generated_image_urls:
                prev_image_url = ComflyChatGPTApi._last_generated_image_urls.split('\n')[0].strip()
                if prev_image_url:
                    print(f"Using previous image URL: {prev_image_url}")
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": prev_image_url}
                    })
            
            elif clear_chats:
                if images is not None:
                    batch_size = images.shape[0]
                    max_images = min(batch_size, 4)  
                    for i in range(max_images):
                        pil_image = tensor2pil(images)[i]
                        image_base64 = self.image_to_base64(pil_image)
                        content.append({
                            "type": "image_url", 
                            "image_url": {"url": f"data:image/png;base64,{image_base64}"}
                        })
                    if batch_size > max_images:
                        content.append({
                            "type": "text",
                            "text": f"\n(Note: {batch_size-max_images} additional images were omitted due to API limitations)"
                        })
                
                elif image_url:
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": image_url}
                    })
        
            elif image_url:
                content.append({
                    "type": "image_url",
                    "image_url": {"url": image_url}
                })
            
            if files:
                file_paths = files if isinstance(files, list) else [files]
                for file_path in file_paths:
                    encoded_content, mime_type = self.file_to_base64(file_path)
                    if encoded_content and mime_type:
                       
                        if mime_type.startswith('image/'):
                           
                            content.append({
                                "type": "image_url", 
                                "image_url": {"url": f"data:{mime_type};base64,{encoded_content}"}
                            })
                        else:
                            
                            content.append({
                                "type": "text", 
                                "text": f"\n\nI've attached a file ({os.path.basename(file_path)}) for analysis."
                            })
                            content.append({
                                "type": "file_url",
                                "file_url": {
                                    "url": f"data:{mime_type};base64,{encoded_content}",
                                    "name": os.path.basename(file_path)
                                }
                            })
        
            messages = []
        
            messages.append({
                "role": "user",
                "content": content
            })
        
            payload = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "top_p": top_p,
                "frequency_penalty": frequency_penalty,
                "presence_penalty": presence_penalty,
                "seed": seed,
                "stream": True  
            }
        
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            response_text = loop.run_until_complete(self.stream_response(payload, pbar))
            loop.close()
        
            self.conversation_history.append({
                "user": prompt,
                "ai": response_text
            })
        
            technical_response = f"**Model**: {model}\n**Temperature**: {temperature}\n**Seed**: {seed}\n**Time**: {timestamp}"
        
            image_urls = self.extract_image_urls(response_text)
            image_urls_string = "\n".join(image_urls) if image_urls else ""
            
            if image_urls:
                ComflyChatGPTApi._last_generated_image_urls = image_urls_string
     
            chat_history = self.format_conversation_history()
            if image_urls:
                try:
                    
                    img_tensors = []
                    successful_downloads = 0
                    for i, url in enumerate(image_urls):
                        print(f"Attempting to download image {i+1}/{len(image_urls)} from: {url}")
                
                        pbar.update_absolute(min(80, 40 + (i+1) * 40 // len(image_urls)))
                        img_tensor = self.download_image(url, self.image_download_timeout)
                        if img_tensor is not None:
                            img_tensors.append(img_tensor)
                            successful_downloads += 1
                    print(f"Successfully downloaded {successful_downloads} out of {len(image_urls)} images")
                    if img_tensors:
                
                        combined_tensor = torch.cat(img_tensors, dim=0)
                        pbar.update_absolute(100)
                        return (combined_tensor, technical_response, image_urls_string, chat_history)
                except Exception as e:
                    print(f"Error processing image URLs: {str(e)}")
        
            if images is not None:
                pbar.update_absolute(100)
                return (images, technical_response, image_urls_string, chat_history)  
            else:
                blank_img = Image.new('RGB', (512, 512), color='white')
                blank_tensor = pil2tensor(blank_img)
                pbar.update_absolute(100)
                return (blank_tensor, technical_response, image_urls_string, chat_history)  
                
        except Exception as e:
            error_message = f"Error calling ChatGPT API: {str(e)}"
            print(error_message)
        
            if images is not None:
                return (images, error_message, "", self.format_conversation_history())  
            else:
                blank_img = Image.new('RGB', (512, 512), color='white')
                blank_tensor = pil2tensor(blank_img)
                return (blank_tensor, error_message, "", self.format_conversation_history()) 

    async def stream_response(self, payload, pbar):
        """Stream response from API"""
        full_response = ""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.api_endpoint, 
                    headers=self.get_headers(), 
                    json=payload,
                    timeout=self.timeout
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise Exception(f"API Error {response.status}: {error_text}")
                    
                    async for line in response.content:
                        line = line.decode('utf-8').strip()
                        if line.startswith('data: '):
                            data = line[6:]  
                            if data == '[DONE]':
                                break
                            try:
                                chunk = json.loads(data)
                                if 'choices' in chunk and chunk['choices']:
                                    delta = chunk['choices'][0].get('delta', {})
                                    if 'content' in delta:
                                        content = delta['content']
                                        full_response += content
                                       
                                        pbar.update_absolute(min(40, 20 + len(full_response) // 100))
                            except json.JSONDecodeError:
                                continue
            return full_response
        
        except asyncio.TimeoutError:
            raise TimeoutError(f"API request timed out after {self.timeout} seconds")
        except Exception as e:
            raise Exception(f"Error in streaming response: {str(e)}")



############################# Flux ###########################

class Comfly_Flux_Kontext:
    _last_image_url = ""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
            },
            "optional": {
                "input_image": ("IMAGE",),
                "model": (["flux-kontext-dev", "flux-kontext-pro", "flux-kontext-max"], {"default": "flux-kontext-pro"}),
                "apikey": ("STRING", {"default": ""}),
                "aspect_ratio": (["Default", "21:9", "16:9", "4:3", "3:2", "1:1", "2:3", "3:4", "9:16", "9:21"], 
                         {"default": "Default"}),
                "guidance": ("FLOAT", {"default": 3.5, "min": 1.0, "max": 20.0, "step": 0.5}),
                "num_of_images": ("INT", {"default": 1, "min": 1, "max": 4}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 2147483647}),
                "clear_image": ("BOOLEAN", {"default": True})
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
                "https://ai.comfly.chat/v1/files",
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
    
    def generate_image(self, prompt, input_image=None, model="flux-kontext-pro", 
                  apikey="", aspect_ratio="Default", guidance=3.5, num_of_images=1,
                  seed=-1, clear_image=True):
        if apikey.strip():
            self.api_key = apikey
            config = get_config()
            config['api_key'] = apikey
            save_config(config)
            
        if not self.api_key:
            error_message = "API key not found in Comflyapi.json"
            print(error_message)

            if input_image is None:
                blank_image = Image.new('RGB', (1024, 1024), color='white')
                blank_tensor = pil2tensor(blank_image)
                return (blank_tensor, "")
            return (input_image, "")
        
        try:
            pbar = comfy.utils.ProgressBar(100)
            pbar.update_absolute(10)

            final_prompt = prompt
            custom_dimensions = None

            if input_image is not None:
                batch_size = input_image.shape[0]
                all_image_urls = []
                
                for i in range(batch_size):
                    single_image = input_image[i:i+1]
                    pbar.update_absolute(10 + (i * 10) // batch_size)
                    image_url = self.upload_image(single_image)
                    if image_url:
                        all_image_urls.append(image_url)

                if all_image_urls:
                    image_urls_text = " ".join(all_image_urls)
                    final_prompt = f"{image_urls_text} {prompt}"
                    if aspect_ratio == "match_input_image" and batch_size > 0:
                        pil_image = tensor2pil(input_image)[0]
                        width, height = pil_image.size
                        custom_dimensions = {"width": width, "height": height}
                else:
                    print("Failed to upload any images")
                    if input_image is None:
                        blank_image = Image.new('RGB', (1024, 1024), color='white')
                        blank_tensor = pil2tensor(blank_image)
                        return (blank_tensor, "")
                    return (input_image, "")
 
            elif not clear_image and Comfly_Flux_Kontext._last_image_url:
                final_prompt = f"{Comfly_Flux_Kontext._last_image_url} {prompt}"

            payload = {
                "prompt": final_prompt,
                "model": model,
                "n": num_of_images,  
                "guidance_scale": guidance  
            }

            if custom_dimensions and aspect_ratio == "match_input_image":
                payload.update(custom_dimensions)
            else:
                payload["aspect_ratio"] = aspect_ratio

            if seed != -1:
                payload["seed"] = seed

            response = requests.post(
                "https://ai.comfly.chat/v1/images/generations",
                headers=self.get_headers(),
                json=payload,
                timeout=self.timeout
            )
            
            pbar.update_absolute(50)
            
            if response.status_code != 200:
                error_message = f"API Error: {response.status_code} - {response.text}"
                print(error_message)
                if input_image is None:
                    blank_image = Image.new('RGB', (1024, 1024), color='white')
                    blank_tensor = pil2tensor(blank_image)
                    return (blank_tensor, "")
                return (input_image, "")
                
            result = response.json()

            if not result.get("data") or not result["data"]:
                error_message = "No image data in response"
                print(error_message)
                if input_image is None:
                    blank_image = Image.new('RGB', (1024, 1024), color='white')
                    blank_tensor = pil2tensor(blank_image)
                    return (blank_tensor, "")
                return (input_image, "")

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

                if image_urls:
                    Comfly_Flux_Kontext._last_image_url = image_urls[0]
                
                return (combined_tensor, "\n".join(image_urls))
            else:
                error_message = "Failed to process any images"
                print(error_message)
                if input_image is None:
                    blank_image = Image.new('RGB', (1024, 1024), color='white')
                    blank_tensor = pil2tensor(blank_image)
                    return (blank_tensor, "")
                return (input_image, "")
            
        except Exception as e:
            error_message = f"Error in image generation: {str(e)}"
            print(error_message)
            if input_image is None:
                blank_image = Image.new('RGB', (1024, 1024), color='white')
                blank_tensor = pil2tensor(blank_image)
                return (blank_tensor, "")
            return (input_image, "")
         
        
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
            save_config(config)
            
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
                    "https://ai.comfly.chat/v1/images/edits",
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
                    "https://ai.comfly.chat/v1/images/generations",
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
            save_config(config)

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

        api_endpoint = f"https://ai.comfly.chat/bfl/v1/{model}"
        
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
                        f"https://ai.comfly.chat/bfl/v1/get_result?id={task_id}",
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



############################# Googel ###########################

class Comfly_Googel_Veo3:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "model": (["veo3", "veo3-fast", "veo3-pro", "veo3-fast-frames", "veo3-pro-frames"], {"default": "veo3"}),
                "enhance_prompt": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "apikey": ("STRING", {"default": ""}),
                "image": ("IMAGE",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647}),
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
    
    def generate_video(self, prompt, model="veo3", enhance_prompt=False, apikey="", image=None, seed=0):
        if apikey.strip():
            self.api_key = apikey
            config = get_config()
            config['api_key'] = apikey
            save_config(config)
            
        if not self.api_key:
            error_response = {"code": "error", "message": "API key not found in Comflyapi.json"}
            return ("", "", json.dumps(error_response))
            
        try:
            pbar = comfy.utils.ProgressBar(100)
            pbar.update_absolute(10)
 
            is_image_to_video = image is not None
 
            payload = {
                "prompt": prompt,
                "model": model,
                "enhance_prompt": enhance_prompt
            }
 
            if seed > 0:
                payload["seed"] = seed
 
            if is_image_to_video:
                image_base64 = self.image_to_base64(image)
                if image_base64:
                    payload["images"] = [f"data:image/png;base64,{image_base64}"]

            response = requests.post(
                "https://ai.comfly.chat/google/v1/models/veo/videos",
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

            max_attempts = 60  
            attempts = 0
            video_url = None
            
            while attempts < max_attempts:
                time.sleep(10)
                attempts += 1
                
                try:
                    status_response = requests.get(
                        f"https://ai.comfly.chat/google/v1/tasks/{task_id}",
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
                    "video_url": video_url
                }
                
                video_adapter = ComflyVideoAdapter(video_url)
                return (video_adapter, video_url, json.dumps(response_data))
            
        except Exception as e:
            error_message = f"Error generating video: {str(e)}"
            print(error_message)
            return ("", "", json.dumps({"code": "error", "message": error_message}))
    

WEB_DIRECTORY = "./web"    
        
NODE_CLASS_MAPPINGS = {
    "Comfly_Mj": Comfly_Mj,
    "Comfly_mjstyle": Comfly_mjstyle,
    "Comfly_upload": Comfly_upload,
    "Comfly_Mju": Comfly_Mju,
    "Comfly_Mjv": Comfly_Mjv,  
    "Comfly_kling_text2video": Comfly_kling_text2video,
    "Comfly_kling_image2video": Comfly_kling_image2video,
    "Comfly_video_extend": Comfly_video_extend,
    "Comfly_lip_sync": Comfly_lip_sync,
    "ComflyGeminiAPI": ComflyGeminiAPI,
    "ComflySeededit": ComflySeededit,
    "ComflyChatGPTApi": ComflyChatGPTApi,
    "ComflyJimengApi": ComflyJimengApi, 
    "Comfly_gpt_image_1_edit": Comfly_gpt_image_1_edit,
    "Comfly_gpt_image_1": Comfly_gpt_image_1,
    "ComflyJimengVideoApi": ComflyJimengVideoApi,
    "Comfly_Flux_Kontext": Comfly_Flux_Kontext,
    "Comfly_Flux_Kontext_Edit": Comfly_Flux_Kontext_Edit,
    "Comfly_Flux_Kontext_bfl": Comfly_Flux_Kontext_bfl,
    "Comfly_Googel_Veo3": Comfly_Googel_Veo3,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Comfly_Mj": "Comfly_Mj", 
    "Comfly_mjstyle": "Comfly_mjstyle",
    "Comfly_upload": "Comfly_upload",
    "Comfly_Mju": "Comfly_Mju",
    "Comfly_Mjv": "Comfly_Mjv",  
    "Comfly_kling_text2video": "Comfly_kling_text2video",
    "Comfly_kling_image2video": "Comfly_kling_image2video",
    "Comfly_video_extend": "Comfly_video_extend",
    "Comfly_lip_sync": "Comfly_lip_sync",
    "ComflyGeminiAPI": "Comfly Gemini API",
    "ComflySeededit": "Comfly Doubao SeedEdit",
    "ComflyChatGPTApi": "Comfly ChatGPT Api",
    "ComflyJimengApi": "Comfly Jimeng API", 
    "Comfly_gpt_image_1_edit": "Comfly_gpt_image_1_edit",
    "Comfly_gpt_image_1": "Comfly_gpt_image_1", 
    "ComflyJimengVideoApi": "Comfly Jimeng Video API",
    "Comfly_Flux_Kontext": "Comfly_Flux_Kontext",
    "Comfly_Flux_Kontext_Edit": "Comfly_Flux_Kontext_Edit",
    "Comfly_Flux_Kontext_bfl": "Comfly_Flux_Kontext_bfl",
    "Comfly_Googel_Veo3": "Comfly Google Veo3",
}
