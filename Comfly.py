import os
import random
import torch
import requests
import time
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
from .utils import pil2tensor, tensor2pil


current_dir = os.path.dirname(os.path.abspath(__file__))
config_file_path = os.path.join(current_dir, 'Comflyapi.json')

with open(config_file_path, 'r') as f:
    api_config = json.load(f)


############################# Midjourney ###########################

class ComflyBaseNode:
    def __init__(self):
        self.midjourney_api_url = {
            "turbo mode": "https://ai.comfly.chat/mj-turbo",
            "fast mode": "https://ai.comfly.chat/mj-fast",
            "relax mode": "https://ai.comfly.chat/mj-relax"
        }
        self.api_key = api_config.get('api_key', '') 
        self.speed = "fast mode"
        self.timeout = 300 

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
                        data = await response.json()
                        return data
                    else:
                        error_message = f"Error submitting Midjourney action: {response.status}"
                        print(error_message)
                        return error_message
        except asyncio.TimeoutError:
            error_message = f"Timeout error: Request to submit action timed out after {self.timeout} seconds"
            print(error_message)
            raise Exception(error_message)

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
                        data = await response.json()
                        return data["result"]
                    else:
                        error_message = f"Error submitting Midjourney task: {response.status}"
                        print(error_message)
                        raise Exception(error_message)
        except asyncio.TimeoutError:
            error_message = f"Timeout error: Request to submit imagine task timed out after {self.timeout} seconds"
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
                        data = await response.json()
                        return data
                    else:
                        error_message = f"Error fetching Midjourney task result: {response.status}"
                        print(error_message)
                        raise Exception(error_message)
        except asyncio.TimeoutError:
            error_message = f"Timeout error: Request to fetch task result timed out after {self.timeout} seconds"
            print(error_message)
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
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("url",)
    FUNCTION = "upload_image"
    CATEGORY = "Comfly/Midjourney"

    async def upload_image_to_midjourney(self, image):
        # Convert Tensor to PIL Image
        image = tensor2pil(image)[0]
        # Encode the image as base64
        buffered = BytesIO()
        image_format = "PNG"  # Specify the desired image format
        image.save(buffered, format=image_format)
        image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        # Prepare the request payload
        payload = {
            "base64Array": [f"data:image/{image_format.lower()};base64,{image_base64}"],
            "instanceId": "",
            "notifyHook": ""
        }
        # Send the POST request to upload the image
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(f"{self.midjourney_api_url[self.speed]}/mj/submit/upload-discord-images", headers=self.get_headers(), json=payload, timeout=self.timeout) as response:
                    if response.status == 200:
                        data = await response.json()
                        if "result" in data and data["result"]:
                            return data["result"][0]
                        else:
                            error_message = f"Unexpected response from Midjourney API: {data}"
                            raise Exception(error_message)
                    else:
                        error_message = f"Error uploading image to Midjourney: {response.status}"
                        raise Exception(error_message)
        except asyncio.TimeoutError:
            error_message = f"Timeout error: Request to upload image timed out after {self.timeout} seconds"
            print(error_message)
            raise Exception(error_message)

    def upload_image(self, image):
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
                "speed": (["turbo mode", "fast mode", "relax mode"], {"default": "fast mode"}),
                "text": ("STRING", {"multiline": True}),
            },
            "optional": {
                "text_en": ("STRING", {"multiline": True, "default": ""}),
                "model_version": (["v 6.1", "v 6.0", "v 5.2", "v 5.1", "niji 6", "niji 5", "niji 4"], {"default": "v 6.1"}),
                "ar": ("STRING", {"default": "1:1"}),
                "no": ("STRING", {"default": "", "forceInput": True}),
                "c": ("INT", {"default": 0, "min": 0, "max": 100, "forceInput": True}),
                "s": ("INT", {"default": 0, "min": 0, "max": 1000, "forceInput": True}),
                "iw": ("FLOAT", {"default": 0, "min": 0, "max": 2, "forceInput": True}),
                "r": ("INT", {"default": 1, "min": 1, "max": 40, "forceInput": True}),
                "sw": ("INT", {"default": 0, "min": 0, "max": 1000, "forceInput": True}),
                "cw": ("INT", {"default": 0, "min": 0, "max": 100, "forceInput": True}),
                "sv": (["1", "2", "3", "4"], {"default": "1", "forceInput": True}),
                "video": ("BOOLEAN", {"default": False}),
                "tile": ("BOOLEAN", {"default": False}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647}),
                "cref": ("STRING", {"default": "none", "forceInput": True}),
                "sref": ("STRING", {"default": "none", "forceInput": True}),
                "positive": ("STRING", {"default": "", "forceInput": True}),
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

    def process_input(self, speed, text, text_en="", image=None, model_version=None, ar=None, no=None, c=None, s=None, iw=None, r=None, sw=None, cw=None, sv=None, video=False, tile=False, seed=0, cref="none", sref="none", positive=""):
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
        taskId = await self.midjourney_submit_imagine_task(text, ar, no, c, s, iw, tile, r, video, sw, cw, sv, seed)
        self.taskId = taskId 
        task_result = None

        while not task_result or task_result.get("status") != "SUCCESS":
            await asyncio.sleep(1)
            task_result = await self.midjourney_fetch_task_result(taskId)
            progress = task_result.get("progress", 0)
            try:
                progress_int = int(progress[:-1])
            except (ValueError, TypeError):
                progress_int = 0
            pbar.update_absolute(progress_int)

        image_url = task_result["imageUrl"]
        prompt = task_result["prompt"]
        U1 = self.generate_custom_id(task_result["id"], "upsample", 1)
        U2 = self.generate_custom_id(task_result["id"], "upsample", 2)
        U3 = self.generate_custom_id(task_result["id"], "upsample", 3)
        U4 = self.generate_custom_id(task_result["id"], "upsample", 4)

        return image_url, prompt, U1, U2, U3, U4

    def process_text(self, pbar, ar, no, c, s, iw, tile, r, video, sw, cw, sv, seed):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        image_url, text, U1, U2, U3, U4 = loop.run_until_complete(self.process_text_midjourney(self.text, pbar, ar, no, c, s, iw, tile, r, video, sw, cw, sv, seed))
        loop.close()
        
        U = json.dumps({"U1": U1, "U2": U2, "U3": U3, "U4": U4})
        
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
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")  
    RETURN_NAMES = ("image", "taskId")  
    FUNCTION = "run"
    CATEGORY = "Comfly/Midjourney"

    def run(self, taskId, U1=False, U2=False, U3=False, U4=False):
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

            message_hash = task_result["properties"]["messageHash"]
            custom_id = self.generate_custom_id(action, index, message_hash)

            response = await self.midjourney_submit_action(action, taskId, index, custom_id)

            if isinstance(response, str) and response.startswith("Error"):
                print(f"Midjourney API returned an error: {response}")
                raise self.MidjourneyError(response)

            if "result" not in response:
                print(f"Unexpected response from Midjourney API: {response}")
                raise self.MidjourneyError(f"Unexpected response from Midjourney API: {response}")

            task_result = None
            task_id = response["result"]
            while not task_result or task_result.get("status") != "SUCCESS" or task_result.get("progress") != "100%":
                await asyncio.sleep(1)
                task_result = await self.midjourney_fetch_task_result(task_id)

            if task_result.get("code") == 5 and task_result.get("description") == "task_no_found":
                print(f"Task not found for taskId: {taskId}")
                raise self.MidjourneyError(f"Task not found for taskId: {taskId}")

            response = requests.get(task_result["imageUrl"])
            image = Image.open(BytesIO(response.content))
            tensor_image = pil2tensor(image)
            return tensor_image, task_id 

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
                        data = await response.json()
                        return data
                    else:
                        error_message = f"Error submitting Midjourney action: {response.status}"
                        print(error_message)
                        return error_message
        except asyncio.TimeoutError:
            error_message = f"Timeout error: Request to submit action timed out after {self.timeout} seconds"
            print(error_message)
            raise Exception(error_message)

    async def midjourney_fetch_task_result(self, taskId):
        headers = self.get_headers()

        async with aiohttp.ClientSession() as session:
            async with session.get(f"{self.midjourney_api_url[self.speed]}/mj/task/{taskId}/fetch", headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    return data
                else:
                    error_message = f"Error fetching Midjourney task result: {response.status}"
                    print(error_message)
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
                        data = await response.json()
                        return data["result"]
                    else:
                        error_message = f"Error submitting Midjourney task: {response.status}"
                        print(error_message)
                        raise Exception(error_message)
        except asyncio.TimeoutError:
            error_message = f"Timeout error: Request to submit imagine task timed out after {self.timeout} seconds"
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
                        data = await response.json()
                        return data
                    else:
                        error_message = f"Error fetching Midjourney task result: {response.status}"
                        print(error_message)
                        raise Exception(error_message)
        except asyncio.TimeoutError:
            error_message = f"Timeout error: Request to fetch task result timed out after {self.timeout} seconds"
            print(error_message)
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
                    data = await response.json()
                    return data
                else:
                    error_message = f"Error submitting Midjourney action: {response.status}"
                    print(error_message)
                    raise Exception(error_message)

    def run(self, taskId, upsample_v6_2x_subtle=False, upsample_v6_2x_creative=False, costume_zoom=False, zoom="", pan_left=False, pan_right=False, pan_up=False, pan_down=False):
       
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
        task_result = None
        while not task_result or task_result["status"] != "SUCCESS":
            await asyncio.sleep(1)
            try:
                task_result = await self.midjourney_fetch_task_result(taskId)
            except Exception as e:
                error_message = f"Error fetching task result: {str(e)}"
                print(error_message)
                raise Exception(error_message)
        return task_result["imageUrl"]

    async def midjourney_fetch_task_result(self, taskId):
        headers = self.get_headers()

        async with aiohttp.ClientSession() as session:
            async with session.get(f"{self.midjourney_api_url[self.speed]}/mj/task/{taskId}/fetch", headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    return data
                else:
                    error_message = f"Error fetching Midjourney task result: {response.status}"
                    print(error_message)
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



############################# Kling ###########################

class Comfly_kling_text2video:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "model_name": (["kling-v1-6", "kling-v1-5", "kling-v1"], {"default": "kling-v1-6"}),
                "imagination": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
                "aspect_ratio": (["1:1", "16:9", "9:16"], {"default": "1:1"}),
                "mode": (["std", "pro"], {"default": "std"}),
                "duration": (["5", "10"], {"default": "5"}),
                "num_videos": ("INT", {"default": 1, "min": 1, "max": 4}),
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647})
            },
            "optional": {
                "camera": (["none", "horizontal", "vertical", "zoom", "vertical_shake", "horizontal_shake", 
                          "rotate", "master_down_zoom", "master_zoom_up", "master_right_rotate_zoom", 
                          "master_left_rotate_zoom"], {"default": "none"}),
                "camera_value": ("FLOAT", {"default": 0, "min": -10, "max": 10, "step": 0.1})
            }
        }

    RETURN_TYPES = ("VIDEO", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("video", "video_url", "task_id", "video_id", "response")
    FUNCTION = "generate_video"
    CATEGORY = "Comfly/Comfly_kling"

    def __init__(self):
        super().__init__()
        self.api_key = self.load_config_file().get('api_key', '')

    def load_config_file(self):
        config_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "Comflyapi.json")
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading config file: {str(e)}")
            return {}

    def generate_video(self, prompt, model_name, imagination, aspect_ratio, mode, duration, num_videos, 
                  negative_prompt="", camera="none", camera_value=0, seed=0):
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
            "mode": mode,
            "duration": duration,
            "model_name": model_name,
            "imagination": imagination,
            "num_videos": num_videos,
            "camera_json": camera_json,
            "seed": seed
        }
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
                    video_path = self.download_video(video_url)
                    
                    # Format the response with task status information
                    response_data = {
                        "task_status": "succeed",
                        "task_status_msg": "Video generated successfully",
                        "progress": 100,
                    }
                    
                    return (video_path, video_url, task_id, video_id, json.dumps(response_data))
                
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

    def get_headers(self):
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

    def download_video(self, video_url):
        input_path = folder_paths.get_output_directory()
        video_filename = f"{str(uuid.uuid4())}.mp4"
        video_path = os.path.join(input_path, video_filename)

        response = requests.get(video_url, stream=True)
        with open(video_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        return video_path

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


class Comfly_kling_image2video:
    @classmethod 
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "prompt": ("STRING", {"multiline": True}),
                "model_name": (["kling-v1-6", "kling-v1-5", "kling-v1"], {"default": "kling-v1-6"}),
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
                "camera": (["none", "horizontal", "vertical", "zoom", "vertical_shake", "horizontal_shake", 
                          "rotate", "master_down_zoom", "master_zoom_up", "master_right_rotate_zoom", 
                          "master_left_rotate_zoom"], {"default": "none"}),
                "camera_value": ("FLOAT", {"default": 0, "min": -10, "max": 10, "step": 0.1})
            }
        }

    RETURN_TYPES = ("VIDEO", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("video", "video_url", "task_id", "video_id", "response")
    FUNCTION = "generate_video"
    CATEGORY = "Comfly/Comfly_kling"

    def __init__(self):
        super().__init__()
        self.api_key = self.load_config_file().get('api_key', '')

    def load_config_file(self):
        config_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "Comflyapi.json")
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading config file: {str(e)}")
            return {}

    def generate_video(self, image, prompt, model_name, imagination, aspect_ratio, mode, duration, 
                  num_videos, negative_prompt="", camera="none", camera_value=0, seed=0, image_tail=None):
        if not self.api_key:
            error_response = {"task_status": "failed", "task_status_msg": "API key not found in Comflyapi.json"}
            return ("", "", "", "", json.dumps(error_response))
            
        camera_json = {}
        if model_name == "kling-v1-5" and mode == "pro": 
            camera_json = self.get_camera_json(camera, camera_value)
        else:
            camera_json = self.get_camera_json("none", 0)
            
        try:
            image_base64 = self.image_to_base64(tensor2pil(image)[0])
            image_tail_base64 = ""
            if image_tail is not None:
                image_tail_base64 = self.image_to_base64(tensor2pil(image_tail)[0])
                
            payload = {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "image": image_base64,
                "image_tail": image_tail_base64,
                "aspect_ratio": aspect_ratio,
                "mode": mode,
                "duration": duration,
                "model_name": model_name,
                "imagination": imagination,
                "num_videos": num_videos,
                "camera_json": camera_json,
                "seed": seed
            }
            
            pbar = comfy.utils.ProgressBar(100)
            pbar.update_absolute(5)  
            response = requests.post(
                "https://ai.comfly.chat/kling/v1/videos/image2video",
                headers=self.get_headers(),
                json=payload
            )
            response.raise_for_status()
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
                    video_path = self.download_video(video_url)
                    
                    # Format the response with task status information
                    response_data = {
                        "task_status": "succeed",
                        "task_status_msg": "Video generated successfully",
                        "progress": 100,
                    }
                    
                    return (video_path, video_url, task_id, video_id, json.dumps(response_data))
                
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

    def get_headers(self):
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

    def download_video(self, video_url):
        input_path = folder_paths.get_output_directory()
        video_filename = f"{str(uuid.uuid4())}.mp4"
        video_path = os.path.join(input_path, video_filename)

        response = requests.get(video_url, stream=True)
        with open(video_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        return video_path

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

    def image_to_base64(self, image):
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')



class Comfly_video_extend:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_id": ("STRING", {"default": "", "multiline": False, "forceInput": True}),
                "prompt": ("STRING", {"default": "", "multiline": True}),
            }
        }

    RETURN_TYPES = ("VIDEO", "STRING", "STRING")
    RETURN_NAMES = ("video", "video_id", "response")
    FUNCTION = "extend_video"
    CATEGORY = "Comfly/Comfly_kling"

    def __init__(self):
        super().__init__()
        self.api_key = self.load_config_file().get('api_key', '')

    def load_config_file(self):
        config_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "Comflyapi.json")
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading config file: {str(e)}")
            return {}

    def extend_video(self, video_id, prompt=""):
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
                json=payload
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
                    headers=headers
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
                    video_path = self.download_video(video_url)
                    
                    # Format the response with task status information
                    response_data = {
                        "task_status": "succeed",
                        "task_status_msg": "Video extended successfully",
                        "progress": 100,
                    }
                    
                    return (video_path, new_video_id, json.dumps(response_data))
                
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

    def download_video(self, video_url):
        input_path = folder_paths.get_output_directory()
        video_filename = f"{str(uuid.uuid4())}.mp4"
        video_path = os.path.join(input_path, video_filename)

        response = requests.get(video_url, stream=True)
        with open(video_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        return video_path
   

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
                "video_url": ("STRING", {"default": ""}),
                "audio_type": (["file", "url"], {"default": "file"}),
                "audio_file": ("STRING", {"default": ""}),
                "audio_url": ("STRING", {"default": ""})
            }
        }

    RETURN_TYPES = ("VIDEO", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("video", "video_url", "task_id", "response")
    FUNCTION = "process_lip_sync"
    CATEGORY = "Comfly/Comfly_kling"

    def __init__(self):
        super().__init__()
        self.api_key = self.load_config_file().get('api_key', '')
        self.zh_voice_map = {name: voice_id for name, voice_id in self.__class__.zh_voices}
        self.en_voice_map = {name: voice_id for name, voice_id in self.__class__.en_voices}

    def load_config_file(self):
        config_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "Comflyapi.json")
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading config file: {str(e)}")
            return {}

        def process_lip_sync(self, video_id, task_id, mode, text, voice_language, zh_voice, en_voice, voice_speed, seed=0,
                    video_url="", audio_type="file", audio_file="", audio_url=""):
    
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
                    json=payload
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
                        headers=headers
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
                        video_path = self.download_video(video_url)
                        
                        # Format the response with task status information
                        response_data = {
                            "task_status": "succeed",
                            "task_status_msg": "Lip sync completed successfully",
                            "progress": 100,
                        }
                        
                        return (video_path, video_url, task_id, json.dumps(response_data))
                        
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

        def download_video(self, video_url):
            input_path = folder_paths.get_output_directory()
            video_filename = f"{str(uuid.uuid4())}.mp4"
            video_path = os.path.join(input_path, video_filename)

            response = requests.get(video_url, stream=True)
            with open(video_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            return video_path


class Comfly_kling_videoPreview:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":{
            "video":("VIDEO",),
        }}
    
    CATEGORY = "Comfly/Comfly_kling"
    DESCRIPTION = "Preview the generated video."

    RETURN_TYPES = ()

    OUTPUT_NODE = True

    FUNCTION = "Preview_video"

    def Preview_video(self, video):
        video_name = os.path.basename(video)
        video_path_name = os.path.basename(os.path.dirname(video))
        return {"ui":{"video":[video_name,video_path_name]}}
    


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
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_file_path = os.path.join(current_dir, 'Comflyapi.json')
        
        with open(config_file_path, 'r') as f:
            api_config = json.load(f)
            
        self.api_key = api_config.get('api_key', '')
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
                object_image=None, subject_image=None, scene_image=None):
        try:
            # Set the timeout value from user input
            self.timeout = timeout
            
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
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_file_path = os.path.join(current_dir, 'Comflyapi.json')
        
        try:
            with open(config_file_path, 'r') as f:
                api_config = json.load(f)
                self.api_key = api_config.get('api_key', '')
        except Exception as e:
            print(f"Error loading API config: {str(e)}")
            self.api_key = ""
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
    def edit_image(self, image, prompt, scale=0.5, seed=-1, add_logo=False, logo_position="右下角", logo_language="中文", logo_text=""):
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

class ComflyChatGPTApi:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "model": ("STRING", {"default": "gpt-4o-all", "multiline": False}),             
            },
            "optional": {
                "files": ("FILES",), 
                "image_url": ("STRING", {"multiline": False, "default": ""}),
                "images": ("IMAGE", {"default": None}),  
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.01}),
                "max_tokens": ("INT", {"default": 4096, "min": 1, "max": 16384, "step": 1}),
                "top_p": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "frequency_penalty": ("FLOAT", {"default": 0.0, "min": -2.0, "max": 2.0, "step": 0.01}),
                "presence_penalty": ("FLOAT", {"default": 0.0, "min": -2.0, "max": 2.0, "step": 0.01}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 2147483647}),
                "image_download_timeout": ("INT", {"default": 30, "min": 5, "max": 120, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("images", "response", "image_urls")
    FUNCTION = "process"
    CATEGORY = "Comfly/Chatgpt"

    def __init__(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_file_path = os.path.join(current_dir, 'Comflyapi.json')
        
        try:
            with open(config_file_path, 'r') as f:
                api_config = json.load(f)
                self.api_key = api_config.get('api_key', '')
        except Exception as e:
            print(f"Error loading API config: {str(e)}")
            self.api_key = ""
        
        # Fixed timeout of 300 seconds for API calls
        self.timeout = 300
        # Default timeout for image downloads
        self.image_download_timeout = 30
        self.api_endpoint = "https://ai.comfly.chat/v1/chat/completions"

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
        import re
        # Extract URLs from markdown image format: ![description](url)
        image_pattern = r'!\[.*?\]\((.*?)\)'
        matches = re.findall(image_pattern, response_text)
        
        # If no markdown format found, extract raw URLs that look like image links
        if not matches:
            url_pattern = r'https?://\S+\.(?:jpg|jpeg|png|gif|webp)'
            matches = re.findall(url_pattern, response_text)
            
        # Extract all URLs that look like links
        if not matches:
            all_urls_pattern = r'https?://\S+'
            matches = re.findall(all_urls_pattern, response_text)
            
        return matches if matches else []

    def download_image(self, url, timeout=30):
        """Download image from URL and convert to tensor with improved error handling"""
        try:
            # Custom headers to mimic a browser
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.9',
                'Referer': 'https://comfyui.com/'
            }
            
            # Try to download with custom headers
            response = requests.get(url, headers=headers, timeout=timeout)
            response.raise_for_status()
            
            # Check content type to verify it's an image
            content_type = response.headers.get('Content-Type', '')
            if not content_type.startswith('image/'):
                print(f"Warning: URL doesn't point to an image. Content-Type: {content_type}")
                # Still try to open it anyway
            
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

    def process(self, model, prompt, files=None, image_url="", images=None, temperature=0.7, 
               max_tokens=4096, top_p=1.0, frequency_penalty=0.0, presence_penalty=0.0, seed=-1,
               image_download_timeout=30):
        try:
            # Set image download timeout from parameter
            self.image_download_timeout = image_download_timeout
            
            if not self.api_key:
                error_message = "API key not found in Comflyapi.json"
                print(error_message)
                # Return a blank image if no input image
                blank_img = Image.new('RGB', (512, 512), color='white')
                return (pil2tensor(blank_img), error_message, "")  
            
            # Initialize progress bar
            pbar = comfy.utils.ProgressBar(100)
            pbar.update_absolute(10)
            
            # Get current timestamp for formatting
            import time
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            
            # Handle seed
            if seed < 0:
                seed = random.randint(0, 2147483647)
                print(f"Using random seed: {seed}")
            
            # Prepare message content
            content = []
            
            # Add prompt text
            content.append({"type": "text", "text": prompt})
            
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
            
            # Add image URL if provided and no images
            elif image_url:
                content.append({
                    "type": "image_url",
                    "image_url": {"url": image_url}
                })
            
            # Add files if provided
            if files:
                file_paths = files if isinstance(files, list) else [files]
                for file_path in file_paths:
                    encoded_content, mime_type = self.file_to_base64(file_path)
                    if encoded_content and mime_type:
                        # Add file as attachment
                        if mime_type.startswith('image/'):
                            # If it's an image, use image_url type
                            content.append({
                                "type": "image_url", 
                                "image_url": {"url": f"data:{mime_type};base64,{encoded_content}"}
                            })
                        else:
                            # For non-image files
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
                "max_tokens": max_tokens,
                "top_p": top_p,
                "frequency_penalty": frequency_penalty,
                "presence_penalty": presence_penalty,
                "seed": seed,
                "stream": True  # Use streaming for better user experience
            }
            
            # Use asyncio for streaming response
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            response_text = loop.run_until_complete(self.stream_response(payload, pbar))
            loop.close()
            
            # Format the response
            formatted_response = f"**User prompt**: {prompt}\n\n**Response** ({timestamp}):\n{response_text}"
            
            # Check if response contains image URLs
            image_urls = self.extract_image_urls(response_text)
            image_urls_string = "\n".join(image_urls) if image_urls else ""
            
            if image_urls:
                try:
                    # Download and convert all found images
                    img_tensors = []
                    successful_downloads = 0
                    
                    for i, url in enumerate(image_urls):
                        print(f"Attempting to download image {i+1}/{len(image_urls)} from: {url}")
                        
                        # Update progress bar for each image
                        pbar.update_absolute(min(80, 40 + (i+1) * 40 // len(image_urls)))
                        
                        img_tensor = self.download_image(url, self.image_download_timeout)
                        if img_tensor is not None:
                            img_tensors.append(img_tensor)
                            successful_downloads += 1
                    
                    print(f"Successfully downloaded {successful_downloads} out of {len(image_urls)} images")
                    
                    if img_tensors:
                        # Combine all images into a single tensor batch
                        combined_tensor = torch.cat(img_tensors, dim=0)
                        pbar.update_absolute(100)
                        return (combined_tensor, formatted_response, image_urls_string)
                except Exception as e:
                    print(f"Error processing image URLs: {str(e)}")
            
            # Return appropriate response if no image was found or downloads failed
            if images is not None:
                # Return input images with text response
                pbar.update_absolute(100)
                return (images, formatted_response, image_urls_string)
            else:
                # Create a default blank image
                blank_img = Image.new('RGB', (512, 512), color='white')
                blank_tensor = pil2tensor(blank_img)
                pbar.update_absolute(100)
                return (blank_tensor, formatted_response, image_urls_string)
            
        except Exception as e:
            error_message = f"Error calling ChatGPT API: {str(e)}"
            print(error_message)
            
            # Return error with original image or blank image
            if images is not None:
                return (images, error_message, "")
            else:
                blank_img = Image.new('RGB', (512, 512), color='white')
                blank_tensor = pil2tensor(blank_img)
                return (blank_tensor, error_message, "")

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
                    
                    # Process streaming response
                    async for line in response.content:
                        line = line.decode('utf-8').strip()
                        if line.startswith('data: '):
                            data = line[6:]  # Remove 'data: ' prefix
                            if data == '[DONE]':
                                break
                            
                            try:
                                chunk = json.loads(data)
                                if 'choices' in chunk and chunk['choices']:
                                    delta = chunk['choices'][0].get('delta', {})
                                    if 'content' in delta:
                                        content = delta['content']
                                        full_response += content
                                        
                                        # Update progress (approximate progress)
                                        pbar.update_absolute(min(40, 20 + len(full_response) // 100))
                            except json.JSONDecodeError:
                                continue
            
            return full_response
            
        except asyncio.TimeoutError:
            raise TimeoutError(f"API request timed out after {self.timeout} seconds")
        except Exception as e:
            raise Exception(f"Error in streaming response: {str(e)}")
            

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
    "Comfly_kling_videoPreview": Comfly_kling_videoPreview,  
    "ComflyGeminiAPI": ComflyGeminiAPI,
    "ComflySeededit": ComflySeededit,
    "ComflyChatGPTApi": ComflyChatGPTApi,
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
    "Comfly_kling_videoPreview": "Comfly_kling_videoPreview",  
    "ComflyGeminiAPI": "Comfly Gemini API",
    "ComflySeededit": "Comfly Doubao SeedEdit",
    "ComflyChatGPTApi": "Comfly ChatGPT Api",
}
