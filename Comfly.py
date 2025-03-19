import os
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

        async with aiohttp.ClientSession() as session:
            async with session.post(f"{self.midjourney_api_url[self.speed]}/mj/submit/action", headers=headers, json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    return data
                else:
                    error_message = f"Error submitting Midjourney action: {response.status}"
                    print(error_message)
                    return error_message

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

        async with aiohttp.ClientSession() as session:
            async with session.post(f"{self.midjourney_api_url[self.speed]}/mj/submit/imagine", headers=headers, json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    return data["result"]
                else:
                    error_message = f"Error submitting Midjourney task: {response.status}"
                    print(error_message)
                    raise Exception(error_message)

    async def midjourney_fetch_task_result(self, taskId):
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer " + self.api_key
        }

        async with aiohttp.ClientSession() as session:
            async with session.get(f"{self.midjourney_api_url[self.speed]}/mj/task/{taskId}/fetch", headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    return data
                else:
                    error_message = f"Error fetching Midjourney task result: {response.status}"
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
        async with aiohttp.ClientSession() as session:
            async with session.post(f"{self.midjourney_api_url[self.speed]}/mj/submit/upload-discord-images", headers=self.get_headers(), json=payload) as response:
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

        async with aiohttp.ClientSession() as session:
            async with session.post(f"{self.midjourney_api_url[self.speed]}/mj/submit/action", headers=headers, json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    return data
                else:
                    error_message = f"Error submitting Midjourney action: {response.status}"
                    print(error_message)
                    return error_message

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


        async with aiohttp.ClientSession() as session:
            async with session.post(f"{self.midjourney_api_url[self.speed]}/mj/submit/imagine", headers=headers, json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    return data["result"]
                else:
                    error_message = f"Error submitting Midjourney task: {response.status}"
                    print(error_message)
                    raise Exception(error_message)

    async def midjourney_fetch_task_result(self, taskId):
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer " + self.api_key  
        }

        async with aiohttp.ClientSession() as session:
            async with session.get(f"{self.midjourney_api_url[self.speed]}/mj/task/{taskId}/fetch", headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    return data
                else:
                    error_message = f"Error fetching Midjourney task result: {response.status}"
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

    RETURN_TYPES = ("VIDEO", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("video", "video_url", "task_id", "video_id")
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
            raise ValueError("API key not found in Comflyapi.json")

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
                raise Exception(f"API Error: {result['message']}")

            task_id = result["data"]["task_id"]
            pbar.update_absolute(5)  

            while True:
                time.sleep(2)
                status_response = requests.get(
                    f"https://ai.comfly.chat/kling/v1/videos/text2video/{task_id}",
                    headers=self.get_headers()
                )
                status_response.raise_for_status()
                status_result = status_response.json()

                progress = status_result["data"].get("progress", 0)
                pbar.update_absolute(progress)

                if status_result["data"]["task_status"] == "succeed":
                    pbar.update_absolute(100) 
                    video_url = status_result["data"]["task_result"]["videos"][0]["url"]
                    video_id = status_result["data"]["task_result"]["videos"][0]["id"]
                    video_path = self.download_video(video_url)
                    return (video_path, video_url, task_id, video_id)
                
                elif status_result["data"]["task_status"] == "failed":
                    raise Exception("Video generation failed")

        except Exception as e:
            print(f"Error generating video: {str(e)}")
            return ("", "", "", "")

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

    RETURN_TYPES = ("VIDEO", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("video", "video_url", "task_id", "video_id")
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
            raise ValueError("API key not found in Comflyapi.json")

        camera_json = {}
        if model_name == "kling-v1-5" and mode == "pro": 
            camera_json = self.get_camera_json(camera, camera_value)
        else:
            camera_json = self.get_camera_json("none", 0)

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

        try:
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
                raise Exception(f"API Error: {result['message']}")

            task_id = result["data"]["task_id"]
            pbar.update_absolute(10) 

            while True:
                time.sleep(2)
                status_response = requests.get(
                    f"https://ai.comfly.chat/kling/v1/videos/image2video/{task_id}",
                    headers=self.get_headers()
                )
                status_response.raise_for_status()
                status_result = status_response.json()

                progress = status_result["data"].get("progress", 0)
                pbar.update_absolute(progress)

                if status_result["data"]["task_status"] == "succeed":
                    pbar.update_absolute(100) 
                    video_url = status_result["data"]["task_result"]["videos"][0]["url"]
                    video_id = status_result["data"]["task_result"]["videos"][0]["id"]
                    video_path = self.download_video(video_url)
                    return (video_path, video_url, task_id, video_id)
                
                elif status_result["data"]["task_status"] == "failed":
                    raise Exception("Video generation failed")

        except Exception as e:
            print(f"Error generating video: {str(e)}")
            return ("", "", "", "")

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

    RETURN_TYPES = ("VIDEO", "STRING")
    RETURN_NAMES = ("video", "video_id")
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
            raise ValueError("API key not found in Comflyapi.json")

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
                raise Exception(f"API Error: {result['message']}")

            task_id = result["data"]["task_id"]
            pbar = comfy.utils.ProgressBar(100)

            while True:
                time.sleep(2)
                status_response = requests.get(
                    f"https://ai.comfly.chat/kling/v1/videos/video-extend/{task_id}",
                    headers=headers
                )
                status_response.raise_for_status()
                status_result = status_response.json()

                if status_result["data"]["task_status"] == "succeed":
                    video_url = status_result["data"]["task_result"]["videos"][0]["url"]
                    new_video_id = status_result["data"]["task_result"]["videos"][0]["id"]
                    video_path = self.download_video(video_url)
                    return (video_path, new_video_id)
                
                elif status_result["data"]["task_status"] == "failed":
                    raise Exception(f"Video extension failed: {status_result['data'].get('task_status_msg', 'Unknown error')}")

                progress = 0
                if status_result["data"]["task_status"] == "processing":
                    progress = 50
                elif status_result["data"]["task_status"] == "succeed":
                    progress = 100
                pbar.update_absolute(progress)

        except Exception as e:
            print(f"Error extending video: {str(e)}")
            return ("", "")

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
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "top_p": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 1.0, "step": 0.01}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647}),
            },
            "optional": {
                "image": ("IMAGE",),  
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "response")
    FUNCTION = "process"
    CATEGORY = "Comfly/Comfly_Gemini"

    def __init__(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_file_path = os.path.join(current_dir, 'Comflyapi.json')
        
        with open(config_file_path, 'r') as f:
            api_config = json.load(f)
            
        self.api_key = api_config.get('api_key', '')

    def get_headers(self):
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

    def image_to_base64(self, image):
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    def base64_to_image(self, base64_str):
        # Remove data URL header if present
        if "base64," in base64_str:
            base64_str = base64_str.split("base64,")[1]
        
        # Decode base64 to image
        image_data = base64.b64decode(base64_str)
        image = Image.open(BytesIO(image_data))
        return image

    def extract_base64_images(self, response_text):
        # Extract base64 image data from response
        base64_pattern = r'data:image/[^;]+;base64,([a-zA-Z0-9+/=]+)'
        matches = re.findall(base64_pattern, response_text)
        return matches

    def format_response(self, raw_response, model_name, timestamp, num_images=0):
        """Format the response with model info and timestamp"""
        # Clean response to remove base64 data
        clean_response = re.sub(r'data:image/[^;]+;base64,[a-zA-Z0-9+/=]+', '[Image]', raw_response)
        
        header = f"**Model**: {model_name}\n**Time**: {timestamp}\n"
        if num_images > 0:
            header += f"**Generated Images**: {num_images}\n"
        header += "\n"
        
        return header + clean_response

    def process(self, prompt, model, temperature, top_p, seed, image=None):
        try:
            # Get current timestamp for response formatting
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            
            # Prepare message content (always includes text)
            content = [{"type": "text", "text": prompt}]
            
            # Add image to content if provided
            if image is not None:
                # Convert tensor to PIL image
                pil_image = tensor2pil(image)[0]
                image_base64 = self.image_to_base64(pil_image)
                content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}})
            
            # Create API payload
            payload = {
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": content
                    }
                ],
                "temperature": temperature,
                "top_p": top_p,
                "seed": seed if seed > 0 else None,
                "max_tokens": 8192
            }
            
            # Make API request with progress bar
            pbar = comfy.utils.ProgressBar(2)
            pbar.update(0)
            
            response = requests.post(
                "https://ai.comfly.chat/v1/chat/completions",
                headers=self.get_headers(),
                json=payload
            )
            response.raise_for_status()
            result = response.json()
            
            pbar.update(1)
            
            # Extract response text
            response_text = result["choices"][0]["message"]["content"]
            
            # Extract base64 images if present
            base64_images = self.extract_base64_images(response_text)
            
            if base64_images:
                # Use first base64 image to create the image output
                try:
                    result_image = self.base64_to_image(base64_images[0])
                    result_tensor = pil2tensor(result_image)
                    
                    # Format response text without the base64 data
                    formatted_response = self.format_response(response_text, model, timestamp, len(base64_images))
                    
                    return (result_tensor, formatted_response)
                except Exception as e:
                    print(f"Error processing base64 image: {str(e)}")
            
            pbar.update(2)
            
            # If no base64 images found, return appropriate response
            formatted_response = self.format_response(response_text, model, timestamp)
            
            # Return appropriate response based on input
            if image is not None:
                # If input image was provided, return it with the text response
                return (image, formatted_response)
            else:
                # If no input image, create a blank default image
                default_image = Image.new('RGB', (512, 512), color='white')
                default_tensor = pil2tensor(default_image)
                return (default_tensor, formatted_response)
            
        except Exception as e:
            error_message = f"Error calling Gemini API: {str(e)}"
            print(error_message)
            
            # Handle errors with appropriate image output
            if image is not None:
                return (image, error_message)
            else:
                default_image = Image.new('RGB', (512, 512), color='white')
                default_tensor = pil2tensor(default_image)
                return (default_tensor, error_message)


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
    "Comfly_kling_videoPreview": Comfly_kling_videoPreview, 
    "ComflyGeminiAPI": ComflyGeminiAPI,
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
    "Comfly_kling_videoPreview": "Comfly_kling_videoPreview",  
    "ComflyGeminiAPI": "Comfly Gemini API",
}
