import os
import requests
import time
import numpy as np
import torch
from PIL import Image, ImageOps, UnidentifiedImageError
from io import BytesIO
import json
import comfy.utils
from typing import List, Union
import re
import aiohttp
import asyncio
import base64
import uuid
from split_image import split_image
import tempfile


def pil2tensor(image: Union[Image.Image, List[Image.Image]]) -> torch.Tensor:
    if isinstance(image, list):
        return torch.cat([pil2tensor(img) for img in image], dim=0)

    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

class Comfly_to_image:
    """
    Comfly_to_image node

    Processes image inputs from URL and returns the selected image.

    Inputs:
        image_url (STRING): URL of the image.

    Outputs:
        image (IMAGE): Selected image.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "url": (
                    "STRING",
                    {
                        "default": "https://comfly.chat/image.jpg"
                    },
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process_input"
    CATEGORY = "Comfly"

    def process_input(self, url):
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            image = Image.open(response.raw)
            image = ImageOps.exif_transpose(image)
            return (pil2tensor(image),)
        except (requests.exceptions.RequestException, IOError) as e:
            print(f"Error fetching or processing image: {str(e)}")
            placeholder_image = Image.new("RGB", (512, 512), (255, 255, 255))  
            return (pil2tensor(placeholder_image),)
    

def pil2tensor(image: Union[Image.Image, List[Image.Image]]) -> torch.Tensor:
    if isinstance(image, list):
        if len(image) == 0:
            return torch.empty(0)
        return torch.cat([pil2tensor(img) for img in image], dim=0)

    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)


class Comfly_split_image:
    """Comfly_split_image node
    Splits an input image into specified number of rows and columns.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "rows": ("INT", {"default": 2, "min": 1}),
                "columns": ("INT", {"default": 2, "min": 1}),
                "should_square": ("BOOLEAN", {"default": False}),
                "should_cleanup": ("BOOLEAN", {"default": False}),
                "should_quiet": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "output_dir": ("STRING", {"default": os.path.join(os.getcwd(), "ComfyUI", "output", "Comfly_split_image")}),
                "image_url": ("STRING", {"default": ""}),
                "image_path": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "INT")
    RETURN_NAMES = ("IMAGE", "MASK", "IMAGE_COUNT")
    FUNCTION = "split"
    CATEGORY = "Comfly"

    def __init__(self):
        self.output_dir = ""

    def split(self, rows: int, columns: int, should_square: bool, should_cleanup: bool, should_quiet: bool, output_dir: str = "", image_url: str = "", image_path: str = "") -> List[Image.Image]:
        if image_url:
            output_path = os.path.join(os.getcwd(), "ComfyUI", "output", "original_image.png")
            try:
                response = requests.get(image_url)
                response.raise_for_status()
                with open(output_path, "wb") as f:
                    f.write(response.content)
                print(f"Original image saved to: {output_path}")
                image_path = output_path
            except Exception as e:
                print(f"Error saving original image: {str(e)}")
                return (torch.empty(0), torch.empty(0), 0)
        
        if not image_path:
            print("No image path or URL provided.")
            return (torch.empty(0), torch.empty(0), 0)
        
        try:
            image = self.load_image_from_path(image_path)
            print(f"Loaded image: {image}")

            if should_square:
                print("Squaring image...")
                image = self.make_square(image)
                image.save(image_path)
                print(f"Squared image saved to: {image_path}")

            if not output_dir:
                output_dir = os.path.join(os.getcwd(), "ComfyUI", "output", "Comfly_split_image")
            os.makedirs(output_dir, exist_ok=True)
            print(f"Output directory: {output_dir}")

            print("Splitting image...")
            split_images_dir = os.path.join(output_dir, "split_images")
            os.makedirs(split_images_dir, exist_ok=True)
            split_image(image_path, rows, columns, should_square, should_cleanup, should_quiet, split_images_dir)

            images, masks = self.load_split_images(split_images_dir)

            image_count = len(images)
            if image_count == 0:
                raise ValueError("No split images found in output directory.")
            
            image_tensor = torch.cat(images, dim=0)
            mask_tensor = torch.stack(masks, dim=0)
            
            self.output_dir = output_dir
            
            return (image_tensor, mask_tensor, image_count)
        
        except Exception as e:
            print(f"Error during image split: {str(e)}")
            return (torch.empty(0), torch.empty(0), 0)

    def load_image_from_path(self, path: str) -> Image.Image:
        if not os.path.exists(path):
            raise FileNotFoundError(f"The file {path} does not exist.")
        print(f"Loading image from path: {path}")
        return Image.open(path)

    def make_square(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        print(f"Image size: {width}x{height}")
        if width == height:
            print("Image is already square.")
            return image
        new_size = max(width, height)
        print(f"New square size: {new_size}x{new_size}")
        new_image = Image.new("RGB", (new_size, new_size), (255, 255, 255))
        new_image.paste(image, ((new_size - width) // 2, (new_size - height) // 2))
        return new_image

    def load_split_images(self, split_images_dir):
        images = []
        masks = []
        for file in os.listdir(split_images_dir):
            if file.endswith(".jpg") or file.endswith(".jpeg") or file.endswith(".png"):
                path = os.path.join(split_images_dir, file)
                i = Image.open(path)
                i = ImageOps.exif_transpose(i)
                image = i.convert("RGB")
                image = np.array(image).astype(np.float32) / 255.0
                image = torch.from_numpy(image)[None,]

                if 'A' in i.getbands():
                    mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
                    mask = 1. - torch.from_numpy(mask)
                else:
                    mask = torch.zeros((image.shape[1], image.shape[2]), dtype=torch.float32, device="cpu")

                images.append(image)
                masks.append(mask)

        return images, masks

    def getOutputDir(self):
        return self.output_dir
    
    @classmethod
    def IS_CHANGED(cls, image_path: str, rows: int, columns: int, should_square: bool, should_cleanup: bool, should_quiet: bool, output_dir: str = ""):
        return True

current_dir = os.path.dirname(os.path.abspath(__file__))
config_file_path = os.path.join(current_dir, 'Comflyapi.json')

with open(config_file_path, 'r') as f:
    api_config = json.load(f)

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

def tensor2pil(image: torch.Tensor) -> List[Image.Image]:
    batch_count = image.size(0) if len(image.shape) > 3 else 1
    if batch_count > 1:
        out = []
        for i in range(batch_count):
            out.extend(tensor2pil(image[i]))
        return out

    return [
        Image.fromarray(
            np.clip(255.0 * image.cpu().numpy().squeeze(), 0, 255).astype(
                np.uint8
            )
        )
    ]

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
        model_version (STRING): Selected Midjourney model version (v 6.0, v 5.2, v 5.1, niji 6, niji 5, niji 4).
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
                "model_version": (["v 6.0", "v 5.2", "v 5.1", "niji 6", "niji 5", "niji 4"], {"default": "v 6.0"}),
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

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("image_url", "text", "taskId")
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

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("image_url", "taskId")
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
            while not task_result or task_result.get("status") != "SUCCESS" or task_result.get("progress") != "100%":
                await asyncio.sleep(1)
                task_result = await self.midjourney_fetch_task_result(response["result"])


            if task_result.get("code") == 5 and task_result.get("description") == "task_no_found":
                print(f"Task not found for taskId: {taskId}")
                raise self.MidjourneyError(f"Task not found for taskId: {taskId}")

            return task_result["imageUrl"], response["result"]

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

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("image_url",)
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
        
        result_str = f"Run method completed with results: taskId={taskId}, upsample_v6_2x_subtle={upsample_v6_2x_subtle}, upsample_v6_2x_creative={upsample_v6_2x_creative}, costume_zoom={costume_zoom}, zoom={zoom}, pan_left={pan_left}, pan_right={pan_right}, pan_up={pan_up}, pan_down={pan_down}, image_url={image_url}"
        print(result_str)
        
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
                else:
                    image_url = task_result["imageUrl"]    
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
    


class Comfly_coze:
    """
    Comfly_coze node

    Interacts with the Coze API to process text input and returns processed output.

    Inputs:
        text (STRING, optional): Input text for processing.
        seed (INT): Seed value for reproducibility.

    Outputs:
        image (IMAGE): Processed image output.
        text (STRING): Processed text output.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "text": ("STRING", {"multiline": True}),
            },
            "required": {
                "seed": ("INT", {"default": 42, "min": 0}),
            },
            "hidden": {
                "stream": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("output_image", "output_text")
    FUNCTION = "process_input"
    CATEGORY = "Comfly"

    def __init__(self):
        self.api_url = "https://api.coze.cn/open_api/v2/chat"
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_file_path = os.path.join(current_dir, 'Comflyapi.json')

        with open(config_file_path, 'r') as f:
            api_config = json.load(f)

        self.auth_token = api_config.get('coze_auth_token', '') 
        self.bot_id = api_config.get('bot_id', '')
        self.user_id = api_config.get('user_id', '')

    async def process_coze_api(self, payload):
        headers = {
            "Authorization": f"Bearer {self.auth_token}",
            "Content-Type": "application/json",
            "Accept": "*/*",
            "Host": "api.coze.cn",
            "Connection": "keep-alive"
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(self.api_url, headers=headers, json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    return data
                else:
                    raise Exception(f"Error from Coze API: {response.status}")

    def process_input(self, text="", seed=42, stream=False):
        print("Input values:")
        print(f"  text: {text}")
        print(f"  seed: {seed}")
        print(f"  stream: {stream}")

        payload = {
            "conversation_id": "123",
            "bot_id": self.bot_id,
            "user": self.user_id,
            "query": "",
            "stream": stream
        }

        if text:
            print(f"Processing text: {text}")
            payload["query"] = text

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(self.process_coze_api(payload))
        loop.close()

        output_image = None
        output_text = ""
        plugin_name = ""

        if "messages" in result:
            for message in result["messages"]:
                if message["role"] == "assistant" and message["type"] == "function_call":
                    content = json.loads(message["content"])
                    if "plugin_name" in content:
                        plugin_name = content["plugin_name"]
                if message["role"] == "assistant" and message["type"] == "tool_response":
                    content = json.loads(message["content"])
                    if content is None:
                        raise ValueError("Content cannot be None")
                    if "data" in content and "image_urls" in content["data"]:
                        image_url = content["data"]["image_urls"][0]
                        text = content["data"].get("text", "")
                        response = requests.get(image_url)
                        image = Image.open(BytesIO(response.content))
                        output_image = pil2tensor(image)
                elif message["role"] == "assistant" and message["type"] == "answer":
                    output_text = message["content"]
                    print(f"Output text: {output_text}")

        if plugin_name:
            output_text = f"Plugin: {plugin_name}\n{output_text}"

        return output_image, output_text

def pil2tensor(image: Union[Image.Image, List[Image.Image]]) -> torch.Tensor:
    if isinstance(image, list):
        return torch.cat([pil2tensor(img) for img in image], dim=0)

    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)


class Comfly_luma:
    """
    Comfly_luma node

    Processes image and/or text inputs using Luma API and returns the processed video file path.

    Inputs:
        image (IMAGE, optional): Input image for processing.
        text (STRING, optional): Input text prompt for processing.

    Outputs:
        video_path (STRING): Local file path of the processed video.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"default": "", "multiline": True}),
            },
            "optional": {
                "image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("video_path",)
    FUNCTION = "process_input"
    CATEGORY = "Comfly"

    def __init__(self):
        self.api_url = "https://ai.comfly.chat/luma/generations/"
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_file_path = os.path.join(current_dir, 'Comflyapi.json')

        with open(config_file_path, 'r') as f:
            api_config = json.load(f)

        self.api_key = api_config.get('api_key', '')

    async def process_luma_api(self, payload):
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(self.api_url, headers=headers, json=payload) as response:
                if response.status == 201:
                    data = await response.json()
                    return data
                else:
                    raise Exception(f"Error from Luma API: {response.status}")

    def process_input(self, text="", image=None):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        video_path = loop.run_until_complete(self.process_image_and_text(image, text))
        loop.close()

        return (video_path,)

    async def process_image_and_text(self, image, text):
        payload = {
            "aspect_ratio": "16:9",
            "expand_prompt": True,
            "user_prompt": text
        }

        if image is not None:
            # Convert the image to base64
            buffered = BytesIO()
            image = tensor2pil(image)[0]
            image.save(buffered, format="PNG")
            image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
            payload["image_url"] = f"data:image/png;base64,{image_base64}"

        # Submit the task to Luma API
        response = await self.process_luma_api(payload)
        task_id = response["id"]

        # Fetch the task result
        while True:
            await asyncio.sleep(1)
            response = await self.fetch_task_result(task_id)
            if response["state"] == "completed":
                video_url = response["video"]["url"]
                break

        # Download the video to a local temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
            video_path = temp_file.name
            response = requests.get(video_url)
            temp_file.write(response.content)

        return video_path

    async def fetch_task_result(self, task_id):
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        async with aiohttp.ClientSession() as session:
            async with session.get(f"{self.api_url}{task_id}", headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    return data
                else:
                    raise Exception(f"Error fetching task result from Luma API: {response.status}")
                
class Comfly_kling_image:
    """
    Comfly_kling_image node

    Generates images from text inputs using the Kling AI API.

    Inputs:
        text (STRING, optional): Input text for text-to-image generation.
        aspect_ratio (STRING): Aspect ratio of the generated images (1:1, 16:9, 4:3, 3:2, 2:3, 3:4, 9:16).
        image_count (INT): Number of images to generate (1-8).

    Outputs:
        image (IMAGE): Generated images.
    """

    api_url = "https://klingai.kuaishou.com"

    def __init__(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_file_path = os.path.join(current_dir, 'Comflyapi.json')

        with open(config_file_path, 'r') as f:
            config = json.load(f)

        self.cookie = config.get('cookie', '')

        if not self.cookie:
            raise ValueError("Cookie is required. Please enter the correct cookie in Comflyapi.json")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True}),
                "aspect_ratio": (["1:1", "16:9", "4:3", "3:2", "2:3", "3:4", "9:16"], {"default": "1:1"}),
                "image_count": ("INT", {"default": 4, "min": 1, "max": 8, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate_image"
    CATEGORY = "Comfly/Comfly_kling"

    def generate_image(self, text, aspect_ratio, image_count):
        if text is None:
            raise ValueError("Text input must be provided.")

        payload = {
            "arguments": [
                {"name": "prompt", "value": text},
                {"name": "style", "value": "默认"},
                {"name": "aspect_ratio", "value": aspect_ratio},
                {"name": "imageCount", "value": str(image_count)}, 
                {"name": "biz", "value": "klingai"},
            ],
            "inputs": [],
            "type": "mmu_txt2img_aiweb",
        }

        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            task_id = loop.run_until_complete(self.submit_task(payload))
        finally:
            loop.close()

        if task_id is None:
            print("Failed to submit task to Kling AI API.")
            return (torch.zeros((4, 512, 512, 3)),)

        output_images = self.wait_for_task_completion(task_id)
        return (output_images,)

    async def submit_task(self, payload):
        headers = {
            "Content-Type": "application/json;charset=UTF-8",
            "Cookie": self.cookie,
        }
        max_retries = 3
        retry_delay = 1

        for attempt in range(max_retries):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(f"{self.api_url}/api/task/submit", json=payload, headers=headers) as response:
                        if response.status == 200:
                            data = await response.json()
                            if data and "data" in data and "task" in data["data"] and "id" in data["data"]["task"]:
                                return data["data"]["task"]["id"]
                            else:
                                print(f"Unexpected response from Kling AI API: {data}")
                                return None
                        else:
                            print(f"Error submitting task: {response.status}")
                            return None
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"Attempt {attempt + 1} failed. Retrying in {retry_delay} seconds. Error: {str(e)}")
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    print(f"Failed to submit task after {max_retries} attempts. Last error: {str(e)}")
                    return None

    def wait_for_task_completion(self, task_id):
        headers = {
            "Content-Type": "application/json, text/plain, */*",
            "Accept-Encoding": "gzip, deflate, br, zstd",
            "Cookie": self.cookie,
        }
        max_retries = 5
        retry_delay = 1

        for attempt in range(max_retries):
            try:
                response = requests.get(f"{self.api_url}/api/task/status?taskId={task_id}", headers=headers)
                data = response.json()

                if data["data"]["status"] == 99:
                    output_images = []
                    for work in data["data"]["works"]:
                        image_url = work["resource"]["resource"]
                        image_response = requests.get(image_url, stream=True)
                        try:
                            output_images.append(pil2tensor(Image.open(image_response.raw)))
                        except UnidentifiedImageError as e:
                            print(f"Error opening image: {e}")
                            output_images.append(torch.zeros((1, 512, 512, 3), dtype=torch.uint8))
                    return torch.cat(output_images, dim=0)

                time.sleep(retry_delay)
                retry_delay *= 1.5
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"Attempt {attempt + 1} failed. Retrying in {retry_delay} seconds. Error: {str(e)}")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    print(f"Failed to get task result after {max_retries} attempts. Last error: {str(e)}")
                    return torch.zeros((4, 512, 512, 3))
        


   
WEB_DIRECTORY = "./web"    
        
NODE_CLASS_MAPPINGS = {
    "Comfly_to_image": Comfly_to_image,
    "Comfly_split_image": Comfly_split_image, 
    "Comfly_Mj": Comfly_Mj,
    "Comfly_mjstyle": Comfly_mjstyle,
    "Comfly_upload": Comfly_upload,
    "Comfly_Mju": Comfly_Mju,
    "Comfly_Mjv": Comfly_Mjv,  
    "Comfly_coze": Comfly_coze,
    "Comfly_luma": Comfly_luma,
    "Comfly_kling_image": Comfly_kling_image,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Comfly_to_image": "Comfly_to_image",
    "Comfly_split_image": "Comfly_split_image",
    "Comfly_Mj": "Comfly_Mj", 
    "Comfly_mjstyle": "Comfly_mjstyle",
    "Comfly_upload": "Comfly_upload",
    "Comfly_Mju": "Comfly_Mju",
    "Comfly_Mjv": "Comfly_Mjv",  
    "Comfly_coze": "Comfly_coze",
    "Comfly_luma": "Comfly_luma",
    "Comfly_kling_image": "Comfly_kling_image",
}
        
        
