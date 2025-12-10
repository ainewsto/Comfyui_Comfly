import os
import json


baseurl = "https://ai.comfly.chat"

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

class Comfly_api_set:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_base": (["comfly", "ip", "hk", "us"], {"default": "comfly"}),
                "apikey": ("STRING", {"default": ""}),
            },
            "optional": {
                "custom_ip": ("STRING", {"default": "", "placeholder": "Enter IP when using 'ip' option"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("apikey",)
    FUNCTION = "set_api_base"
    CATEGORY = "Comfly"

    def set_api_base(self, api_base, apikey="", custom_ip=""):
        global baseurl
        
        base_url_mapping = {
            "comfly": "https://ai.comfly.chat",
            "ip": custom_ip,
            "hk": "https://hk-api.gptbest.vip",
            "us": "https://api.gptbest.vip"
        }
        
        if api_base == "ip" and not custom_ip.strip():
            raise ValueError("When selecting 'ip' option, you must provide a custom IP address in the 'custom_ip' field")
        
        if api_base in base_url_mapping:
            baseurl = base_url_mapping[api_base]
            
        if apikey.strip():
            config = get_config()
            config['api_key'] = apikey
            
        message = f"API Base URL set to: {baseurl}"
        if apikey.strip():
            message += "\nAPI key has been updated"
            
        print(message)
        return (apikey,)