import asyncio
import aiohttp
from aiohttp import web
from aiohttp_cors import setup, ResourceOptions
import subprocess
import os
import json
import shutil
import git
import requests
import threading
import logging
import time
import numpy as np
import torch
from PIL import Image
from io import BytesIO
import re
import sys
import shutil
import zipfile
import sysconfig
import hashlib
from io import StringIO
from contextlib import redirect_stdout, redirect_stderr



async def on_prepare(request, response):
    request.start_time = time.time()

async def on_response(request, response):
    pass

async def on_request_start(request, *args, **kwargs):
    pass

async def on_request_end(request, *args, **kwargs):
    pass


async def get_python_path():
    current_path = os.path.abspath(__file__)
    comfyui_path = os.path.abspath(os.path.join(current_path, "..", "..", ".."))
    python_paths = [
        os.path.join(comfyui_path, "python_miniconda", "python.exe"),
        os.path.join(comfyui_path, "python_miniconda", "bin", "python"),
        os.path.join(comfyui_path, "venv", "bin", "python")
    ]
    for path in python_paths:
        if os.path.exists(path):
            return path
    return "python"

async def get_dependencies(request):
    try:
        python_path = await get_python_path()
        output = subprocess.check_output([python_path, '-m', 'pip', 'list', '--format=json'])
        dependencies = json.loads(output)
        formatted_dependencies = [f"{dep['name']}=={dep['version']}" for dep in dependencies]
        return web.json_response(formatted_dependencies)
    except subprocess.CalledProcessError as e:
        return web.json_response({'error': str(e)}, status=500)

async def install_dependency(request):
    name = request.query.get('name')
    if not name:
        return web.json_response({'error': 'Name is required'}, status=400)
    try:
        python_path = await get_python_path()
        
        process = subprocess.Popen([python_path, '-m', 'pip', 'install', name], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)

        output = []
        while True:
            line = process.stdout.readline()
            if line:
                print(line.strip())  
                output.append(line)
            else:
                break

        return_code = process.wait()  

        if return_code == 0:
            response_data = {'message': 'Installation successful', 'output': ''.join(output)}
            print(f"Message: {response_data}")
            return web.json_response(response_data)
        else:
            response_data = {'error': 'Installation failed', 'output': ''.join(output)}
            print(f"Message: {response_data}")
            return web.json_response(response_data, status=500)

    except subprocess.CalledProcessError as e:
        print(f"Error installing dependency: {name}")
        print(e.output.decode('utf-8'))  
        return web.json_response({'error': str(e)}, status=500)
    

async def manage_dependency(request):
    name = request.query.get('name')
    action = request.query.get('action')
    if not name or not action:
        return web.json_response({'error': 'Name and action are required'}, status=400)
    try:
        python_path = await get_python_path()
        if action == 'uninstall':
            subprocess.check_call([python_path, '-m', 'pip', 'uninstall', '-y', name])
        else:
            return web.json_response({'error': f'Invalid action: {action}'}, status=400)
        return web.json_response({'message': f'{action.capitalize()} successful'})
    except subprocess.CalledProcessError as e:
        return web.json_response({'error': str(e)}, status=500)

async def replace_dependency(request):
    name = request.query.get('name')
    version = request.query.get('version')
    if not name or not version:
        return web.json_response({'error': 'Name and version are required'}, status=400)
    try:
        python_path = await get_python_path()
        subprocess.check_call([python_path, '-m', 'pip', 'install', f'{name}=={version}'])
        return web.json_response({'message': 'Replacement successful'})
    except subprocess.CalledProcessError as e:
        return web.json_response({'error': str(e)}, status=500)

comfyui_versions_cache = None

async def get_comfyui_versions(request):
    global comfyui_versions_cache
    if comfyui_versions_cache is not None:
        return web.json_response(comfyui_versions_cache)
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get("https://api.github.com/repos/comfyanonymous/ComfyUI/commits?per_page=1000") as response:
                if response.status == 200:
                    commits = await response.json()
                    versions = [{'id': commit['sha'][:7], 'message': commit['commit']['message'], 'date': commit['commit']['committer']['date']} for commit in commits]
                    comfyui_versions_cache = versions
                    return web.json_response(versions)
                else:
                    error_message = f"Error fetching ComfyUI versions: {response.status}"
                    print(error_message)
                    raise Exception(error_message)
    except Exception as e:
        error_message = f"An unexpected error occurred while fetching ComfyUI versions: {str(e)}"
        print(error_message)
        return web.json_response({'error': error_message}, status=500)

async def select_comfyui_version(request):
    version_id = request.query.get('version_id')
    if not version_id:
        return web.json_response({'error': 'Version ID is required'}, status=400)
    try:
        comfyui_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        repo = git.Repo(comfyui_path)

        repo.remotes.origin.fetch()
        repo.git.reset('--hard', 'origin/master') 

        try:
            repo.commit(version_id)
        except git.BadName:
            error_message = f'Version {version_id} not found'
            print(error_message)
            return web.json_response({'error': error_message}, status=404)

        repo.git.checkout(version_id)

        global comfyui_versions_cache
        comfyui_versions_cache = None
        return web.json_response({'message': f'ComfyUI version switched to {version_id}'})
    except Exception as e:
        error_message = f"Error selecting ComfyUI version: {str(e)}"
        print(error_message)
        return web.json_response({'error': error_message}, status=500)

current_comfyui_version_cache = None

async def get_current_comfyui_version(request):
    global current_comfyui_version_cache
    if current_comfyui_version_cache is not None:
        return web.Response(text=current_comfyui_version_cache)
    try:
        comfyui_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        repo = git.Repo(comfyui_path)
        current_version = repo.head.object.hexsha[:7]
        current_comfyui_version_cache = current_version
        return web.Response(text=current_version)
    except Exception as e:
        error_message = f"Error getting current ComfyUI version: {str(e)}"
        print(error_message)
        return web.json_response({'error': error_message}, status=500)   

async def get_plugins(request):
    try:
        current_path = os.path.dirname(os.path.abspath(__file__))
        plugins_dir = os.path.abspath(os.path.join(current_path, ".."))
        plugins = []
        exclude_files = ["__pycache__", "example_node.py.example", "websocket_image_save.py"]
        for entry in os.listdir(plugins_dir):
            if entry in exclude_files:
                continue
            plugin_path = os.path.join(plugins_dir, entry)
            if os.path.isdir(plugin_path):
                plugin_name = entry
                if plugin_name.endswith(".disabled"):
                    plugin_name = plugin_name[:-9]
                    enabled = False
                else:
                    enabled = True
                git_config_path = os.path.join(plugin_path, ".git", "config")
                if os.path.exists(git_config_path):
                    with open(git_config_path, "r") as git_config_file:
                        git_config = git_config_file.read()
                        url_match = re.search(r'url = (.*)', git_config)
                        if url_match:
                            url = url_match.group(1)
                        else:
                            url = ''
                else:
                    url = ''
                plugin = {
                    'name': plugin_name,
                    'type': 'directory',
                    'url': url,
                    'version': '',
                    'date': '',
                    'enabled': enabled
                }
                plugins.append(plugin)
            elif os.path.isfile(plugin_path) and entry.endswith(".py"):
                plugin_name = os.path.splitext(entry)[0]
                if plugin_name.endswith(".disabled"):
                    plugin_name = plugin_name[:-9]
                    enabled = False
                else:
                    enabled = True
                plugin = {
                    'name': plugin_name,
                    'type': 'file',
                    'url': '',
                    'version': '',
                    'date': '',
                    'enabled': enabled
                }
                plugins.append(plugin)
        return web.json_response(plugins)
    except Exception as e:
        error_message = f"An unexpected error occurred while fetching plugins: {str(e)}"
        print(error_message)
        return web.json_response({'error': error_message}, status=500)

async def install_plugin(request):
    git_url = request.query.get('git_url')
    if not git_url:
        return web.json_response({'error': 'Git URL is required'}, status=400)
    try:
        plugin_name = request.query.get('plugin_name')
        overwrite = request.query.get('overwrite') == 'true'
        print(f"Installing plugin: {plugin_name} from {git_url}")
        
        current_path = os.path.dirname(os.path.abspath(__file__))
        plugins_dir = os.path.abspath(os.path.join(current_path, "..")) 
        
        plugin_path = os.path.join(plugins_dir, plugin_name)

        if os.path.exists(plugin_path):
            if overwrite:
                print(f"Plugin {plugin_name} already exists, overwriting")
                shutil.rmtree(plugin_path)
            else:
                print(f"Plugin {plugin_name} already exists, skipping installation")
                return web.json_response({'error': f'Plugin {plugin_name} already exists'}, status=400)
        
        process = subprocess.Popen(['git', 'clone', git_url, plugin_path], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)

        output = []
        while True:
            line = process.stdout.readline()
            if line:
                print(line.strip())  
                output.append(line)
            else:
                break

        return_code = process.wait()  

        if return_code == 0:
            print(f"Plugin {plugin_name} installed successfully")
            return web.json_response({'message': 'Plugin installed successfully', 'output': ''.join(output)})
        else:
            print(f"Error installing plugin: {plugin_name}")
            return web.json_response({'error': 'Plugin installation failed', 'output': ''.join(output)}, status=500)

    except Exception as e:
        print(f"Error installing plugin: {str(e)}")
        return web.json_response({'error': str(e)}, status=500)

async def select_plugin_version(request):
    plugin_name = request.query.get('plugin_name')
    version = request.query.get('version')
    if not plugin_name or not version:
        return web.json_response({'error': 'Plugin name and version are required'}, status=400)
    try:
        current_path = os.path.dirname(os.path.abspath(__file__))
        plugins_dir = os.path.abspath(os.path.join(current_path, ".."))
        plugin_path = os.path.join(plugins_dir, plugin_name)
        if not os.path.exists(plugin_path):
            error_message = f'Plugin {plugin_name} does not exist'
            print(error_message)
            return web.json_response({'error': error_message}, status=400)
        repo = git.Repo(plugin_path)
        repo.git.checkout(version)
        return web.json_response({'message': f'Plugin version switched to {version}'})
    except Exception as e:
        print(f"Error selecting plugin version: {str(e)}")
        return web.json_response({'error': str(e)}, status=500)

async def update_plugin(request):
    plugin_name = request.query.get('plugin_name')
    overwrite = request.query.get('overwrite') == 'true'
    if not plugin_name:
        return web.json_response({'error': 'Plugin name is required'}, status=400)
    try:
        current_path = os.path.dirname(os.path.abspath(__file__))
        plugins_dir = os.path.abspath(os.path.join(current_path, ".."))
        plugin_path = os.path.join(plugins_dir, plugin_name)
        if not os.path.exists(plugin_path):
            error_message = f'Plugin {plugin_name} does not exist'
            print(error_message)
            return web.json_response({'error': error_message}, status=400)
        repo = git.Repo(plugin_path)
        repo.remotes.origin.pull()

        return web.json_response({'message': 'Plugin updated successfully'})
    except Exception as e:
        error_message = f"Error updating plugin: {plugin_name}. {str(e)}"
        print(error_message)
        return web.json_response({'error': str(e)}, status=500)

async def get_plugin_versions(request):
    plugin_name = request.query.get('plugin_name')
    if not plugin_name:
        return web.json_response({'error': 'Plugin name is required'}, status=400)
    try:
        current_path = os.path.dirname(os.path.abspath(__file__))
        plugins_dir = os.path.abspath(os.path.join(current_path, ".."))
        plugin_path = os.path.join(plugins_dir, plugin_name)
        if not os.path.exists(plugin_path):
            return web.json_response({'error': f'Plugin {plugin_name} does not exist'}, status=400)
        repo = git.Repo(plugin_path)
        commits = list(repo.iter_commits())
        versions = [{"id": c.hexsha[:7], "message": c.message.strip(), "date": c.committed_datetime.strftime("%Y-%m-%d %H:%M:%S")} for c in commits]
        return web.json_response(versions)
    except Exception as e:
        return web.json_response({'error': str(e)}, status=500)

async def view_plugin_requirements(request):
    plugin_name = request.query.get('plugin_name')
    if not plugin_name:
        return web.json_response({'error': 'Plugin name is required'}, status=400)
    try:
        current_path = os.path.dirname(os.path.abspath(__file__))
        plugins_dir = os.path.abspath(os.path.join(current_path, ".."))
        plugin_path = os.path.join(plugins_dir, plugin_name)
        
        if not os.path.exists(plugin_path):
            return web.json_response({'error': f'Plugin {plugin_name} does not exist'}, status=400)
        requirements_path = os.path.join(plugin_path, 'requirements.txt')
        if not os.path.exists(requirements_path):
            return web.json_response({'message': 'No requirements found'})
        with open(requirements_path, 'r') as file:
            requirements = file.read()
        return web.json_response(requirements)
    except Exception as e:
        return web.json_response({'error': str(e)}, status=500)

async def edit_plugin_requirements(request):
    plugin_name = request.query.get('plugin_name')
    if not plugin_name:
        return web.json_response({'error': 'Plugin name is required'}, status=400)
    try:
        current_path = os.path.dirname(os.path.abspath(__file__))
        plugins_dir = os.path.abspath(os.path.join(current_path, ".."))

        plugin_path = os.path.join(plugins_dir, plugin_name)

        if not os.path.exists(plugin_path):
            return web.json_response({'error': f'Plugin {plugin_name} does not exist'}, status=400)

        requirements_path = os.path.join(plugin_path, 'requirements.txt')

        requirements = await request.text()
        with open(requirements_path, 'w') as file:
            file.write(requirements)
        return web.json_response({'message': 'Requirements updated successfully'})
    except Exception as e:
        return web.json_response({'error': str(e)}, status=500)
   
async def toggle_plugin(request):
    plugin_name = request.query.get('plugin_name')
    enabled = request.query.get('enabled')
    if not plugin_name or enabled is None:
        return web.json_response({'error': 'Plugin name and enabled state are required'}, status=400)
    try:
        current_path = os.path.dirname(os.path.abspath(__file__))
        plugins_dir = os.path.abspath(os.path.join(current_path, ".."))
        plugin_path = os.path.join(plugins_dir, plugin_name)
        disabled_plugin_path = plugin_path + '.disabled'
        if not os.path.exists(plugin_path) and not os.path.exists(disabled_plugin_path):
            return web.json_response({'error': f'Plugin {plugin_name} does not exist'}, status=400)
 
        if enabled == 'true':   
            if os.path.exists(disabled_plugin_path):
                os.rename(disabled_plugin_path, plugin_path)  
        else:    
            if os.path.exists(plugin_path):
                os.rename(plugin_path, disabled_plugin_path)  
        
        return web.json_response({'message': f'Plugin {plugin_name} {"enabled" if enabled == "true" else "disabled"} successfully'})
    except Exception as e:
        return web.json_response({'error': str(e)}, status=500)
    
async def open_plugin_folder(request):
    plugin_name = request.query.get('plugin_name')
    if not plugin_name:
        return web.json_response({'error': 'Plugin name is required'}, status=400)
    try:
        current_path = os.path.dirname(os.path.abspath(__file__))
        plugins_dir = os.path.abspath(os.path.join(current_path, ".."))
        plugin_path = os.path.join(plugins_dir, plugin_name)

        if not os.path.exists(plugin_path):
            return web.json_response({'error': f'Plugin {plugin_name} does not exist'}, status=400)


        if sys.platform == "win32":
            os.startfile(plugin_path)
        else:
            subprocess.Popen(["xdg-open", plugin_path])

        return web.json_response({'message': 'Plugin folder opened successfully'})
    except Exception as e:
        return web.json_response({'error': str(e)}, status=500)

async def open_site_packages_folder(request):
    try:
        site_packages_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "python_miniconda", "Lib", "site-packages"))

        if sys.platform == "win32":
            os.startfile(site_packages_path)
        else:
            subprocess.Popen(["xdg-open", site_packages_path])

        return web.json_response({'message': 'Site-packages folder opened successfully'})
    except Exception as e:
        return web.json_response({'error': str(e)}, status=500)
    
async def check_dependency_conflicts(request):
    try:
        python_path = await get_python_path()
        print("Checking dependency conflicts...")
        process = subprocess.Popen([python_path, '-m', 'pip', 'check'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        if stdout:
            return web.Response(text=stdout.decode('utf-8', errors='replace'))
        elif stderr:
            print("Error checking dependency conflicts:")
            return web.Response(text=stderr.decode('utf-8', errors='replace'))
        else:
            print("No conflicts found.")
            return web.Response(text="No conflicts found.")
    except Exception as e:
        return web.json_response({'error': str(e)}, status=500)
    

async def get_dependency_versions(request):
    name = request.query.get('name')
    if not name:
        return web.json_response({'error': 'Name is required'}, status=400)
    
    if name in version_cache:
        return web.json_response(version_cache[name])
    
    try:
        python_path = await get_python_path()
        process = subprocess.Popen([python_path, '-m', 'pip', 'install', f'{name}=='], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        _, stderr = process.communicate()
        output_text = stderr.decode('utf-8').strip()
        
        if '(from versions:' in output_text:
            start_index = output_text.index('(from versions:') + len('(from versions:')
            end_index = output_text.index(')', start_index)
            versions_text = output_text[start_index:end_index].strip()
            versions = [v.strip() for v in versions_text.split(',')]
            formatted_versions = [f"{name}=={version}" for version in versions]
            
            version_cache[name] = formatted_versions
            
            return web.json_response(formatted_versions)
        else:
            return web.json_response([])
    except Exception as e:
        return web.json_response([])

version_cache = {}

async def install_dependency_version(request):
    name = request.query.get('name')
    version = request.query.get('version')
    if not name or not version:
        return web.json_response({'error': 'Name and version are required'}, status=400)
    try:
        python_path = await get_python_path()
        process = subprocess.Popen([python_path, '-m', 'pip', 'install', f'{name}=={version}'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)

        output = []
        while True:
            line = process.stdout.readline()
            if line:
                print(line.strip())  
                output.append(line)
            else:
                break
                
        if process.returncode == 0:
            filtered_output = [line for line in output if not line.startswith('WARNING')]
            response_data = {'message': 'Installation successful', 'output': ''.join(filtered_output), 'error': ''}
            print(f"Message: {response_data}")
            return web.json_response(response_data)
        else:
            response_data = {'message': 'Installation failed', 'output': ''.join(output), 'error': ''}
            print(f"Message: {response_data}")
            return web.json_response(response_data, status=500)
    except subprocess.CalledProcessError as e:
        error_message = f"Error installing dependency: {name}=={version}. {str(e)}"
        print(error_message)
        print(e.output.decode('utf-8')) 
        return web.json_response({'error': error_message}, status=500)
    
    
async def install_plugin_requirements(request):
    plugin_name = request.query.get('plugin_name')
    if not plugin_name:
        return web.json_response({'error': 'Plugin name is required'}, status=400)
    try:
        current_path = os.path.dirname(os.path.abspath(__file__))
        plugins_dir = os.path.abspath(os.path.join(current_path, ".."))
        plugin_path = os.path.join(plugins_dir, plugin_name)
        requirements_path = os.path.join(plugin_path, 'requirements.txt')
        
        if not os.path.exists(plugin_path):
            error_message = f'Plugin {plugin_name} does not exist'
            print(error_message)
            return web.json_response({'error': error_message}, status=400)
        
        if not os.path.exists(requirements_path):
            print(f'No requirements found for plugin {plugin_name}')
            return web.json_response({'message': 'No requirements found'})

        python_path = await get_python_path()
        
        print(f"Installing requirements for plugin: {plugin_name}")
        
        response = web.StreamResponse()
        response.content_type = 'text/plain'
        await response.prepare(request)

        process = await asyncio.create_subprocess_exec(
            python_path, '-m', 'pip', 'install', '-r', requirements_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT
        )

        while True:
            line = await process.stdout.readline()
            if not line:
                break
            await response.write(line)

        await process.wait()

        if process.returncode == 0:
            print(f"Successfully installed requirements for plugin: {plugin_name}")
            await response.write(b'Requirements installed successfully')
        else:
            print(f"Failed to install requirements for plugin: {plugin_name}")
            await response.write(b'Requirements installation failed')

        return response

    except Exception as e:
        error_message = f"Error installing requirements for plugin: {plugin_name}. {str(e)}"
        print(error_message)
        return web.json_response({'error': error_message}, status=500)
    
async def checkout_plugin_branch(request):
    plugin_name = request.query.get('plugin_name')
    branch = request.query.get('branch')
    if not plugin_name or not branch:
        return web.json_response({'error': 'Plugin name and branch are required'}, status=400)
    try:
        current_path = os.path.dirname(os.path.abspath(__file__))
        plugins_dir = os.path.abspath(os.path.join(current_path, ".."))
        plugin_path = os.path.join(plugins_dir, plugin_name)
        if not os.path.exists(plugin_path):
            error_message = f'Plugin {plugin_name} does not exist'
            print(error_message)
            return web.json_response({'error': error_message}, status=400)
        repo = git.Repo(plugin_path)
        repo.git.checkout(branch)
        return web.json_response({'message': f'Switched to {branch} branch for plugin: {plugin_name}'})
    except Exception as e:
        error_message = f"Error switching to {branch} branch for plugin: {plugin_name}. {str(e)}"
        print(error_message)
        return web.json_response({'error': error_message}, status=500)    
    
async def get_plugin_default_branch(request):
    plugin_name = request.query.get('plugin_name')
    if not plugin_name:
        return web.json_response({'error': 'Plugin name is required'}, status=400)
    try:
        current_path = os.path.dirname(os.path.abspath(__file__))
        plugins_dir = os.path.abspath(os.path.join(current_path, ".."))
        plugin_path = os.path.join(plugins_dir, plugin_name)
        if not os.path.exists(plugin_path):
            error_message = f'Plugin {plugin_name} does not exist'
            print(error_message)
            return web.json_response({'error': error_message}, status=400)
        repo = git.Repo(plugin_path)

        if repo.head.is_detached:
            # If HEAD is detached, try to return 'master' branch
            try:
                default_branch = 'master'
                repo.git.checkout(default_branch)
            except git.GitCommandError:
                # If 'master' branch doesn't exist, return the first available branch
                default_branch = repo.refs[0].name
        else:
            default_branch = repo.head.ref.name

        return web.Response(text=default_branch)
    except Exception as e:
        error_message = f"Error getting default branch for plugin: {plugin_name}. {str(e)}"
        print(error_message)
        return web.json_response({'error': error_message}, status=500)
    
def get_package_version(module_path):
    try:
        parts = module_path.split('.')
        package_name = parts[0]

        url = f"https://pypi.org/pypi/{package_name}/json"
        response = requests.get(url)
        data = response.json()

        versions = list(data["releases"].keys())
        latest_version = versions[-1]

        return f"{package_name}=={latest_version}"
    except requests.exceptions.RequestException:
        return f"Error: Failed to fetch package metadata for {package_name}"
    except KeyError:
        return f"Error: Package {package_name} not found on PyPI"
    except Exception as e:
        return f"Error: {str(e)}"

async def get_module_version(request):
    module_path = request.query.get('module_path')
    if not module_path:
        return web.json_response({'error': 'Module path is required'}, status=400)
    try:
        result = get_package_version(module_path)
        return web.Response(text=result)
    except Exception as e:
        return web.json_response({'error': str(e)}, status=500)    

  
async def get_mjstyle_json(request):
    name = request.match_info['name']
    base_path = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_path, 'docs', 'mjstyle', f'{name}.json')
        
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            json_data = f.read()
        return web.Response(body=json_data, content_type='application/json')
    else:
        return web.Response(status=404)
    
async def get_marked_js(request):
    base_path = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_path, 'web', 'lib', 'marked.min.js')    
    
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            js_data = f.read()
        return web.Response(body=js_data, content_type='application/javascript')
    else:
        return web.Response(status=404)

async def get_purify_js(request):
    base_path = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_path, 'web', 'lib', 'purify.min.js')     
    
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            js_data = f.read()
        return web.Response(body=js_data, content_type='application/javascript')
    else:
        return web.Response(status=404)
          

app = web.Application()

# Configure CORS
cors = setup(app, defaults={
    "*": ResourceOptions(
        allow_credentials=True,
        expose_headers="*",
        allow_headers="*",
        allow_methods="*",
    )
})


# Define routes
routes = [
    ("/view_plugin_requirements", view_plugin_requirements),
    ("/lib/marked.min.js", get_marked_js), 
    ("/lib/purify.min.js", get_purify_js),
    ("/mjstyle/{name}.json", get_mjstyle_json),
    ("/get_dependencies", get_dependencies),
    ("/install_dependency", install_dependency),
    ("/manage_dependency", manage_dependency),
    ("/replace_dependency", replace_dependency),
    ("/get_comfyui_versions", get_comfyui_versions),
    ("/select_comfyui_version", select_comfyui_version),
    ("/get_current_comfyui_version", get_current_comfyui_version),
    ("/get_plugins", get_plugins),
    ("/install_plugin", install_plugin),
    ("/select_plugin_version", select_plugin_version),
    ("/update_plugin", update_plugin),
    ("/get_plugin_versions", get_plugin_versions),
    ("/view_plugin_requirements", view_plugin_requirements),
    ("/edit_plugin_requirements", edit_plugin_requirements),
    ("/open_plugin_folder", open_plugin_folder),
    ("/open_site_packages_folder", open_site_packages_folder),
    ("/toggle_plugin", toggle_plugin),
    ("/check_dependency_conflicts", check_dependency_conflicts),
    ("/get_dependency_versions", get_dependency_versions),
    ("/install_dependency_version", install_dependency_version),
    ("/install_plugin_requirements", install_plugin_requirements),
    ("/get_plugin_default_branch", get_plugin_default_branch),
    ("/checkout_plugin_branch", checkout_plugin_branch),
    ("/get_module_version", get_module_version),
]

# Add routes to the application with CORS
for route in routes:
    resource = cors.add(app.router.add_resource(route[0]))
    cors.add(resource.add_route("GET", route[1]))
    cors.add(resource.add_route("POST", route[1]))

def start_api_server():
    web.run_app(app, port=8080, access_log=None)

if __name__ == '__main__':
    print("\033[32m ** Comfly Loaded :\033[33m fly, just fly ")
    web.run_app(app, port=8080, access_log=None, print=None)



WEB_DIRECTORY = "./web"

NODE_CLASS_MAPPINGS = {

}

NODE_DISPLAY_NAME_MAPPINGS = {

}
     

