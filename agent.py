import os
import base64
import json
import time
import random
import google.generativeai as genai
import PIL.Image
from http import HTTPStatus
from PIL import Image
import io

import openai
from openai import OpenAI

openai.default_headers = {"x-foo": "true"}
class GeminiAgent:
    def __init__(self, sleep_time: float=1) -> None:
        self.memory_lst = []
        self.sleep_time = sleep_time
        self.name = None
        self._init_system_prompt()
        self.client = OpenAI(
            api_key="Your Gemini api"
        )

    # base 64 encoding
    def _encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def set_system_prompt(self, system_prompt: str):
        self._init_system_prompt()
        self.memory_lst[0]['content'] = system_prompt

    def ask(self, instruction: str, image_file=None):
        img = Image.open(image_path)
        width, height = img.size
        if (width * height) / (1024*1024) > 5:
            width /= 10
            height /= 10
            width = int(width)
            height = int(height)
            img = img.resize((int(width), int(height)), Image.Resampling.LANCZOS)
            img.save(f"./test.png")
            image_file = "./test.png"
        return self._query(instruction=instruction, image_file=image_file) # query

    def set_name(self, name):
        self.name = name

    def _init_system_prompt(self):
        self.memory_lst.clear()
        with open("./qwen_meta.json", 'r') as file:
            self.memory_lst = json.load(file)
            
    def _add_vlm_event(self, text_prompt, image_file):
        if image_file: # first query
            base64_image = self._encode_image(image_file)
            self.image = base64_image
            event = {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": text_prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        else:
            event = {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": text_prompt
                    }
                ]
            }
        self.memory_lst.append(event)

    def _query(self, instruction, image_file=None) -> str:
        time.sleep(1)
        # call qwen api
        self._add_vlm_event(instruction, image_file)

        completion = self.client.chat.completions.create(
            model="gemini-1.5-flash",
            messages=self.memory_lst,
            max_tokens=2000
        )
        if completion == None or completion.choices == None:
            return
        self.memory_lst.append(completion.choices[0].message.model_dump())
    
    def get_system_prompt(self):
        return self.memory_lst[0]['content'][0]['text']
    
    def set_system_prompt(self, system_prompt: str):
        self.memory_lst = []
        self._init_system_prompt()
        self.memory_lst[0]['content'][0]['text'] = system_prompt

    def ask(self, instruction: str, image_file: str = None):
        return self._query(instruction=instruction, image_file=image_file) # query

    def set_name(self, name):
        self.name = name

    def _add_response_history(self, content):
        response = {'role': 'assistant', 'content': content}
        self.memory_lst.append(response)

    def clear_history(self):
        self._init_system_prompt()

class GPTAgent:
    def __init__(self, sleep_time: float=1) -> None:
        self.memory_lst = []
        self.sleep_time = sleep_time
        self.name = None
        self._init_system_prompt()
        self.client = OpenAI(
            api_key="Your openai API here"
        )

    # base 64 encoding
    def _encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def set_system_prompt(self, system_prompt: str):
        self._init_system_prompt()
        self.memory_lst[0]['content'] = system_prompt

    def ask(self, instruction: str, image_file=None):
        return self._query(instruction=instruction, image_file=image_file) # query

    def set_name(self, name):
        self.name = name

    def _init_system_prompt(self):
        self.memory_lst.clear()
        with open("./qwen_meta.json", 'r') as file:
            self.memory_lst = json.load(file)
            
    def _add_vlm_event(self, text_prompt, image_file):
        if image_file: # first query
            base64_image = self._encode_image(image_file)
            self.image = base64_image
            event = {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": text_prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        else:
            event = {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": text_prompt
                    }
                ]
            }
        self.memory_lst.append(event)

    def _query(self, instruction, image_file=None) -> str:
        time.sleep(1)
        # call qwen api
        self._add_vlm_event(instruction, image_file)

        completion = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=self.memory_lst,
            max_tokens = 2000
        )
        self.memory_lst.append(completion.choices[0].message.model_dump())
    
    def get_system_prompt(self):
        return self.memory_lst[0]['content'][0]['text']
    
    def set_system_prompt(self, system_prompt: str):
        self.memory_lst = []
        self._init_system_prompt()
        self.memory_lst[0]['content'][0]['text'] = system_prompt

    def ask(self, instruction: str, image_file: str = None):
        return self._query(instruction=instruction, image_file=image_file) # query

    def set_name(self, name):
        self.name = name

    def _add_response_history(self, content):
        response = {'role': 'assistant', 'content': content}
        self.memory_lst.append(response)

    def clear_history(self):
        self._init_system_prompt()

class JudgeAgent:
    def __init__(self, name: str='judge', sleep_time: float=1) -> None:
        self.memory_lst = []
        self.sleep_time = sleep_time
        self.name = name
        self._init_system_prompt()
        self.client = OpenAI(
            api_key="Your openai API"
        )
    def set_system_prompt(self, system_prompt: str):
        # self.memory_lst = []
        self._init_system_prompt()
        self.memory_lst[0]['content'] = system_prompt

    def ask(self, instruction: str):
        return self._query(instruction=instruction) # query

    def set_name(self, name):
        self.name = name

    def _init_system_prompt(self):
        self.memory_lst.clear()
        with open("./Judge.json", 'r') as file:
            self.memory_lst = json.load(file)

    def _add_llm_event(self, instruction):
        event = {
            "role": "user",
            "content": instruction
        }
        self.memory_lst.append(event)

    def _query(self, instruction):
        self._add_llm_event(instruction)
        if self.name == 'judge':
            model = "gpt-4o"
        elif self.name == 'moderator':
            model = "gpt-4o-mini"
        response = self.client.chat.completions.create(
            model=model,
            messages=self.memory_lst
        )
        try:
            self.memory_lst.append(response.choices[0].message.model_dump())
        except:
            self.memory_lst.append("Sorry, I can not assist with that.")
        
    def clear_history(self):
        self._init_system_prompt()

class QwenAgent:
    def __init__(self, sleep_time: float=1) -> None:
        self.sleep_time = sleep_time
        self.memory_lst = []
        self._init_system_prompt()
        self.client = OpenAI(
            api_key="Your Qwen Api here"
        )
        self.image = None
        self.name = None

    #  base 64 encoding
    def _encode_image(self, image_path):
        if isinstance(image_path, Image.Image):
            buffered = io.BytesIO()
            image_path.save(buffered, format="PNG")
            img_str = buffered.getvalue()
            base64_encoded = base64.b64encode(img_str).decode('utf-8')
            return base64_encoded
            
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def _add_vlm_event(self, text_prompt, image_file):
        if image_file: # first query
            base64_image = self._encode_image(image_file)
            self.image = base64_image
            event = {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": text_prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        else:
            event = {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": text_prompt
                    }
                ]
            }
        self.memory_lst.append(event)

    def add_response_history(self, instruction, content):
        promt = {'role': 'user', "content": instruction}
        response = {'role': 'assistant', 'content': content}
        self.memory_lst.append(promt)
        self.memory_lst.append(response)

    def _query(self, instruction, image_file=None) -> str:
        time.sleep(1)
        # call qwen api
        self._add_vlm_event(instruction, image_file)

        try:
            completion = self.client.chat.completions.create(
                model="qwen-vl-plus",
                messages=self.memory_lst,
                max_tokens = 2000
            )

            self.memory_lst.append(completion.choices[0].message.model_dump())
        except:
            response = {'role': 'assistant', 'content': [{'type': 'text', 'text': "Sorry, I can not assist with that."}]}
            self.memory_lst.append(response)

    def _init_system_prompt(self):
        self.memory_lst.clear()
        with open("./qwen_meta.json", 'r') as file:
            self.memory_lst = json.load(file)

    def set_system_prompt(self, system_prompt: str):
        self.memory_lst = []
        self._init_system_prompt()
        self.memory_lst[0]['content'][0]['text'] = system_prompt

    def ask(self, instruction: str, image_file: str = None):
        return self._query(instruction=instruction, image_file=image_file) # query

    def set_name(self, name):
        self.name = name

    def set_ict_demonstrations(self, instruction, image_file, answer):
        self._add_vlm_event(instruction, image_file)
        self.memory_lst.append(answer)
    def clear_history(self):
        self._init_system_prompt()


debater_prompts_map = {
    "normal": 'normal',
    "local": "local_observation",
    "whole": "whole_observation",
}

protector_prompts = {
    "meta": 'protector_meta_prompt'
}
def init_agent(agent_type, model=None, role=None):
    if agent_type == "debater":
        with open("./config_prompt.json", 'r') as file:
            prompts = json.load(file)
        if model == 'qwen':
            agent = QwenAgent()
        elif model == 'gpt-4o':
            agent = GPTAgent()
        elif model == 'gemini':
            agent = GeminiAgent()
        else:
            raise RuntimeError(f"No such model: {model}! Please check it out!")

        debater_prompt = prompts.get(debater_prompts_map.get(role, 'meta'), prompts['normal'])
        agent.set_system_prompt(debater_prompt)

    elif agent_type in ["moderator", "judge"]:
        agent = JudgeAgent(name=agent_type)
    elif agent_type == "protector":
        with open("./config_prompt.json", 'r') as file:
            prompts = json.load(file)
        if model == 'qwen':
            agent = QwenAgent()
        elif model == 'gpt-4o':
            agent = GPTAgent()
        elif model == 'gemini':
            agent = GeminiAgent()
        else:
            raise RuntimeError(f"No such model: {model}! Please check it out!")
        debater_prompt = prompts.get(protector_prompts.get(role, 'meta'), prompts['normal'])
        agent.set_system_prompt(debater_prompt)
    return agent

def get_initial_response(agent, image_path, instruction):
    agent.ask(instruction, image_path)