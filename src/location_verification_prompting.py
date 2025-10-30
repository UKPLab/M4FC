import time
from openai import AzureOpenAI
import os
import base64
import google.generativeai as genai
import mimetypes
from utils import *


def verify_location_closed_source(model, image_path, candidate_location='', evidence_images=[], max_tokens=100, sleep=3):
    #Prompt for the task of location verification
    prompt= "You are a helpful assistant designed to support fact-checking. Your task is to verify whether a candidate location is accurate for an image.\n"
    if len(evidence_images)==0:
        prompt+= "You are given the image and the candidate location as input.\n"
    else:
        if 'map' in [ev[0] for ev in evidence_images]:
            if 'satellite' in [ev[0] for ev in evidence_images]:
                prompt+= "You are given the image, and a map and a satellite view of the candidate location as input.\n"
            else:
                prompt+= "You are given the image and a map of the candidate location as input.\n"

        else:
            prompt+= "You are given the image and a satellite view of the candidate location as input.\n"
    prompt+= "Answer only with true or false.\n"
    if candidate_location!='':
        prompt += f'Candidate location: {candidate_location}'
    if model=='gpt4':
        output, usage_in, usage_ou = gpt4_prompting(prompt,max_tokens=max_tokens, image_path=image_path, evidence_images=evidence_images)
    else:
        output, usage_in, usage_ou = gemini_prompting(prompt,max_tokens=max_tokens, image_path=image_path, evidence_images=evidence_images)
    time.sleep(sleep)
    return output, usage_in, usage_ou



def encode_image_gpt4(image_path):
  with open(image_path, "rb") as image_file:
    return f"data:image/png;base64,{base64.b64encode(image_file.read()).decode('utf-8')}"

def gpt4_prompting(prompt, max_tokens=100, image_path=None, evidence_images=[]):
    client = AzureOpenAI(
    api_key=os.getenv("YOUR_AZURE_OPENAI_API_KEY"),  
    api_version="2023-10-01-preview",
    azure_endpoint = os.getenv("YOUR_AZURE_OPENAI_ENDPOINT")
    )
    #Generic function for gpt4o mini
    deployment_name='gpt-4o-mini' 
    content = [{"type": "text", "text": prompt}]
    if image_path:
        image64 = encode_image_gpt4(image_path)
        content += [{"type":"image_url","image_url":{"url":image64}}] 
    if len(evidence_images)!=0:
        #each evidence image is a tuple with the accompanying text and the image
        for ev in evidence_images:
            if ev[0]=='map':
                content += [{"type": "text", "text": 'A map of the candidate location.'}]
            elif ev[0]=='satellite':
                content += [{"type": "text", "text": 'A satellite image of the candidate location.'}]
            else:
                content += [{"type": "text", "text": ev[0]}]
            if ev[0] in ['map', 'satellite']:
                image64 = encode_image_gpt4(ev[1])
                content += [{"type":"image_url","image_url":{"url":image64}}]

    messages=[
        {
            "role": "user",
            'content':content
        }]
    completion = client.chat.completions.create(model=deployment_name,
                                                messages=messages, 
                                                max_tokens=max_tokens,
                                                temperature=0
                                                )
    output = completion.choices[0].message.content
    usage_input = completion.usage.prompt_tokens
    usage_output = completion.usage.completion_tokens


    return output, usage_input, usage_output


def gemini_prompting(prompt, max_tokens=100, image_path=None, evidence_images = [], demonstrations=None):
    genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
    gemini_client = genai.GenerativeModel(model_name='gemini-1.5-flash')
    try:
        # Prepare the inputs as a list of parts
        parts = []
        # Add the actual user prompt and image
        if image_path:
            image_data = load_image(image_path)
            media_type, _ = mimetypes.guess_type(image_path)
            parts.append({
                "inline_data": {
                    "mime_type": media_type,  
                    "data": image_data
                }
            })

        # Add the user prompt
        parts.append({"text": prompt})

        if len(evidence_images)!=0:
            #each evidence image is a tuple with the accompanying text and the image
            for ev in evidence_images:
                if ev[0]=='map':
                    parts += [{"text": 'A map of the candidate location.'}]
                elif ev[0]=='satellite':
                    parts += [{"text": 'A satellite image of the candidate location.'}]
                else:
                    parts += [{"text": ev[0]}]
                if ev[0] in ['map', 'satellite']:
                    image_data = load_image(ev[1])
                    media_type, _ = mimetypes.guess_type(ev[1])
                    parts.append({
                        "inline_data": {
                            "mime_type": media_type,  
                            "data": image_data
                        }
                    })

        # Generate content using the provided parts
        response = gemini_client.generate_content({"parts": parts}, generation_config={
                                                    "temperature": 0.0,
                                                    "top_p": 1,
                                                    "top_k": 1,
                                                    "max_output_tokens": max_tokens
                                                }
                                                )
        output = response.text
        usage_input = response.usage_metadata.prompt_token_count
        usage_output = response.usage_metadata.candidates_token_count
        return output, usage_input, usage_output

    except Exception as e:
        raise RuntimeError(f"An error occurred while querying the Gemini model: {e}")