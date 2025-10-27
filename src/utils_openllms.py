import torch
import os
from PIL import Image
from typing import Dict, Any, List, Tuple, Optional


def resolve_asset_path(rel_path: str) -> str:
    rel_path = (str(rel_path) or "").strip().replace("\\", "/")
    path = os.path.normpath(os.path.join("assets", *rel_path.split("/")))
    return path.replace("\\", "/")


def get_open_model(model_name: str) -> Dict[str, Any]:
    """
    Load and initialize a vision-language model.
    
    Args:
        model_name: One of 'llama', 'internvl', or 'qwenvl'
    
    Returns:
        Dictionary containing model components (model, processor, etc.)
    """
    model_name = model_name.lower()
    
    if model_name == 'llama':
        return _load_llama_model()
    elif model_name == 'internvl':
        return _load_internvl_model()
    elif model_name == 'qwenvl':
        return _load_qwenvl_model()
    else:
        raise ValueError(f"Unsupported model: {model_name}. Choose from 'llama', 'internvl', or 'qwenvl'")

def prompt_open_model(
    model_components: Dict[str, Any], 
    query: str, 
    image_path: str,
    demonstrations: Optional[List[Tuple[str, str, str]]] = None
) -> str:
    """
    Prompt a pre-loaded vision-language model with a query and image, optionally using demonstrations.
    
    Args:
        model_components: Dictionary with model components returned by get_open_model()
        query: Text query to ask about the image
        image_path: Path to local image file
        demonstrations: Optional list of (query, image_path, response) tuples for in-context learning
    
    Returns:
        Response text from the model
    """
    # Load the local image
    image = Image.open(image_path)
    
    model_type = model_components.get("model_type", "")
    
    if model_type == "llama":
        return _prompt_llama_model(model_components, query, image, image_path, demonstrations)
    elif model_type == "internvl":
        return _prompt_internvl_model(model_components, query, image_path, demonstrations)
    elif model_type == "qwenvl":
        return _prompt_qwenvl_model(model_components, query, image_path, demonstrations)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

# Helper functions for loading models
def _load_llama_model() -> Dict[str, Any]:
    """Load and return Llama model components"""
    from transformers import MllamaForConditionalGeneration, AutoProcessor
    from transformers import BitsAndBytesConfig
    
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True
    )

    model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"
    model = MllamaForConditionalGeneration.from_pretrained(
        model_id,
        quantization_config=quantization_config,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(model_id)
    
    return {
        "model_type": "llama",
        "model": model,
        "processor": processor
    }

def _load_qwenvl_model() -> Dict[str, Any]:
    """Load and return QwenVL model components"""
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )
    
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
    
    return {
        "model_type": "qwenvl",
        "model": model,
        "processor": processor
    }

def _load_internvl_model() -> Dict[str, Any]:
    """Load and return InternVL model components"""
    from lmdeploy import pipeline, TurbomindEngineConfig
    
    model_name = 'OpenGVLab/InternVL2_5-8B'
    engine_config = TurbomindEngineConfig(
        session_len=8192,  # Adjust session length as needed
        quant_policy=4,    # Enable 4-bit quantization
    )
    pipe = pipeline(model_name, backend_config=engine_config, gen_config=dict(
        temperature=0,
        max_new_tokens=4096
    ))
    
    return {
        "model_type": "internvl",
        "pipe": pipe
    }
    

# Helper functions for prompting loaded models with demonstrations
def _prompt_llama_model(
    components: Dict[str, Any], 
    query: str, 
    image: Image.Image, 
    image_path: str,
    demonstrations: Optional[List[Tuple[str, str, str]]] = None
) -> str:
    """
    Use loaded Llama model to answer a query about an image, with optional demonstrations
    """
    model = components["model"]
    processor = components["processor"]
    
    if not demonstrations:
        # Simple case: no demonstrations
        messages = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": query}
            ]}
        ]
        input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
        
        inputs = processor(
            image,
            input_text,
            add_special_tokens=False,
            return_tensors="pt"
        ).to(model.device)
    else:
        # With demonstrations: we need to include multiple images in the message
        all_content = []
        
        # Add demonstration examples
        demo_index = 0
        for demo_query, demo_img_path, demo_response in demonstrations:
            demo_img = Image.open(demo_img_path)
            
            # Add the demonstration image and query
            all_content.append({"type": "text", "text": f"Example {demo_index + 1}:"})
            all_content.append({"type": "image"})
            all_content.append({"type": "text", "text": f"Example Input: {demo_query}"})
            all_content.append({"type": "text", "text": f"Example Output: {demo_response}"})
            all_content.append({"type": "text", "text": "\n"})
            
            demo_index += 1
        
        # Add the actual query with its image
        all_content.append({"type": "text", "text": "Now, please answer the following question:"})
        all_content.append({"type": "image"})
        all_content.append({"type": "text", "text": f"Input: {query}"})
        all_content.append({"type": "text", "text": "Output:"})
        
        messages = [{"role": "user", "content": all_content}]
        input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
        
        # Process all images
        all_images = [Image.open(demo[1]) for demo in demonstrations] + [image]
        
        # Need to use a different approach for multiple images
        # This is a simplified approximation - might need adjustments based on model's exact API
        inputs = processor(
            all_images,
            input_text,
            add_special_tokens=False,
            return_tensors="pt"
        ).to(model.device)
    
    output = model.generate(
        **inputs,
        max_new_tokens=4096,
        temperature=0.00001,
        return_dict_in_generate=True,
        output_scores=False
    )

    # Slice only the new tokens
    generated_tokens = output.sequences[0][inputs["input_ids"].shape[1]:]
    result = processor.decode(generated_tokens, skip_special_tokens=True)
    
    return result

def _prompt_qwenvl_model(
    components: Dict[str, Any], 
    query: str, 
    image_path: str,
    demonstrations: Optional[List[Tuple[str, str, str]]] = None
) -> str:
    """
    Use loaded QwenVL model to answer a query about an image, with optional demonstrations
    """
    from qwen_vl_utils import process_vision_info
    
    model = components["model"]
    processor = components["processor"]
    
    if not demonstrations:
        # Simple case: no demonstrations
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},  
                    {"type": "text", "text": f"{query}"},
                ],
            }
        ]
    else:
        # With demonstrations
        all_content = []
        
        # Add demonstration examples
        for i, (demo_query, demo_img_path, demo_response) in enumerate(demonstrations):
            # Add the demonstration image and query
            all_content.append({"type": "image", "image": demo_img_path})
            all_content.append({"type": "text", "text": f"Example {i+1} Input: {demo_query}"})
            all_content.append({"type": "text", "text": f"Example {i+1} Output: {demo_response}"})
            all_content.append({"type": "text", "text": "\n"})
        
        # Add the actual query with its image
        all_content.append({"type": "text", "text": "Now, please answer the following question:"})
        all_content.append({"type": "image", "image": image_path})
        all_content.append({"type": "text", "text": f"Main Input: {query}"})
        
        messages = [{"role": "user", "content": all_content}]
    
    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")
    
    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=4096, temperature=0.00001)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    
    return output_text

def _prompt_internvl_model(
    components: Dict[str, Any], 
    query: str, 
    image_path: str,
    demonstrations: Optional[List[Tuple[str, str, str]]] = None
) -> str:
    """
    Use loaded InternVL model to answer a query about an image, with optional demonstrations
    """
    from lmdeploy.vl import load_image
    from lmdeploy.vl.constants import IMAGE_TOKEN
    
    pipe = components["pipe"]
    
    # Prepare the full prompt with demonstrations and main query
    full_prompt = ""
    
    # Add demonstrations if provided
    if demonstrations:
        for i, (demo_query, demo_image_path, demo_response) in enumerate(demonstrations, 1):
            demo_image = load_image(demo_image_path)
            full_prompt += f"Example-{i} Input: {demo_query}\n"
            full_prompt += f"Example-{i} Image: {IMAGE_TOKEN}\n"
            full_prompt += f"Example-{i} Output: {demo_response}\n\n"
    
    # Add the main query and image
    full_prompt += f"Main Input: {query}\n"
    full_prompt += f"Main Image: {IMAGE_TOKEN}\n"
    full_prompt += "Response:"
    
    # Load the main image
    main_image = load_image(image_path)
    
    # Combine images from demonstrations and main image
    all_images = []
    if demonstrations:
        all_images.extend([load_image(demo[1]) for demo in demonstrations])
    all_images.append(main_image)
    
    # Generate response
    response = pipe((full_prompt, all_images))
    
    return response.text

    
if __name__ == "__main__":
    for model_name in ["qwenvl", "internvl", "llama"]: #"qwenvl", "internvl", "llama"
        model = get_open_model(model_name)
        image_path = "assets/M4FC.png"
        text_prompt = "describe the image"
        resp = prompt_open_model(model, text_prompt, image_path)
        print(resp)