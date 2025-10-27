import torchvision.transforms as T
from openai import OpenAI
import base64
import google.generativeai as genai
import pandas as pd
import os
import json

def get_gpt_model():
    model = OpenAI(api_key="YOUR_OPENAI_API_KEY")
    return model



def prompt_gpt4o(model, prompt, image_path=None, demonstrations=None):
    """
    Generates a response from the GPT-4o model using optional image input,
    and supports adding demonstration examples.

    Args:
        model: The GPT-4o model instance.
        prompt (str): The text prompt for the model.
        image_path (str, optional): The file path of the image. Defaults to None.
        demonstrations (list of tuples, optional):
            A list of (demo_query, demo_image_path, demo_answer).
            Each tuple is used as a demonstration example before the final prompt.

    Returns:
        str: The model's response text.
    """
    try:
        messages = []

        # 1. Add demonstration pairs, if any
        if demonstrations:
            demo_index = 0
            for (demo_query, demo_image_path, demo_answer) in demonstrations:
                # (A) User demonstration message
                user_content = [{"type": "text", "text": f"Example {demo_index} Input: {demo_query}"}]

                if demo_image_path:
                    # Convert demo image to base64 (normalize local path)
                    local_demo_path = os.path.normpath(str(demo_image_path).strip().replace("\\", "/"))
                    with open(local_demo_path, "rb") as image_file:
                        base64_image = f"data:image/png;base64,{base64.b64encode(image_file.read()).decode('utf-8')}"
                    user_content.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": base64_image},
                        }
                    )

                messages.append({"role": "user", "content": user_content})

                # (B) Assistant demonstration message (the "answer")
                assistant_content = [{"type": "text", "text": f"Example {demo_index} Output: {demo_answer}"}]
                messages.append({"role": "assistant", "content": assistant_content})

        # 2. Add the final user prompt
        final_user_content = [{"type": "text", "text": prompt}]

        if image_path:
            # Convert final prompt image to base64 (normalize local path)
            
            
            with open(image_path, "rb") as image_file:
                base64_image = f"data:image/png;base64,{base64.b64encode(image_file.read()).decode('utf-8')}"
            final_user_content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": base64_image},
                }
            )

        messages.append({"role": "user", "content": final_user_content})

        # 3. Generate response from the model
        response = model.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0,
            max_tokens=4096,
        )
        print(response)
        return response.choices[0].message.content

    except Exception as e:
        raise RuntimeError(f"An error occurred while querying the GPT-4o model: {e}")



def get_gemini_model():
    genai.configure(api_key="YOUR_GEMINI_API_KEY")
    model = genai.GenerativeModel('gemini-1.5-flash')
    return model

def prompt_gemini(client, prompt, image_path=None, demonstrations=None):
    """
    Prompt Gemini with optional in-context learning using demonstrations.

    Args:
        client: The initialized Gemini model client
        prompt (str): The text prompt to send to Gemini
        image_path (str, optional): Path to an image file to include in the prompt
        demonstrations (list, optional): List of tuples (prompt, image_path, answer) for in-context learning

    Returns:
        str: Gemini's response
    """
    import PIL.Image

    # Initialize the content list that will be sent to Gemini
    content_parts = []

    # Add demonstrations for in-context learning if provided
    if demonstrations and len(demonstrations) > 0:
        # Format the demonstrations
        demo_index = 0
        for demo_prompt, demo_image_path, demo_answer in demonstrations:
            # Add a separator for each demonstration
            content_parts.append(f"Example {demo_index} Input:")

            # Add demonstration image if provided
            if demo_image_path:
                try:
                    demo_image = PIL.Image.open(demo_image_path)
                    content_parts.append(demo_image)
                except Exception as e:
                    content_parts.append(f"[Image could not be loaded: {str(e)}]")

            # Add demonstration prompt
            content_parts.append(demo_prompt)

            # Add demonstration answer
            content_parts.append("Example Output:")
            content_parts.append(demo_answer)

            # Add a separator between demonstrations
            content_parts.append("---")
            demo_index += 1

        # Add a final separator before the actual query
        content_parts.append("Now, please analyze the following:")

    # Add the current image if provided
    if image_path:
        try:
            image = PIL.Image.open(image_path)
            content_parts.append(image)
        except Exception as e:
            return f"Error loading image: {str(e)}"

    # Add the current prompt
    content_parts.append(prompt)

    # Generate the response
    try:
        response = client.generate_content(content_parts, generation_config={"temperature": 0.0, "top_p": 1, "top_k": 1, "max_output_tokens": 4096})
        return response.text
    except Exception as e:
        return f"Error generating response: {str(e)}"


def json_to_xlsx(json_list, xlsx_file):
    df = pd.DataFrame(json_list)
    df.to_excel(xlsx_file, index=False, engine='openpyxl')
    print(f"Successfully saved to {xlsx_file}")


def load_json_file(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def init_cols(df_input, new_cols):
    for col in new_cols:
        if col not in df_input.columns:
            df_input[col] = None
    return df_input


def get_commercial_model(model_name):
    model_map = {
        "gpt4o": get_gpt_model,
        "gemini": get_gemini_model,
    }
    if model_name in model_map:
        return model_map[model_name]()
    raise ValueError(f"Unknown model name: {model_name}")


def prompt_commercial_model(client, model_name, prompt, image_id, demonstrations=None):
    prompt_map = {
        "gpt4o": prompt_gpt4o,
        "gemini": prompt_gemini,
    }
    if model_name in prompt_map:
        try:
            res = prompt_map[model_name](client, prompt, image_id, demonstrations)
            print(res)
            return res
        except Exception as e:
            print(image_id, str(e))
            return ""
    raise ValueError(f"Unknown model name: {model_name}")


if __name__ == "__main__":
    gpt_client = get_gpt_model()
    gemini_client = get_gemini_model()
    image_path = "assets/M4FC.png"
    text_prompt = "describe the image"

    # Process and get response
    response = prompt_gpt4o(gpt_client, text_prompt, image_path,)
    print("\ngpt-4o Response:")
    print(response)
