import os
from openai import AzureOpenAI
import time
import base64

from dotenv import load_dotenv
load_dotenv()  

AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")

deployment = "gpt-5-chat"
api_version = "2025-04-01-preview"

client = AzureOpenAI(
    api_version=api_version,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
)

def call_GPT5(payload):
    try:
        response = client.chat.completions.create(**payload)
    except Exception as e:
        print(f"Error during GPT5 query: {e}")
        raise e
    return response.choices[0].message.content


def response_GPT5(user_prompt, system_prompt, image_path, temperature=0.0, top_p=0.95):
    max_retries = 3
    retries = 0
    delay = 5
    encoded_image = base64.b64encode(open(image_path, "rb").read()).decode("ascii")
    payload = {
        "messages": [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": system_prompt
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"},
                    },
                    {   "type": "text",
                        "text": user_prompt
                    },
                    
                ],
            },
        ],
        "temperature": 0.5 * retries,
        "top_p": top_p,
        "max_tokens": 800,
        "model": deployment
    }
    while retries <= max_retries:
        try:
            response = call_GPT5(payload)
            return response
        except Exception  as e:
                    last_error = e  # Store the caught error
                    if retries < max_retries:
                        # print(f"Error calling API (attempt {retries + 1}): {e}")
                        # print(f"Retrying in {delay} seconds...")
                        time.sleep(delay)
                        retries += 1
                    else:
                        print(f"Error with image at {image_path}: {e}")
                        raise last_error              
