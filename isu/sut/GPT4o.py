import os
import requests
import base64
import time
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

GPT4o_KEY = os.getenv("GPT4o_KEY")
GPT4o_ENDPOINT = os.getenv("GPT4o_ENDPOINT")

headers = {
    "Content-Type": "application/json",
    "api-key": GPT4o_KEY
}

def call_oracle(payload, time_limit=10):
    try:
        response = requests.post(
                                GPT4o_ENDPOINT,
                                headers=headers,
                                json=payload,
                                timeout=(50, 100)  # (connect timeout, read timeout)
                            )
        response.raise_for_status()
    except requests.Timeout as e:
        print(f"Request timed out after {time_limit} seconds")
        raise e   
    except requests.RequestException as e:
        print(f"Failed to make the request. Error: {e}")
        raise e

    return response.json()["choices"][0]["message"]["content"]


def response_GPT4o( user_prompt, system_prompt, image_path, temperature=0.1, top_p=0.95):
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
                    {   "type": "text",
                        "text": user_prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{encoded_image}"},
                    }
                ],
            },
        ],
        "temperature": 0.5 * retries,
        "top_p": top_p,
        "max_tokens": 800,
    }
    while retries <= max_retries:
        try:
            response = call_oracle(payload)
            return response
        except Exception  as e:
                    last_error = e  # Store the caught error
                    if retries < max_retries:
                        time.sleep(delay)
                        retries += 1
                    else:
                        raise last_error              