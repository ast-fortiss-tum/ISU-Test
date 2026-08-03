from google import genai
from google.genai import types
from dotenv import load_dotenv
import os
import time

load_dotenv()  

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

client = genai.Client(api_key=GEMINI_API_KEY)

def call_Gemini(user_prompt, image_bytes):
    try:
        response = client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=[
                types.Part.from_bytes(
                    data=image_bytes,
                    mime_type='image/jpeg',
                ),
                user_prompt
            ]
        )
    except Exception as e:
        print(f"Error during Gemini query: {e}")
        raise e
    return response.text

def response_Gemini(user_prompt, image_path):
    with open(image_path, "rb") as image_file:
        image_bytes = image_file.read()
    max_retries = 5
    retries = 0
    delay = 5
    while retries <= max_retries:
        try:
            response = call_Gemini(user_prompt, image_bytes)
            return response
        except Exception  as e:
                    last_error = e  # Store the caught error
                    if retries < max_retries:
                        print(f"Error calling API (attempt {retries + 1}): {e}")
                        print(f"Retrying in {delay} seconds...")
                        time.sleep(delay)
                        retries += 1
                    else:
                        raise last_error    