from io import BytesIO
import matplotlib.pyplot as plt
import json
from PIL import Image
from google.oauth2 import service_account
import google.genai as genai
from google.genai.types import GenerateContentConfig, Modality# Path to the service account JSON file
import time

service_account_path = "gemini-key.json.json"

# Load the JSON file
try:
    with open(service_account_path, 'r') as file:
        service_account_data = json.load(file)
        print("Service account data loaded successfully.")
        # print(service_account_data)  # Print the contents for verification
except FileNotFoundError:
    print(f"File not found: {service_account_path}")
except json.JSONDecodeError:
    print("Error decoding JSON file.")

creds = service_account.Credentials.from_service_account_info(service_account_data, scopes=["https://www.googleapis.com/auth/cloud-platform"],)

client = genai.Client(
    vertexai=True,
    credentials=creds,
    project=service_account_data["project_id"],
    location="global", # "us-central1" # "europe-west4" # "global"
) 

def call_gemini(payload, version):
    if version == "2.0":
        model = "gemini-2.0-flash"
    elif version == "2.5":
        model = "gemini-2.5-flash"
    else:
        raise ValueError("Unsupported Gemini version specified.")
    try:
        response = client.models.generate_content(
            model=model, 
            contents=payload,
            config=GenerateContentConfig(
                response_modalities=[Modality.TEXT],
            ),
        )
    except Exception as e:
        print(f"Error during Gemini query: {e}")
        raise e
    return response.text


def response_gemini(path_1, prompt, version):
    max_retries = 3
    retries = 0
    delay = 5

    with open(path_1, "rb") as f:
        img_bytes_1 = f.read()
    input_img_1 = Image.open(BytesIO(img_bytes_1)).copy()
    payload = [input_img_1, prompt]

    while retries <= max_retries:
        try:
            response = call_gemini(payload, version)
            return response
        except Exception  as e:
                    last_error = e  # Store the caught error
                    if retries < max_retries:
                        # print(f"Error calling API (attempt {retries + 1}): {e}")
                        # print(f"Retrying in {delay} seconds...")
                        time.sleep(delay)
                        retries += 1
                    else:
                        raise last_error              
