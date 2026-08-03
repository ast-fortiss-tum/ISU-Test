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


def resize_and_pad(ref_img: Image.Image, base_w: int, base_h: int, bg=(255, 255, 255)) -> Image.Image:
    """
    Resize and pad the reference image to match the base image's aspect ratio.
    - ref_img: your reference outfit image (portrait, etc.)
    - base_w, base_h: base image dimensions (e.g., 960x540)
    - scale_frac: how much of the canvas the ref should occupy (0.6 = 60%)
    - bg: padding color (usually white)
    """
    bw, bh = base_w, base_h
    rw, rh = ref_img.size

    # Step 1. Resize proportionally so it fits comfortably in the target ratio
    max_size = 540
    if rw > max_size or rh > max_size:
        scale = max_size / max(rw, rh)
        new_w = int(rw * scale)
        new_h = int(rh * scale)
    ref_resized = ref_img.resize((new_w, new_h), Image.LANCZOS)

    # Step 2. Create a landscape canvas of the same size as base
    canvas = Image.new("RGB", (bw, bh), bg)
    offset_x = (bw - new_w) // 2
    offset_y = (bh - new_h) // 2
    canvas.paste(ref_resized, (offset_x, offset_y))
    return canvas

def call_nanobanana(payload):
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash-image", # gemini-2.5-flash-image-preview # "gemini-2.0-flash-preview-image-generation"
            contents=payload,
            config=GenerateContentConfig(
                response_modalities=[Modality.IMAGE],
            ),
        )
    except Exception as e:
        print(f"Error during Nanobanana query: {e}")
        raise e
    return response.candidates[0].content


def response_nanobanana(path_1, path_2, prompt):
    max_retries = 3
    retries = 0
    delay = 5

    with open(path_1, "rb") as f:
        img_bytes_1 = f.read()
    input_img_1 = Image.open(BytesIO(img_bytes_1)).copy()

    if path_2 != "":
        with open(path_2, "rb") as f:
            img_bytes_2 = f.read()
        input_img_2 = Image.open(BytesIO(img_bytes_2)).copy()

        W, H = input_img_1.size         # your base (e.g., 960x540)
        input_img_2 = resize_and_pad(input_img_2, W, H)

        payload = [input_img_1, prompt, input_img_2]
    else:
        payload = [input_img_1, prompt]

    while retries <= max_retries:
        try:
            response = call_nanobanana(payload)
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
