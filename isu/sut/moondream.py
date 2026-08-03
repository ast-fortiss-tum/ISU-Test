from transformers import AutoModelForCausalLM
import time

moondream_model = AutoModelForCausalLM.from_pretrained(
    "vikhyatk/moondream2",
    revision="2025-06-21",
    trust_remote_code=True,
    device_map="mps"
    )
print("Moondream model loaded")

def call_Moondream(prompt, image):
    try:
        response = moondream_model.query(image, prompt)["answer"]
    except Exception as e:
        print(f"Error during Moondream query: {e}")
        raise e
    return response

def response_Moondream(user_prompt, image):
    max_retries = 3
    retries = 0
    delay = 1
    while retries <= max_retries:
        try:
            response = call_Moondream(user_prompt, image)
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