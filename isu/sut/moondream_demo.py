from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image

model = AutoModelForCausalLM.from_pretrained(
    "vikhyatk/moondream2",
    revision="2025-06-21",
    trust_remote_code=True,
    device_map="mps"
)

#import image
image = Image.open("/Users/id/Library/CloudStorage/OneDrive-BMWGroup/opensbt-ISU/isu/blender/images/20250811_131530.png")

# Captioning
print("Short caption:")
print(model.caption(image, length="short")["caption"])

print("\nNormal caption:")
for t in model.caption(image, length="normal", stream=True)["caption"]:
    # Streaming generation example, supported for caption() and detect()
    print(t, end="", flush=True)
print(model.caption(image, length="normal"))

# Visual Querying
user_prompt = (
    "You are a security expert evaluating images for distracting driving detection. "
    "Your task is to analyze the image and provide a score for distracting driving behavior. "
    "Pay attention to the driver's physical condition regarding facial expressions, head position, and other visible signs of distraction. "
    "The score must be a float between 0 and 1, where 0 means no distracting driving and 1 means clear signs of distraction. "
    "Additionally, explain why you made this decision in a single sentence. "
    "Return your result in **strict JSON format** with the following structure:\n\n"
    '{\n  "score": <float>,\n  "explanation": "<string>"\n}\n'
    # "For example, if the score is 0.8, you should return:\n"
    # '{\n  "score": 0.8,\n  "explanation": "Because the driver closed his eyes and appeared to be drowsy."\n}'
)
#print("\nVisual query: 'How many people are in the image?'")
print(model.query(image, user_prompt)["answer"])

# Object Detection
print("\nObject detection: 'face'")
objects = model.detect(image, "face")["objects"]
print(f"Found {len(objects)} face(s)")

# Pointing
print("\nPointing: 'person'")
points = model.point(image, "person")["points"]
print(f"Found {len(points)} person(s)")
