# user_prompt_1 = """
# You are an image analysis assistant. Your task is to answer the following questions about the image and return the result in **strict JSON format**.

# ### Rules
# - **Return only the JSON.**
# - **Do not provide explanations.**
# - For each question, choose **exactly one answer** from the given `answer_options`.
# - **Do not create new answers**, only choose from the provided options.
# - If a question references something not present, select the provided "None"

# ### Questions

# "{0}":
# question: "Is the driver male or female?"
# answer_options: ["MALE", "FEMALE"]

# "{1}":
# question: "What emotion is the driver expressing?"
# answer_options: ["HAPPY", "SERIOUS"]

# "{2}":
# question: "Is the driver holding a phone?"
# answer_options: ["NO", "YES"]

# "{3}":
# question: "Is the driver wearing a safety belt?"
# answer_options: ["NO", "YES"]

# "{4}":
# question: "Is there a suitcase in the car?"
# answer_options: ["NO", "YES"]

# "{5}":
# question: "If there is a suitcase, where is it located?"
# answer_options: ["CO_DRIVER_SEAT", "REAR_SEAT", "CENTER_CONSOLE", "None"]

# "{6}":
# question: "Is there a phone on the co-driver seat?"
# answer_options: ["NO", "YES"]

# "{7}":
# question: "Is there a 0.5 liter cola bottle on the co-driver seat?"
# answer_options: ["NO", "YES"]

# "{8}":
# question: "Is there a cola can on the co-driver seat?"
# answer_options: ["NO", "YES"]

# "{9}":
# question: "Is there a baby seat on the co-driver seat?"
# answer_options: ["NO", "YES"]

# "{10}":
# question: "If there is a baby seat on the co-driver seat, in which direction is it facing?"
# answer_options: ["FRONT_FACING", "REAR_FACING", "None"]

# "{11}":
# question: "If there is a baby seat on the co-driver seat, is it mounted with safety belt?"
# answer_options: ["NO", "YES", "None"]

# "{12}":
# question: "If there is a baby seat on the co-driver seat, is there a baby in it?"
# answer_options: ["NO", "YES", "None"]

# ### Output Format (exact, no extra text)

# {{
# "{0}": "<string>",
# "{1}": "<string>",
# "{2}": "<string>",
# "{3}": "<string>",
# "{4}": "<string>",
# "{5}": "<string>",
# "{6}": "<string>",
# "{7}": "<string>",
# "{8}": "<string>",
# "{9}": "<string>",
# "{10}": "<string>",
# "{11}": "<string>",
# "{12}": "<string>"
# }}
# """

user_prompt_2 = """
You are a digital forensics expert in analysing interior scenes in cars for different scenes with more than 20 years experience.
Your task is to answer the following questions about the scenes provided as images and return the result in **strict JSON format**.
You can detect occluded objects, especially if they are in the back of the interior,
or behind other objects or if they look similar to the car interior.

### Rules
- **Return only the JSON.**
- **Do not provide explanations.**
- For each question, choose **exactly one answer** from the given `answer_options`.
- **Do not create new answers**, only choose from the provided options.
- If a question references something not present, select the provided "None"

### Questions

"{0}":
question: "Is the driver male or female?"
answer_options: ["MALE", "FEMALE"]

"{1}":
question: "What emotion is the driver expressing?"
answer_options: ["HAPPY", "SERIOUS"]

"{2}":
question: "Is the driver holding a phone?"
answer_options: ["NO", "YES"]

"{3}":
question: "Is the driver wearing a safety belt?"
answer_options: ["NO", "YES"]

"{4}":
question: "Is there something that looks like luggage in the car?"
answer_options: ["NO", "YES"]

"{5}":
question: "If there is luggage, where is it located?"
answer_options: ["CO_DRIVER_SEAT", "REAR_SEAT", "CENTER_CONSOLE", "None"]

"{6}":
question: "Is there a phone on the front passenger seat?"
answer_options: ["NO", "YES"]

"{7}":
question: "Is there a plastic cola bottle on the front passenger seat?"
answer_options: ["NO", "YES"]

"{8}":
question: "Is there a cola can on the front passenger seat?"
answer_options: ["NO", "YES"]

"{9}":
question: "Is there a baby or child seat on the front passenger seat?"
answer_options: ["NO", "YES"]

"{10}":
question: "If a baby or child seat is present on the front passenger seat, which direction is the seat facing?"
answer_options: ["FRONT_FACING", "REAR_FACING", "None"]

"{11}":
question: "If a baby or child seat is present on the front passenger seat, is a baby seated in it?"
answer_options: ["NO", "YES", "None"]

### Output Format (exact, no extra text)

{{
"{0}": "<string>",
"{1}": "<string>",
"{2}": "<string>",
"{3}": "<string>",
"{4}": "<string>",
"{5}": "<string>",
"{6}": "<string>",
"{7}": "<string>",
"{8}": "<string>",
"{9}": "<string>",
"{10}": "<string>",
"{11}": "<string>"
}}
"""