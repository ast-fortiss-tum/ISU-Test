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
question: "What is the driver's gender?"
answer_options: ["MALE", "FEMALE"]

"{1}":
question: "What color is the driver's shirt?"
answer_options: ["BLACK", "WHITE"]

"{2}":
question: "What emotion is the driver expressing?"
answer_options: ["HAPPY", "SERIOUS"]

"{3}":
question: "Is the driver holding a phone?"
answer_options: ["NO", "YES"]

"{4}":
question: "Is the driver wearing a safety belt?"
answer_options: ["NO", "YES"]

"{5}":
question: "Is there a front passenger?"
answer_options: ["NO", "YES"]

"{6}":
question: "What color is the front passenger's shirt?"
answer_options: ["BLACK", "WHITE", "None"]

"{7}":
question: "What emotion is the front passenger expressing?"
answer_options: ["HAPPY", "SERIOUS", "None"]

"{8}":
question: "Is the front passenger wearing a safety belt?"
answer_options: ["NO", "YES", "None"]

"{9}":
question: "Is there a rear-left passenger?"
answer_options: ["NO", "YES"]

"{10}":
question: "What color is the rear-left passenger's shirt?"
answer_options: ["BLACK", "WHITE", "None"]

"{11}":
question: "What emotion is the rear-left passenger expressing?"
answer_options: ["HAPPY", "SERIOUS", "None"]

"{12}":
question: "Is the rear-left passenger wearing a safety belt?"
answer_options: ["NO", "YES", "None"]

"{13}":
question: "Is there a rear-right passenger?"
answer_options: ["NO", "YES"]

"{14}":
question: "What color is the rear-right passenger's shirt?"
answer_options: ["BLACK", "WHITE", "None"]

"{15}":
question: "What emotion is the rear-right passenger expressing?"
answer_options: ["HAPPY", "SERIOUS", "None"]

"{16}":
question: "Is the rear-right passenger wearing a safety belt?"
answer_options: ["NO", "YES", "None"]

"{17}":
question: "Is there a suitcase in the car?"
answer_options: ["NO", "YES"]

"{18}":
question: "What color is the suitcase?"
answer_options: ["ANTRACITE", "YELLOW", "None"]

"{19}":
question: "Where is the suitcase located?"
answer_options: ["CO_DRIVER_SEAT", "REAR_SEAT", "None"]

"{20}":
question: "What is the suitcase pose?"
answer_options: ["UPWARDS", "FLAT", "None"]

"{21}":
question: "Is there a phone on the front passenger seat?"
answer_options: ["NO", "YES"]

"{22}":
question: "What color is the phone on the front passenger seat?"
answer_options: ["BLACK", "WHITE", "None"]

"{23}":
question: "Is there a cola bottle on the front passenger seat?"
answer_options: ["NO", "YES"]

"{24}":
question: "Is there a cola can on the front passenger seat?"
answer_options: ["NO", "YES"]

"{25}":
question: "Is there a baby seat on the front passenger seat?"
answer_options: ["NO", "YES"]

"{26}":
question: "Which direction is the baby seat facing?"
answer_options: ["FRONT_FACING", "REAR_FACING", "None"]

"{27}":
question: "Is there a baby in the baby seat?"
answer_options: ["NO", "YES", "None"]

### Output Format (exact, no extra text)

{{
"{0}": "<string>", "{1}": "<string>", "{2}": "<string>",
"{3}": "<string>", "{4}": "<string>", "{5}": "<string>",
"{6}": "<string>", "{7}": "<string>", "{8}": "<string>",
"{9}": "<string>", "{10}": "<string>", "{11}": "<string>",
"{12}": "<string>", "{13}": "<string>", "{14}": "<string>",
"{15}": "<string>", "{16}": "<string>", "{17}": "<string>",
"{18}": "<string>", "{19}": "<string>", "{20}": "<string>",
"{21}": "<string>", "{22}": "<string>", "{23}": "<string>",
"{24}": "<string>", "{25}": "<string>", "{26}": "<string>",
"{27}": "<string>"
}}
"""

user_prompt_1 = user_prompt_2