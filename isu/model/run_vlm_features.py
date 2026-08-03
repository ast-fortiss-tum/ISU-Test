import os
import json
from isu.sut.nanobanana import response_nanobanana
from io import BytesIO
from PIL import Image

def run_vlm_features(json_path: str, image_path:str, dummy: bool ) -> None:
        if dummy:
            return
        # if safety belt is YES in json, call nanobanana to add safety belt to the driver
        with open(json_path, "r") as f:
            params = json.load(f)
        

        safety_belt = params.get("safety_belt", None)

        gender = params.get("gender")
        if gender == "FEMALE":
            driver_cloth = params.get("female_driver_cloth", None)
        else:
            driver_cloth = params.get("male_driver_cloth", None)

        participant_driver = params.get("participant_driver")
        if participant_driver != "NO":
            driver_cloth = participant_driver

        folder_cloth = "isu/blender/assets/clothes"
        driver_cloth_path = os.path.join(folder_cloth, f"{driver_cloth}.png")


        if driver_cloth != "DEFAULT" and safety_belt == "YES":
            cloth_safety_belt_prompt = f""""
            image[0] = BASE canvas (keep framing). image[1] = outfit reference on a landscape canvas.
            Follow image[0] for canvas/field of view. Do NOT crop or change aspect ratio, which is 16:9, horizontal.
            Keep all semantics, human pose, facial expressions, lighting, reflections, background, and perspective identical of the base image.
            Update BASE Image, replace ONLY the driver’s clothing using the outfit from image[1]. 
            Additionally, add a safety belt, fastened to the driver on the driver seat, ensure the safety belt looks realistic and properly fitted.
            """
            try:
                response = response_nanobanana(path_1=image_path, path_2=driver_cloth_path, prompt=cloth_safety_belt_prompt)
                for part in response.parts:
                    if part.inline_data:
                        edited = Image.open(BytesIO(part.inline_data.data)).copy()
                        edited.save(image_path)
                        break
            except Exception as e:
                print(f"Nanobanana failed to change clothing for image: {image_path} with error: {e}")


        else:
            if driver_cloth != "DEFAULT":
                cloth_prompt = f""""
                image[0] = BASE canvas (keep framing). image[1] = outfit reference on a landscape canvas.
                Follow image[0] for canvas/field of view. Do NOT crop or change aspect ratio, which is 16:9, horizontal.
                Keep all semantics, human pose, facial expressions, lighting, reflections, background, and perspective identical of the base image.
                Update BASE Image, replace ONLY the driver’s clothing using the outfit from image[1]. 
                """
                try:
                    response = response_nanobanana(path_1=image_path, path_2=driver_cloth_path, prompt=cloth_prompt)
                    for part in response.parts:
                        if part.inline_data:
                            edited = Image.open(BytesIO(part.inline_data.data)).copy()
                            edited.save(image_path)
                            break
                except Exception as e:
                    print(f"Nanobanana failed to change clothing for image: {image_path} with error: {e}")

            if safety_belt == "YES":
                safety_belt_prompt = """"
                Follow the image for canvas/field of view. Do NOT crop or change aspect ratio, which is 16:9, horizontal.
                Keep all semantics, human pose, facial expressions, lighting, reflections, background, and perspective identical of the base image.
                Add a safety belt to the driver, fastened on the driver seat, ensure the safety belt looks realistic and properly fitted.
                """
                try:
                    response = response_nanobanana(path_1=image_path, path_2="", prompt=safety_belt_prompt)
                    for part in response.parts:
                        if part.inline_data:
                            edited = Image.open(BytesIO(part.inline_data.data)).copy()
                            edited.save(image_path)
                            break
                except Exception as e:
                    print(f"Nanobanana failed to add safety belt for image: {image_path} with error: {e}")