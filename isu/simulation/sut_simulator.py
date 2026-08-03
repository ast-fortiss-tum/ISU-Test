
from typing import Dict, List, Optional
import json
from pathlib import Path

from PIL import Image
from isu.model.models import Scenario
from opensbt.simulation.simulator import Simulator, SimulationOutputBase
from dataclasses import dataclass
from json_repair import repair_json
        
#from isu.sut.moondream import response_Moondream
from isu.sut.GPT4o import response_GPT4o
from isu.sut.gpt5 import response_GPT5
from isu.sut.isu_bmw import response_ISU
from isu.sut.gemini import response_gemini
import time
import os
from isu.simulation.prompt import *

feature_weight_map = {
    "gender": 0.05,
    "emotion": 0.05,
    "phone_driver": 0.1,
    "safety_belt": 0.1,
    "suitcase": 0.1,
    "suitcase_location": 0.1,
    "phone_codriver_seat": 0.1,
    "colabottle_codriver_seat": 0.05,
    "colacan_codriver_seat": 0.05,
    "baby_seat": 0.1,
    "baby_seat_orientation": 0.1,
    "baby": 0.1
}
feature_names = list(feature_weight_map.keys())

@dataclass
class ISUSimulationOutput(SimulationOutputBase):
    scenario: Scenario
    raw_result: str
    name: str
    last_error: str 

    def to_dict(self) -> dict:
        return {
            "scenario": self.scenario.model_dump(),
            "raw_result": self.raw_result,
            "name": self.name,
            "last_error": self.last_error,
        }

class ISUSimulator(Simulator):

    @staticmethod
    def simulate(variable_names: List[str], 
                 list_individuals: List[List[Scenario]], 
                 name: str, 
                 sut: str) -> List[ISUSimulationOutput]:

        results = []  
        
        for scenario in list_individuals: 
            if sut == "isu-bmw":
                wait_time = 1  # seconds
                print(f"Waiting for {wait_time} seconds before evaluating the next image...")
                time.sleep(wait_time)
            scenario = scenario[0]
            predicts_dict, raw_result, last_error = ISUSimulator.evaluate_image(scenario.image_path_sim2real, sut=sut)
            ISUSimulator.log_simulation_output(raw_result, scenario.image_path_sim, tartget_folder="output_predictions" )
            scenario.predicts_dict = predicts_dict
            result = ISUSimulationOutput(
                scenario = scenario,
                raw_result = raw_result,
                name = name,
                last_error = last_error
            )
            results.append(result)
            
        return results
    
    @staticmethod
    def log_simulation_output(raw_result: str, image_path: str, tartget_folder: str):
        folder_path = Path(image_path).parent.parent / tartget_folder
        folder_path.mkdir(parents=True, exist_ok=True)
        path = folder_path / (Path(image_path).stem + ".txt")
        with open(path, "w") as f:
            f.write(str(raw_result))
        #also save image to the folder
        # image = Image.open(image_path)
        # image.save(folder_path / Path(image_path).name)

    @staticmethod
    def get_user_prompt(prompt_version: int, feature_names: List[str]) -> str:
        if prompt_version == 1:
            return user_prompt_1.format(*feature_names)
        elif prompt_version == 2:
            return user_prompt_2.format(*feature_names)
        else:
            raise ValueError(f"Prompt version {prompt_version} not recognized.")
    
    @staticmethod
    def evaluate_image(image_path: str, sut: str, prompt_version: int = 2):
        fallback_result = {feature: "IND" for feature in feature_names}

        if sut == "dummy":
            return fallback_result, fallback_result, None

        if sut == "isu-bmw":
            db_json = "/Users/id/Documents/isu_db.json"
            start_time = time.time()
            res = response_ISU(image_path)
            # elapsed_time = time.time() - start_time
            # print(f"ISU response completed in {elapsed_time:.2f} seconds.")
            timeout = 100  # max seconds to wait
            # poll_interval = 0.01

            end_time = start_time + timeout
            while time.time() < end_time:
                try:
                    mtime = os.path.getmtime(db_json)
                    if mtime > start_time + 1:
                        with open(db_json, "r") as f:
                            try:
                                db_data = json.load(f)
                                break  # Successful load
                            except json.JSONDecodeError:
                                # File might still be in the middle of writing
                                pass
                except FileNotFoundError:
                    pass
                # time.sleep(poll_interval)
            else:
                raise TimeoutError("Timed out waiting for isu_db.json to update.")
            ##########################

            with open(db_json, "r") as f:
                db_data = json.load(f)

            
            print("image_path:", image_path)
            print("db_data:", db_data)

            try:
                driver = db_data["DRIVER"]
                gender = driver["sex"]
                emotion = driver["emotion"]
                phone_driver = driver["phone"]
                safety_belt = driver["wearing_seatbelt"]
            except KeyError as e:
                print(f"KeyError when accessing driver information: {e}")
                gender = "IND"
                emotion = "IND"
                phone_driver = "IND"
                safety_belt = "IND"

            try:
                # boolean features initialization as "NO"
                suitcase = "NO"
                suitcase_location = set()
                phone_codriver_seat = "NO"
                colabottle_codriver_seat = "NO"
                colacan_codriver_seat = "NO"
                baby_seat = "NO"

                # conditional prediction intialization as "None"
                baby_seat_orientation = "None"
                baby = "None"

                try:
                    backseat_area = db_data["BACK_SEAT_AREA"]["object_description"]
                except KeyError as e:
                    backseat_area = []

                try:
                    center_area = db_data["CENTER_CONSOLE_AREA"]["object_description"]
                except KeyError as e:
                    center_area = []
                
                try:
                    codriver_area = db_data["CO_DRIVER_SEAT_AREA"]["object_description"]
                except KeyError as e:
                    codriver_area = []

                if len(backseat_area) > 0:
                    for obj in backseat_area:
                        try:
                            if obj["category"] == "SUITCASE":
                                suitcase = "YES"
                                suitcase_location.add("REAR_SEAT")
                        except KeyError:
                            print("KeyError in backseat_area object:", obj)
                            pass
                if len(center_area) > 0:
                    for obj in center_area:
                        try:
                            if obj["category"] == "SUITCASE":
                                suitcase = "YES"
                                suitcase_location.add("CENTER_CONSOLE")
                        except KeyError:
                            print("KeyError in center_area object:", obj)
                            pass
                if len(codriver_area) > 0:
                    for obj in codriver_area:
                        try:
                            if obj["category"] == "SUITCASE":
                                suitcase = "YES"
                                suitcase_location.add("CO_DRIVER_SEAT")
                            elif obj["category"] == "PHONE":
                                phone_codriver_seat = "YES"
                            elif obj["category"] == "COLA_PLASTIC_BOTTLE":
                                colabottle_codriver_seat = "YES"
                            elif obj["category"] == "COLA_CAN":
                                colacan_codriver_seat = "YES"
                            elif obj["category"] == "BABY_SEAT":
                                baby_seat = "YES"
                                baby_seat_orientation = obj.get("babyseat_orientation")
                                if obj.get("baby_in_baby_seat") == "YES":
                                    baby = "YES"
                                else:
                                    baby = "NO"
                        except KeyError:
                            print("KeyError in codriver_area object:", obj)
                            pass
                
                suitcase_location = ", ".join(f'{x}' for x in suitcase_location)

                if suitcase_location == "":
                    suitcase_location = "None"
            
            except KeyError as e:
                phone_codriver_seat = "IND"
                suitcase = "IND"
                suitcase_location = "IND"
                print(f"KeyError encountered: {e}")

            result = {
                "gender": gender,
                "emotion": emotion,
                "phone_driver": phone_driver,
                "safety_belt": safety_belt,
                "suitcase": suitcase,
                "suitcase_location": suitcase_location,
                "phone_codriver_seat": phone_codriver_seat,
                "colabottle_codriver_seat": colabottle_codriver_seat,
                "colacan_codriver_seat": colacan_codriver_seat,
                "baby_seat": baby_seat,
                "baby_seat_orientation": baby_seat_orientation,
                "baby": baby
            }
            return result, result, None
        
        # other SUTs
        user_prompt = ISUSimulator.get_user_prompt(prompt_version, feature_names)
        last_error = None
        raw_result = None
        if sut == "gpt4o":
            try:
                raw_result = response_GPT4o(user_prompt, "", image_path)
            except Exception as e:
                last_error = e
        elif sut == "gpt5":
            try:
                raw_result = response_GPT5(user_prompt, "", image_path)
            except Exception as e:
                last_error = e
        elif sut == "gemini-2.5" or sut == "gemini-2.0":
            try:
                raw_result = response_gemini(image_path, user_prompt, version=sut.split("-")[1])
            except Exception as e:
                last_error = e
        # elif sut == "moondream":
        #     image = Image.open(image_path)
        #     raw_result = response_Moondream(user_prompt, image)
        else:
            print(f"SUT {sut} not recognized.")
            last_error = ValueError(f"SUT {sut} not recognized.")

        #print("Raw result:", image_path, raw_result)
        if raw_result:
            try:
                result = json.loads(repair_json(raw_result))
                if all(feature in result for feature in feature_names):
                    return result, raw_result, str(last_error)
                else:
                    print(f"Missing features in result: {[feature for feature in feature_names if feature not in result]}")
                    last_error = ValueError("Missing features in result.")
            except Exception as e:
                last_error = e

        return fallback_result, raw_result, str(last_error)