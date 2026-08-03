
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
S        
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
S        
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