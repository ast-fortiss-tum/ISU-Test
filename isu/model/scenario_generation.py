from abc import abstractmethod, ABC
from typing import List, Optional, Dict, Any, Tuple

from isu.model.models import Scenario
from isu.features import FeatureHandler
from isu.model.blender_contentinput import BlenderContentInput
import os
import time
import json
from .run_blender import run_blender
from .call_sim2real import call_sim2real
from .run_vlm_features import run_vlm_features
from isu.features.models import FeatureType

from opensbt.config import RESULTS_FOLDER
from isu.config import SIM2REAL, VLM_FEATURES
output_dir = str(os.getcwd()) + os.sep + RESULTS_FOLDER

class ScenarioGenerator(ABC):
    def __init__(self, feature_handler: Optional[FeatureHandler] = None):
        self.feature_handler = feature_handler

    @abstractmethod
    def generate(
        self,
        ordinal_vars: List[float],
        categorical_vars: List[int],
        continuous_vars: List[float],
        image_output_dir: str,
        json_output_dir: str
    ) -> Scenario:
        pass

class BlenderScenarioGenerator(ScenarioGenerator):
    def __init__(self,
                 problem_name: str,
                 json_output_dir: str,
                 image_output_dir: str,
                 feature_handler: Optional[FeatureHandler] = None,
                 apply_constrains_to_vars: bool = True):
        super().__init__(feature_handler=feature_handler)
        self.problem_name = problem_name
        self.json_output_dir = json_output_dir
        self.image_output_dir = image_output_dir
        self.apply_constrains_to_vars = apply_constrains_to_vars

    def _get_content_input(self, feature_values: Dict[str, Any]) -> BlenderContentInput:
        return BlenderContentInput.model_validate(feature_values)
    
    def run_blender_mock(self, json_path: str) -> str:
        return "isu/dummy_image.png"

    def write_params_to_json(self, params: dict, json_output_dir: str) -> str:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        json_path = os.path.join(json_output_dir, f"{timestamp}.json")
        with open(json_path, "w") as f:
            json.dump(params, f, indent=2)
        return json_path
    
    def generate(
        self,
        ordinal_vars: List[float],
        categorical_vars: List[int],
        continuous_vars: List[float],
        render: bool
    ) -> Scenario:
        feature_values = self.feature_handler.get_feature_values_dict(
            ordinal_feature_scores=ordinal_vars,
            categorical_feature_indices=categorical_vars,
            continuous_feature_values=continuous_vars
        )
        content_input = self._get_content_input(feature_values)
        self.apply_constraints(content_input)

        if self.apply_constrains_to_vars:
            self._update_features_from_content_input(
                ordinal_vars,
                categorical_vars,
                continuous_vars,
                content_input,
            )

        if render:
            json_path = self.write_params_to_json(params=content_input.model_dump(), json_output_dir=self.json_output_dir)
            image_path_sim = run_blender(json_path=json_path, image_output_dir=self.image_output_dir)
            run_vlm_features(json_path=json_path, image_path=image_path_sim, dummy=not VLM_FEATURES)
            image_path_sim2real = call_sim2real(image_path_sim, dummy=not SIM2REAL, gt=content_input.model_dump())
        else:
            json_path = ""
            image_path_sim = ""
            image_path_sim2real = ""

        return Scenario(
            params_dict=content_input.model_dump(),
            ordinal_vars=ordinal_vars,
            categorical_vars=categorical_vars,
            continuous_vars=continuous_vars,
            image_path_sim=image_path_sim,
            image_path_sim2real=image_path_sim2real
        )

        
    def apply_constraints(self, content_input: BlenderContentInput) -> BlenderContentInput:
        if content_input.participant_driver == "LEV":
            content_input.gender = "MALE"
            content_input.height_m = 1.94
            content_input.weight_kg = 86
        elif content_input.participant_driver == "STEPAN":
            content_input.gender = "MALE"
            content_input.height_m = 1.89
            content_input.weight_kg = 75
        elif content_input.participant_driver == "IVAN":
            content_input.gender = "MALE"
            content_input.height_m = 1.78
            content_input.weight_kg = 100
        elif content_input.participant_driver == "KEN":
            content_input.gender = "MALE"
            content_input.height_m = 1.80
            content_input.weight_kg = 85
        elif content_input.participant_driver == "CHEN":
            content_input.gender = "FEMALE"
            content_input.height_m = 1.67
            content_input.weight_kg = 52

        if content_input.phone_driver == "NO":
            content_input.phone_driver_pose_frame = None
            
        if content_input.suitcase == "NO":
            content_input.suitcase_color = None
            content_input.suitcase_location = None

        if content_input.baby_seat == "NO":
            content_input.baby_seat_orientation = None
            # content_input.baby_seat_safety_belt = None
            content_input.baby = None
        # if baby seat is present, no other objects allowed on co-driver seat
        else:
            content_input.phone_codriver_seat = "NO"
            content_input.colabottle_codriver_seat = "NO"
            content_input.colacan_codriver_seat = "NO"
            if content_input.suitcase_location == "CO_DRIVER_SEAT":
                content_input.suitcase_location = "REAR_SEAT"

        if content_input.suitcase_location == "CO_DRIVER_SEAT":
            if content_input.phone_codriver_seat == "YES" or content_input.colabottle_codriver_seat == "YES" or content_input.colacan_codriver_seat == "YES":
                content_input.suitcase_location = "REAR_SEAT"

        if content_input.suitcase_location != "REAR_SEAT":
            content_input.suitcase_pose = None
            content_input.suitcase_rotation = None
        elif content_input.suitcase_location == "REAR_SEAT" and content_input.suitcase_pose == "UPWARDS":
            content_input.suitcase_rotation = None
        
        if content_input.phone_codriver_seat == "NO":
            content_input.phone_codriver_seat_color = None

        if content_input.env_light_discrete == "MEDIUM":
            content_input.env_strength = 0.5
            content_input.exposure_compensation = 2.0
        elif content_input.env_light_discrete == "HIGH":
            content_input.env_strength = 1.0
            content_input.exposure_compensation = 0.0

        # if content_input.env_strength > 2.0:
        #     content_input.light_front = 0.0
        #     content_input.light_back_left = 0.0
        #     content_input.light_back_right = 0.0
        # elif content_input.env_strength < 0.5:
        #     content_input.in_car_light = "ON"


        off_strength = 0.0
        on_strength = 0.5
        if content_input.in_car_light == "ON":
            content_input.light_front = on_strength
            content_input.light_back_left = on_strength
            content_input.light_back_right = on_strength
        elif content_input.in_car_light == "OFF":
            content_input.light_front = off_strength
            content_input.light_back_left = off_strength
            content_input.light_back_right = off_strength

        return content_input
    

    def _update_features_from_content_input(
            self,
            ordinal_vars: List[float],
            categorical_vars: List[int],
            continuous_vars: List[float],
            content_input: BlenderContentInput,
    ) -> Tuple[List[float], List[int], List[float]]:
        for i, feature in enumerate(self.feature_handler.ordinal_features.values()):
            if not hasattr(content_input, feature.name):
                continue
            new_value = getattr(content_input, feature.name, None)
            new_var = self.feature_handler.get_var_from_feature_value(
                feature,
                new_value,
                feature_type=FeatureType.ORDINAL
            )
            if new_var is not None:
                ordinal_vars[i] = new_var
        for i, feature in enumerate(self.feature_handler.categorical_features.values()):
            if not hasattr(content_input, feature.name):
                continue
            new_value = getattr(content_input, feature.name, None)
            new_var = self.feature_handler.get_var_from_feature_value(
                feature,
                new_value,
                feature_type=FeatureType.CATEGORICAL
            )
            if new_var is not None:
                categorical_vars[i] = new_var
        for i, feature in enumerate(self.feature_handler.continuous_features.values()):
            if not hasattr(content_input, feature.name):
                continue
            new_value = getattr(content_input, feature.name, None)
            new_var = self.feature_handler.get_var_from_feature_value(
                feature,
                new_value,
                feature_type=FeatureType.CONTINUOUS
            )
            if new_var is not None:
                continuous_vars[i] = new_var
        return ordinal_vars, categorical_vars, continuous_vars
