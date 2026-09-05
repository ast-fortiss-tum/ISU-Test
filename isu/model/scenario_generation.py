from abc import abstractmethod, ABC
from typing import List, Optional, Dict, Any, Tuple

from isu.model.models import Scenario
from isu.features import FeatureHandler
from isu.model.blender_contentinput import BlenderContentInput
import os
import time
import json
from .run_blender import run_blender
# from .call_sim2real import call_sim2real
from isu.features.models import FeatureType

from opensbt.config import RESULTS_FOLDER
from isu.config import BLENDER_FILE_PATH
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
                 apply_constrains_to_vars: bool = True,
                 depth_output_dir: Optional[str] = None,
                 seg_output_dir: Optional[str] = None,
                 canny_output_dir: Optional[str] = None,
                 instance_seg_output_dir: Optional[str] = None,
                 blend_file: Optional[str] = None):
        super().__init__(feature_handler=feature_handler)
        self.problem_name = problem_name
        self.json_output_dir = json_output_dir
        self.image_output_dir = image_output_dir
        self.apply_constrains_to_vars = apply_constrains_to_vars
        self.depth_output_dir = depth_output_dir
        self.seg_output_dir = seg_output_dir
        self.canny_output_dir = canny_output_dir
        self.instance_seg_output_dir = instance_seg_output_dir
        self.blend_file = blend_file

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
            image_path_sim = run_blender(
                json_path=json_path,
                image_output_dir=self.image_output_dir,
                depth_output_dir=self.depth_output_dir,
                seg_output_dir=self.seg_output_dir,
                canny_output_dir=self.canny_output_dir,
                instance_seg_output_dir=self.instance_seg_output_dir,
                blend_file=self.blend_file or BLENDER_FILE_PATH,
            )
            # VLM feature conversion is disabled.
            # Sim2Real conversion is disabled; use the rendered simulation image.
            # image_path_sim2real = call_sim2real(
            #     image_path_sim,
            #     dummy=not SIM2REAL,
            #     gt=content_input.model_dump(),
            # )
            image_path_sim2real = ""
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
        if content_input.suitcase == "NO":
            content_input.suitcase_color = None
            content_input.suitcase_location = None
            content_input.suitcase_pose = None

        if content_input.baby_seat == "NO":
            content_input.baby_seat_orientation = None
            content_input.baby = None
        else:
            content_input.phone_codriver_seat = "NO"
            content_input.colabottle_codriver_seat = "NO"
            content_input.colacan_codriver_seat = "NO"
            content_input.passenger_codriver = "NO"
            content_input.codriver_safety_belt = "NO"
            if content_input.suitcase_location == "CO_DRIVER_SEAT":
                content_input.suitcase_location = "REAR_SEAT"

        if content_input.passenger_codriver == "YES":
            content_input.phone_codriver_seat = "NO"
            content_input.colabottle_codriver_seat = "NO"
            content_input.colacan_codriver_seat = "NO"
            if content_input.suitcase_location == "CO_DRIVER_SEAT":
                content_input.suitcase_location = "REAR_SEAT"
        elif (
            content_input.phone_codriver_seat == "YES"
            or content_input.colabottle_codriver_seat == "YES"
            or content_input.colacan_codriver_seat == "YES"
            or content_input.suitcase_location == "CO_DRIVER_SEAT"
        ):
            content_input.passenger_codriver = "NO"
            content_input.codriver_safety_belt = "NO"
            content_input.passenger_codriver_tshirt_color = None
            content_input.passenger_codriver_emotion = None
            content_input.passenger_codriver_head_angle = None
        elif content_input.passenger_codriver == "NO":
            content_input.codriver_safety_belt = "NO"
            content_input.passenger_codriver_tshirt_color = None
            content_input.passenger_codriver_emotion = None
            content_input.passenger_codriver_head_angle = None

        if content_input.suitcase_location == "CO_DRIVER_SEAT" and (
            content_input.phone_codriver_seat == "YES"
            or content_input.colabottle_codriver_seat == "YES"
            or content_input.colacan_codriver_seat == "YES"
            or content_input.passenger_codriver == "YES"
        ):
            content_input.suitcase_location = "REAR_SEAT"

        if content_input.suitcase_location != "REAR_SEAT":
            content_input.suitcase_pose = None
        
        if content_input.phone_codriver_seat == "NO":
            content_input.phone_codriver_seat_color = None

        if content_input.passenger_back_seat_left != "YES":
            content_input.passenger_back_seat_left = "NO"
            content_input.passenger_rear_left_safety_belt = "NO"
            content_input.passenger_rear_left_tshirt_color = None
            content_input.passenger_rear_left_emotion = None
            content_input.passenger_rear_left_head_angle = None

        if content_input.passenger_back_seat_right != "YES":
            content_input.passenger_back_seat_right = "NO"
            content_input.passenger_rear_right_safety_belt = "NO"
            content_input.passenger_rear_right_tshirt_color = None
            content_input.passenger_rear_right_emotion = None
            content_input.passenger_rear_right_head_angle = None

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
