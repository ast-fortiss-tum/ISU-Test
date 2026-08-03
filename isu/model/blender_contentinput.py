from typing import Optional
from isu.model.models import ContentInput

class BlenderContentInput(ContentInput):
    participant_driver: Optional[str] = None
    in_car_light: Optional[str] = None
    env_light_discrete: Optional[str] = None
    gender: Optional[str] = None
    emotion: Optional[str] = None
    #driver_texture_index: Optional[int] = None
    driver_tshirt_color: Optional[str] = None
    phone_driver: Optional[str] = None
    phone_driver_pose_frame: Optional[int] = None
    safety_belt: Optional[str] = None
    env: Optional[str] = None
    car: Optional[str] = None

    # female_driver_cloth: Optional[str] = None
    # male_driver_cloth: Optional[str] = None
    suitcase: Optional[str] = None
    suitcase_color: Optional[str] = None
    suitcase_location: Optional[str] = None
    suitcase_pose: Optional[str] = None
    suitcase_rotation: Optional[float] = None

    phone_codriver_seat: Optional[str] = None
    phone_codriver_seat_color: Optional[str] = None
    colabottle_codriver_seat: Optional[str] = None
    colacan_codriver_seat: Optional[str] = None

    baby_seat: Optional[str] = None
    baby_seat_orientation: Optional[str] = None
    baby: Optional[str] = None

    height_m: Optional[float] = None
    weight_kg: Optional[float] = None
    light_front: Optional[float] = None
    light_back_left: Optional[float] = None
    light_back_right: Optional[float] = None
    env_strength: Optional[float] = None
    exposure_compensation: Optional[float] = None
