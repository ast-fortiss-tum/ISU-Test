from typing import Optional
from isu.model.models import ContentInput

class BlenderContentInput(ContentInput):
    # Driver
    driver_gender: Optional[str] = None
    driver_tshirt_color: Optional[str] = None
    driver_emotion: Optional[str] = None
    driver_phone: Optional[str] = None
    driver_safety_belt: Optional[str] = None

    # Environment
    env: Optional[str] = None

    # Co-driver seat (front right)
    passenger_codriver: Optional[str] = None
    passenger_codriver_tshirt_color: Optional[str] = None
    passenger_codriver_emotion: Optional[str] = None
    codriver_safety_belt: Optional[str] = None

    # Rear left seat
    passenger_back_seat_left: Optional[str] = None
    passenger_rear_left_tshirt_color: Optional[str] = None
    passenger_rear_left_emotion: Optional[str] = None
    passenger_rear_left_safety_belt: Optional[str] = None

    # Rear right seat
    passenger_back_seat_right: Optional[str] = None
    passenger_rear_right_tshirt_color: Optional[str] = None
    passenger_rear_right_emotion: Optional[str] = None
    passenger_rear_right_safety_belt: Optional[str] = None
    passenger_codriver_head_angle: Optional[int] = None
    passenger_rear_left_head_angle: Optional[int] = None
    passenger_rear_right_head_angle: Optional[int] = None

    # Co-driver seat objects
    suitcase: Optional[str] = None
    suitcase_color: Optional[str] = None
    suitcase_location: Optional[str] = None
    suitcase_pose: Optional[str] = None

    phone_codriver_seat: Optional[str] = None
    phone_codriver_seat_color: Optional[str] = None
    colabottle_codriver_seat: Optional[str] = None
    colacan_codriver_seat: Optional[str] = None

    # Baby seat (co-driver seat)
    baby_seat: Optional[str] = None
    baby_seat_orientation: Optional[str] = None
    baby: Optional[str] = None

    # Continuous features
    driver_height_m: Optional[float] = None
    driver_weight_kg: Optional[float] = None
    height_m: Optional[float] = None
    weight_kg: Optional[float] = None
    light_front: Optional[float] = None
    light_back_left: Optional[float] = None
    light_back_right: Optional[float] = None
    env_strength: Optional[float] = None
    exposure_compensation: Optional[float] = None
