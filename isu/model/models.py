from typing import List, Dict, Any, Optional
import random
from pydantic import BaseModel
from dataclasses import dataclass, field
class ContentInput(BaseModel):
    pass

class ContentOutput(BaseModel):
    pass
class Scenario(BaseModel):
    ordinal_vars: List[float] = field(default_factory=list)
    categorical_vars: List[int] = field(default_factory=list)
    continuous_vars: List[float] = field(default_factory=list)
    
    params_dict: Dict[str, Any] = None
    
    predicts_dict: Dict[str, Any] = None
    image_path_sim: str = ""
    image_path_sim2real: str = ""