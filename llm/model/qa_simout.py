from dataclasses import dataclass, field
from typing import Any, Dict

from opensbt.simulation.simulator import SimulationOutputBase
from llm.model.models import Utterance

@dataclass
class QASimulationOutput(SimulationOutputBase):
    utterance: Utterance
    model: str
    ipa: str = ""
    response: Any = None
    poi_exists: bool = False
    other: Dict = field(default_factory=dict)  # ensures each instance gets its own dict
    timestamp: str = ""

    def to_dict(self) -> dict:
        return {
            "utterance": self.utterance.dict(),
            "model": self.model.value,
            "ipa": self.ipa,
            "response": self.response,
            "poi_exists": self.poi_exists,
            "other": self.other,
            "timestamp": self.timestamp,
        }