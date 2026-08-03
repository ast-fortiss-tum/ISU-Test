from typing import Dict
import numpy as np
from pydantic import BaseModel
from isu.model.models import Scenario

def dist_euclidean_ind(ind_1, ind_2, bounds = None):
    ind_1 = np.array(ind_1)
    ind_2 = np.array(ind_2)
    if bounds is not None:
        # print("Normalizing with bounds:", bounds)
        bounds = np.array(bounds)
        up = bounds[1]
        low = bounds[0]
        # Normalize the vectors ind_1 and ind_2 based on the bounds
        norm_ind_1 = (ind_1 - low) * 1.0 / (up - low)
        norm_ind_2 = (ind_2 - low) * 1.0 / (up - low)

        return np.linalg.norm(norm_ind_1 - norm_ind_2)
        #return np.sum((norm_ind_1 - norm_ind_2)**2)
    else:
        return np.linalg.norm(ind_1 - ind_2)
        #return np.sum((ind_1 - ind_2)**2)

def get_similarity_individual(a, b, bounds):
    return ScenarioDistance.calculate(a.get("X")[0], b.get("X")[0], norm_bounds = bounds).vars_distance

class ScenarioDistance(BaseModel):
    continuous_vars_distance: float = 0.0
    ordinal_vars_distance: float = 0.0
    categorical_vars_distance: float = 0.0
    vars_distance: float = 0.0

    @staticmethod
    def safe_divide(numerator: float, denominator: int) -> float:
        """Safely divide, handling division by zero."""
        return numerator / max(denominator, 1)

    @classmethod
    def calculate(
        cls, a: Scenario, b: Scenario, *, use_local_embeddings: bool = True, norm_bounds: Dict[str, list] = None
    ) -> "ScenarioDistance":
        
        continuous_vars_euclidean = dist_euclidean_ind(a.continuous_vars, b.continuous_vars, bounds=norm_bounds["continuous_vars"] if norm_bounds else None)
        continuous_vars_distance = cls.safe_divide(
            continuous_vars_euclidean, len(a.continuous_vars)
        )
        
        # Ordinal variables distance
        ordinal_vars_euclidean = dist_euclidean_ind(a.ordinal_vars, b.ordinal_vars)
        # ordinal_vars_distance = cls.safe_divide(
        #     ordinal_vars_euclidean, len(a.ordinal_vars)
        # )

        # Categorical variables distance
        categorical_vars_total_distance = np.sum(
            np.array(a.categorical_vars) != np.array(b.categorical_vars)
        )
        categorical_vars_distance = cls.safe_divide(
            categorical_vars_total_distance, len(a.categorical_vars)
        )

        # Combined variable distance
        # total_distance = cls.safe_divide(
        #     continuous_vars_euclidean + ordinal_vars_euclidean + categorical_vars_total_distance,
        #     len(a.continuous_vars) + len(a.ordinal_vars) + len(a.categorical_vars)
        # )
        total_distance = continuous_vars_distance + categorical_vars_distance

        # return cls(
        #     continuous_vars_distance=continuous_vars_distance,
        #     ordinal_vars_distance=ordinal_vars_distance,
        #     categorical_vars_distance=categorical_vars_distance,
        #     vars_distance=total_distance,
        # )
        return cls(
            continuous_vars_distance=continuous_vars_euclidean,
            ordinal_vars_distance=ordinal_vars_euclidean,
            categorical_vars_distance=categorical_vars_total_distance,
            vars_distance=total_distance,
        )