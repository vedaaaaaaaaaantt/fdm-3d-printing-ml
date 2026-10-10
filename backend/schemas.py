
from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    printer_type: str
    material: str

    layer_thickness: float = Field(gt=0)
    infill_density: float = Field(gt=0)

    infill_pattern: str

    speed: float = Field(gt=0)
    width: float = Field(gt=0)
    thickness: float = Field(gt=0)


class PredictionResponse(BaseModel):
    predicted_peak_load_n: float

    predicted_peak_stress_kpa: float
    predicted_peak_stress_mpa: float

    predicted_elastic_modulus_mpa: float

    peak_load_model: str
    peak_stress_model: str
    elastic_modulus_model: str

    # Backward compatibility with the existing React UI.
    model: str = "Optimized XGBoost"
