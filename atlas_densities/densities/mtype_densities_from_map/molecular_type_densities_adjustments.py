from typing import Dict
from atlas_commons.typing import FloatArray


def synthesize_lamp5_molecular_type_densities(molecular_type_densities: Dict[str, FloatArray]):
    molecular_type_densities["lamp5"] = (
        molecular_type_densities["gad67"]
        - molecular_type_densities["vip"]
        - molecular_type_densities["sst"]
        - molecular_type_densities["pv"]
    )
    return molecular_type_densities


MOLECULAR_TYPE_DENSITIES_ADJUSTMENTS = [
    synthesize_lamp5_molecular_type_densities,
]
