from typing import Dict, Type

from .interpolator import ImageSeriesInterpolator
from .utilise import UTILISE

MODELS: Dict[str, Type[UTILISE | ImageSeriesInterpolator]] = {
    "utilise": UTILISE,
    "ImageSeriesInterpolator": ImageSeriesInterpolator,
}
