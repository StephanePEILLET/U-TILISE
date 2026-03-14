from typing import Dict, Type

from .ImageSeriesInterpolator import ImageSeriesInterpolator
from .utilise import UTILISE
from .utilise_multi_stream import UtiliseMultiStream

MODELS = {
    "utilise": UTILISE,
    "ImageSeriesInterpolator": ImageSeriesInterpolator,
    "utilise_multistream": UtiliseMultiStream,
}
