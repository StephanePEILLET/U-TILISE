import numpy as np


class TypeConverter:
    def __init__(self):
        self._from = "float32"
        self._to = "uint8"

    def from_type(self, img_type):
        self._from = img_type
        return self

    def to_type(self, img_type):
        self._to = img_type
        return self

    def convert(self, img, threshold=0.5):
        if self._from == "float32":
            if self._to == "float32":
                return img
            elif self._to == "uint8":
                if img.max() > 1:
                    img = img.astype(np.float32) / img.max()  # normalize the data to 0 - 1
                img = np.iinfo(np.uint8).max * img  # scale by 255
                return img.astype(np.uint8)
            elif self._to == "uint16":
                if img.max() <= 1.0:
                    img = np.iinfo(np.uint16).max * img
                return img.astype(np.uint16)
            elif self._to == "bit":
                img = img > threshold
                return img.astype(np.uint8)
            else:
                return img
