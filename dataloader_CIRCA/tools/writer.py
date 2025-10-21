import math
import os

import numpy as np
import rasterio
from pytorch_lightning.callbacks import BasePredictionWriter
from rasterio import Affine
from rasterio.enums import Resampling
from rasterio.plot import reshape_as_image
from rasterio.plot import reshape_as_raster
from rasterio.warp import aligned_target
from rasterio.windows import from_bounds
from rasterio.windows import transform
from skimage import img_as_float
from skimage.transform import resize

THRESHOLD = 0.5
GDAL_OPTIONS = {
    "compress": "LZW",
    "tiled": True,
    "blockxsize": 256,
    "blockysize": 256,
    "SPARSE_MODE": False,
}


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
                    info = np.idebug(img.dtype)  # Get the information of the incoming image type
                    img = img.astype(np.float32) / info.max  # normalize the data to 0 - 1
                img = 255 * img  # scale by 255
                return img.astype(np.uint8)
            elif self._to == "bit":
                img = img > threshold
                return img.astype(np.uint8)
            else:
                return img


from rasterio.crs import CRS


def convert_string_to_wkt(crs_string: str):
    """
    Convertit une chaîne de caractères représentant un CRS en son format WKT.

    Args:
        crs_string (str): La chaîne du CRS (ex: "EPSG:4326", "+proj=utm ...").

    Returns:
        str: Le CRS au format WKT, ou None si la conversion échoue.
    """
    try:
        # 1. Créer un objet CRS à partir de la chaîne de caractères.
        #    Cette méthode est très flexible et peut interpréter différents formats.
        crs_obj = CRS.from_string(crs_string)

        # # 2. Exporter l'objet CRS au format WKT.
        # #    Vous pouvez utiliser pretty=True pour un affichage plus lisible.
        # wkt_string = crs_obj.to_wkt(pretty=True)

        return crs_obj

    except Exception as e:
        print(f"Erreur lors de la conversion de '{crs_string}': {e}")
        return None


def write_predictions(
    batch,
    y_pred,
    output_file,
):
    """
    Write the predictions to disk with rasterio (georeferenced tiff).
    """
    print(batch)
    print(y_pred)
    output_type = "uint8"
    meta = {
        "driver": "GTiff",
        "dtype": output_type,
        "count": batch["y"].shape[1],
        "width": batch["y"].shape[3],
        "height": batch["y"].shape[2],
        "transform": Affine(*[el.numpy()[0] for el in batch["info"]["transform"]]),
    }
    with rasterio.open(output_file, "w", **meta, **GDAL_OPTIONS) as src:
        converter = TypeConverter()
        pred = converter.from_type("float32").to_type(output_type).convert(y_pred, threshold=THRESHOLD)
        src.write(pred)


class PatchPredictionWriter(BasePredictionWriter):
    def __init__(
        self,
        output_dir,
        output_type,
        write_interval,
        threshold=THRESHOLD,
        img_size_pixel=(256, 256),
        sparse_mode=False,
        gdal_options=None,
    ):
        super().__init__(write_interval)
        self.output_dir = output_dir
        self.output_type = output_type
        self.threshold = threshold
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        self.meta = None
        self.img_size_pixel = img_size_pixel
        self.sparse_mode = sparse_mode
        if gdal_options is None:
            self.gdal_options = {
                "compress": "LZW",
                "tiled": True,
                "blockxsize": None if img_size_pixel is None else self.img_size_pixel[0],
                "blockysize": None if img_size_pixel is None else self.img_size_pixel[1],
                "SPARSE_MODE": self.sparse_mode,
            }
        else:
            self.gdal_options = gdal_options

    def on_predict_start(self, trainer, pl_module):
        if self.img_size_pixel is None:
            self.img_size_pixel = (
                trainer.datamodule.sample_dims["image"][0],
                trainer.datamodule.sample_dims["image"][1],
            )
            self.gdal_options["blockxsize"] = self.img_size_pixel[0]
            self.gdal_options["blockysize"] = self.img_size_pixel[1]

        self.meta = trainer.datamodule.meta["test"]
        self.meta["driver"] = "GTiff"
        self.meta["dtype"] = "uint8" if self.output_type in ["uint8", "bit"] else "float32"
        self.meta["count"] = trainer.datamodule.num_classes
        self.meta["width"] = self.img_size_pixel[0]
        self.meta["height"] = self.img_size_pixel[1]
        if self.output_type == "bit":
            self.gdal_options["bit"] = 1
        return super().on_predict_start(trainer, pl_module)

    def write_on_batch_end(
        self,
        trainer,
        pl_module,
        prediction,
        batch_indices,
        batch,
        batch_idx,
        dataloader_idx,
    ):

        probas, filenames, affines = (
            prediction["proba"],
            prediction["filename"],
            prediction["affine"],
        )

        # Pass prediction and their transformations on CPU
        probas = probas.cpu().numpy()
        affines = affines.cpu().numpy()

        for proba, filename, affine in zip(probas, filenames, affines):
            output_file = os.path.join(self.output_dir, filename)
            self.meta["transform"] = ndarray_to_affine(affine)
            self.meta["transform"], _, _ = aligned_target(
                self.meta["transform"],
                self.meta["width"],
                self.meta["height"],
                trainer.datamodule.resolution["test"],
            )

            with rasterio.open(output_file, "w", **self.meta, **self.gdal_options) as src:
                converter = TypeConverter()
                pred = converter.from_type("float32").to_type(self.output_type).convert(proba, threshold=self.threshold)
                src.write(pred)

    def on_predict_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):

        if not self.interval.on_batch:
            return

        batch_indices = trainer.predict_loop.epoch_loop.current_batch_indices
        self.write_on_batch_end(trainer, pl_module, outputs, batch_indices, batch, batch_idx, dataloader_idx)
