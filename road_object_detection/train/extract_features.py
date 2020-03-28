import logging
import os
from time import time
from typing import Dict, List, Union

import numpy as np
import pandas as pd
from PIL import Image
from tensorflow.keras.applications.resnet import ResNet101, preprocess_input

LOGGER = logging.getLogger(__name__)


def images_to_arrays(input_directory: str) -> List[Dict[str, Union[str, np.array]]]:
    """Load all images and create a list of file name - image array mappings."""
    LOGGER.info("Reading files from %s.", input_directory)
    file_names = os.listdir(input_directory)
    LOGGER.info("Number of files %d.", len(file_names))

    t0 = time()
    image_arrays = []

    for file_name in file_names:
        image = Image.open(f"{input_directory}/{file_name}")
        raw_image = np.array(image)
        image_arrays.append({"file_name": file_name, "raw_image": raw_image})

    LOGGER.info("Loading images and converting to arrays took %d seconds.", time() - t0)

    return image_arrays


def preprocess_data(input_directory: str, output_file: str) -> pd.DataFrame:
    """Load all images, extract features, and store to a csv file."""
    image_arrays = images_to_arrays(input_directory)
    model = ResNet101(include_top=False, weights="imagenet")

    t0 = time()

    image_df = pd.DataFrame(image_arrays)
    image_df["prepped_input"] = preprocess_input(image_df["raw_image"].to_numpy())
    image_df["features"] = model.predict(image_df["prepped_input"].to_numpy())
    image_df.to_csv(output_file, index=False)

    LOGGER.info("Extracting features and dumping to csv took %d seconds.", time() - t0)

    return image_df
