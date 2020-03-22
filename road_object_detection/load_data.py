import json
import logging
import os
from time import time
from typing import List

from road_object_detection.image import RoadImage

LOGGER = logging.getLogger(__name__)


def load_images(image_dir: str, labels_file: str) -> List[RoadImage]:
    """Load all images and set their metadata such as labels."""
    LOGGER.info("Reading files from %s", image_dir)
    file_names = os.listdir(image_dir)
    LOGGER.info("Number of files %d", len(file_names))

    with open(labels_file) as file:
        label_data = json.load(file)

    t0 = time()
    images = []
    for file_name in file_names:
        img = RoadImage(image_dir, file_name)
        img.set_metadata(label_data)
        images.append(img)

    LOGGER.info("Loading images took %d seconds.", time() - t0)

    return images
