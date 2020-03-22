import logging
import sys
import os
from road_object_detection.image import RoadImage
import json

LOGGER = logging.getLogger(__file__)


def main():
    directory = "data/images/100k/val"
    LOGGER.info("Reading files from %s", directory)
    file_names = os.listdir(directory)
    LOGGER.info("Number of files %d", len(file_names))

    with open("data/images/100k/labels/bdd100k_labels_images_val.json") as file:
        label_data = json.load(file)

    for file_name in file_names:
        LOGGER.info(file_name)
        img = RoadImage(directory, file_name)
        img.set_metadata(label_data)
        LOGGER.info(img.raw)
        img.show(with_boxes=True)
        sys.exit()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stdout,
        format="%(asctime)s %(name)-4s: %(module)-4s :%(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    main()
