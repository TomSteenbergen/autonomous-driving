import logging
import sys

from road_object_detection.load_data import load_images

LOGGER = logging.getLogger(__name__)


def main():
    directory = "data/images/100k/val"
    labels_path = "data/images/100k/labels/bdd100k_labels_images_val.json"
    images = load_images(directory, labels_path)

    # Show an example picture.
    images[2352].show(with_boxes=True, actual=True)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stdout,
        format="%(asctime)s %(name)-4s: %(module)-4s :%(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    main()
