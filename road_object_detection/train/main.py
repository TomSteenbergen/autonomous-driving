import logging
import sys
from pathlib import Path

import yaml

from road_object_detection.train.extract_features import preprocess_data

LOGGER = logging.getLogger(__name__)

CONFIG = yaml.safe_load(Path("road_object_detection/train/config.yaml").read_text())


def main():
    image_df = preprocess_data(CONFIG["train_data_path"], CONFIG["train_features_path"])


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stdout,
        format="%(asctime)s %(name)-4s: %(module)-4s :%(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    main()
