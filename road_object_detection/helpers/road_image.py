from typing import Dict, List

import numpy as np
from PIL import Image, ImageDraw


class RoadImage:
    """Convenience class for working with images."""

    OBJECTS = [
        "bike",
        "bus",
        "car",
        "motor",
        "person",
        "rider",
        "traffic light",
        "traffic sign",
        "train",
        "truck",
    ]

    OBJ_TO_COLOR = {
        "bike": "black",
        "bus": "yellow",
        "car": "red",
        "motor": "blue",
        "person": "green",
        "rider": "white",
        "traffic light": "purple",
        "traffic sign": "orange",
        "train": "cyan",
        "truck": "pink",
    }

    def __init__(self, directory: str, file_name: str):
        self.directory = directory
        self.file_name = file_name
        self._image = Image.open(f"{directory}/{file_name}")
        self.raw = np.array(self._image)
        self.objects = None
        self.prediction = None
        self._metadata = None

    def set_metadata(self, label_data: List[Dict]):
        """Set the metadata of this image, such as the labels."""
        self._metadata = list(
            filter(lambda x: x["name"] == self.file_name, label_data)
        )[0]
        self.objects = list(
            filter(lambda x: x["category"] in self.OBJECTS, self._metadata["labels"])
        )

    def show(self, with_boxes: bool = False, actual: bool = True):
        """Show the image, optionally with bounding boxes."""
        if with_boxes is True:
            image = self._image.convert("RGBA")
            draw = ImageDraw.Draw(image)

            # Draw bounding boxes.
            if actual is True:
                for obj in self.objects:
                    draw.rectangle(
                        tuple(obj["box2d"].values()),
                        outline=self.OBJ_TO_COLOR[obj["category"]],
                    )
                    draw.text(
                        tuple(obj["box2d"].values())[:2], obj["category"],
                    )
            elif self.prediction:
                ...  # TODO: Plot the bounding boxes according to the prediction.
            else:
                raise ValueError(
                    "No prediction has been made for this image yet. Unable to plot "
                    "the bounding boxes."
                )

            image.show()

        else:
            self._image.show()
