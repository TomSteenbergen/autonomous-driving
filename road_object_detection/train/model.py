import logging
from typing import Tuple

from tensorflow.keras import Model
from tensorflow.keras.applications.resnet import ResNet101
from tensorflow.keras.layers import Input, Convolution2D, Flatten

LOGGER = logging.getLogger(__name__)


def create_model(input_shape: Tuple[int], n_classes: int, n_anchors: int) -> Model:
    # Backbone
    inputs = Input(shape=input_shape)
    resnet = ResNet101(include_top=False, weights="imagenet")(inputs)

    # Classification subnet
    class_conv1 = Convolution2D(filters=256, kernel_size=3, activation="relu")(resnet)
    class_conv2 = Convolution2D(filters=256, kernel_size=3, activation="relu")(
        class_conv1
    )
    class_conv3 = Convolution2D(filters=256, kernel_size=3, activation="relu")(
        class_conv2
    )
    class_conv4 = Convolution2D(filters=256, kernel_size=3, activation="relu")(
        class_conv3
    )
    class_conv5 = Convolution2D(
        filters=n_classes * n_anchors, kernel_size=3, activation="relu"
    )(class_conv4)
    class_outputs = Flatten()(class_conv5)

    # Bounding box subnet
    bbox_conv1 = Convolution2D(filters=256, kernel_size=3, activation="relu")(resnet)
    bbox_conv2 = Convolution2D(filters=256, kernel_size=3, activation="relu")(
        bbox_conv1
    )
    bbox_conv3 = Convolution2D(filters=256, kernel_size=3, activation="relu")(
        bbox_conv2
    )
    bbox_conv4 = Convolution2D(filters=256, kernel_size=3, activation="relu")(
        bbox_conv3
    )
    bbox_conv5 = Convolution2D(filters=4 * n_anchors, kernel_size=3, activation="relu")(
        bbox_conv4
    )
    bbox_outputs = Flatten()(bbox_conv5)

    model = Model(inputs=inputs, outputs=[class_outputs, bbox_outputs], name="retinanet")
    LOGGER.info(model.summary())

    return model
