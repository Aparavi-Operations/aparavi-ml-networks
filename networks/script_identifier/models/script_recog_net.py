import tensorflow as tf
from tensorflow import keras
import cv2
import numpy as np
import os
import typing
import keras_ocr

# Script Identifier Network implementation from Google's paper titled as:
# "Sequence-to-Label Script identification for Multilingual OCR"

DEFAULT_BUILD_PARAMS = {
    "height": 31,
    "width": 200,
    "color": False,
    "num_scripts": 5,
    "lr": 0.001,
}

PRETRAINED_WEIGHTS: typing.Dict[str, typing.Any] = {
    "max_sum": {
        "build_params": DEFAULT_BUILD_PARAMS,
        "weights": {
            "filename": "max_sum.h5",
        },
    }
}

def inception_module(input_tensor, d1=32, d2=64):
    """The Inception Module is a type of convolutional neural network (CNN) architecture
    that utilizes multiple kernel sizes in parallel to extract features from input images. 
    This particular implementation consists of a 1x1 convolution layer followed by two 
    parallel convolution layers with kernel sizes of (3x3) and (5x5), respectively. Additionally, 
    it includes an average pooling layer followed by a 1x1 convolution layer to extract 
    features at a different scale. The outputs of these layers are concatenated along the 
    last dimension and passed through a ReLU activation function to produce the final output 
    tensor.

    Args:
        input_tensor: A tensor of shape (batch_size, height, width, channels)
        d1: Number of filters in the 1x1 convolution layer. Default is 32.
        d2: Number of filters in the 3x3 and 5x5 convolution layers. Default is 64.

    Returns:
        output_tensor: Output tensor of shape (batch_size, height, width, 4 * d1 + d2)
    """

    conv_1x1 = tf.keras.layers.Conv2D(d1, kernel_size=(1, 1), strides=(1, 1), activation='relu6', padding='same')(input_tensor) # Define 1x1 convolution
    conv_3x3 = tf.keras.layers.Conv2D(d2, kernel_size=(3, 3), strides=(1, 1), activation='relu6', padding='same')(conv_1x1) # Define 3x3 convolution
    conv_5x5 = tf.keras.layers.Conv2D(d2, kernel_size=(5, 5), strides=(1, 1), activation='relu6', padding='same')(conv_1x1) # Define 5x5 convolution

    avg_pool_3x3 = tf.keras.layers.AveragePooling2D((3, 3), strides=(1, 1), padding='same')(input_tensor) # Define avg pooling layer
    conv_1x1_avg_pool = tf.keras.layers.Conv2D(d1, kernel_size=(1, 1), activation='relu6', padding='same')(avg_pool_3x3)

    output_tensor = tf.keras.layers.Concatenate(axis=-1)([conv_1x1, conv_3x3, conv_5x5, conv_1x1_avg_pool]) # concates along the last dimension
    output_tensor = tf.keras.layers.ReLU(6.0)(output_tensor)

    return output_tensor


def encoder_network(input_shape):
    """Constructs an encoder network that takes in an input tensor of shape 
    (batch_size, height, width, channels) and outputs a tensor of shape 
    (batch_size, 1, width/4, 48). The encoder network consists of 
    convolutional and max pooling layers, followed by several inception modules.
    The output of the final inception module is passed through several fully 
    connected layers to produce the final encoded tensor. 

    Args:
        input_shape (tuple): Shape of the input tensor.

    Returns:
        tf.keras.Model: An encoder model that takes in an input tensor and returns an encoded tensor.
    """
    input = tf.keras.layers.Input(input_shape)
    print("Encoder Input.shape: ", input.shape)

    # Convolutional layers
    inp_ops = tf.keras.layers.Conv2D(filters=64, kernel_size=(7, 7), strides=(1, 1), padding='same', 
                                    activation='relu6', input_shape=input_shape)(input)
    inp_ops = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same')(inp_ops)
    inp_ops = inception_module(inp_ops, d1=32, d2=64)
    inp_ops = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same')(inp_ops)
    inp_ops = inception_module(inp_ops, d1=48, d2=96)
    inp_ops = inception_module(inp_ops, d1=64, d2=128)

    # Fully Connected layers
    inp_ops = tf.keras.layers.Conv2D(filters=512, kernel_size=(10, 1), strides=(10, 1), padding='same', activation='relu6')(inp_ops)
    inp_ops = tf.keras.layers.Conv2D(filters=384, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu6')(inp_ops)
    inp_ops = tf.keras.layers.Conv2D(filters=256, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu6')(inp_ops)
    inp_ops = tf.keras.layers.Conv2D(filters=48, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='tanh')(inp_ops)

    encoder_model = tf.keras.Model(input, inp_ops, name='encoder')
    return encoder_model


def max_summarizer_net(input_shape, num_scripts):
    """This network is designed to summarize the output of an encoder network.
    It consists of three convolutional layers that progressively reduce the 
    spatial dimensions of the input while increasing the number of channels.
    The final layer uses global max pooling to produce a fixed-length vector
    that summarizes the input information. The output size is determined by 
    the number of scripts in the dataset. This network outputs a tensor of shape 
    (batch_size, num_scripts)

    Args:
        input_shape (tuple): Shape of the input tensor in the format (height, width, channels).
        num_scripts (int): The number of scripts to summarize into.

    Returns:
        A Keras model representing the max summarizer network.
    """
    
    inputs = tf.keras.layers.Input(input_shape)
    print("Summarizer Input shape: ", inputs.shape)
    inp_ops = tf.keras.layers.Reshape((1, input_shape[0], input_shape[1]))(inputs)
    print("After reshape: ", inp_ops.shape)
    # Convolutional layers
    inp_ops = tf.keras.layers.Conv2D(filters=64, kernel_size=(1, 3), strides=(1, 1), padding='same', activation='relu6')(inp_ops)
    inp_ops = tf.keras.layers.Conv2D(filters=32, kernel_size=(1, 5), strides=(1, 1), padding='same', activation='relu6')(inp_ops)
    inp_ops = tf.keras.layers.Conv2D(filters=num_scripts, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='linear')(inp_ops)
    # inp_ops = tf.reduce_max(inp_ops, axis=1)
    inp_ops = tf.keras.layers.GlobalMaxPooling2D()(inp_ops)
    inp_ops = tf.keras.layers.Activation(tf.keras.activations.softmax)(inp_ops)

    max_summarizer = tf.keras.Model(inputs, inp_ops, name='max_summarizer')
    return max_summarizer

def build_model(
    height,
    width,
    color,
    num_scripts,
    lr
):
    """Build a Keras CRNN model for character recognition.

    Args:
        height (int): The height of cropped images
        width (int): The width of cropped images
        color (bool): Whether the inputs should be in color (RGB)
        num_scripts (int): The number of scripts to summarize into.
        lr (float): Learning rate
    """
#     encoder = encoder_network((height, width, 3 if color else 1)) # Build the encoder model
#     encoder = tf.keras.applications.InceptionResNetV2(
#         include_top=False,
#         weights="imagenet",
#         input_shape=(height, width, 3 if color else 1),
#         pooling='max'
#     )
    recognizer = keras_ocr.recognition.Recognizer(
        alphabet="01234",
        weights='kurapan'
    )
    for layer in recognizer.backbone.layers:
        layer.trainable = False
    encoder = recognizer.backbone
    print("Output shape of the encoder: ", encoder.output_shape)
    summarizer = max_summarizer_net(encoder.output_shape[1:], num_scripts) # Build the max summarizer model
    model = tf.keras.Model(inputs=encoder.input, outputs=summarizer(encoder.output)) # Define the complete model
    return encoder, summarizer, model

class ScriptRecogNet(tf.keras.layers.Layer):
    def __init__(self, weights=None, build_params=None):
        super(ScriptRecogNet, self).__init__()
        self.build_params = build_params or DEFAULT_BUILD_PARAMS
        (
            self.encoder,
            self.summarizer,
            self.model,
        ) = build_model(**self.build_params)
        if weights is not None:
            weights_dict = PRETRAINED_WEIGHTS[weights]
            self.model.load_weights(
                filename=os.join("pre-trained-weights", weights_dict["weights"]["filename"]),
            )
            
    def get_batch_generator(self, image_generator, batch_size=8, lowercase=False):
        """
        Generate batches of training data from an image generator. The generator
        should yield tuples of (image, sentence) where image contains a single
        line of text and sentence is a string representing the contents of
        the image. If a sample weight is desired, it can be provided as a third
        entry in the tuple, making each tuple an (image, sentence, weight) tuple.

        Args:
            image_generator: An image / sentence tuple generator. The images should
                be in color even if the OCR is setup to handle grayscale as they
                will be converted here.
            batch_size: How many images to generate at a time.
            lowercase: Whether to convert all characters to lowercase before
                encoding.
        """
        if self.model is None:
            raise Exception("You must first create model.")
        while True:
            batch = [sample for sample, _ in zip(image_generator, range(batch_size))]
            images: typing.Union[typing.List[np.ndarray], np.ndarray]
            if not self.model.input_shape[-1] == 3:
                images = [
                    cv2.cvtColor(np.array(sample[0]), cv2.COLOR_RGB2GRAY)[..., np.newaxis]
                    for sample in batch
                ]
            else:
                images = [sample[0] for sample in batch]
            y = [sample[1] for sample in batch]
            images = np.array([image.astype("float32") / 255 for image in images])
            yield images, keras.utils.to_categorical(y, num_classes=self.build_params["num_scripts"])
    
    def compile(self, *args, **kwargs):
        if "optimizer" not in kwargs:
            kwargs["optimizer"] = tf.keras.optimizers.Adam(learning_rate=lr)
            print("We're here")
        if "loss" not in kwargs:
            kwargs["loss"] = 'categorical_crossentropy'
        self.model.compile(*args, **kwargs)
        