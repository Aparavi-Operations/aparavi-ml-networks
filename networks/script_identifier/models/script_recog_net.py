import tensorflow as tf

# Script Identifier Network implementation from Google's paper titled as:
# "Sequence-to-Label Script identification for Multilingual OCR"

class ScriptRecogNet(tf.keras.layers.Layer):
    def __init__(self):
        super(ScriptRecogNet, self).__init__()
    
    def inception_module(self, input_tensor, d1=32, d2=64):
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
    

    def encoder_network(self, input_shape, batch_size):
        """Constructs an encoder network that takes in an input tensor of shape 
        (batch_size, height, width, channels) and outputs a tensor of shape 
        (batch_size, 1, width/4, 48). The encoder network consists of 
        convolutional and max pooling layers, followed by several inception modules.
        The output of the final inception module is passed through several fully 
        connected layers to produce the final encoded tensor. 

        Args:
            input_shape (tuple): Shape of the input tensor.
            batch_size (int): Batch size for input tensor.
            
        Returns:
            tf.keras.Model: An encoder model that takes in an input tensor and returns an encoded tensor.
        """
        input = tf.keras.layers.Input(batch_size=batch_size, shape=input_shape)
        print("Encoder Input.shape: ", input.shape)

        # Convolutional layers
        inp_ops = tf.keras.layers.Conv2D(filters=64, kernel_size=(7, 7), strides=(1, 1), padding='same', 
                                        activation='relu6', input_shape=input_shape)(input)
        inp_ops = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same')(inp_ops)
        inp_ops = self.inception_module(inp_ops, d1=32, d2=64)
        print("Inception module output_shpe for d1=32, d2=64: ", inp_ops.shape)
        inp_ops = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same')(inp_ops)
        inp_ops = self.inception_module(inp_ops, d1=48, d2=96)
        print("Inception module output_shpe for d1=48, d2=96: ", inp_ops.shape)
        inp_ops = self.inception_module(inp_ops, d1=64, d2=128)
        print("Inception module output_shpe for d1=64, d2=128: ", inp_ops.shape)

        # Fully Connected layers
        inp_ops = tf.keras.layers.Conv2D(filters=512, kernel_size=(10, 1), strides=(10, 1), padding='same', activation='relu6')(inp_ops)
        inp_ops = tf.keras.layers.Conv2D(filters=384, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu6')(inp_ops)
        inp_ops = tf.keras.layers.Conv2D(filters=256, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu6')(inp_ops)
        inp_ops = tf.keras.layers.Conv2D(filters=48, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='tanh')(inp_ops)

        encoder_model = tf.keras.Model(input, inp_ops, name='encoder')
        encoder_model.summary() 
        return encoder_model
    

    def max_summarizer_net(self, input, batch_size, num_scripts):
        """This network is designed to summarize the output of an encoder network.
        It consists of three convolutional layers that progressively reduce the 
        spatial dimensions of the input while increasing the number of channels.
        The final layer uses global max pooling to produce a fixed-length vector
        that summarizes the input information. The output size is determined by 
        the number of scripts in the dataset. This network outputs a tensor of shape 
        (batch_size, num_scripts)

        Args:
            input (tuple): Shape of the input tensor in the format (height, width, channels).
            batch_size (int): The batch size for the input tensor.
            num_scripts (int): The number of scripts to summarize into.

        Returns:
            A Keras model representing the max summarizer network.
        """
        
        inputs = tf.keras.layers.Input(batch_size=batch_size, shape=input)
        print("Summarizer Input shape: ", inputs.shape)
        print("Summarizer batch_size: ", batch_size)

        # Convolutional layers
        inp_ops = tf.keras.layers.Conv2D(filters=64, kernel_size=(1, 3), strides=(1, 1), padding='same', activation='relu6')(inputs)
        inp_ops = tf.keras.layers.Conv2D(filters=32, kernel_size=(1, 5), strides=(1, 1), padding='same', activation='relu6')(inp_ops)
        inp_ops = tf.keras.layers.Conv2D(filters=num_scripts, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='linear')(inp_ops)
        # inp_ops = tf.reduce_max(inp_ops, axis=1)
        inp_ops = tf.keras.layers.GlobalMaxPooling2D()(inp_ops)
        
        max_summarizer = tf.keras.Model(inputs, inp_ops, name='max_summarizer')
        max_summarizer.summary()
        return max_summarizer
