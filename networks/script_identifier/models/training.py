import os
import sklearn
import numpy as np
import tensorflow as tf

from sklearn.preprocessing import LabelEncoder
from networks.script_identifier.models.script_recog_net import ScriptRecogNet

class TrainPipeline():
    def __init__(self, data_path):
        super(TrainPipeline, self).__init__()
        self.model = None
        self.data_path = data_path
        self.checkpoint_path = "./pre-trained-weights"
        
    def load_dataset(self):
        """
        Load data from the specified directory, where each language has a subdirectory containing
        a 'train' and 'test' folder with images.

        language labels = {0: 'Arabic', 1: 'Bengali', 2: 'Chinese', 3: 'Cyrillic', 4: 'Latin'}

        Returns:
        - train_images: a list of training images
        - train_labels: a list of training labels
        - test_images: a list of testing images
        - test_labels: a list of testing labels
        """

        train_data = []
        train_labels = []
        test_data = []
        test_labels = []

        for lang in os.listdir(self.data_path):
            print("Current Lang fodler: ", lang)
            script_dir = os.path.join(self.data_path, lang)
            if os.path.isdir(script_dir):
                # Load training images and labels
                train_dir = os.path.join(script_dir, 'train')
                for filename in os.listdir(train_dir):
                    img_path = os.path.join(train_dir, filename)
                    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(40, 224), color_mode='grayscale')
                    img = tf.keras.preprocessing.image.img_to_array(img)
                    # img = img.astype('float32') / 255.0
                    train_data.append(img)
                    train_labels.append(lang)

                # Load testing images and labels
                test_dir = os.path.join(script_dir, 'test')
                for filename in os.listdir(test_dir):
                    img_path = os.path.join(test_dir, filename)
                    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(40, 224), color_mode='grayscale')
                    img = tf.keras.preprocessing.image.img_to_array(img)
                    test_data.append(img)
                    test_labels.append(lang)

        # perform one hot encoding of labels
        label_encoder = LabelEncoder()
        train_labels = label_encoder.fit_transform(train_labels)
        test_labels = label_encoder.transform(test_labels)

        num_classes = len(label_encoder.classes_)
        train_labels = tf.keras.utils.to_categorical(train_labels, num_classes=num_classes)
        test_labels = tf.keras.utils.to_categorical(test_labels, num_classes=num_classes)
        print("Shape after one-hot encoding: " , train_labels.shape)

        return train_data, train_labels, test_data, test_labels

    def build_model(self, batch_size):
        input_shape = (40, 224, 1)  # Define input shape of images
        num_scripts = 5 # Define number of scripts to recognize (Arabic, Bengali, Cyrillic, Chinese, Latin)
        net = ScriptRecogNet() # Instantiate the network 
        encoder = net.encoder_network(batch_size=batch_size, input_shape=input_shape) # Build the encoder model
        print("Output shape of the encoder: ", encoder.output_shape )
        max_summarizer = net.max_summarizer_net(encoder.output_shape[1:], batch_size, num_scripts) # Build the max summarizer model
        self.model = tf.keras.Model(inputs=encoder.input, outputs=max_summarizer(encoder.output)) # Define the complete model
        self.model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), metrics=['accuracy'])

    def train(self, epochs, batch_size):
        train_data, train_labels, _, _ = self.load_dataset() # Load train data and label
        self.build_model(batch_size) # Build the model

        print("Train Data.shape: " , len(train_data))
        print("Train Data length: " , len(train_labels))
        print("Train data[0].shape : ", train_data[0].shape)
        
        # convert to numpy arrays
        train_data = np.array(train_data)
        train_labels = np.array(train_labels)
        
        # Check the shapes of train_data and train_labels
        print("Shape of training data: ", train_data.shape)  # should be (n, 40, 224, 3)
        print(train_labels.shape)  # should be (n,)

        # Train the model
        history = self.model.fit(train_data, train_labels, epochs=epochs, batch_size=batch_size)

        # Save the trained model weights
        if not os.path.exists(self.checkpoint_path):
            os.mkdir(self.checkpoint_path)

        self.model.save('./pre-trained-weights/scriptRecogNet_model_weights.h5')
        

    def test(self):
        _, _, test_data, test_labels = self.load_dataset() # Load test data and label
        test_data = np.array(test_data)
        test_labels = np.array(test_labels)

        # Check the shapes of train_data and train_labels
        print("Shape of test data: ", test_data.shape)  # should be (n, 40, 224, 3)
        print(test_labels.shape)  # should be (n,)

        loss, accuracy = self.model.evaluate(test_data, test_labels) # Evaluate model
        print('Test loss:', loss)
        print('Test accuracy:', accuracy)
