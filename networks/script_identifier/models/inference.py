import cv2
import numpy as np
import tensorflow as tf
    
def predict_script():
    language_names = {0: 'Arabic', 1: 'Bengali', 2: 'Chinese', 3: 'Cyrillic', 4: 'Latin'}

    image = cv2.imread('E:/test_dataset/tf_ocr_test/4.png') # Load the image and preprocess it
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # Convert the image to grayscale
    
    resized = cv2.resize(gray, (224, 40)) # Resize the image to the expected size
    normalized = resized.astype(np.float32) / 255.0 # Normalize the pixel values to be between 0 and 1
    print("Normalized shape:" , normalized.shape)
    reshaped = np.expand_dims(normalized, axis=(0, 3)) # Add a channel dimension to the image to match the expected shape
    print("image_shape:" , reshaped.shape) # (1, 40, 224, 1)

    model = tf.keras.models.load_model('./pre-trained-weights/scriptRecogNet_model_weights.h5')

    # Make predictions
    logits = model.predict(reshaped)

    predicted_class = np.argmax(logits, axis=1)[0]

    # Map the predicted class index to a language name
    predicted_language = language_names[predicted_class]
    print("Predicted Language: ", predicted_language)
    

# Main entry to run this standalone python script
if __name__ == '__main__':
    predict_script()