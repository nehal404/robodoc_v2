from keras.models import load_model  
import cv2
import numpy as np


np.set_printoptions(suppress=True)


model = load_model("robodoc_v2/keras_Model.h5", compile=False)


class_names = open("robodoc_v2/labels.txt", "r").readlines()


def predict_image(image_path: str):
    """
    Takes an image path, preprocesses it, and returns the predicted class and confidence.
    """
    # Load the image from path
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image at {image_path}")

    # Resize to match the model's input shape
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)

    # Preprocess: convert to numpy array, reshape, normalize
    image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)
    image = (image / 127.5) - 1

    # Predict
    prediction = model.predict(image)
    index = np.argmax(prediction)
    class_name = class_names[index].strip()
    confidence_score = float(prediction[0][index])

    return class_name, confidence_score
