# predict.py
import numpy as np
import tensorflow as tf
from PIL import Image
import sys

def load_image(img_path, img_size):
    img = Image.open(img_path).convert('RGB')
    img = img.resize((img_size, img_size))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def predict(img_path):
    model = tf.keras.models.load_model('cat_dog_classifier.h5')
    img = load_image(img_path, 128)
    prediction = model.predict(img)
    return "Dog" if prediction[0] > 0.5 else "Cat"

if __name__ == "__main__":
    img_path = sys.argv[1]
    result = predict(img_path)
    print(f"The image is a {result}")
