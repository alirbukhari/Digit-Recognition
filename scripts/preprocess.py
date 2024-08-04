# preprocess.py
import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

def load_images_from_folder(folder, label, img_size):
    images = []
    labels = []
    for filename in os.listdir(folder):
        if filename.endswith('.jpeg') or filename.endswith('.jpg'):
            img_path = os.path.join(folder, filename)
            img = Image.open(img_path).convert('RGB')
            img = img.resize((img_size, img_size))
            img = np.array(img)
            images.append(img)
            labels.append(label)
    return images, labels

def preprocess_data(img_size=128):
    cat_images, cat_labels = load_images_from_folder('data/cats', 0, img_size)
    dog_images, dog_labels = load_images_from_folder('data/dogs', 1, img_size)
    
    X = np.array(cat_images + dog_images)
    y = np.array(cat_labels + dog_labels)
    
    X = X / 255.0
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = preprocess_data()
    np.save('X_train.npy', X_train)
    np.save('X_test.npy', X_test)
    np.save('y_train.npy', y_train)
    np.save('y_test.npy', y_test)

