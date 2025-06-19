
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


captcha_path = os.path.join(os.getcwd(), "CaptchaImages")

CHARS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
CHAR_TO_LABEL = {char: idx for idx, char in enumerate(CHARS)}
LABEL_TO_CHAR = {idx: char for char, idx in CHAR_TO_LABEL.items()}

def checkImgDims():

    image_dim=None
    misaligned=False

    for filename in os.listdir(captcha_path):

        if filename.lower().endswith((".png", ".jpg", ".jpeg")):

            image_path = os.path.join(captcha_path, filename)
            read_image = cv2.imread(image_path)

        if image_dim is None:
            image_dim = read_image.shape

        if read_image.shape != image_dim:
            misaligned=True

    if misaligned:
        print("Not all images have the same dimensions.")
    else:
        print("All images have the same dimensions.")


def preprocess_and_save():


    processed_images=[]
    labels=[]

    for filename in (os.listdir(captcha_path)):


        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):

            full_path = os.path.join(captcha_path, filename)
            read_image = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
            if read_image is None:
                print("Error:Couldn't load image.")
                continue

            read_image = read_image.astype(np.float32) / 255.0
            read_image = np.expand_dims(read_image, axis=-1)

            processed_images.append(read_image)

            string_label=filename.split(".")[0]
            label=encode_label(string_label)

            labels.append(label)


    processed_images = np.array(processed_images)
    labels=np.array(labels)

    np.save("X_data.npy", processed_images)
    np.save("y_labels.npy", labels)




def encode_label(string_label):
    return [CHAR_TO_LABEL[char] for char in string_label]




preprocess_and_save()
