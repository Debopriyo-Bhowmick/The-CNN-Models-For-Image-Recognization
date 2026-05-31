# Cats and Dogs Classifier

A CNN-based image classification model built using TensorFlow and Keras.

## Project Overview

This project classifies images as either:
- Cat 🐱
- Dog 🐶

The model was trained on the PetImages dataset.

## Technologies Used

- Python
- TensorFlow
- Keras
- OpenCV
- NumPy
- Matplotlib

## Dataset

Dog and Cat Classification Dataset:
https://www.kaggle.com/datasets/bhavikjikadara/dog-and-cat-classification-dataset

## Model Architecture

- Conv2D
- Batch Normalization
- MaxPooling2D
- GlobalAveragePooling2D
- Dropout
- Dense Layers

## Results

Test Accuracy: 75.58%

## Files

- Cats_And_Dogs_Classifier.ipynb
- cat_dog_model.keras

#Instruction for using only the model not compile the whole code

-First Download the cat_dog_model.keras
-Then upload that file to the google colab or any other compiler in python
-Paste this code and upload a image of Dog or cat or you can simply download from here some sample and let the code run 

    from tensorflow.keras.models import load_model
    from google.colab import files
    import cv2
    import matplotlib.pyplot as plt

    model = load_model('cat_dog_model.keras')

    uploaded = files.upload()

    for file in uploaded.keys():

        img = cv2.imread(file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
        plt.imshow(img)
        plt.show()

        img = cv2.resize(img,(256,256))

        pred = model.predict(
            img.reshape(1,256,256,3), verbose=0 )[0][0]

        if pred > 0.5:
            print("🐶 Dog")
        else:
            print("🐱 Cat")


## Author

Debopriyo Bhowmick
