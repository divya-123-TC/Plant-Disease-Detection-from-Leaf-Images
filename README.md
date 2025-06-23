# Plant-Disease-Detection-from-Leaf-Images

# 🌿 Plant Disease Detection from Leaf Images using Deep Learning

This project is a deep learning-based application designed to detect plant diseases from leaf images using a Convolutional Neural Network (CNN). The goal is to support early diagnosis and improve crop management in agriculture.

 Project Overview

Plant disease can severely affect agricultural productivity. Traditional methods of disease detection are manual and time-consuming. This project uses image processing and deep learning to classify diseases in plant leaves and provides an easy-to-use web interface built with Streamlit.


## Features

- Detects plant diseases from leaf images
- Trained on the PlantVillage dataset (Kaggle)
- Convolutional Neural Network (CNN) model using Keras & TensorFlow
- Streamlit-based web interface for real-time predictions
- Supports image upload and live prediction
- High accuracy and fast processing



## Tools & Technologies

- *Language:* Python  
- *Libraries:* TensorFlow, Keras, OpenCV, NumPy, Pillow, Streamlit  
- *IDE:* Visual Studio Code  
- *Dataset:* [PlantVillage Dataset (Kaggle)](https://www.kaggle.com/datasets/emmarex/plantdisease)

---

## Project Structure

```bash
├── model/
│   └── plant_disease_model.h5        # Trained CNN model
├── app/
│   ├── main.py                        # Streamlit web app
│   └── utils.py                       # Image preprocessing and helper functions
├── dataset/
│   ├── train/                         # Training images
│   └── test/                          # Testing images
├── README.md
└── requirements.txt
