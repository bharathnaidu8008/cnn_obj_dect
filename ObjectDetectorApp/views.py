from django.core.files.storage import FileSystemStorage
from django.shortcuts import render

# Create your views here.

# -*- coding: utf-8 -*-
import os
import sys
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import shutil
import os
import torch


def load_image(filename):
    # load the image
    img = load_img(filename, target_size=(200, 200))
    # convert to array
    img = img_to_array(img)
    # reshape into a single sample with 3 channels
    img = img.reshape(1, 200, 200, 3)
    # center pixel data
    img = img.astype('float32')
    img = img - [123.68, 116.779, 103.939]
    return img


def predict(image):
    # load the image
    if image.startswith('/'):
        image = image[1:]
    img = load_image(image)
    # load model
    model = load_model(
        r'models/cat_vs_dog.h5')
    # predict the class
    result = model.predict(img)
    return result[0]


def cat_vs_dog(request):
    if request.method == 'POST':
        myfile = request.FILES['myfile']
        fs = FileSystemStorage()
        filename = fs.save(myfile.name, myfile)
        uploaded_file_url = fs.url(filename)

        predicted_value = predict(uploaded_file_url)
        if predicted_value >= 0.7 and predicted_value <= 1.0:
            label = "Dog"
        else:
            label = "Cat"

        return render(request, 'home.html', {
            'image': uploaded_file_url, 'label': label
        })
    return render(request, 'home.html')


def yolovs_object_detector(request):
    if request.method == 'POST':
        myfile = request.FILES['myfile']
        fs = FileSystemStorage()
        filename = fs.save(myfile.name, myfile)
        uploaded_file_url = fs.url(filename)

        if uploaded_file_url.startswith('/'):
            uploaded_file_url = uploaded_file_url[1:]

        model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
        img = uploaded_file_url
        results = model(img)
        results.save(os.path.split(uploaded_file_url)[0])

        return render(request, 'home_yolo.html', {
            'image': "/"+uploaded_file_url
        })
    return render(request, 'home_yolo.html')


def home(request):
    return render(request, 'app.html')
