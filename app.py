import sys
import os
import glob
import re
import numpy as np
import pickle
import keras
import keras.applications.inception_v3 as InceptionV3
from keras.models import Model
from keras.layers import Dense,Flatten
from keras.callbacks import ReduceLROnPlateau
from keras.models import load_model
from keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename

app = Flask(__name__)
model = pickle.load(open('eye.pkl', 'rb'))
MODEL_PATH ='my_model.h5'
model = load_model(MODEL_PATH)

@app.route('/')
def home():
    return render_template('index.html')


def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(256, 256))

    # Preprocessing the image
	
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    preds = model.predict(x)
	
    preds=np.argmax(preds, axis=1)
    if preds==0:
        preds="Normal"
    elif preds==1:
        preds="Catract"
    elif preds==2:
        preds="Glucoma"
    elif preds==3:
        preds="Retinal Disease"
    return preds


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = "The Predicted Outcome is"+" "+str(model_predict(file_path, model))

        return render_template('home.html',result=preds)


if __name__ == '__main__':
    app.run(debug = True)