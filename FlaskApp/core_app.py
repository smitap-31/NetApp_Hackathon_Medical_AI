from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
import keras.backend as K
from datetime import datetime as dt
import numpy as np
import cv2
from cv2 import resize, INTER_AREA
import uuid
from PIL import Image
import os
import tempfile
from keras.models import load_model
import imageio
from keras.preprocessing import image
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
import keras
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.imagenet_utils import decode_predictions


"""Instantiating the flask object"""
app = Flask(__name__)
CORS(app)

@app.route('/home')
def home():
    return render_template('index.html')

@app.route('/checkup')
def checkup():
    return render_template('checkup.html')

@app.route('/retino.html')
def retino():
    return render_template('retino.html')

@app.route('/index.html')
def index_from_checkup():
    return render_template('index.html')

@app.route('/checkup.html')
def checkup_from_any():
    return render_template('checkup.html')

@app.route('/about.html')
def about():
    return render_template('about.html')

@app.route("/", methods = ["POST", "GET"])
def index():
  if request.method == "POST":
    type_ = request.form.get("type", None)
    data = None
    final_json = []
    if 'img' in request.files: 
      file_ = request.files['img']
      name = os.path.join(tempfile.gettempdir(), str(uuid.uuid4().hex[:10]))
      file_.save(name)
      print("[DEBUG: %s]"%dt.now(),name,file_)

      if(type_=='dia_ret'):
        test_image = Image.open(name)                                  #Read image using the PIL library
        test_image = test_image.resize((128,128), Image.ANTIALIAS)     #Resize the images to 128x128 pixels
        test_image = np.array(test_image)                              #Convert the image to numpy array
        test_image = test_image/255                                    #Scale the pixels between 0 and 1
        test_image = np.expand_dims(test_image, axis=0)                #Add another dimension because the model was trained on (n,128,128,3)
        data = test_image
      

      model=get_model(type_)[0]

      if(type_=='dia_ret'):
         preds, pred_val = translate_retinopathy(model["model"].predict_proba(data))
         final_json.append({"empty": False, "type":model["type"], 
                            "mild":preds[0], 
                            "mod":preds[1],
                            "norm":preds[2],
                            "severe":preds[3],
                            "pred_val": pred_val})

    else:
      warn = "Feeding blank image won't work. Please enter an input image to continue."
      pred_val =" "
      final_json.append({"pred_val": warn,"para": " ","unin": " ","tumor": " ", "can":" ",
                         "normal": " ","bac": " ","viral": " ","cnv": " ","dme": " ",
                         "drusen": " ","mild": " ","mod": " ","severe": " ","norm": " ",
                         "top1": " ","top2": " ","top3": " ","top4": " ","top5": " "}) 

    K.clear_session()
    return jsonify(final_json)
  return jsonify({"empty":True})

"""This function is used to load the model from disk."""
def load_model_(model_name):
  model_name = os.path.join("static/weights",model_name)
  model = load_model(model_name)
  return model

"""This function is used to load the specific model for specific request calls. This
function will return a list of dictionary items, where the key will contain the loaded
models and the value will contain the request type."""
def get_model(name = None):
  model_name = []
  if(name=='dia_ret'):
    model_name.append({"model": load_model_("mymodel.h5"), "type": name})
  return model_name

"""preds will contain the predictions made by the model. We will take the class probabalities and 
store them in individual variables. We will return the class probabilities and the final predictions
made by the model to the frontend. The value contained in variables total and prediction will be
displayed in the frontend HTML layout."""

def translate_retinopathy(preds):
  y_proba_Class0 = preds.flatten().tolist()[0] * 100
  y_proba_Class1 = preds.flatten().tolist()[1] * 100
  y_proba_Class2 = preds.flatten().tolist()[2] * 100
  y_proba_Class3 = preds.flatten().tolist()[3] * 100

  mild="Probability of the input image to have Mild Diabetic Retinopathy: {:.2f}%".format(y_proba_Class0)
  mod="Probability of the input image to have Moderate Diabetic Retinopathy: {:.2f}%".format(y_proba_Class1)
  norm="Probability of the input image to be Normal: {:.2f}%".format(y_proba_Class2)
  severe="Probability of the input image to have Severe Diabetic Retinopathy: {:.2f}%".format(y_proba_Class3)

  total = [mild,mod,norm,severe]
  
  list_proba = [y_proba_Class0,y_proba_Class1,y_proba_Class2,y_proba_Class3]
  statements = ["Inference: The image has high evidence for Mild Nonproliferative Diabetic Retinopathy Disease.",
               "Inference: The image has high evidence for Moderate Nonproliferative Diabetic Retinopathy Disease.",
               "Inference: The image has no evidence for Nonproliferative Diabetic Retinopathy Disease.",
               "Inference: The image has high evidence for Severe Nonproliferative Diabetic Retinopathy Disease."]
  
  index = list_proba.index(max(list_proba))
  prediction = statements[index]

  return total, prediction



if __name__=="__main__":
  app.run("127.0.0.1",5000, debug = False)