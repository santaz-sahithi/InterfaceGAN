from keras.models import load_model, model_from_json
from PIL import Image
import json
import numpy as np 

# Load model from Json file 
json_file = open('model.json','r')
loaded_model = json_file.read()
json_file.close()

load_model = model_from_json(loaded_model)
load_model.load_weights('model.h5')

# Load Image 
image = Image.open('000001.jpg') ## Test Image Path
im = image.resize((200,200))

im = np.asarray(im)
im = np.reshape(im,(1,im.shape[0],im.shape[1],im.shape[2]))

# Make Prediction 
prediction = load_model.predict(im)
if prediction == 1:
  print('Real Face')
else:
  print('Fake Face')
