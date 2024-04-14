from flask import Flask,request
import tensorflow as tf
import base64
import cv2
import numpy as np
from PIL import Image, ImageOps


model = tf.keras.models.load_model("keras_model.h5")
class_names = open("labels.txt", "r").readlines()



app = Flask(__name__)



@app.route('/api',methods = ['Put'] )
def index():
       inputchar = request.get_data()
       imgdata = base64.b64decode(inputchar)
       filename = 'somthing.jpg'  
       with open(filename, 'wb') as f:
        f.write(imgdata)       
       image = Image.open("somthing.jpg").convert("RGB")
       data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
       size = (224, 224)
       image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
       image_array = np.asarray(image)
       normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
       data[0] = normalized_image_array
       prediction = model.predict(data)
       index = np.argmax(prediction)
       class_name = class_names[index]
       confidence_score = prediction[0][index]

       result = ""
       if index == 0: 
              result= f"{class_name[2:]}"
       elif index == 1:
           result=  f"{class_name[2:]}"
       elif index == 2:
           result=  f"{class_name[2:]}"
       elif index == 3:
              result=  f"{class_name[2:]}"  
       elif index == 4:
              result=  f"{class_name[2:]}" 
       else:
             result=  f"{class_name[2:]}"

    


       return result



if __name__ == "__main__":
    app.run(debug=True)





