import cv2
import numpy as np
import keras.utils as image
from tensorflow.keras.models import load_model
import datetime
import csv
import os
import re

# VIDEO

# Read the videos in the LIRIS dataset
directory = r'C:\Users\Bengi\Desktop\ABO_paper\LIRISChildrenSpontaneousFacialExpressionVideoDatabase\videos_208'                
for file_name in os.listdir(directory):
    
    if file_name[-4:] == '.mp4':
        print(file_name)


# Define the emotion name from video name
name = "S8_confusing.mp4"
emotion_name = re.search('S[0-9]+_(.+?)_[0-9]+', name)
if emotion_name:
    emotion_name = emotion_name.group(1)


cap = cv2.VideoCapture(name)

directory = emotion_name
# Check whether the specified path exists or not
isExist = os.path.exists(directory)
if not isExist:

   # Create a new directory because it does not exist
   os.makedirs(directory)


filename = 'ecrin_demir_7_Emotions.csv'

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

count = 1

# Find the faces in the video and resize it
while True:   
    ret, frame = cap.read()
    if ret:
        
        face_rectangle = face_cascade.detectMultiScale(frame, 1.3, 7)
        
        for (x,y,w,h) in face_rectangle:

            resize_image = cv2.resize(frame, (1000,700))
            cv2.imshow('Emotion',resize_image)
            
            faces = frame[y:(y+h), x:(x+w)]
            faces = cv2.resize(faces,(48,48))

            # Save the resized faces .jpg format every 5 frame.
            if(count % 5 == 0):
                name = emotion_name+str(count)+".jpg"
                cv2.imwrite(os.path.join(directory , name), faces)

            count+=1
        
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
    
cap.release()
cv2.destroyAllWindows()
