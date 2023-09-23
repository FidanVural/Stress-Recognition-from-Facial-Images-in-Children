import cv2
import numpy as np
import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import csv

# VIDEO
# There are 52 videos in the AKTIVES data. Code runs seperately for each video and saves in different csv files.

# Capture the video
cap = cv2.VideoCapture("C:\\Users\\LENOVO\\Desktop\\AKTİVES\\Brachial Pleksus\\Azra Çelik\\CatchAPet\\2022-01-29_02-10-16_WebCam.mp4")
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Get the model path
model_path = ".//models//kids_8.h5"

# Read the model
model = tensorflow.keras.models.load_model(model_path)

emotion_dict = {0: "angry", 1: "fear", 2: "happy", 3: "neutral", 4: "sad", 5: "surprise"} 

csv_path = "C:\\Users\\LENOVO\\Desktop\\AKTİVES\\Brachial Pleksus\\Azra Çelik\\CatchAPet\\1c_kids.csv"

# Write the header
with open(csv_path, "w", newline="") as file:
    writer = csv.writer(file)
    field = ["kid_id", "disease_id", "sex", "frame_number", "prediction"]
    writer.writerow(field)

with open(csv_path, "r") as file:
    csvreader = csv.reader(file, delimiter=";")
    # Skip the header part
    next(csvreader, None)

kid_id = 3   
disease_id = 1
sex = 0 # 0-female 1-male
frame_number = 0


# Predict the label and write the kid info to thw csv file
while True:
    ret, frame = cap.read()
    
    if ret:
        
        face_rectangle = face_cascade.detectMultiScale(frame, 1.4, 7)

        for (x,y,w,h) in face_rectangle:
            cv2.rectangle(frame, (x,y), (x+w, y+h), (255,255,255), 10)
            roi = frame[y:y+h, x:x+w]
            roi = cv2.resize(roi, (48,48))
            img_pixels = image.img_to_array(roi)
            img_pixels = np.expand_dims(img_pixels, axis=0)
            
            predictions = model.predict(img_pixels)
            predictions = predictions.flatten() # [[]] output -> []
            add = [kid_id, disease_id, sex, frame_number, predictions]
            
            with open(csv_path, "a", newline="") as f:
                wr = csv.writer(f)
                wr.writerow(add)
            
            print(predictions)
            emotion_label = int(np.argmax(predictions))
            
            emotion_prediction = emotion_dict[emotion_label]
            
            cv2.putText(frame, emotion_prediction, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 1, cv2.LINE_AA)

            resize_image = cv2.resize(frame, (1000,700))
            cv2.imshow('Emotion',resize_image)
            break # tek bir yüz bulsun 
        
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
    
cap.release()
cv2.destroyAllWindows()
