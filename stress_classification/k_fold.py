import csv
# Read the FER2013 data csv file
path = 'adults.csv'
with open(path, 'r') as file:
    csvreader = csv.reader(file, delimiter=';')
    # Skip the header part
    next(csvreader, None)     
    prev_emotion = "angry"
    # Number of images per label in FER2013 except "disgust" class
    emotion_dict = {
      "angry": 4953,
      "fear": 5121,
      "neutral": 6198,
      "happy": 8989,
      "sad": 6077,
      "surprise": 4002
    }
    
    j=0
    count = 0
    fold =0
    for i,row in enumerate(csvreader):
        img_name = row[0]
        
        emotion = row[1]
        # Border represents the number of elements in each fold, since we use 10-fold we divide
        # the number of samples for each class by 10
        border = int(emotion_dict[emotion]/10)
        # We increase the fold number if the count reaches the border
        if( count== border) :
            count = 0
            fold+=1
        
        # Reset the fold and count number if the label changes.
        if(prev_emotion!= emotion):
            fold = 0
            count = 0

        # Save the file with the information we get.
        with open('last_version.csv', 'a', newline='') as file_2:
            writer = csv.writer(file_2)
            data = [img_name,emotion,fold]
            writer.writerow(data)
        
        count+=1
        prev_emotion = emotion

        
file.close()
file_2.close()