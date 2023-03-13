import cv2, pandas as pd, numpy, os
import pyttsx3
from datetime import date
import time
import csv

today = date.today()
#time = time.localtime()
file = open(r'C:\Users\HP\Desktop\pythonProject3\database.txt','a')

engine = pyttsx3.init('sapi5')
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[0].id)
def speak(audio):
    engine.say(audio)
    engine.runAndWait()
haar_file = 'haarcascade_frontalface_default.xml'
datasets = 'datasets'
print("1.New Entry.\n2.Face recognition.\n3.Show attendance.")
que=int(input("Enter your choice What you to do:"))
if que==1:
    name=input("Enter your name:")
    sub_data = name
    path = os.path.join(datasets, sub_data)
    if not os.path.isdir(path):
        os.mkdir(path)
    data = {
        'Name': [name],
        'Date': [today],
        #'Time': [time],
        'Present': ["P"]

    }
    df = pd.DataFrame(data)
    df.to_csv('Attendence.csv', mode='a', index=False, header=False)
    (width, height) = (130, 100)
    face_cascade = cv2.CascadeClassifier(haar_file)
    camera_port = 0
    webcam = cv2.VideoCapture(camera_port, cv2.CAP_DSHOW)

    # The program loops until it has 30 images of the face.
    count = 1
    while count < 150:
        (_, im) = webcam.read()
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 4)
        for (x, y, w, h) in faces:
            cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 2)
            face = gray[y:y + h, x:x + w]
            face_resize = cv2.resize(face, (width, height))
            cv2.imwrite('% s/% s.png' % (path, count), face_resize)
        count += 1

        cv2.imshow('OpenCV', im)
        key = cv2.waitKey(10)
        if key == 27:
            break
    speak("Your name has been recorded.")
    file.write(name)
elif que==3:
   
   with open(r'C:\Users\HP\Desktop\pythonProject3\Attendence.csv') as file:
    reader = csv.reader(file)
    for row in reader:
        print(row)
   

else:
    size = 4
    haar_file = 'haarcascade_frontalface_default.xml'
    datasets = 'datasets'

    # Part 1: Create fisherRecognizer
    print('Recognizing Face Please Be in sufficient Lights...')

    # Create a list of images and a list of corresponding names
    (images, labels, names, id) = ([], [], {}, 0)
    for (subdirs, dirs, files) in os.walk(datasets):
        for subdir in dirs:
            names[id] = subdir
            subjectpath = os.path.join(datasets, subdir)
            for filename in os.listdir(subjectpath):
                path = subjectpath + '/' + filename
                label = id
                images.append(cv2.imread(path, 0))
                labels.append(int(label))
            id += 1
    (width, height) = (130, 100)

    # Create a Numpy array from the two lists above
    (images, labels) = [numpy.array(lis) for lis in [images, labels]]

    # OpenCV trains a model from the images
    model = cv2.face.LBPHFaceRecognizer_create()
    model.train(images, labels)

    # Part 2: Use fisherRecognizer on camera stream
    face_cascade = cv2.CascadeClassifier(haar_file)
    camera_port = 0
    webcam = cv2.VideoCapture(camera_port, cv2.CAP_DSHOW)
    while True:
        (_, im) = webcam.read()
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale( gray, scaleFactor=1.1, minNeighbors=10, minSize=(50, 50) )
        for (x, y, w, h) in faces:
            cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 2)
            face = gray[y:y + h, x:x + w]
            face_resize = cv2.resize(face, (width, height))
            # Try to recognize the face
            prediction = model.predict(face_resize)
            cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 3)

            if prediction[1] < 500:

                cv2.putText(im, '% s - %.0f' %
                            (names[prediction[0]], prediction[1]), (x - 10, y - 10),
                            cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))
            else:
                cv2.putText(im, 'not recognized',
                            (x - 10, y - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))

        cv2.imshow('OpenCV', im)
        cv2.setWindowProperty('OpenCV', cv2.WND_PROP_TOPMOST, 1)
        
        key = cv2.waitKey(5)
        if key == 27:
            break
    data = {
        'Name': [names[prediction[0]]],
        'Date': [today],
        #'Time': [time],
        'Present': ["P"]
    }
    df = pd.DataFrame(data)
    df.to_csv('Attendence.csv', mode='a', index=False, header=False)
    text="welcome"+names[prediction[0]]
    speak(text)
    file.close()