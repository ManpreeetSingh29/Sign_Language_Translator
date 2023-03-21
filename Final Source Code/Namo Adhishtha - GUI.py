# -*- coding: utf-8 -*-

from tkinter import *
from PIL import Image,ImageTk

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard

from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import numpy as np
import os
import cv2
from matplotlib import pyplot as plt
import time
import mediapipe as mp

# from Sign_detection import *
import warnings
warnings.filterwarnings('ignore')

mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities

def draw_styled_landmarks(image, results):
    
    # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                             ) 
    # Draw right hand connections  
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                             ) 

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, lh, rh])

DATA_PATH = os.path.join('MP_Data') 
actions = np.array(['Namaste','Chandigarh','University'])

# actions = np.array(['Hello','Everyone','We','are','from','Chandigarh University','designed','this','project',
#                     'named','Namo Adhishta','Thankyou','Nice','to meet','you']) # Actions that we try to detect

No_sequences = 30 # 10 videos worth of data
sequence_length = 30 # Videos are going to be 30 frames in length
Start_folder = 0 # Folder start

label_map = {label:num for num, label in enumerate(actions)}
print(label_map)

model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,258)))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))

model.load_weights("Small.h5")

flag = True

def stTranslate():
    global flag
    print(flag)
    if flag==True:
        flag = False
    else:
        flag= True
        
def custWords():      
    print("Code2")

try:
    window = Tk()
    window.title("Sign Language Translator")
    window.geometry('1500x800')
    
    lbl = Label(window, text="NAMO ADHISHTHA",font=("Times New Roman", 42,"bold"),fg='Green',  pady=10, padx=10).place(x=500, y=20)
    
    cam_frame = LabelFrame(window,bg= "red").pack()
    cam_label = Label(cam_frame,bg = "green",width='700',height='500')
    cam_label.place(x=600,y=175)
    cap = cv2.VideoCapture(0)
    
    btn1 = Button(window,text='Toggle Translating',font=("Arial", 12),bg = '#728FCE',fg='white',width=30,pady=10,padx=10,command=stTranslate).place(x = 125, y=175)
    btn2 = Button(window,text='Add Customised Words',font=("Arial", 12),bg = '#728FCE',fg='white',width=30,pady=10,padx=10,command=custWords).place(x = 125, y=375)
    btn3 = Button(window,text='Exit',font=("Arial", 12),bg = '#728FCE',fg='white',width=30,pady=10,padx=10,command=lambda:[cap.release(),window.destroy()]).place(x = 125, y=550)
    sequence = []
    sentence = []
    threshold = 0.7
    while True:
        if(flag):    
            print(flag)
            image = cap.read()[1] #read() returns true if cam is open nd array returns the image
            image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
            
            
        else:
            # print(flag)
            
            # Set mediapipe model 
            with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
                # Read feed
                ret, image = cap.read()

                # Make detections
                results = holistic.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                # print(results)
                
                # Draw landmarks
                draw_styled_landmarks(image, results)
                
                # 2. Prediction logic
                keypoints = extract_keypoints(results)
                sequence.append(keypoints)
                sequence = sequence[-30:]
                
                if len(sequence) == 30:
                    res = model.predict(np.expand_dims(sequence, axis=0))[0]
                    print(actions[np.argmax(res)])
                    
                    
                #3. Viz logic
                    if res[np.argmax(res)] > threshold:
                        if len(sentence) > 0: 
                            if actions[np.argmax(res)] != sentence[-1]:
                                sentence.append(actions[np.argmax(res)])
                        else:
                            sentence.append(actions[np.argmax(res)])

                    if len(sentence) > 5: 
                        sentence = sentence[-5:]
                # print(sentence)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)     
                cv2.putText(image, ' '.join(sentence), (10,450), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        message = Label(window, text = str(sentence)).place(x = 40,y = 60)  
        image = ImageTk.PhotoImage(Image.fromarray(image))
        cam_label['image'] = image
        # cv2.putText(image, '  '.join(sentence), (10,10), cv2.FONT_HERSHEY_SIMPLEX, 10, (255, 255, 255), 2, cv2.LINE_AA)
        window.update()
    
    window.mainloop()
    
except Exception:
    print("\n The application is closed")
    