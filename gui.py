from tkinter import *
from PIL import Image,ImageTk
from tkinter import filedialog
from tensorflow.keras.models import model_from_json
import numpy as np
import tensorflow as tf
import cv2
import numpy as np


class FacialExpressionModel(object):

    EMOTIONS_LIST = ["Angry", "Disgust",
                    "Fear", "Happy",
                    "Neutral", "Sad",
                    "Surprise"]

    def __init__(self, model_json_file, model_weights_file):
        # load model from JSON file
        with open(model_json_file, "r") as json_file:
            loaded_model_json = json_file.read()
            self.loaded_model = model_from_json(loaded_model_json)

        # load weights into the new model
        self.loaded_model.load_weights(model_weights_file)
        self.loaded_model.make_predict_function()

    def predict_emotion(self, img):
        self.preds = self.loaded_model.predict(img)
        return FacialExpressionModel.EMOTIONS_LIST[np.argmax(self.preds)]

facec = cv2.CascadeClassifier('E:/facial_expression_gui/Facial-Expression-GUI/haarcascade_frontalface_default.xml')
model = FacialExpressionModel("E:/facial_expression_gui/Facial-Expression-GUI/model.json", "E:/facial_expression_gui/Facial-Expression-GUI/model_weights.h5")
font = cv2.FONT_HERSHEY_SIMPLEX

class VideoCamera(object):
    def __init__(self,file_name):
        self.video = cv2.VideoCapture(file_name)

    def __del__(self):
        self.video.release()

    # returns camera frames along with bounding boxes and predictions
    def get_frame(self):
        _, fr = self.video.read()
        gray_fr = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        faces = facec.detectMultiScale(gray_fr, 1.3, 5)

        for (x, y, w, h) in faces:
            fc = gray_fr[y:y+h, x:x+w]

            roi = cv2.resize(fc, (48, 48))
            pred = model.predict_emotion(roi[np.newaxis, :, :, np.newaxis])

            cv2.putText(fr, pred, (x, y), font, 1, (255, 255, 0), 2)
            cv2.rectangle(fr,(x,y),(x+w,y+h),(255,0,0),2)

        return fr

def gen(camera):
    while True:
        frame = camera.get_frame()
        cv2.imshow('Facial Expression Recognition (TO QUIT PRESS KEY "q")',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()

root = Tk()
root.title("Facial Expression Recognition")
root.iconbitmap("E:/facial_expression_gui/Facial-Expression-GUI/Images/Emoji.ico")
root.geometry("430x500")

status = Label(root,text=" Facial Expression Recognition ", bd=1,justify=CENTER,height=3,font="Helvetica 10 bold",fg="#2aa387",bg="#eec6f7")
status.grid(row=0,column=2,pady=10)

img = ImageTk.PhotoImage(Image.open("E:/facial_expression_gui/Facial-Expression-GUI/Images/image.png"))
label = Label(image=img,justify=CENTER)
label.grid(row=1,column=1,pady=20,columnspan=3)

def hover(e):
    e.widget["bg"] = "white"

def leave(e):
    e.widget["bg"] = "SystemButtonFace"

def predict(type):
    if type=="file":
        root.filename = filedialog.askopenfilename(initialdir = "C:/",title="Select a file")
    else:
        root.filename = 0
    gen(VideoCamera(root.filename))


button_file = Button(root,text="Open File",pady=20,command=lambda: predict("file"),justify=LEFT,font="Helvetica 10 bold",cursor="hand2")
button_camera = Button(root,text="Open Camera",pady=20,command=lambda: predict("camera"),justify=RIGHT,font="Helvetica 10 bold",cursor="hand2")
button_quit = Button(root,text="EXIT",command=root.quit,justify=CENTER,font="Helvetica 10 bold",cursor="hand2")

button_file.bind("<Enter>",hover)
button_quit.bind("<Enter>",hover)
button_camera.bind("<Enter>",hover)

button_file.bind("<Leave>",leave)
button_quit.bind("<Leave>",leave)
button_camera.bind("<Leave>",leave)

button_file.grid(row=2,column=1,pady=20,padx=10)
button_camera.grid(row=2,column=3,pady=20)
button_quit.grid(row=2,column=2,pady=30,padx=10)

root.mainloop()
