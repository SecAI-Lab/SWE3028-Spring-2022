import numpy as np
import cv2
import torch
import models
from torchvision import transforms
import random

# This is the DEMO version of the capstone team F : AI,GO DOODLE!
# We used 50 classes for the model classification on this DEMO
#   which is trained on Quick, Draw! dataset with resnet34
# The DEMO provides three functionalities
# 1) Random generate of the KEYWORDS
# 2) User can draw his/her doodling on  the palette for the given keyword
# 3) DEMO predicts users doodling and gives its prediction and corresonding accuracy
# Version - Ubuntu 20.04.4 LTS (64 bit)

# load the pretrained model
model=models.resnet34()
model.load_state_dict(torch.load('./resexp.pt','cpu'))
model.eval()

# classes that model is trained for classification
classes=['airplane','apple','axe','banana','baseball bat','bee','boomerang','bus','cake','candle',
        'cloud','cup','diamond','dog','ear','eye','frog','frying pan','hammer','hat',
        'horse','hot dog','hourglass','house','ice cream','kangaroo','key','knife','ladder','light bulb',
        'lion','lollipop','map','megaphone','monkey','moon','mountain','mushroom','necklace','octopus',
        'palm tree','pants','peanut','parachute','pencil','rainbow','scissors','spoon','tiger','umbrella']


# print lables
print('*********KEYWORDS*********')
print(classes[:10])
print(classes[10:20])
print(classes[20:30])
print(classes[30:40])
print(classes[40:50])


# calculate result and corresonding accuracy of the model
def predict(model,x):
    model.eval()
    pred=model(x)
    result=torch.argmax(pred).item()
    exp=torch.exp(pred)
    accuracy=(torch.max(exp).item()/torch.sum(exp).item())*100
    return result,accuracy


onDown=False
xprev,yprev=None,None

# event : 'onmouse'
def onmouse(event,x,y,flags,params):
    global onDown,img,xprev,yprev
    if event==cv2.EVENT_LBUTTONDOWN:
        onDown=True
    elif event==cv2.EVENT_MOUSEMOVE:
        if onDown==True:
            cv2.line(img,(xprev,yprev),(x,y),255,10)
    elif event==cv2.EVENT_LBUTTONUP:
        onDown=False
    xprev,yprev=x,y


# create window for drawing
cv2.namedWindow("DEMO")
cv2.setMouseCallback("image",onmouse)

# create empty palette
img=np.zeros((300,300,1),np.uint8)

# random keyword generatation
keywordidx=random.randint(0,49)
print('\n'*3)
print(f'Keyword : {classes[keywordidx]}')

# start DEMO
while True:
    cv2.imshow("DEMO",img)
    key=cv2.waitKey(1)

    # clear the palette
    if key==ord('r'):
        img=np.zeros((336,336,1),np.uint8)

    # submit user answer
    elif key==ord(' '):
        #x=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        x=cv2.resize(img,dsize=(28,28))
        x=cv2.resize(x,dsize=(112,112))
        x=x.reshape((1,1,112,112))
        x=x.astype('float32')/255.0
        x=torch.from_numpy(x)
        result,accuracy=predict(model,x)
        print(f'prediction({classes[result]})  score({accuracy:.2f}) ')
        keywordidx=random.randint(0,49)
        print('*'*40)
        print()
        print('nextkeyword : '+f'{classes[keywordidx]}')
    
    # quit program
    elif key==ord('q'):
        print("Quit")
        break

cv2.destroyAllWindows()
