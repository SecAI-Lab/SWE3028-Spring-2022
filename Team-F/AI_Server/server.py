from flask import Flask, request, make_response, jsonify, render_template
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import base64
import io
from PIL import Image
import cv2
import numpy as np
import requests
import json
import cgi
import torch
import asyncio
import utils
import models
from io import StringIO


def threshold1(pix):
    if pix>=0:
        return pix
    else:
        return 0

def threshold2(pix,len):
    if pix<len:
        return pix
    else:
        return len

def get_keywords_and_models(path='./',num_models=4):
    keywords_list=[]
    models_list=[]

    for n in range(num_models):
        # get keywords
        keywords_path=path+f'categories{n+1}.txt'

        with open(keywords_path,'r') as f:
            keywords=f.readlines()

        keywords=list(map(lambda s:s.rstrip(),keywords))
        keywords_list.append(keywords)

        # get models
        pretrained_path=path+f'model_params{n+1}.pt'

        model=models.resnet34()
        model.load_state_dict(torch.load(pretrained_path,'cpu'))
        models_list.append(model)

    return keywords_list,models_list

def preprocess(user_image):
    src = cv2.imread(user_image, cv2.IMREAD_GRAYSCALE)
    cv2.imwrite(user_image, src)
    img = np.array(Image.open(user_image))

    image=255-img

    # check if the user_image is empty
    if np.sum(np.array(image))==0:
        return None

    res,thr=cv2.threshold(image,127,255,cv2.THRESH_BINARY)
    contours,_=cv2.findContours(thr,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    cnt=contours[0]
    x,y,w,h=cv2.boundingRect(cnt)

    b_h,b_w=image.shape

#    image=image[y:y+h,x:x+w]

    if w>=h: # 가로가 더 긴 경우 -> 세로로 bounding box 확장
        extend=int((w-h)/2)
        image=image[threshold1(y-extend):threshold2(y+h+extend,b_h),x:x+w]
    else: # 세로가 더 긴 경우 -> 가로로 bounding box 확장
        extend=int((h-w)/2)
        image=image[y:y+h,threshold1(x-extend):threshold2(x+w+extend,b_w)]

    image=cv2.resize(image,dsize=(32,32))
    image=np.array(image)
    
    image=image.reshape(1,1,32,32)
    image=image.astype('float32')/255.0
    image=torch.from_numpy(image)

    return image


def accuracy_for_keyword(keyword,user_image,keywords_list,models_list):
    model_idx=0
    key_idx=0

    for i,keywords in enumerate(keywords_list):
        if keyword in keywords:
            model_idx=i
            key_idx=keywords.index(keyword)
            break

    model=models_list[model_idx]
    model.eval()

    x=preprocess(user_image)

    if x==None:
        return 0.0

    y=model(x)

    y=y.detach().numpy()

    exp_y=np.exp(y[0])
    exp_usr=exp_y[key_idx]
    acc=exp_usr/np.sum(exp_y)*100

    return acc


app = Flask(__name__)
keywords_list,models_list = get_keywords_and_models()

# 3) 위의 predict() 함수에 1)에서 선언한 model과 서버로 받은 image를 대입하면 됨
# pred,acc=predict(model,image)


@app.route('/', methods=['GET'])
def main():
    return render_template('index_text.html')

@app.route('/', methods=['POST'])
def get_image():
    input_uri=request.form["name"] 
    base64_string = input_uri.split(',')[1]
    imgdata = base64.b64decode(base64_string)
    imagefile = 'new_image.jpg'
    image_path = "./images/" + imagefile
    with open(image_path, 'wb') as f:
        f.write(imgdata)


    keyword = "clock"

    keywords_list,models_list = get_keywords_and_models()
    acc = accuracy_for_keyword(keyword, image_path, keywords_list, models_list)
    
    return render_template('index_text.html', prediction=acc)


#input이 keyword, userlist [nickname, image]의 json 일 때
@app.route('/get/userlist', methods=['POST'])
def get_userlist():
    input = request.get_json(silent=True)

    keywords_list,models_list = get_keywords_and_models()

    dic = {}
    dic['result'] = []

    keyword = input["keyword"]
    userlist = input["users"]
    for user in userlist:
        nickname = user['nickname']
        image = user['image']
        base64_string = image.split(',')[1]
        imgdata = base64.b64decode(base64_string)

        io = StringIO()
        print(f'new_image_{nickname}.jpg', file=io, end="")
        imagefile =  io.getvalue()

        image_path = "./images/" + imagefile
        with open(image_path, 'wb') as f:
            f.write(imgdata)

        acc = accuracy_for_keyword(keyword, image_path, keywords_list, models_list)

        dic['result'].append({
            "nickname": nickname,
            "predict": keyword,
            "accuracy": int(acc),
            "rank": 0
        })

    result = sorted(dic['result'], key=lambda k: k['accuracy'], reverse = True)

    if result[0]["accuracy"] != result[-1]["accuracy"]:
        i = 1
        point = result[0]["accuracy"]
        for item in result:
            if item["accuracy"] == point:
                item["rank"] = i
            else:
                i += 1
                item["rank"] = i
                point = item["accuracy"]

    output = {}
    output['result'] = result

    return jsonify(output)


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)