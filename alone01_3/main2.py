import datetime

import numpy as np

from flask import Flask, render_template, request
import cv2

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/upload", methods=['POST'])
def upload():
    image = request.files['file']
    # print(image.filename)
    filepath = f"static/{image.filename}"
    image.save(filepath)

    img1 = cv2.imread(filepath)
    height, width = img1.shape[:2]
    # # numpy로 이미지 처리
    # npimage = cv2.imread(filepath)
    # filepath2 = f"static/np{image.filename}"
    # # npimage[:100, :100] = [0, 0, 0]
    # # npimage[101:200, 101:200] = [255, 255, 255]
    # roi = npimage[50:100, 50:100]
    # npimage[:50, :50] = roi
    # npimage[:, :, 2] = 0
    # cv2.imwrite(filepath2, npimage)
    #
    # img1 = cv2.imread(filepath)
    # filepath3 = f"static/exp_{image.filename}"
    # filepath4 = f"static/shr_{image.filename}"
    # expand = cv2.resize(img1, None, fx=1.3, fy=1.3, interpolation=cv2.INTER_CUBIC)
    # shrink = cv2.resize(img1, None, fx=0.7, fy=0.7, interpolation=cv2.INTER_AREA)
    # cv2.imwrite(filepath3, expand)
    # cv2.imwrite(filepath4, shrink)
    #
    # img2 = cv2.imread(filepath)
    # filepath5 = f"static/m_{image.filename}"
    # m = np.float32([[1, 0, 50], [0, 1, 100]])
    # mimg2 = cv2.warpAffine(img2, m, (300, 300))
    # cv2.imwrite(filepath5, mimg2)

    filepath30 = f"static/rot30_{image.filename}"
    filepath60 = f"static/rot60_{image.filename}"
    filepath90 = f"static/rot90_{image.filename}"
    filepath120 = f"static/rot120_{image.filename}"
    filepath150 = f"static/rot150_{image.filename}"
    filepath180 = f"static/rot180_{image.filename}"
    m30 = cv2.getRotationMatrix2D((width/2, height/2), 30, 1)
    m60 = cv2.getRotationMatrix2D((width/2, height/2), 60, 1)
    m90 = cv2.getRotationMatrix2D((width/2, height/2), 90, 1)
    m120 = cv2.getRotationMatrix2D((width/2, height/2), 120, 1)
    m150 = cv2.getRotationMatrix2D((width/2, height/2), 150, 1)
    m180 = cv2.getRotationMatrix2D((width/2, height/2), 180, 1)
    # print(m30)
    # print(m60)
    # print(m90)
    # print(m120)
    # print(m150)
    # print(m180)
    rot30 = cv2.warpAffine(img1, m30, (width, height))
    rot60 = cv2.warpAffine(img1, m60, (width, height))
    rot90 = cv2.warpAffine(img1, m90, (width, height))
    rot120 = cv2.warpAffine(img1, m120, (width, height))
    rot150 = cv2.warpAffine(img1, m150, (width, height))
    rot180 = cv2.warpAffine(img1, m180, (width, height))
    cv2.imwrite(filepath30, rot30)
    cv2.imwrite(filepath60, rot60)
    cv2.imwrite(filepath90, rot90)
    cv2.imwrite(filepath120, rot120)
    cv2.imwrite(filepath150, rot150)
    cv2.imwrite(filepath180, rot180)
    return render_template("print.html",
                           image=filepath,
                           # npimage=filepath2,
                           # exp=filepath3,
                           # shr=filepath4,
                           # mimg=filepath5,
                           rot30=filepath30,
                           rot60=filepath60,
                           rot90=filepath90,
                           rot120=filepath120,
                           rot150=filepath150,
                           rot180=filepath180
                           )


@app.route("/addimg", methods=['POST'])
def add():
    # img1 = cv2.imdecode(np.fromstring(request.files['file1'].read(), np.uint8), cv2.IMREAD_UNCHANGED)
    # img2 = cv2.imdecode(np.fromstring(request.files['file2'].read(), np.uint8), cv2.IMREAD_UNCHANGED)
    # height1, width1 = img1.shape[:2]
    # height2, width2 = img2.shape[:2]

    param1 = request.files['file1']
    param2 = request.files['file2']
    filepath1 = f"static/{param1.filename}_org1"
    filepath2 = f"static/{param2.filename}_org2"
    filepath3 = f"static/{param1.filename}_{param2.filename}_added"
    param1.save(filepath1)
    param2.save(filepath2)
    img1 = cv2.imread(filepath1)
    img2 = cv2.imread(filepath2)
    height1, width1 = img1.shape[:2]
    height2, width2 = img2.shape[:2]

    addcv2 = cv2.add(img1[:100, :100], img2[:100, :100])
    # cv2.imshow('test', addcv2)
    cv2.imwrite(filepath3, addcv2)
    return render_template("print.html",
                           addimg=filepath3)


app.run(host='127.0.0.1', port=5000, debug=True)
