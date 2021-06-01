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

    img1 = cv2.imread(filepath)
    filepath30 = f"static/rot30_{image.filename}"
    filepath60 = f"static/rot60_{image.filename}"
    filepath90 = f"static/rot90_{image.filename}"
    filepath120 = f"static/rot120_{image.filename}"
    filepath150 = f"static/rot150_{image.filename}"
    filepath180 = f"static/rot180_{image.filename}"
    m30 = cv2.getRotationMatrix2D((240, 150), 30, 1)
    m60 = cv2.getRotationMatrix2D((240, 150), 60, 1)
    m90 = cv2.getRotationMatrix2D((240, 150), 90, 1)
    m120 = cv2.getRotationMatrix2D((240, 150), 120, 1)
    m150 = cv2.getRotationMatrix2D((240, 150), 150, 1)
    m180 = cv2.getRotationMatrix2D((240, 150), 180, 1)
    print(m30)
    print(m60)
    print(m90)
    print(m120)
    print(m150)
    print(m180)

    rot30 = cv2.warpAffine(img1, m30, (480, 478))
    rot60 = cv2.warpAffine(img1, m60, (480, 478))
    rot90 = cv2.warpAffine(img1, m90, (480, 478))
    rot120 = cv2.warpAffine(img1, m120, (480, 478))
    rot150 = cv2.warpAffine(img1, m150, (480, 478))
    rot180 = cv2.warpAffine(img1, m180, (480, 478))
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


app.run(host='127.0.0.1', port=5000)
