from flask import Flask, render_template, request
import cv2

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/upload", methods=['POST'])
def upload():
    image = request.files['file']
    print(image.filename)
    filepath = f"static/{image.filename}"
    filepath2 = f"static/np{image.filename}"
    image.save(filepath)
    # numpy로 이미지 처리
    npimage = cv2.imread(filepath)
    # npimage[:100, :100] = [0, 0, 0]
    # npimage[101:200, 101:200] = [255, 255, 255]

    roi = npimage[50:100, 50:100]
    npimage[:50, :50] = roi
    npimage[:, :, 2] = 0

    cv2.imwrite(filepath2, npimage)
    return render_template("print.html", image=filepath, npimage=filepath2)


app.run(host='127.0.0.1', port=5000)
