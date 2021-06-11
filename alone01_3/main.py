from flask import Flask, render_template, request
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

app = Flask(__name__)
kn = KNeighborsClassifier()


@app.route("/")
def home():
    # parameters
    param_len = float(request.args.get("length", default=0))
    param_wei = float(request.args.get("weight", default=0))

    # 생선(도미) 데이터
    bream_length = [25.4, 26.3, 26.5, 29.0, 29.0, 29.7, 29.7, 30.0, 30.0, 30.7, 31.0, 31.0, 31.5, 32.0, 32.0, 32.0,
                    33.0, 33.0, 33.5, 33.5, 34.0, 34.0, 34.5, 35.0, 35.0, 35.0, 35.0, 36.0, 36.0, 37.0, 38.5, 38.5,
                    39.5, 41.0, 41.0]
    bream_weight = [242.0, 290.0, 340.0, 363.0, 430.0, 450.0, 500.0, 390.0, 450.0, 500.0, 475.0, 500.0, 500.0, 340.0,
                    600.0, 600.0, 700.0, 700.0, 610.0, 650.0, 575.0, 685.0, 620.0, 680.0, 700.0, 725.0, 720.0, 714.0,
                    850.0, 1000.0, 920.0, 955.0, 925.0, 975.0, 950.0]
    bream_cnt = len(bream_length)

    # 생선(빙어) 데이터
    smelt_length = [9.8, 10.5, 10.6, 11.0, 11.2, 11.3, 11.8, 11.8, 12.0, 12.2, 12.4, 13.0, 14.3, 15.0]
    smelt_weight = [6.7, 7.5, 7.0, 9.7, 9.8, 8.7, 10.0, 9.9, 9.8, 12.2, 13.4, 12.2, 19.7, 19.9]
    smelt_cnt = len(smelt_length)

    plt.xlabel('length')
    plt.ylabel('weight')
    plt.scatter(bream_length, bream_weight, marker='o')
    plt.scatter(smelt_length, smelt_weight, marker='*')

    # 도미, 빙어 데이터 통합?
    length = bream_length + smelt_length
    weight = bream_weight + smelt_weight

    fish_data = [[l, w] for l, w in zip(length, weight)]
    print(fish_data)
    # 도미(1), 빙어(0)로 정의
    fish_target = ([1] * bream_cnt) + ([0] * smelt_cnt)
    print(fish_target)
    kn.fit(fish_data, fish_target)
    kn.score(fish_data, fish_target)

    plt.scatter([param_len], [param_wei], marker='^')
    result = kn.predict([[param_len, param_wei]])
    if result == 1:
        pre_val = "도미?"
    else:
        pre_val = "빙어?"

    path = f"static/fish_img.png"
    plt.savefig(path)
    plt.close()

    return render_template("index.html",
                           fish_img=path,
                           pre_val=pre_val)


app.run(host="127.0.0.1", port=5000)
