'''
@author - Dereck Jos and Harish Natarajan
Rest API Deployed To Azure Cloud
'''

from flask import Flask, render_template, request, session
import pickle
import requests
from PIL import Image
import cv2
import numpy as np
import base64
import io
from skimage import feature


app = Flask(__name__)

app.secret_key = "secret_key"

class HSVDescriptor:
    def __init__(self, bins):
        self.bins = bins

    def describe(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        features = []

        (h, w) = image.shape[:2]
        (cX, cY) = (int(w * 0.5), int(h * 0.5))

        segments = [(0, cX, 0, cY), (cX, w, 0, cY), (cX, w, cY, h),
                    (0, cX, cY, h)]

        (axesX, axesY) = (int(w * 0.75) // 2, int(h * 0.75) // 2)
        ellipMask = np.zeros(image.shape[:2], dtype="uint8")
        cv2.ellipse(ellipMask, (cX, cY), (axesX, axesY), 0, 0, 360, 255, -1)

        for (startX, endX, startY, endY) in segments:
            cornerMask = np.zeros(image.shape[:2], dtype="uint8")
            cv2.rectangle(cornerMask, (startX, startY), (endX, endY), 255, -1)
            cornerMask = cv2.subtract(cornerMask, ellipMask)
            hist = self.histogram(image, cornerMask)
            features.extend(hist)

        hist = self.histogram(image, ellipMask)
        features.extend(hist)

        return np.array(features)

    def histogram(self, image, mask=None):
        hist = cv2.calcHist([image], [0, 1, 2], mask, self.bins,
                            [0, 180, 0, 256, 0, 256])

        hist = cv2.normalize(hist, hist).flatten()
        return hist

color_hist = HSVDescriptor((4, 6, 3))

@app.get("/")
def index():
    model_params = pickle.load(open('model/model_params.pkl', "rb"))
    N = model_params["N"]
    K = model_params["K"]
    S = model_params["S"]
    cu = model_params["cu"]
    P = model_params["P"]
    fe = model_params["fe"]
    Mn = model_params["Mn"]
    B = model_params["B"]
    ph = model_params["ph"]
    ec = model_params["ec"]
    oc = model_params["oc"]

    return render_template("soil_index_2.html", N=N, P=P, K=K, ph=ph, ec=ec, oc=oc, S=S, fe=fe, cu=cu, Mn=Mn, B=B)


@app.post("/model_pred")
def model_pred():
    N = request.form["N"]
    P = request.form["P"]
    K = request.form["K"]
    ph = request.form["ph"]
    ec = request.form["ec"]
    oc = request.form["oc"]
    S = request.form["S"]
    fe = request.form["fe"]
    cu = request.form["cu"]
    Mn = request.form["Mn"]
    B = request.form["B"]

    # response = {"Nutrients": [N, P, K, ph, ec, oc, S, fe, cu, Mn, B]}
    # pred = requests.post(url="https://rest-api-soil.azurewebsites.net/model_pred", json=response)

    X = [[N, P, K, ph, ec, oc, S, fe, cu, Mn, B]]
    model = pickle.load(open("model/model.cpickle","rb"))
    pred = model.predict(X)[0]

    # pickle.dump(pred, open("model_prediction.pkl", "wb"))
    session["pred"] = str(pred)
    pred1 = ""
    if pred == 0:
        pred1 = "Less Fertile"

    elif pred == 1:
        pred1 = "Fertile"

    else:
        pred1 = "Highly Fertile"

    return render_template("soil_prediction_2.html", pred=pred1)


@app.get("/crop_recom")
def crop_recom():
    min_rainfall = pickle.load(open("model/rainfall_stats.pkl", "rb"))["min_rainfall"]
    max_rainfall = pickle.load(open("model/rainfall_stats.pkl", "rb"))["max_rainfall"]
    return render_template("crop_recommend.html", flag=False, min_rainfall=min_rainfall, max_rainfall=max_rainfall)


@app.post("/crop_recom_res")
def crop_recom_res():
    pred = int(session["pred"])
    min_rainfall = pickle.load(open("model/rainfall_stats.pkl", "rb"))["min_rainfall"]
    max_rainfall = pickle.load(open("model/rainfall_stats.pkl", "rb"))["max_rainfall"]
    model = pickle.load(open("model/knn.cpickle", "rb"))
    fertility = pred
    crop_recommend_list = pickle.load(open("model/crop_recom.cpickle", "rb"))
    crop_recommend_list = crop_recommend_list["crop"].to_dict()
    min_rainfall_form = request.form["min"]
    max_rainfall_form = request.form["max"]
    x = [[min_rainfall_form, max_rainfall_form, fertility]]
    pred = model.predict(x)[0]

    crop = crop_recommend_list[pred]

    return render_template("crop_recommend.html", flag=True, min_rainfall=min_rainfall, max_rainfall=max_rainfall,
                           crop=crop)


@app.get("/soil_upload")
def soil_upload():
    return render_template("soil_upload.html")

@app.get("/soil_type")
def soil_type():
    return render_template("soil_type.html")

@app.post("/soil_type_response")
def soil_type_response():
    f = request.files['files']
    img = Image.open(f)
    img.save("test.jpg")
    img_soil = Image.open('test.jpg')
    data = io.BytesIO()
    img_soil.save(data, "JPEG")
    encoded_img_data = base64.b64encode(data.getvalue())
    img = np.array(img)
    img = img[:, :, ::-1]

    feature_vectors = color_hist.describe(img)
    model = pickle.load(open("./model/soil_type100.pkl", 'rb'))
    pred = model.predict([feature_vectors])[0]
    return render_template("soil_type.html", soil_type=pred, img_soil=encoded_img_data.decode('utf-8'))



if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0')
