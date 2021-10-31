from sklearn.ensemble import RandomForestClassifier
import cv2
import pickle
from imutils import paths
import progressbar
import numpy as np

soil_features = []
label = []


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

red_soil_path = list(paths.list_images("Web scrapping/red_soil"))
black_soil_path = list(paths.list_images("Web scrapping/black_soil"))
yellow_soil_path = list(paths.list_images("Web scrapping/yellow_soil"))

widgets = ["Extracting LBPH on Red Soil.... ", progressbar.Bar(), "", progressbar.Percentage(), "", progressbar.ETA()]
pbar = progressbar.ProgressBar(maxval=len(red_soil_path), widgets=widgets)
pbar.start()

for i, imagePath in enumerate(red_soil_path):
    image = cv2.imread(imagePath)
    feature_vectors = color_hist.describe(image)
    soil_features.append(feature_vectors)
    label.append("Red Soil")
    pbar.update(i)
pbar.finish()

widgets = ["Extracting LBPH on Black Soil.... ", progressbar.Bar(), "", progressbar.Percentage(), "", progressbar.ETA()]
pbar = progressbar.ProgressBar(maxval=len(black_soil_path), widgets=widgets)
pbar.start()

for i, imagePath in enumerate(black_soil_path):
    image = cv2.imread(imagePath)
    feature_vectors = color_hist.describe(image)
    soil_features.append(feature_vectors)
    label.append("Black Soil")
    pbar.update(i)

pbar.finish()

widgets = ["Extracting LBPH on Yellow Soil.... ", progressbar.Bar(), "", progressbar.Percentage(), "",
           progressbar.ETA()]
pbar = progressbar.ProgressBar(maxval=len(yellow_soil_path), widgets=widgets)
pbar.start()

for i, imagePath in enumerate(yellow_soil_path):
    image = cv2.imread(imagePath)
    feature_vectors = color_hist.describe(image)
    soil_features.append(feature_vectors)
    label.append("Yellow Soil")
    pbar.update(i)
pbar.finish()

rfc = RandomForestClassifier()
rfc.fit(soil_features, label)

pickle.dump(rfc, open("./model/soil_type100.pkl", 'wb'))

############################## Model Testing ######################################
# model = pickle.load(open("./model/soil_type100.pkl", 'rb'))
# test = list(paths.list_images("test"))

# for i in test:
#     image = cv2.imread(i)
#     feature_vectors = color_hist.describe(image)
#     pred = model.predict([feature_vectors])[0]
#     cv2.putText(image, pred, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#     cv2.imshow("Test", image)
#     cv2.waitKey(0)

####################################################################################



