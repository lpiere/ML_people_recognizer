import numpy as np

import tensorflow as tf
import cv2
import time
import dlib

_CLASS = ["Bruno Neutral", "Escalante Neutral", "karen_neutral"]
P = "./shape_predictor_68_face_landmarks.dat"


def load():
    file = open('try_to_know_who_is/people_reconizer.json', 'r')
    estrutura_rede = file.read()
    file.close()

    classifier = tf.keras.models.model_from_json(estrutura_rede)
    classifier.load_weights('try_to_know_who_is/people_reconizer.h5')
    return classifier


def use_predict(frame, classifier):
    test_image = frame[..., ::-1].astype(np.float32)

    test_image = tf.keras.preprocessing.image.img_to_array(test_image)
    test_image /= 255
    test_image = np.expand_dims(test_image, axis=0)

    predict = classifier.predict(test_image)

    result = np.array(predict)
    print(_CLASS[np.argmax(result)], result)


clasifier = load()
detector = dlib.get_frontal_face_detector()

cap = cv2.VideoCapture("http://192.168.100.145:8080/video")
ret, frame = cap.read()
while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces_found = detector(frame_gray, 0)
    frame = cv2.resize(frame, (64, 64))

    if len(faces_found) == 1:
        use_predict(frame, clasifier)
        print("one Face")
    elif len(faces_found) > 1:
        print("tooo many faces")
    else:
        print("where is the face?")
    
    cv2.imshow("Frame", frame)
    ret, frame = cap.read()

cap.release()
cv2.destroyAllWindows()
