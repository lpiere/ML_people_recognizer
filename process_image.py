import os
import dlib
import cv2

def create_path():
    dirs_raw = os.listdir("./data_raw")
    dirs = os.listdir("./data")
    for dir in dirs_raw:
        if dir in dirs:
            continue
        os.mkdir(f"data/{dir}")

def cut_face_from_dir():
    create_path()
    dirs_raw = os.listdir("./data_raw")
    for dir in dirs_raw:
        pics = os.listdir(f"./data_raw/{dir}")
        
        for pic in pics:
            img = cv2.imread(f"./data_raw/{dir}/{pic}")
            face = get_only_face(img)
            if face is None:
                print("where is the face?")
                continue

            cv2.imwrite(f"./data/{dir}/{pic}", face)
            


def get_only_face(img):
    face_detector = dlib.get_frontal_face_detector()
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rects = face_detector(img_gray, 0)
    
    if len(rects) == 0:
        return None

    for rect in rects:
        x = rect.tl_corner().x
        y = rect.tl_corner().y
        x2 = rect.br_corner().x
        y2 = rect.br_corner().y

        image_to_show = img[y:y2, x:x2]

    return image_to_show