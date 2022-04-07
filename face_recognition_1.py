from re import I
from deepface import DeepFace
import cv2 as cv

models = ["VGG-Face", "Facenet", "Facenet512", "OpenFace", "DeepFace", "DeepID", "ArcFace", "Dlib"]

metrics = ["cosine", "euclidean", "euclidean_l2"]

backends = ['opencv', 'ssd', 'dlib', 'mtcnn', 'retinaface', 'mediapipe']

# euclidean_l2 is more accurate based on experiences 
def verify():
    result = DeepFace.verify(img1_path = "img_5.jpg", img2_path = "./database/Kheybar/Kheybar_3.jpg", model_name=models[6],distance_metric = metrics[2])
    print(result)

def stream():
    stream_res = DeepFace.stream(db_path = "./database", enable_face_analysis = False, model_name=models[6],distance_metric = metrics[2], time_threshold = 2)
    print(stream_res)

def detect():
    img_det = DeepFace.detectFace(img_path = "img_4.jpg", detector_backend=backends[1])
    
    cv.imshow("Detected face",img_det)

    cv.waitKey(0)
    cv.destroyAllWindows()


stream()
#detect()
#verify()