import cv2
import time
import os
import requests
from yolov5 import detect

CANTEEN_ID = 1
SERVER_URL = 'http://testurl.com'

cap = cv2.VideoCapture(0)

while True:
    time.sleep(5) # wait 5 seconds
    ret, frame = cap.read() # capture image

    if not ret: # break if capture was not successful
        break

    cv2.imwrite('test.png' , frame) # save captured image

    detect.run(source='test.png', save_txt=True, classes=0, conf_thres=0.5, exist_ok=True)

    num_persons = sum(1 for _ in open('./yolov5/runs/detect/exp/labels/test.txt'))
    current_timestamp = time.time()
    snapshot = {
        "date": current_timestamp,
        "count": num_persons,
        "canteen": CANTEEN_ID
    }
    print(snapshot)

    req = requests.post(SERVER_URL, json=snapshot)

    os.remove('./yolov5/runs/detect/exp/labels/test.txt')
    os.remove('test.png')

cap.release()
cv2.destroyAllWindows()

