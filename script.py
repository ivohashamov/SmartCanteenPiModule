import cv2
import time
import os
import requests
from yolov5 import detect

CANTEEN_ID = 1
SERVER_URL = 'http://testurl.com'
INTERVAL_IN_SECONDS = 5


cap = cv2.VideoCapture(0)

while True:
    time.sleep(INTERVAL_IN_SECONDS) # wait a specific interval seconds
    ret, frame = cap.read() # capture image

    if not ret: # break if capture was not successful
        break

    cv2.imwrite('test.png' , frame) # save captured image

    # run person detection script on captured image
    # source - the image to analyse
    # save_txt - whether coordiantes of detected objects should be saved in a txt file
    # classes - which classes should be detected in the image - 0 belongs to the 'person' class
    # conf_thresh - threshold for when an object is labeled as its class - 0.5 means 50% confidence
    # exist_ok - whether the files from previous runs should be overwritten
    detect.run(source='test.png', save_txt=True, classes=0, conf_thres=0.5, exist_ok=True)

    # count how many lines are in the detection file
    # this is equivalent to the number of people detected
    num_persons = sum(1 for _ in open('./yolov5/runs/detect/exp/labels/test.txt'))
    current_timestamp = time.time()
    snapshot = {
        "date": current_timestamp,
        "count": num_persons,
        "canteen": CANTEEN_ID
    }
    print(snapshot)

    # send snapshot to the server
    req = requests.post(SERVER_URL, json=snapshot)

    # cleanup old files
    os.remove('./yolov5/runs/detect/exp/labels/test.txt')
    os.remove('test.png')

cap.release()
cv2.destroyAllWindows()

