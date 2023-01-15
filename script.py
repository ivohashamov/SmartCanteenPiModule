import cv2
import time
import datetime
import os
import requests
from yolov5 import detect
from dotenv import load_dotenv

load_dotenv() # read environment variables

CANTEEN_ID = int(os.environ.get("CANTEEN_ID"))
SERVER_URL = os.environ.get("SERVER_URL")
INTERVAL_IN_SECONDS = int(os.environ.get("INTERVAL_IN_SECONDS"))
MODE = os.environ.get("MODE")

OUTPUT_FILE = os.environ.get("OUTPUT_FILE")
SNAPSHOT_NAME = os.environ.get("SNAPSHOT_NAME")

print('Program starting')

cap = cv2.VideoCapture(0)

try:
    while True:
        time.sleep(INTERVAL_IN_SECONDS) # wait a specific interval seconds
        ret, frame = cap.read() # capture image

        if not ret: # break if capture was not successful
            break

        cv2.imwrite(SNAPSHOT_NAME, frame) # save captured image

        # run person detection script on captured image
        # source - the image to analyse
        # save_txt - whether coordiantes of detected objects should be saved in a txt file
        # classes - which classes should be detected in the image - 0 belongs to the 'person' class
        # conf_thresh - threshold for when an object is labeled as its class - 0.5 means 50% confidence
        # exist_ok - whether the files from previous runs should be overwritten
        detect.run(source='snapshot.png', save_txt=True, classes=0, conf_thres=0.5, exist_ok=True)

        # count how many lines are in the detection file
        # this is equivalent to the number of people detected
        num_persons = 0
        try:
            with open(OUTPUT_FILE, 'r') as f:
                num_persons = sum(1 for _ in f)
                print('People detected:', num_persons)
        except:
            print("No people detected")

        # get current timestamp in the YYYY-mm-ddTHH:MM:SS.msZ format
        current_timestamp = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%fZ")

        snapshot = {
            "date": current_timestamp,
            "count": num_persons,
            "canteen_ID": CANTEEN_ID,
            "mode": MODE
        }
        print(snapshot)

        # send snapshot to the server
        req = requests.post(SERVER_URL, json=snapshot)

        # cleanup old files
        if num_persons > 0:
            os.remove(OUTPUT_FILE)
        
        os.remove(SNAPSHOT_NAME)
except KeyboardInterrupt:
    print('\nProgram interrupted')

cap.release()
cv2.destroyAllWindows()

