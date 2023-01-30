import cv2
import time
import datetime
import os
import glob
import requests
import shutil
from yolov5 import detect
from dotenv import load_dotenv

load_dotenv() # read environment variables

ID = int(os.environ.get("ID"))
SERVER_URL = os.environ.get("SERVER_URL")
INTERVAL_IN_SECONDS = int(os.environ.get("INTERVAL_IN_SECONDS"))
MODE = os.environ.get("MODE")

OUTPUT_FILE = os.environ.get("OUTPUT_FILE")
OUTPUT_IMAGE = os.environ.get("OUTPUT_IMAGE")
SNAPSHOT_NAME = os.environ.get("SNAPSHOT_NAME")
IMAGES_DIRECTORY = os.environ.get("IMAGES_DIRECTORY")

counter = 1

print('------- Program starting -------')

files = glob.glob('images/*.png')

# clear images from previous run
for f in files:
    try:
        os.remove(f)
    except OSError as e:
        print("Error: %s : %s" % (f, e.strerror))

try:
    while True:
        print("------- Iteration %d starting -------" % counter)
        counter += 1

        cap = cv2.VideoCapture(0)
        time.sleep(2)
        ret, frame = cap.read() # capture image

        if not ret: # break if capture was not successful
            break

        cv2.imwrite(SNAPSHOT_NAME, frame) # save captured image
        print('Image captured')

        cap.release()
        cv2.destroyAllWindows()

        # run person detection script on captured image
        # source - the image to analyse
        # save_txt - whether coordiantes of detected objects should be saved in a txt file
        # classes - which classes should be detected in the image - 0 belongs to the 'person' class
        # conf_thresh - threshold for when an object is labeled as its class - 0.3 means 30% confidence
        # exist_ok - whether the files from previous runs should be overwritten
        # weights - indicates model size, the yolov5n is the smallest model for the fastest inference
        # imgsz - determines input image size, smaller image sizes lead to faster preprocessing and inference
        detect.run(source=SNAPSHOT_NAME, save_txt=True, classes=0, conf_thres=0.3, exist_ok=True, weights='yolov5/yolov5n.pt', imgsz=(320,320))

        num_persons = 0
        coordinates = []
        try:
            # open output file which contains information about the detected people
            with open(OUTPUT_FILE, 'r') as f:
                while True:
                    # each detected person's informationn is saved on a separate line in the file
                    line = f.readline()
                    if not line:
                        break
                    num_persons += 1
                    
                    # read and save the coordinates, detected for the person in the format is (l, x, y, w, h):
                    # l - the label of the detecte object - in our case it is always 0 - a person
                    # x - the x coordinate of the edge of the detected object
                    # y - the y coordinate of the edge of the detected object
                    # w - the width of the detected object, starting from x
                    # h - the height of the detect object, starting from y
                    # each of the x, y, w, h is a decimal from 0 to 1, representing a proportion of the width/height of the photo
                    coordinates_arr = line.split()
                    coordinates_snapshot = {
                        "x": coordinates_arr[1],
                        "y": coordinates_arr[2],
                        "w": coordinates_arr[3],
                        "h": coordinates_arr[4]
                    }
                    coordinates.append(coordinates_snapshot)
                    
                print('People detected:', num_persons)
                f.close()
        except:
            print("No people detected")

        
        current_date = datetime.datetime.now()

        # get current timestamp in the YYYY-mm-ddTHH:MM:SS.msZ format
        current_timestamp = current_date.strftime("%Y-%m-%dT%H:%M:%S.%fZ")

        weekdays = dict([(0, 'Monday'), (1, 'Tuesday'), (2, 'Wednesday'), (3, 'Thursday'), (4, 'Friday'), (5, 'Saturday'), (6, 'Sunday')])

        # map the integer return value of the current weekday to string
        current_weekday = weekdays[current_date.weekday()]

        current_hour = current_date.hour

        snapshot = {
            "date": current_timestamp,
            "count": num_persons,
            "entity_ID": ID,
            "coordinates": coordinates,
            "weekday": current_weekday,
            "hour": current_hour
        }
        print(snapshot)

        url = SERVER_URL + "/" + MODE
        print("Sending data to", url)
        # send snapshot to the server
        try:
            res = requests.post(url, json=snapshot)
            print("Request was successful")
        except:
            print("Server not reachable")
        
        COPIED_IMAGE_LOCATION = IMAGES_DIRECTORY + "/snapshot_" + str(counter - 1) + ".png"

        shutil.copyfile(OUTPUT_IMAGE, COPIED_IMAGE_LOCATION)

        # cleanup old files
        if num_persons > 0:
            os.remove(OUTPUT_FILE)
        
        os.remove(SNAPSHOT_NAME)

        time.sleep(INTERVAL_IN_SECONDS) # wait a specific interval seconds
except KeyboardInterrupt:
    print('\nProgram interrupted')

cap.release()
cv2.destroyAllWindows()
