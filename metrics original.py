import cv2
import time
import numpy as np
from datetime import datetime
from ultralytics import YOLO
from sort import Sort

# BLUE_LINE = [(320, 190), (600, 190)]
# GREEN_LINE = [(270, 220), (580, 220)]
# RED_LINE = [(220, 250), (560, 250)]

BLUE_LINE = [(450, 350), (850, 350)]
GREEN_LINE = [(400, 400), (900, 400)]
RED_LINE = [(350, 450), (950, 450)]

cross_blue_line = {}
cross_green_line = {}
cross_red_line = {}

avg_speeds = {}

VIDEO_FPS = 20
FACTOR_KM = 3.6
LATENCY_FPS = 7

def euclidean_distance(point1: tuple, point2: tuple):
    x1, y1 = point1
    x2, y2 = point2
    distance = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
    return distance

def calculate_avg_speed(track_id):
    time_bg = (cross_green_line[track_id]["time"] - cross_blue_line[track_id]["time"]).total_seconds()
    time_gr = (cross_red_line[track_id]["time"] - cross_green_line[track_id]["time"]).total_seconds()

    distance_bg = euclidean_distance(cross_green_line[track_id]["point"], cross_blue_line[track_id]["point"])
    distance_gr = euclidean_distance(cross_red_line[track_id]["point"], cross_green_line[track_id]["point"])

    speed_bg = round((distance_bg / (time_bg * VIDEO_FPS)) * (FACTOR_KM * LATENCY_FPS), 2)
    speed_gr = round((distance_gr / (time_bg * VIDEO_FPS)) * (FACTOR_KM * LATENCY_FPS), 2)

    return round((speed_bg + speed_gr) / 2, 2)


if __name__ == '__main__':
    cap = cv2.VideoCapture("traffic3.mp4")

    model = YOLO("yolov8n.pt")

    tracker = Sort()

    while cap.isOpened():
        status, frame = cap.read()

        if not status:
            break

        results = model(frame, stream=True)

        for res in results:
            filtered_indices = np.where((np.isin(res.boxes.cls.cpu().numpy(), [2,3,5,2])) & (res.boxes.conf.cpu().numpy() > 0.3))[0]
            boxes = res.boxes.xyxy.cpu().numpy()[filtered_indices].astype(int)

            # SORT
            tracks = tracker.update(boxes)
            tracks = tracks.astype(int)

            for xmin, ymin, xmax, ymax, track_id in tracks:
                # Get buttom center of bbox
                xc, yc = int((xmin + xmax) / 2), ymax

                if track_id not in cross_blue_line:
                    cross_blue = (BLUE_LINE[1][0] - BLUE_LINE[0][0]) * (yc - BLUE_LINE[0][1]) - (BLUE_LINE[1][1] - BLUE_LINE[0][1]) * (xc - BLUE_LINE[0][0])
                    if cross_blue>= 0:
                        cross_blue_line[track_id] = {
                            "time": datetime.now(),
                            "point": (xc, yc)
                        }

                elif track_id not in cross_green_line and track_id in cross_blue_line:
                    cross_green = (GREEN_LINE[1][0] - GREEN_LINE[0][0]) * (yc - GREEN_LINE[0][1]) - (GREEN_LINE[1][1] - GREEN_LINE[0][1]) * (xc - GREEN_LINE[0][0])
                    if cross_green>= 0:
                        cross_green_line[track_id] = {
                            "time": datetime.now(),
                            "point": (xc, yc)
                        }

                elif track_id not in cross_red_line and track_id in cross_green_line:
                    cross_red = (RED_LINE[1][0] - RED_LINE[0][0]) * (yc - RED_LINE[0][1]) - (RED_LINE[1][1] - RED_LINE[0][1]) * (xc - RED_LINE[0][0])
                    if cross_red>= 0:
                        cross_red_line[track_id] = {
                            "time": datetime.now(),
                            "point": (xc, yc)
                        }

                        avg_speed = calculate_avg_speed(track_id)
                        avg_speeds[track_id] = f"{avg_speed} Km/h"

                if track_id in avg_speeds:
                    cv2.putText(img=frame, text=avg_speeds[track_id], org=(xmin, ymin-10), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1, color=(0,255,0), thickness=2)

                cv2.circle(img=frame, center=(xc, yc), radius=5, color=(0,255,0), thickness=-1)
                cv2.rectangle(img=frame, pt1=(xmin, ymin), pt2=(xmax, ymax), color=(255, 255, 0), thickness=2)

        cv2.line(frame, BLUE_LINE[0], BLUE_LINE[1], (255,0,0), 3)
        cv2.line(frame, GREEN_LINE[0], GREEN_LINE[1], (0,255,0), 3)
        cv2.line(frame, RED_LINE[0], RED_LINE[1], (0,0,255), 3)

        cv2.imshow("frame", frame)
        
        if cv2.waitKey(1) &0xFF == ord("q"):
            break

    cap.release()
