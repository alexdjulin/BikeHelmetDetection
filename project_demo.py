# -*- coding: utf-8 -*-
"""
Created on May 2024

@author: Alexandre Donciu-Julin
https://github.com/alexdjulin/BikeHelmetDetection

The goal of this project file is to demonstrate the combination of different YOLOv8 model for
detecting cyclists wearing a helmet or not in a video file. By sending them a visual feedback
(red or green smiley face), we aim at sensibilizing them to the importance of wearing a helmet.

"""

# %% Module imports

import os
import cv2
from ultralytics import YOLO
import threading
from statistics import mean
from datetime import datetime, timedelta

# %% Settings and constants
video_w = 640  # video width
video_h = 360  # video height
video_fps = 30  # video frames per second

tracking_confidence = 0.25  # minimum confidence to consider a detection
person_confidence = 0.25  # minimum confidence to consider a person detection
bike_confidence = 0.25  # minimum confidence to consider a bike detection
helmet_confidence = 0.75  # minimum confidence to consider a helmet detection

debug = False  # display all bounding boxes on the video
debug_color = (255, 255, 255)  # white

combined_box = True  # display the combined bounding box on the video
combined_box_padding = 3  # padding around the combined bounding box
red_color = (0, 255, 0)  # green
green_color = (0, 0, 255)  # red

show_fps = False  # display the FPS counter on the video
fps_color = (255, 255, 255)  # white
fps_font_size = 0.5  # text size of the FPS counter
fps_thickness = 1  # text thickness of the FPS counter

show_smiley = True  # display the smiley face on the video
smiley_size = 100  # pixel size of the smiley face

label_text_size = 0.5  # text size of the labels
label_text_thickness = 2  # text thickness of the labels

separator = 50 * '-'  # separator for the console output

# %% Define project paths

# project root dir
project_dir = os.getcwd()
# images and video files to test the models
test_img_dir = os.path.join(project_dir, "test", "images")
test_videos_dir = os.path.join(project_dir, "test", "videos")
# fine-tuned versions of the model
models_dir = os.path.join(project_dir, "models")
# medias folder containing the smileys
medias_dir = os.path.join(project_dir, "medias")
# prediction folder to save predictions on image/video
predict_dir = os.path.join(project_dir, "predict")


# %% Load and resize smiley images
green_smiley_path = os.path.join(medias_dir, 'smiley_green.png')
green_smiley = cv2.resize(cv2.imread(green_smiley_path), (smiley_size, smiley_size))
red_smiley_path = os.path.join(medias_dir, 'smiley_red.png')
red_smiley = cv2.resize(cv2.imread(red_smiley_path), (smiley_size, smiley_size))


# %% Load YOLO models and list the classes we are interested in.
# Models will be downloaded if not found in the project directory

print(separator)

# yolov8n pre-trained on COCO
coco_yolo = YOLO('yolov8n.pt')
print("COCO yolo loaded successfully")
print(f"{len(coco_yolo.names)} classes")
for i in [0, 1]:  # person, bycicle
    print(f"- {i}: {coco_yolo.names[i]}")

print(separator)

# yolov8n pre-trained on OpenImageV7
oiv7_yolo = YOLO('yolov8n-oiv7.pt')
print("OpenImageV7 YOLO loaded successfully", )
print(f"{len(oiv7_yolo.names)} classes")
for i in [42, 43, 322, 381, 594]:  # bicycle, bicycle helmet, man, woman, person
    print(f"- {i}: {oiv7_yolo.names[i]}")

print(separator)

# load my model, pretrained on COCO and fine-tuned on helmet dataset
my_yolo = YOLO(os.path.join(models_dir, 'best_260424_0028.pt'))
print("Fine-Tuned YOLO loaded successfully")
print(f"{len(my_yolo.names)} classes")
for i in [0, 1]:  # without helmet, with helmet
    print(f"- {i}: {my_yolo.names[i]}")

print(separator)


# %% Define helper methods to detect cyclists and helmets on images
def parse_results(results, model, classes=None):
    '''
    Parse the tracking or prediction results of a YOLO model and store them in a list of dicts
    Each dict contains the class, confidence and bounding box coordinates

    Args:
        results: YOLO results object
        model: YOLO: YOLO model used to predict the results
        classes: list: list of classes to filter the results (None by default)

    Return:
        results_list: list: list of dictionaries containing class, confidence and bounding box coordinates
    '''

    if classes is None:
        classes = model.names

    results_list = []

    for result in results:
        boxes = result.boxes
        for box in boxes:
            pred = {}
            box_cls = int(box.cls[0])
            if box_cls not in classes:
                continue
            pred['class'] = model.names[box_cls]
            pred['confidence'] = round(float(box.conf[0]), 3)
            pred['bbox'] = tuple(map(int, box.xyxy[0]))
            results_list.append(pred)

    return results_list


def draw_result_on_frame(frame, result, color, label=None, position='top-left'):
    '''
    Draw a track/prediction result label and bounding box on an image

    Args:
        frame: np.array: frame to draw the bounding box on
        result: dict: result dict containing class, confidence and bounding box coordinates
        color: tuple: BGR color of the bounding box
        label: str: label to display (None by default)
        position: str: position of the label (top-left, top-right, bottom-left, bottom-right)

    Return:
        frame: np.array: modified frame with the bounding box and label drawn on it
    '''

    x1, y1, x2, y2 = result['bbox']

    # draw bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # draw label
    if label and position in ['top-left', 'top-right', 'bottom-left', 'bottom-right']:
        if position == 'top-left':
            label_pos = x1, (y1 - 5)
        elif position == 'top-right':
            label_pos = x2, y1
        elif position == 'bottom-left':
            label_pos = x1, y2
        elif position == 'bottom-right':
            label_pos = x2, (y2 - 5)

        cv2.putText(
            frame,
            f"{label} {result['confidence']}",
            label_pos,
            cv2.FONT_HERSHEY_SIMPLEX,
            label_text_size,
            color,
            label_text_thickness
        )

    return frame


def get_bbox_center(bbox):
    '''
    Get the center coordinates of a bounding box

    Args:
        bbox: tuple: bounding box coordinates (x1, y1, x2, y2)

    Return:
        _: tuple: center of the bounding box (x, y)
    '''

    x1, y1, x2, y2 = bbox
    x = int((x1 + x2) / 2)
    y = int((y1 + y2) / 2)
    return (x, y)


def bbox_inside_bbox(bbox1, bbox2):
    '''
    Check if a bounding box center is inside another bounding box

    Args:
        bbox1: tuple: bounding box coordinates (x1, y1, x2, y2)
        bbox2: tuple: bounding box coordinates (x1, y1, x2, y2)

    Return:
        inside: bool: True if the center of bbox2 is inside bbox1, False otherwise
    '''

    x1, y1, x2, y2 = bbox1
    x_center, y_center = get_bbox_center(bbox2)

    inside = x1 < x_center < x2 and y1 < y_center < y2
    return inside


def combine_bboxes(*args, padding=0):
    '''
    Combine multiple bounding boxes into a single one framing all of them

    Args:
        args: tuples: bounding box coordinates (x1, y1, x2, y2)
        padding: int: padding to add or substract to the combined bounding box (0 by default)

    Return:
        combined_bbox: tuple: combined bounding box coordinates (x1, y1, x2, y2)
    '''

    global video_w, video_h
    combined_bbox = ()

    for bbox in args:
        if not combined_bbox:
            combined_bbox = bbox
        else:
            x1, y1, x2, y2 = bbox
            x1_min, y1_min, x2_max, y2_max = combined_bbox
            combined_bbox = (
                max(min(x1, x1_min) - padding, 0),
                max(min(y1, y1_min) - padding, 0),
                min(max(x2, x2_max) + padding, video_w),
                min(max(y2, y2_max) + padding, video_h)
            )

    return combined_bbox


def track_cyclist_with_helmet(frame, results_list):
    '''
    Split classification results into 3 groups and detect if a person is riding a bike AND wearing a helmet
    - Group1 classes: person, man, woman
    - Group2 classes: bicycle
    - Group3 classes: bicycle helmet, with helmet

    Args:
        frame: np.array: image
        results_list: list: result dictionaries parsed by the parse_results function

        Return:
            frame: np.array: modified frame with the bounding boxes and labels drawn on it
    '''

    global debug, combine_box

    # Split the results into 3 groups: persons, bycicles, helmets
    person_results = [result for result in results_list if result['class'].lower() in ['person', 'man', 'woman']]
    bike_results = [result for result in results_list if result['class'].lower() in ['bicycle']]
    helmet_results = [result for result in results_list if result['class'].lower() in ['bicycle helmet', 'with helmet']]

    # remove person detections when the top of the bounding touches the frame (helmet disapearing, false negative)
    person_results = [person for person in person_results if person['bbox'][1] > 2]

    # exit function if no person detected
    if not person_results:
        return frame

    # Order the results by decreasing confidence, to check the most confident predictions first
    # in case of overlapping bounding boxes
    person_results = sorted(person_results, key=lambda x: x['confidence'], reverse=True)
    bike_results = sorted(bike_results, key=lambda x: x['confidence'], reverse=True)
    helmet_results = sorted(helmet_results, key=lambda x: x['confidence'], reverse=True)

    # Only take the first person with the max confidence (we can only show one smiley face per frame)
    person = person_results[0]

    # Check if the person's bounding box contains a bike and a helmet bbox
    # We consider that a bounding box is contained inside another if its center is inside the other bbox
    bike_found = False
    helmet_found = False

    if debug:
        # draw the person bbox
        frame = draw_result_on_frame(frame, person, color=debug_color, label='Person', position='bottom-right')

    for bike in bike_results:
        # look for a bike bbox inside the person bbox with a confidence above the threshold
        if bike['confidence'] > bike_confidence and bbox_inside_bbox(person['bbox'], bike['bbox']):
            bike_found = True
            if debug:
                frame = draw_result_on_frame(frame, bike, color=debug_color, label='Bicycle', position='bottom-right')
            break

    for helmet in helmet_results:
        # look for a helmet bbox inside the person bbox with a confidence above the threshold
        if helmet['confidence'] > helmet_confidence and bbox_inside_bbox(person['bbox'], helmet['bbox']):
            helmet_found = True
            if debug:
                frame = draw_result_on_frame(frame, helmet, color=debug_color, label='Helmet', position='bottom-right')
            break

    # We found a biker with helmet :)
    if bike_found and helmet_found:
        # Create a new result dict with the combined bbox and the average confidence of the person, bike and helmet detections
        cyclist = {
            'class': 'Cyclist with Helmet',
            'confidence': round(mean([person['confidence'], bike['confidence'], helmet['confidence']]), 3),
            'bbox': combine_bboxes(person['bbox'], bike['bbox'], helmet['bbox'], padding=combined_box_padding)
        }
        combined_bbox_color = red_color  # green
        smiley = green_smiley

    # We found a biker without helmet :(
    elif bike_found and not helmet_found:
        # Create a new result dict with the combined bbox and the average confidence of the person and bike detections
        cyclist = {
            'class': 'Cyclist without Helmet', 
            'confidence': round(mean([person['confidence'], bike['confidence']]), 3),
            'bbox': combine_bboxes(person['bbox'], bike['bbox'], padding=combined_box_padding)
        }
        combined_bbox_color = green_color  # red
        smiley = red_smiley

    # We found a pedestrian or another person object
    else:
        return frame

    # draw the cyclist bbox and label, greed if cyclist with helmet, red if cyclist without helmet
    if combined_box:
        frame = draw_result_on_frame(frame, cyclist, color=combined_bbox_color, label=cyclist['class'], position='top-left')

    # draw a smiley face in the bottom right corner
    if show_smiley:
        frame[-smiley_size:, -smiley_size:] = smiley

    return frame


def tracking_thread(frame, model, result_dict, classes, conf):
    '''
    Track objects on a frame using a given model and store the results in a given dictionary.
    We modify the results dictionary in place, as we cannot return it and get the results from the thread.

    Args:
        frame: np.array: video frame
        model: YOLO: YOLO model used to track the objects
        result_dict: dict: dictionary passed as reference to store the tracking results
        classes: list: list of classes to track
        conf: float: confidence threshold
    '''

    result_dict[model] = model.track(frame, verbose=False, persist=True, classes=classes, conf=conf)[0]


# %% Track cyclists on a video file using OpenCV
def track_on_video(video_path, tracking_assignment, conf, output_path=None):
    '''
    Track objects on a video using one or multiple YOLO models and classes.
    Each tracking assignment is a tuple containing a YOLO model and a list of classes to track.
    The tracking is done on separate threads to speed up the process.

    Args:
        video_path: str: path to the video file
        tracking_assignment: list: list of tuples (model, [classes]) containing the YOLO model and
            the classes to track
        conf: float: confidence threshold
        output_path: str: path to save the tracked video (Skip saving if None)
    '''

    global debug, show_fps, show_smiley, combined_box, video_w, video_h, video_fps

    cap = cv2.VideoCapture(video_path)

    if not video_w and video_h and video_fps:
        video_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        video_fps = cap.get(cv2.CAP_PROP_FPS)

    # FPS counter reinitialized every second
    fps_count = 0
    fps_value = 0
    fps_time_counter = datetime.now()

    if output_path:
        out = cv2.VideoWriter(output_path, -1, video_fps, (video_w, video_h))

    while cap.isOpened():

        # load current frame
        success, frame = cap.read()

        # exit video when reaching the end
        if frame is None:
            break

        # Calculate FPS
        fps_count += 1
        if datetime.now() - fps_time_counter > timedelta(seconds=1):
            fps_value = fps_count
            fps_count = 0
            fps_time_counter = datetime.now()

        # Keyboard shortcuts
        key = cv2.waitKey(1) & 0xFF
        # toggle debug mode
        if key == ord('d'):
            debug = not debug
        # rewind video
        elif key == ord('r'):
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        # display fps
        elif key == ord('f'):
            show_fps = not show_fps
        # display smiley
        elif key == ord('s'):
            show_smiley = not show_smiley
        # display combined bbox
        elif key == ord('b'):
            combined_box = not combined_box
        # quit video
        elif key == ord('q'):
            break

        if success:

            # track each model on the frame on a separate thread
            results = {}
            track_threads = []

            # create and start threads
            for model, classes in tracking_assignment:
                thread = threading.Thread(target=tracking_thread, args=(frame, model, results, classes, conf))
                thread.start()
                track_threads.append(thread)

            # Wait for the threads to finish
            for thread in track_threads:
                thread.join()

            # combine results from all models
            result_list = []
            for model, result in results.items():
                result_list += parse_results(result, model)

            # send the combined results to the tracking function to display the bounding boxes
            frame = track_cyclist_with_helmet(frame, result_list)

            # display FPS counter on a black background
            if show_fps:
                cv2.rectangle(frame, (0, video_h - 20), (70, video_h), (0, 0, 0), -1)
                cv2.putText(frame, f"FPS: {fps_value}", (0, video_h - 5), cv2.FONT_HERSHEY_SIMPLEX, fps_font_size, fps_color, fps_thickness)

            # display the final frame
            cv2.imshow(os.path.split(video_path)[0], frame)

            # write to video file if output path is provided
            if output_path:
                out.write(frame)

    # release and destroy all windows
    cap.release()
    if output_path:
        out.release()

    cv2.destroyAllWindows()


# %% Main
if __name__ == '__main__':

    video_file = 'video4.mp4'
    video_path = os.path.join(predict_dir, video_file)

    # Define the tracking assignment (which model should track which classes)
    tracking_assignment = [
        # (oiv7_yolo, [322, 594, 381, 42]),  # person, man, woman and bike
        (coco_yolo, [0, 1]),  # person, bike
        (my_yolo, [1])  # helmet
    ]

    # define output path to save the tracked video (None to skip saving)
    output_path = os.path.join(test_videos_dir, f"tracked_{video_file}")
    # output_path = None

    # Predict on the frame
    track_on_video(video_path, tracking_assignment, conf=tracking_confidence, output_path=output_path)
