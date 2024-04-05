# Alex Hecksher (2024)
# Code is free to copy and use for non-commericial uses as long as it is properly attributed 
# to me and the other people, whose code and knowledge I am used in creating this harpsicorpse.
# Please reach out to me (alexhecksher@gmail.com) if you have any questions
# 
# Sections of this code were written with the help of or copied directly from the following sources:
#   - "Tensorflow Multi-Person Pose Estimation with Python // Machine Learning Tutorial" 
#       by Nicholas: https://www.youtube.com/watch?v=KC7nJtBHBqg
#   - The mediapipe guide : https://developers.google.com/mediapipe/solutions/guide
#   - "SuperCollider Tutorial: 21. FM Synthesis, Part I" by Eli Fieldsteel: 
#       https://www.youtube.com/watch?v=UoXMUQIqFk4&list=PLPYzvS8A_rTaNDweXe6PX4CXSGq4iEWYC&index=22
#   - "SuperCollider Tutorial: 22. FM Synthesis, Part II" by Eli Fieldsteel:
#       https://www.youtube.com/watch?v=dLMSR2Kjq6Y&list=PLPYzvS8A_rTaNDweXe6PX4CXSGq4iEWYC&index=23
#   - "SuperCollider Tutorial: 23. Wavetable Synthesis, Part I" by Eli Fieldsteel:
#       https://www.youtube.com/watch?v=8EK9sq_9gFI&list=PLPYzvS8A_rTaNDweXe6PX4CXSGq4iEWYC&index=24
#   - "SuperCollider Tutorial: 24. Wavetable Synthesis, Part II" by Eli Fieldsteel:
#       https://www.youtube.com/watch?v=7nrUBbmY1hE&list=PLPYzvS8A_rTaNDweXe6PX4CXSGq4iEWYC&index=25
#   - "Delays, Reverbs, Harmonizers - Week 7 Spring 2021 MUS 499C - Intermediate SuperCollider"
#       by Eli Fieldsteel: https://www.youtube.com/watch?v=eEyYFt3sIWs&list=PLPYzvS8A_rTbTAn-ZExGuVFZgVMwYi1kJ&index=7
# 
# Huge thanks to Eli Fieldsteel in particular, whose tutorials on supercollider have been instrumental
# in helping me build this instrument. Here is a link to his youtube page: https://www.youtube.com/@elifieldsteel
# 
# Also please excuse any typos and feel free to bring them to my attention for correcting :)

import mediapipe as mp

import tensorflow as tf
import tensorflow_hub as hub

import numpy as np

import cv2

import math
import time

from pythonosc import udp_client
from pythonosc import osc_message_builder

# Setting constant variables for face detector
FACE_LANDMARK_NUMS = [4, 13, 14, 50, 280, 105, 334, 61, 291, 468, 473]
FACE_MODEL_PATH = 'face_landmarker.task'
FACE_OSC_FREQ = (60/120) * 0.5

# Setting constant variables for pose detector
POSE_EDGES = [
    (5, 7),
    (6, 8),
    (5, 6),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16)
]
POSE_LANDMARK_NUMS = [5, 6, 7, 8, 11, 12, 13, 14, 15, 16]
MAX_DIM = 256
POSE_OSC_FREQ = (60/120) * 0.25

# Setting constant variables for hand detector
HAND_EDGES = [
    (0, 4),
    (0, 8),
    (0, 12),
    (0, 16),
    (0, 20),
] 
HAND_LANDMARK_NUMS = [0, 4, 8, 12, 16, 20]
HAND_MODEL_PATH = 'hand_landmarker.task'
HAND_OSC_FREQ = 60/120

# Setting constant variable for image detector
IMG_OSC_FREQ = (60/120)

# Setting video source. 0 will use the onboard webcam.
# To use a video set VIDEO_SOURCE="path/to/the/video"
VIDEO_SOURCE = 0

# Functions to draw points and edges and to build the 
# OSC message that is sent to the supercollider program
def draw_faces(image, face_results):
    faces = face_results.face_landmarks
    dimY, dimX, c = image.shape

    for face_landmarks in faces:
        for i in FACE_LANDMARK_NUMS:
            x = face_landmarks[i].x * dimX
            y = face_landmarks[i].y * dimY

            cv2.circle(image, (int(x), int(y)), 6, (0, 0, 255), -1)

def build_face_msg(face_results):
    msg = osc_message_builder.OscMessageBuilder(address="/faces")

    faces = face_results.face_landmarks

    for face_landmarks in faces:
        for i in FACE_LANDMARK_NUMS:
            x = face_landmarks[i].x
            y = face_landmarks[i].y

            msg.add_arg(x, 'f')
            msg.add_arg(y, 'f')

        lm468 = face_landmarks[468]
        lm473 = face_landmarks[473]
        z = math.dist([lm468.x, lm468.y], [lm473.x, lm473.y])

        msg.add_arg(z, 'f')
    
    return msg.build()

def draw_hands(image, hand_results):
    hands = hand_results.hand_landmarks
    dimY, dimX, c = image.shape


    for hand_landmarks in hands:
        for i in HAND_LANDMARK_NUMS:
            x = hand_landmarks[i].x * dimX
            y = hand_landmarks[i].y * dimY

            cv2.circle(image, (int(x), int(y)), 6, (255, 0, 0), -1)

        for edge in HAND_EDGES:
            i, j = edge
            x1 = hand_landmarks[i].x * dimX
            y1 = hand_landmarks[i].y * dimY
            x2 = hand_landmarks[j].x * dimX
            y2 = hand_landmarks[j].y * dimY
        
            cv2.line(image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 4)

def build_hand_msg(hand_results):
    msg = osc_message_builder.OscMessageBuilder(address="/hands")

    hands = hand_results.hand_landmarks


    for hand_landmarks in hands:
        for i in HAND_LANDMARK_NUMS:
            x = hand_landmarks[i].x
            y = hand_landmarks[i].y

            msg.add_arg(x, 'f')
            msg.add_arg(y, 'f')

        lm0 = hand_landmarks[0]
        lm5 = hand_landmarks[5]
        z = math.dist([lm0.x, lm0.y], [lm5.x, lm5.y])

        msg.add_arg(z, 'f')
    
    return msg.build()

def draw_poses(pose_results, image, confidence_threshold):
    dimY, dimX, c = image.shape

    for pose in pose_results:
        for i in POSE_LANDMARK_NUMS:
            y, x, c = pose[i]

            x = x * dimX
            y = y * dimY

            if c > confidence_threshold:
                cv2.circle(image, (int(x), int(y)), 6, (0, 255, 0), -1)

        for edge in POSE_EDGES:
            i, j = edge

            y1, x1, c1 = pose[i]
            y2, x2, c2 = pose[j]

            x1 = x1 * dimX
            y1 = y1 * dimY
            x2 = x2 * dimX
            y2 = y2 * dimY

            if (c1 > confidence_threshold) & (c2 > confidence_threshold):      
                cv2.line(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
            
def trim_poses(pose_results, confidence_threshold):
    trimed_pose_results = []

    for pose in pose_results:
        pose_conf = False

        for i in POSE_LANDMARK_NUMS:
            y, x, c = pose[i]

            if(c > confidence_threshold):
                pose_conf = True

        if(pose_conf):
            trimed_pose_results.append(pose)

    return trimed_pose_results

def build_pose_msg(pose_results, confidence_threshold):
    msg = osc_message_builder.OscMessageBuilder(address="/poses")

    for pose in pose_results:
        for i in POSE_LANDMARK_NUMS:
            y, x, c = pose[i]
            
            if(c > confidence_threshold):
                msg.add_arg(x, 'f')
                msg.add_arg(y, 'f')
            else:
                msg.add_arg(0, 'f')
                msg.add_arg(0, 'f')
            
        y1, x1, c1 = pose[5]
        y2, x2, c2 = pose[6]
        z = math.dist([x1, y1], [x2, y2])
        msg.add_arg(z, 'f')

    return msg.build()

def draw_image_points(image, annotated_image):
    dimY, dimX, c = image.shape

    for i in range(3):
        for j in range(3):
            y = int(((dimY - 50)/2) * i) + 25
            x = int(((dimX - 50)/2) * j) + 25

            p = image[y, x]
            color = (int(p[0]), int(p[1]), int(p[2]))

            cv2.circle(annotated_image, (x, y), 10, color, -1)
            cv2.circle(annotated_image, (x, y), 10, (255, 255, 0), 2)

def build_image_msg(image):
    msg = osc_message_builder.OscMessageBuilder(address="/image")
    dimY, dimX, c = image.shape


    for i in range(3):
        for j in range(3):
            y = int(((dimY - 50)/2) * i) + 25
            x = int(((dimX - 50)/2) * j) + 25

            p = image[y, x]/255

            msg.add_arg(p[0], 'f')
            msg.add_arg(p[1], 'f')
            msg.add_arg(p[2], 'f')
    
    return msg.build()


# Creating the OSC client to send the OSC messages
client = udp_client.SimpleUDPClient("127.0.0.1", 57120)

# Creating the video capture object
cap = cv2.VideoCapture(VIDEO_SOURCE)

# Loading the pose model using tensorflow hub 
# and creating the pose detector
pose_model = hub.load("https://www.kaggle.com/models/google/movenet/frameworks/TensorFlow2/variations/multipose-lightning/versions/1")
pose_detector = pose_model.signatures['serving_default']

first_frame = True
previous_pose_time = time.time()
previous_hand_time = time.time()
previous_face_time = time.time()
previous_img_time = time.time()

# Creating the face and hand detector based on the given options
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions

face_options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=FACE_MODEL_PATH),
    running_mode=VisionRunningMode.IMAGE,
    num_faces=4)

face_detector = FaceLandmarker.create_from_options(face_options)

hand_options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=HAND_MODEL_PATH),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=4)

hand_detector = HandLandmarker.create_from_options(hand_options)

# Main loop that is run on each frame of the video
while cap.isOpened():
    ret, frame = cap.read()

    # check if the video capture is still returning valid frames
    # Will return false and stop the loop once we reach the end of a video
    if(not ret):
        break

    # On the first frame, find out the resize width and height, 
    # which needs to be a multiple of 32 and stay as close to the original 
    # aspect ratio as possible
    if(first_frame):
        h,w,d = frame.shape
        aspect_ratio = h/w
        resize_h = MAX_DIM 
        resize_w = MAX_DIM
        if(aspect_ratio <= 1):
            resize_h = aspect_ratio * MAX_DIM
            mult32 = math.floor((resize_h/32) + 0.5)
            resize_h = mult32 * 32
        else:
            aspect_ratio = 1 / aspect_ratio
            resize_w = aspect_ratio * MAX_DIM
            mult32 = math.floor((resize_w/32) + 0.5)
            resize_w = mult32 * 32
        
        first_frame = False

    # Make two copies, one to be used for detection and one to draw 
    # the landmark points and connection edges
    image = frame.copy()
    annotated_image = frame.copy()

    # Resize image before pose detection
    resized_image = tf.image.resize_with_pad(tf.expand_dims(image, axis=0), resize_h, resize_w)
    input_image = tf.cast(resized_image, dtype=tf.int32)

    # Detect poses
    pose_results = pose_detector(input_image)
    pose_results = pose_results['output_0'].numpy()[:,:,:51].reshape((6,17,3))
    pose_results = trim_poses(pose_results, 0.3)
    
    # Draw poses 
    draw_poses(pose_results, annotated_image, 0.3)

    # If enough time has passed then send pose OSC message
    # How often OSC messages are sent is determined by the OSC frequency
    current_pose_time = time.time()
    pose_time_diff = current_pose_time - previous_pose_time
    if(pose_time_diff >= POSE_OSC_FREQ):
        previous_pose_time = current_pose_time
        pose_msg = build_pose_msg(pose_results, 0.3)
        client.send(pose_msg)

    # Recolor images for the mediapipe detectors
    recolor_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    recolor_image.flags.writeable = False

    # Detect hands
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=recolor_image)
    hand_results = hand_detector.detect(mp_image)

    # Draw hands
    draw_hands(annotated_image, hand_results)

    # Send hand OSC message
    current_hand_time = time.time()
    hand_time_diff = current_hand_time - previous_hand_time
    if(hand_time_diff >= HAND_OSC_FREQ):
        previous_hand_time = current_hand_time
        hand_msg = build_hand_msg(hand_results)
        client.send(hand_msg)

    # Detect faces
    face_results = face_detector.detect(mp_image)

    # Draw faces
    draw_faces(annotated_image, face_results)

    # Send face OSC message
    current_face_time = time.time()
    face_time_diff = current_face_time - previous_face_time
    if(face_time_diff >= FACE_OSC_FREQ):
        previous_face_time = current_face_time
        face_msg = build_face_msg(face_results)
        client.send(face_msg)

    # Draw the image points
    draw_image_points(image, annotated_image)

    # Send image OSC message
    current_img_time = time.time()
    img_time_diff = current_img_time - previous_img_time
    if(img_time_diff >= IMG_OSC_FREQ):
        previous_img_time = current_img_time
        img_msg = build_image_msg(image)
        client.send(img_msg)

    # If the source is a webcam, mirror the image
    if(VIDEO_SOURCE == 0):
        image = cv2.flip(image, 1)
        annotated_image = cv2.flip(annotated_image, 1)

    # Show the original image, the annoted image, 
    cv2.imshow('Harpsicorpse Original Image', image)
    cv2.imshow('Harpsicorpse Annotated Image', annotated_image)

    # The number inside waitKey() will determine how long between each main loop iteration
    # And 0xFF == ord('q') lets you quit the program by pressing "q"
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Send empty OSC messages to stop all the synths
client.send_message('/faces', [])
client.send_message('/hands', [])
client.send_message('/poses', [])
client.send_message('/image', [])