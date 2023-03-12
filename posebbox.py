# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 12:31:08 2022

@author: eko my
"""



import cv2
import numpy as np
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
pose = mp_pose.Pose( static_image_mode=True,  model_complexity=2, enable_segmentation=True,min_detection_confidence=0.5)

# define a video capture object
vid = cv2.VideoCapture('input4.mp4')
i=0
fps = int(vid.get(cv2.CAP_PROP_FPS))
frame_width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))

# define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'avc1') # can use other codecs as well
out = cv2.VideoWriter('output_video.mp4', fourcc, fps, (frame_width, frame_height))

while vid.isOpened():

      
    # Capture the video frame
    # by frame
        
    #vid.set(3,1280)SS
    #vid.set(4,720)
    
    ret, image = vid.read()
    if not ret:
        break
    
    black = np.zeros(image.shape , np.uint8)
    image_height, image_width, _ = image.shape
    results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    # Retrieve the height and width of the input image.
    height, width, _ = image.shape
    
    # Bbox padd amount
    padd_amount = 5
    
    # Initialize a list to store the detected landmarks.
    landmarks = []
    mp_drawing.draw_landmarks(
        black,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
    mp_drawing.draw_landmarks(
        image,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
    
    
    try:
        # Create bounding box for skeleton
        if results.pose_landmarks:
            # Iterate over the detected landmarks.
            for landmark in results.pose_landmarks.landmark:
                # Append the landmark into the list.
                landmarks.append((int(landmark.x * width), int(landmark.y * height),
                                      (landmark.z * width)))
                
            x_coordinates = np.array(landmarks)[:,0]
            
            y_coordinates = np.array(landmarks)[:,1]
            
            x1  = int(np.min(x_coordinates) - padd_amount)
            y1  = int(np.min(y_coordinates) - padd_amount)
            x2  = int(np.max(x_coordinates) + padd_amount)
            y2  = int(np.max(y_coordinates) + padd_amount)
            
            cv2.rectangle(image, (x1, y1), (x2, y2), (155, 0, 255), 3, cv2.LINE_8)
        
     
        processed_frame = cv2.cvtColor(black, cv2.COLOR_RGB2BGR)
        #result = cv2.add(processed_frame, black)  
        crop = black[y1:y2, x1:x2]
        crop2=image[y1-100:y2+100,x1-100:x2+100]
        processed_frame = cv2.resize(processed_frame, (1280,720))
        out.write(processed_frame)
        # Display the resulting frame
        cv2.imshow('camera', processed_frame)
        #cv2.imwrite('inputempat'+str(i)+'.jpg',crop2)
        #cv2.imwrite('inputempat'+str(i)+'('+str(i)+')'+'.jpg',crop)
        i=i+1
    
    except AttributeError:
        print("Koordinat Tidak Ditemukan")
    
    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
  
# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()