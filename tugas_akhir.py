import cv2
import mediapipe as mp
import os

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

# #Membuat Label untuk Folder
# Labels = ["TanganA","TanganB"]
# #Now create folders for each label to store images
# for label in Labels:
#     if not os.path.exists(label):
#         os.mkdir(label)

# For webcam input:
cap = cv2.VideoCapture(0)
with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as pose:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    #bikin kotak buat mapping wilayahnya 
    image_height, image_width, _ = image.shape
    coords = []
    coordsy = []

    results = pose.process(image)

    # Draw the pose annotation on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    #membuat kotak box 
    if results.pose_landmarks:
        pose_landmarks = results.pose_landmarks.landmark
        for A in pose_landmarks:
          cx, cy = A.x * image_width, A.y*image_height                                                                                                          
          
          coords.append(cx) 
          coordsy.append(cy) 
        
        x_max = max(coords)
        y_max = max(coordsy)
        x_min = min(coords)
        y_min = min(coordsy)

        cv2.rectangle(image, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)

    mp_drawing.draw_landmarks(
        image,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())


    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Pose', cv2.flip(image, 1))
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()