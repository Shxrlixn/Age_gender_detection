import cv2
import numpy as np
import os

face_cascade_path = 'Person_Detection/haarcascade_frontalface_default.xml'
age_proto_path = 'Person_Detection/deploy_age.prototxt'
age_model_path = 'Person_Detection/age_net.caffemodel'
gender_proto_path = 'Person_Detection/deploy_gender.prototxt'
gender_model_path = 'Person_Detection/gender_net.caffemodel'

if not os.path.exists(face_cascade_path):
    print(f"Error: File '{face_cascade_path}' not found.")
    exit()

face_cascade = cv2.CascadeClassifier(face_cascade_path)

if not os.path.exists(age_proto_path) or not os.path.exists(age_model_path):
    print(f"Error: Age model files not found.")
    exit()

if not os.path.exists(gender_proto_path) or not os.path.exists(gender_model_path):
    print(f"Error: Gender model files not found.")
    exit()

age_net = cv2.dnn.readNetFromCaffe(age_proto_path, age_model_path)
gender_net = cv2.dnn.readNetFromCaffe(gender_proto_path, gender_model_path)

AGE_BUCKETS = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
GENDER_LIST = ['Male', 'Female']

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video capture.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        face_img = frame[y:y + h, x:x + w]
        blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227),
                                     (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)

        age_net.setInput(blob)
        gender_net.setInput(blob)

        age_predictions = age_net.forward()
        gender_predictions = gender_net.forward()

        age_index = age_predictions[0].argmax()
        gender_index = gender_predictions[0].argmax()

        age = AGE_BUCKETS[age_index]
        gender = GENDER_LIST[gender_index]

        label = f"Gender: {gender}, Age: {age}"
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow('Person Detection, Age and Gender Estimation', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
