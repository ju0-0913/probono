import tensorflow as tf
import numpy as np
import time
import cv2
import serial
import time

INPUT_SIZE = (224, 224)

# 웹캠 영상 불러오기
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 5)

model = tf.keras.models.load_model("saved_model.h5") # 학습시킨 모델 불러오기

ser = serial.Serial('COM7', 9600)

while cap.isOpened():
    start_time = time.time()

    ret, frame = cap.read() # 웹캠 영상 저장하기
    if not ret:
        break

    model_frame = cv2.resize(frame, INPUT_SIZE)
    model_frame = np.expand_dims(model_frame, axis = 0) / 255.0

    transport = model.predict(model_frame)[0]
    is_transport = np.argmax(transport)

    inference_time = time.time() - start_time
    fps = 1 / inference_time
    fps_msg = "Time : {:05.1f}ms {:.1f} FPS".format(inference_time * 1000, fps)

    if is_transport == 0: # 목발
        result = "crutches"
        response = str(0).encode('utf-8')
        ser.write(response)
    elif is_transport == 1: # 일반
        result = "no_crutches_wheelchair"
        response = str(1).encode('utf-8')
        ser.write(response)
    else: # 휠체어
        result = "wheelchair"
        response = str(2).encode('utf-8')
        ser.write(response)

    result += " ({:.1f})%".format(transport[is_transport] * 100)

    cv2.putText(frame, fps_msg, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, ))
    cv2.putText(frame, result, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, ))

    cv2.imshow('test image', frame)

    cv2.imshow('debug', model_frame[0])

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

ser.close()