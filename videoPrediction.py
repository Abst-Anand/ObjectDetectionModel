import cv2
from y_yolo_predictions import YOLO_Pred

cap = cv2.VideoCapture('Scenic Drive.mp4') #o for webcam
yolo = YOLO_Pred('Model4/weights/best.onnx','data.yaml')
while True:
    ret, frame = cap.read()

    if ret == False:
        print('Unable to read video')
        break

    pred_image = yolo.predictions(frame)
    cv2.imshow('Real Time Object Detection',pred_image)

    if cv2.waitKey(1) == 27: #esc key
        break
cv2.destroyAllWindows()
cap.release()