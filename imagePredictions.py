import cv2
from y_yolo_predictions import YOLO_Pred

yolo = YOLO_Pred('Model4/weights/best.onnx','data.yaml')

img = cv2.imread('test.jpg')

cv2.imshow('Actual',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

#predictions
img_pred = yolo.predictions(img)
cv2.imshow('Predicted',img_pred)
cv2.waitKey(0)
cv2.destroyAllWindows()