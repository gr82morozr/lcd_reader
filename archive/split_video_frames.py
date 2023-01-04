import cv2
 
capture = cv2.VideoCapture('C:/Users/N/Desktop/video.avi')
capture = cv2.VideoCapture(0)

cap_width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
cap_height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
cap_fps = capture.get(cv2.CAP_PROP_FPS)
frameNr = 0
 
while (True):
 
    success, frame = capture.read()
 
    if success:
        cv2.imwrite(f'R:/Temp2/frame_{frameNr}.jpg', frame)
 
    else:
        break
 
    frameNr = frameNr+1
 
capture.release()