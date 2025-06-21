import cv2
import numpy as np
cap = cv2.VideoCapture("EyeMovements.mp4")
while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    roi=frame[80:250,20:200]
    rows,cols,_=roi.shape
    gray_roi=cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
    gray_roi=cv2.GaussianBlur(gray_roi,(7,7),0)
    _,threshold=cv2.threshold(gray_roi,50,255,cv2.THRESH_BINARY_INV)
    contours,_=cv2.findContours(threshold,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    contours=sorted(contours,key=lambda x:cv2.contourArea(x),reverse=True)
    #frame=cv2.rectangle(frame,(20,80),(200,250),(0,255,0),3)
    for cnt in contours:
        (x,y,w,h)=cv2.boundingRect(cnt)
        cv2.rectangle(roi,(x,y),(x+w,y+h),(255,0,0),2)
        cv2.line(roi,(x+int(w/2),0),(x+int(w/2),rows),(0,255,0),2)
        cv2.line(roi,(0,y+int(h/2)),(cols,y+int(h/2)),(0,255,0),2)
        break
    cv2.imshow("Frame",frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()