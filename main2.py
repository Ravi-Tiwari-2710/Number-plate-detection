import cv2
import numpy as np


min_width_rect = 80
min_height_rect = 80

#web camera
cap = cv2.VideoCapture('video.mp4')


count_line_position = 550
# initialize the subtractor
algo = cv2.createBackgroundSubtractorMOG2()


def centre_handle(x,y,w,h):
    x1 =int(w/2)
    y1 =int(h/2)
    cx =x+x1
    cy =y+y1
    return cx ,cy

detect = [] 
offset = 6
counter = 0



while True:
    ret,frame1 = cap.read()
    gray = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(3,3),5)
    #applying on each frame
    img_sub = algo.apply(blur)
    dilat = cv2.dilate(img_sub,np.ones((5,5)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    dilatada = cv2.morphologyEx(dilat,cv2.MORPH_CLOSE,kernel)
    countershape,h = cv2.findContours(dilatada,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    cv2.line(frame1,(25,count_line_position),(1200,count_line_position),(225,127,0),3)


    for (i,c) in enumerate(countershape):
        (x,y,w,h) = cv2.boundingRect(c)
        validate_counter = (w>= min_width_rect) and (h>= min_height_rect)
        if not validate_counter:
            continue

        cv2.rectangle(frame1,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.putText(frame1,"Vehicle"+str(counter),(x,y-20),cv2.FONT_HERSHEY_TRIPLEX,1,(225,244,0),2)        

        centre = centre_handle(x,y,w,h)
        detect.append(centre)
        cv2.circle(frame1,centre,4,(0,0,255),-1)

        for (x,y) in detect:
            if y<(count_line_position+offset) and y>(count_line_position-offset):
                counter+=1
            cv2.line(frame1,(25,count_line_position),(1200,count_line_position),(0,127,255),3)
            detect.remove((x,y))
     #       print('vehicle counter:'+str(counter))

    cv2.putText(frame1,"VEHICLE COUTER:"+str(counter),(550,150),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,225),5)  
    cv2.putText(frame1,"SPEED LIMIT:60KMPH",(10,70),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,225),5)      
    

    cv2.imshow('Video Original',frame1)

    if cv2.waitKey(1) ==13:
        break
    