import cv2
import pygame.mixer

#? นำไฟล์พวก cascade เข้ามาใช้
mouth_cascade = cv2.CascadeClassifier('haarcascade_mcs_mouth.xml')
nose_cascade = cv2.CascadeClassifier('haarcascade_mcs_nose.xml')
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture("mask.mp4")

pygame.mixer.init()
soundplay=pygame.mixer.music.load("eiei.mp3")
pygame.mixer.music.play(-1)

while True:
    ret,frame = cap.read()
    frame = cv2.resize(frame,(1024,768)) #ปรับขนาดรูปภาพให้แสดงที่ 1024x768 frame = cv2.resize(frame,(1024,768))

    mouth_detect = mouth_cascade.detectMultiScale(frame, 1.55, 15) #ให้มันตรวจพบปาก
    nose_detect = nose_cascade.detectMultiScale(frame, 1.3, 16,minSize=(20,20)) #ให้มันตรวจพบจมูก
    faces_detect = face_cascade.detectMultiScale(frame,1.25, 1, minSize=(30,30)) #ให้มันตรวจพบใบหน้า
    if(len(faces_detect) == 0 ):
        pygame.mixer.music.pause()
        cv2.putText(frame,"no face",(20,30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
    else:
        if(len(mouth_detect) > 0 and len(nose_detect) > 0 and len(faces_detect) > 0): #ตรวจพบเจอทุกอย่าง
            pygame.mixer.music.unpause()
            for (x,y,w,h) in mouth_detect:
                y = int(y - 5) #ปรับขนาดแกน y ให้มันตรงกับตรงปาก
                cv2.rectangle (frame,(x,y),(x+w,y+h),(92,92,205),2) #วาดสี่เหลี่ยมตรงปาก แสดงสีแดง ความหนา 2      
            for (x,y,w,h) in nose_detect:
                cv2.rectangle(frame, (x,y), (x+w,y+h), (92,92,205), 2) #วาดสี่เหลี่ยมตรงจมูก
            cv2.putText(frame,"You don't wear a mask.",(20,30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
        elif(len(mouth_detect) == 0 and len(nose_detect) > 0 and len(faces_detect) > 0): #ตรวจพบเจอจมูกและใบหน้า
            pygame.mixer.music.pause()
            for (x,y,w,h) in nose_detect:
                cv2.rectangle(frame, (x,y), (x+w,y+h), (92,92,205), 2)
            cv2.putText(frame,"Please wear to your nose",(20,30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
            
        elif(len(mouth_detect) == 0 and len(nose_detect) == 0 and len(faces_detect) > 0): #ไม่ตรวจพบเจออะไรเลย
            pygame.mixer.music.pause()
            cv2.putText(frame,"Thank for wear a mask.",(20,30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
            
    cv2.imshow('Mask Detection', frame) #โชว์ตัวแปลเฟรมขึ้นมา

    if (cv2.waitKey(1) == ord('c')): #กด c เพื่อทำการเบรก 
        break
cap.release()
cv2.destroyAllWindows()