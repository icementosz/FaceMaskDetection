import cv2
import pygame.mixer

#? นำไฟล์พวก cascade เข้ามาใช้
mouth_cascade = cv2.CascadeClassifier('haarcascade_mcs_mouth.xml') 
nose_cascade = cv2.CascadeClassifier('haarcascade_mcs_nose.xml')
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

pygame.mixer.init()
soundplay=pygame.mixer.music.load("eiei.mp3")
pygame.mixer.music.play(-1)

while True:
    ret,frame = cap.read()
    frame = cv2.flip(frame,1) #สลับเฟรมให้มันตรงกับชีวิตจริง
    frame = cv2.resize(frame,(1024,768)) #ปรับขนาดรูปภาพให้แสดงที่ 1024x768 frame = cv2.resize(frame,(1024,768))
    
    #? ค่าต่างๆเกิดจากการทดลองรันแล้วปรับไปเรื่อยๆจนมันเข้าที่
    mouth_detect = mouth_cascade.detectMultiScale(frame, 2, 21,maxSize=(100,60)) #ให้มันตรวจพบปาก
    nose_detect = nose_cascade.detectMultiScale(frame, 2.1, 14) #ให้มันตรวจพบจมูก
    faces_detect = face_cascade.detectMultiScale(frame, 1.1, 5,minSize=(35,35)) #ให้มันตรวจพบหน้า
    if(len(faces_detect) == 0):
        pygame.mixer.music.pause()
        cv2.putText(frame,"no face",(20,30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2) #ไม่ตรวจพบหน้า
    else:
        if(len(mouth_detect) > 0 and len(nose_detect) > 0 and len(faces_detect) > 0):
            pygame.mixer.music.unpause()
            for (x,y,w,h) in mouth_detect:
                y = int(y - 5) #ปรับแกน y และ h ให้มันตรงกับปาก
                h=int(h-10)
                cv2.rectangle (frame,(x,y),(x+w,y+h),(92,92,205),2) #วาดสี่เหลี่ยมผืนผ้าตรงปาก แสดงสีแดง ความหนา 2     
            for (x,y,w,h) in nose_detect:
                y=int(y-27) #ปรับแกนทั้งหมดให้ตรงตำแหน่งจมูก
                w=int(w-30)
                x=int(x+13)
                h=int(h-1)
                cv2.rectangle(frame, (x,y), (x+w,y+h), (92,92,205), 2) #วาดสี่เหลี่ยมผืนผ้าตรงจมูก แสดงสีแดง ความหนา 2  
            cv2.putText(frame,"You don't wear a mask.",(20,30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2) #เจอจมูกและปาก ก็ให้แสดงว่าไม่ใส่แมสก์
        elif(len(mouth_detect) == 0 and len(nose_detect) > 0 and len(faces_detect) > 0):
            pygame.mixer.music.pause()
            for (x,y,w,h) in nose_detect:
                y=int(y-27) 
                w=int(w-30)
                x=int(x+13)
                h=int(h-1)
                cv2.rectangle(frame, (x,y), (x+w,y+h), (92,92,205), 2)
            cv2.putText(frame,"Please wear to your nose.",(20,30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2) #เจอจมูก ก็ให้แสดงว่าสวมให้ถึงจมูก
        elif(len(mouth_detect) == 0 and len(nose_detect) == 0 and len(faces_detect) > 0):
            pygame.mixer.music.pause()
            cv2.putText(frame,"Thank for wear a mask.",(20,30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2) #ไม่เจอปากกับจมูกก็ให้แสดงว่าขอบคุณสำหรับการใส่แมสก์

    cv2.imshow('Mask Detection', frame) #ทำการโชว์ตัวแปลเฟรมขึ้นมา

    if (cv2.waitKey(1) == ord('c')): #กด c เพื่อทำการเบรก 
        break
cap.release()
cv2.destroyAllWindows()