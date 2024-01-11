import face_recognition
import cv2
import os
import numpy as np

path = 'face-recognition-using-python/src'
images = []
names = []
imglist = os.listdir(path)

for img in imglist:
    images.append(cv2.imread(f"{path}/{img}"))
    names.append(os.path.splitext(img)[0])
    

def Encoding(images):
    encodelist = []
    for img in images:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodelist.append(encode)
    return encodelist

known_face_encodings = Encoding(images)

video = cv2.VideoCapture(0)

if not video.isOpened():
    print("ERROR : Video is not opened.")
    exit()

while True:
    ret,frame = video.read()

    if not ret:
        break
    frame = cv2.flip(frame,1)

    frames = cv2.resize(frame,(0,0),fx=0.25,fy=0.25)
    frames = cv2.cvtColor(frames,cv2.COLOR_BGR2RGB)

    loc_current_frame = face_recognition.face_locations(frames)
    encode_current_frame = face_recognition.face_encodings(frames,loc_current_frame)

    for encode_face,facloc in zip(encode_current_frame,loc_current_frame):
        matches = face_recognition.compare_faces(known_face_encodings,encode_face)
        face_distance = face_recognition.face_distance(known_face_encodings,encode_face)

        match_index = np.argmin(face_distance)

        if matches[match_index]:
            name = names[match_index].upper()

            y1,x2,y2,x1 = facloc
            y1,x2,y2,x1 = y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(frame,(x1-1,y1-35),(x2+1,y1),(0,255,0),-1)
            cv2.putText(frame,name,(x1+6,y1-6),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,255,255),2)

    cv2.imshow("Window Video",frame)
    if cv2.waitKey(1) == ord('q'):
        break
video.release()
cv2.destroyAllWindows()




