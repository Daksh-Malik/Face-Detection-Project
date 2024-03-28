import cv2

face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
smileCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

#accessing webcam (0 is webcam)
video_capture = cv2.VideoCapture(0)

#defining a function to detect image from a particular video frame
def detect_bounding_box(vid):
    gray_image = cv2.cvtColor(vid, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray_image, 1.1, 5, minSize=(40, 40))
    for (x, y, w, h) in faces:
        cv2.rectangle(vid, (x, y), (x + w, y + h), (0, 255, 0), 4)
        roi_gray = gray_image[y:y+h, x:x+w]
        roi_color = vid[y:y+h, x:x+w]
        smile = smileCascade.detectMultiScale(
        roi_gray,
        scaleFactor= 1.5,
        minNeighbors=15,
        minSize=(25, 25),
        )
        
        for (xx, yy, ww, hh) in smile:
            cv2.rectangle(roi_color, (xx, yy), (xx + ww, yy + hh), (0, 255, 0), 2)
        
        if(len(smile)>0):
            cv2.putText(vid, "I am a Happy Human", (x, y + h + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(vid, "I am a Human", (x, y + h + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
    return faces

#real time face detecting
while(True):
    result, video_frame = video_capture.read()
    if result==False:
        break

    #calling detection function for the current video frame
    detect_bounding_box(video_frame)

    cv2.imshow("My Face Detection Project", video_frame)

    #making a quit key using 'q' key 
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video_capture.release()
cv2.destroyAllWindows()