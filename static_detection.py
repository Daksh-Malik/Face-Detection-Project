import cv2
import matplotlib.pyplot as plt

#reading image
imagePath = 'D:\CODING\PROJECTS\Project 4 - Face Detection\input_image.jpeg'
color_img = cv2.imread(imagePath)

print(color_img.shape)
#converting image to gray scale to improve efficiency
gray_image = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
print(gray_image.shape)

#creating a classifier object which uses a pretrained model
face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"  #this is a path to the data, 
)  #first part is where data is stored in cv2 library and '+' is used to concatenate the path

#detecting face
face = face_classifier.detectMultiScale(
    gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(40,40) 
) #here first parameter is image, second is image scale 1.1 means reducing the size by 10% for better detection, 
# third is the sliding window size through detection take place, fourth is minsize of face otherwise not detected

#creating rectangle
for (x, y, w, h) in face:
    cv2.rectangle(color_img, (x,y), (x+w, y+h), (0,255,0), 2)

#converting detected image to RGB format as openCV uses BGR format by default
img_rgb = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)

#using a library to display the image
plt.figure(figsize=(20,10))
plt.imshow(color_img)
plt.axis('off')
plt.show()