# IP

#Python program to explain cv2.imshow() method.<br>
import cv2<br>
path='BUTTERFLY3.jpg'<br>
i=cv2.imread(path,1)<br>
cv2.imshow('image',i)<br>
cv2.waitKey(0)<br>
cv2.destroyAllWindows()<br>

OUTPUT:<br>
![image](https://user-images.githubusercontent.com/97940475/173816997-24596b5d-4e42-46bb-855d-6d5be00da6ca.png)<br>

#Develop a program to display grey scale image using read and write operations.<br>
img=cv2.imread('BUTTERFLY1.jpg',0)<br>
cv2.imshow('image',img)<br>
cv2.waitKey(0)<br>
cv2.destroyAllWindows()<br>

OUPUT:<br>
![image](https://user-images.githubusercontent.com/97940475/173811858-b3894fbb-9298-4d18-8e4b-6b3e23ca7ec8.png)<br>

#Develop a program to display the image using matplotlib.<br>
import matplotlib.image as mping<br>
import matplotlib.pyplot as plt<br>
img=mping.imread('FLOWER1.jpg')<br>
plt.imshow(img)<br>

OUTPUT:<br>
![download](https://user-images.githubusercontent.com/97940475/173810784-5e5b6688-e5b2-4d5b-8f63-42c4a06d4c3f.png)<br>

#Develop a program to perform linear transformation.<br>
#1)Rotation<br>
#2)Scalling<br>
from PIL import Image<br>
img=Image.open("LEAF1.jpg")<br>
img=img.rotate(60)<br>
img.show()<br>
cv2.waitKey(0)<br>
cv2.destroyAllWindows()<br>

OUTPUT:<br>
![image](https://user-images.githubusercontent.com/97940475/173812270-14803676-cc20-45ca-bf79-59e5875ce08e.png)<br>

#Develop a program to convert color string to RGB color values.<br>
from PIL import ImageColor<br>
img1=ImageColor.getrgb("pink")<br>
print(img1)<br>
img2=ImageColor.getrgb("blue")<br>
print(img2)<br>

OUTPUT:<br>
(255, 192, 203)<br>
(0, 0, 255)<br>

#Write a program to create image using colors spaces.<br>
from PIL import Image<br>
img=Image.new('RGB',(200,400),(255,255,0))<br>
img.show()<br>

OUTPUT:<br>
![image](https://user-images.githubusercontent.com/97940475/173813213-de5acae2-6ef6-4202-94fa-0da206139145.png)<br>

#Develop a program to visualize the image using various color.<br>
import cv2<br>
import matplotlib.pyplot as plt<br>
import numpy as np<br>
img=cv2.imread('PLANT1.jpeg')<br>
plt.imshow(img)<br>
plt.show()<br>
img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)<br>
plt.imshow(img)<br>
plt.show()<br>
img=cv2.cvtColor(img,cv2.COLOR_RGB2HSV)<br>
plt.imshow(img)<br>
plt.show()<br>

OUTPUT:<br>
![download](https://user-images.githubusercontent.com/97940475/173812693-fee609fd-1ec3-43c4-9a9a-5586ccae2b41.png)<br>
![download](https://user-images.githubusercontent.com/97940475/173812723-c5b43eb9-e2af-4f52-809c-855512fdd217.png)<br>
![download](https://user-images.githubusercontent.com/97940475/173812756-1df9fc22-364c-4e6d-a6da-ba9cf24af554.png)<br>


#Write a program to display the image attributes.<br>
from PIL import Image<br>
image=Image.open('BUTTERFLY3.jpg')<br>
print("Filename:",image.filename)<br>
print("Format:",image.format)<br>
print("Mode:",image.mode)<br>
print("size:",image.size)<br>
print("Width:",image.width)<br>
print("Height:",image.height)<br>
image.close()<br>

OUTPUT:<br>
Filename: BUTTERFLY3.jpg<br>
Format: JPEG<br>
Mode: RGB<br>
size: (770, 662)<br>
Width: 770<br>
Height: 662<br>


#Convert the original image to gray scale and then to binary....<br>
import cv2<br>
img=cv2.imread('FLOWER2.jpg')<br>
print('Original image length width',img.shape)<br>
cv2.imshow('Original image',img)<br>
cv2.waitKey(0)<br>
imgresize=cv2.resize(img,(150,160))<br>
cv2.imshow('Resized image',imgresize)<br>
print('Resized image length width',imgresize.shape)<br>
cv2.waitKey(0)<br>

OUTPUT:<br>
![image](https://user-images.githubusercontent.com/97940475/174043738-b1cffb6b-41c9-4ce5-bcbf-d8413411edd2.png)<br>
![image](https://user-images.githubusercontent.com/97940475/174043822-d56fdf6b-7f5a-4584-abdb-144f70eb612e.png)<br>
Original image length width (668, 800, 3)<br>
Resized image length width (160, 150, 3)<br>


#Resize the original image<br>
import cv2<br>
img=cv2.imread('FLOWER3.jpeg')<br>
cv2.imshow("RGB",img)<br>
cv2.waitKey(0)<br>
img=cv2.imread('FLOWER3.jpeg',0)<br>
cv2.imshow("Gray",img)<br>
cv2.waitKey(0)<br>
ret,bw_img=cv2.threshold(img,127,255,cv2.THRESH_BINARY)<br>
cv2.imshow("Binary",bw_img)<br>
cv2.waitKey(0)<br>
cv2.destroyAllWindows()<br>

OUTPUT:<br>
![image](https://user-images.githubusercontent.com/97940475/174058087-2bd8b4e6-1cbc-4ef5-8f94-422d48cfdd77.png)<br>
![image](https://user-images.githubusercontent.com/97940475/174046096-495e1297-5069-4b38-9e69-75a50961ca82.png)<br>
![image](https://user-images.githubusercontent.com/97940475/174046187-aed38085-5874-4bc5-9854-85b5c7195113.png)<br>

#Develop a program to readimage using URL.<br>
from skimage import io<br>
import matplotlib.pyplot as plt<br>
url='https://cdn.theatlantic.com/thumbor/viW9N1IQLbCrJ0HMtPRvXPXShkU=/0x131:2555x1568/976x549/media/img/mt/2017/06/shutterstock_319985324/original.jpg'<br>
image=io.imread(url)<br>
plt.imshow(image)<br>
plt.show()<br>

OUTPUT:<br>
![download](https://user-images.githubusercontent.com/97940475/175264633-b00283fc-ac7c-4374-a52b-eb8ef2da7f94.png)<br>

#Write a program to mask and blur the image.<br>
import cv2<br>
import matplotlib.image as mping<br>
import matplotlib.pyplot as plt<br>
img=mping.imread('R.jpg')<br>
plt.imshow(img)<br>
plt.show()<br>
![download](https://user-images.githubusercontent.com/97940475/175264974-f98220d0-4d07-4eaf-80fc-904dcd6f0afc.png)<br>
hsv_img=cv2.cvtColor(img,cv2.COLOR_RGB2HSV)<br>
light_orange=(1, 190, 200)<br>
dark_orange=(18, 255, 255)<br>
mask=cv2.inRange(hsv_img,light_orange,dark_orange)<br>
result=cv2.bitwise_and(img,img,mask=mask)<br>
plt.subplot(1,2,1)<br>
plt.imshow(mask,cmap="gray")<br>
plt.subplot(1,2,2)<br>
plt.imshow(result)<br>
plt.show()<br>
![download](https://user-images.githubusercontent.com/97940475/175265164-61aa903f-f23b-4041-bf82-8f8fa1431b4e.png)<br>
light_white=(0,0,200)<br>
dark_white=(145,60,255)<br>
mask_white=cv2.inRange(hsv_img,light_white,dark_white)<br>
result_white=cv2.bitwise_and(img,img,mask=mask_white)<br>
plt.subplot(1,2,1)<br>
plt.imshow(mask_white,cmap="gray")<br>
plt.subplot(1,2,2)<br>
plt.imshow(result_white)<br>
plt.show()<br>
![download](https://user-images.githubusercontent.com/97940475/175265249-358a969d-837f-43f1-92c2-ac424f66a57e.png)<br>
final_mask=mask+mask_white<br>
final_result=cv2.bitwise_and(img,img,mask=final_mask)<br>
plt.subplot(1,2,1)<br>
plt.imshow(final_mask,cmap="gray")<br>
plt.subplot(1,2,2)<br>
plt.imshow(final_result)<br>
plt.show()<br>
![download](https://user-images.githubusercontent.com/97940475/175265334-c8a492a1-29c1-44c2-8283-3f1fab2ab1bd.png)<br>
blur=cv2.GaussianBlur(final_result, (7,7), 0)<br>
plt.imshow(blur)<br>
plt.show()<br>
![download](https://user-images.githubusercontent.com/97940475/175265449-7ee022b2-d1b9-4fbd-b77d-583c133ca31e.png)<br>

#Write a program to mask and blur the image.<br>
import cv2<br>
import matplotlib.image as mping<br>
import matplotlib.pyplot as plt<br>
img=mping.imread('img.jpg')<br>
plt.imshow(img)<br>
plt.show()<br>
![download](https://user-images.githubusercontent.com/97940475/175265765-0035f402-4038-4aba-a256-7d6638d1917c.png)<br>
light_white=(0,0,200)<br>
dark_white=(145,60,255)<br>
mask_white=cv2.inRange(hsv_img,light_white,dark_white)<br>
result_white=cv2.bitwise_and(img,img,mask=mask_white)<br>
plt.subplot(1,2,1)<br>
plt.imshow(mask_white,cmap="gray")<br>
plt.subplot(1,2,2)<br>
plt.imshow(result_white)<br>
plt.show()<br>
![download](https://user-images.githubusercontent.com/97940475/175265866-69c284b7-3ebe-4fc1-8bdb-bb97a9b9094e.png)<br>
blur=cv2.GaussianBlur(result_white, (7,7), 0)<br>
plt.imshow(blur)<br>
plt.show()<br>
![download](https://user-images.githubusercontent.com/97940475/175265927-7ca7c9df-e4aa-4551-9e38-d0f02b402d67.png)<br>

#Write a program to perform arithmatic operations on images<br>
import cv2<br>
import matplotlib.image as mping<br>
import matplotlib.pyplot as plt<br>

#Reading image file<br>
img1=cv2.imread('FLOWER1.jpg')<br>
img2=cv2.imread('BUTTERFLY3.jpg')<br>

#Applying numpy addition on images<br>
fimg1 = img1 + img2<br>
plt.imshow(fimg1)<br>
plt.show()<br>

#saving the output images<br>
cv2.imwrite('output.jpg',fimg1)<br>
fimg2 = img1 - img2<br>
plt.imshow(fimg2)<br>
plt.show()<br>

#saving the output image<br>
cv2.imwrite('output.jpg',fimg2)<br>
fimg3 = img1 * img2<br>
plt.imshow(fimg3)<br>
plt.show()<br>

#saving the output image<br>
cv2.imwrite('output.jpg',fimg3)<br>
fimg4 = img1 / img2<br>
plt.imshow(fimg4)<br>
plt.show()<br>

#saving the output image<br>
cv2.imwrite('output.jpg',fimg4)<br>

OUTPUT:<br>
![download](https://user-images.githubusercontent.com/97940475/175266340-77562e3d-46e1-47a3-bce0-06e1115edc7b.png)<br>
![download](https://user-images.githubusercontent.com/97940475/175266390-7b4b0af4-0ecb-4b60-bb4e-f67a7b53f3c6.png)<br>
![download](https://user-images.githubusercontent.com/97940475/175266441-6fc61c5e-4eb5-4540-909d-d8f05efe33c5.png)<br>
![download](https://user-images.githubusercontent.com/97940475/175266508-9e240717-910e-439b-9719-beb6a97b96b9.png)<br>

#Develop the program to change the image to different color spaces.<br>
import cv2 <br>
img=cv2.imread("PLANT5.jpg")<br>
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)<br>
hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)<br>
lab=cv2.cvtColor(img,cv2.COLOR_BGR2LAB)<br>
hls=cv2.cvtColor(img,cv2.COLOR_BGR2HLS)<br>
yuv=cv2.cvtColor(img,cv2.COLOR_BGR2YUV)<br>
cv2.imshow("GRAY image",gray)<br>
cv2.imshow("HSV image",hsv)<br>
cv2.imshow("LAB image",lab)
cv2.imshow("HLS image",hls)<br>
cv2.imshow("YUV image",yuv)<br>
cv2.waitKey(0)<br>
cv2.destroyAllWindows()<br>

OUTPUT:<br>
![image](https://user-images.githubusercontent.com/97940475/175266819-0c7f15f6-d04e-4065-9baa-86e3a93d83e2.png)<br>
![image](https://user-images.githubusercontent.com/97940475/175266936-c8cedc80-94c2-45f0-ac13-a9ba7f1b5c85.png)<br>
![image](https://user-images.githubusercontent.com/97940475/175267057-08276b48-6976-4691-b99f-38e396e39478.png)<br>
![image](https://user-images.githubusercontent.com/97940475/175267228-3c8c3fef-f28a-42e2-8b36-316f25c2dd32.png)<br>
![image](https://user-images.githubusercontent.com/97940475/175267310-752ec506-b426-4088-bbef-933e68b2e6be.png)<br>

#Program to create an image using 2D array<br>
import cv2 as  c<br>
import numpy as np<br>
from PIL import Image<br>
array=np.zeros([100,200,3],dtype=np.uint8)<br>
array[:,:100]=[255,130,0]<br>
array[:,100:]=[0,0,255]<br>
img=Image.fromarray(array)<br>
img.save('IMAGES.jpg')<br>
img.show()<br>
c.waitKey(0)<br>

OUTPUT:<br>
![image](https://user-images.githubusercontent.com/97940475/175268623-22cefe1a-fb53-46e1-9e3d-fce04be704b5.png)<br>




