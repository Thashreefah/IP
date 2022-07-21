# IP

1)Python program to explain cv2.imshow() method.<br>
import cv2<br>
path='BUTTERFLY3.jpg'<br>
i=cv2.imread(path,1)<br>
cv2.imshow('image',i)<br>
cv2.waitKey(0)<br>
cv2.destroyAllWindows()<br>

OUTPUT:<br>
![image](https://user-images.githubusercontent.com/97940475/173816997-24596b5d-4e42-46bb-855d-6d5be00da6ca.png)<br>

2)Develop a program to display grey scale image using read and write operations.<br>
import cv2<br>
img=cv2.imread('BUTTERFLY1.jpg',0)<br>
cv2.imshow('image',img)<br>
cv2.waitKey(0)<br>
cv2.destroyAllWindows()<br>

OUPUT:<br>
![image](https://user-images.githubusercontent.com/97940475/178718561-fb71e4fb-17e9-43e4-9a29-8ac8e21cdb35.png)<br>

3)Develop a program to display the image using matplotlib.<br>
import matplotlib.image as mping<br>
import matplotlib.pyplot as plt<br>
img=mping.imread('FLOWER1.jpg')<br>
plt.imshow(img)<br>

OUTPUT:<br>
![download](https://user-images.githubusercontent.com/97940475/173810784-5e5b6688-e5b2-4d5b-8f63-42c4a06d4c3f.png)<br>

4)Develop a program to perform linear transformation.<br>
1-Rotation<br>
2-Scalling<br>
from PIL import Image<br>
img=Image.open("LEAF1.jpg")<br>
img=img.rotate(60)<br>
img.show()<br>
cv2.waitKey(0)<br>
cv2.destroyAllWindows()<br>

OUTPUT:<br>
![image](https://user-images.githubusercontent.com/97940475/173812270-14803676-cc20-45ca-bf79-59e5875ce08e.png)<br>

5)Develop a program to convert color string to RGB color values.<br>
from PIL import ImageColor<br>
img1=ImageColor.getrgb("pink")<br>
print(img1)<br>
img2=ImageColor.getrgb("blue")<br>
print(img2)<br>

OUTPUT:<br>
(255, 192, 203)<br>
(0, 0, 255)<br>

6)Write a program to create image using colors spaces.<br>
from PIL import Image<br>
img=Image.new('RGB',(200,400),(255,255,0))<br>
img.show()<br>

OUTPUT:<br>
![image](https://user-images.githubusercontent.com/97940475/173813213-de5acae2-6ef6-4202-94fa-0da206139145.png)<br>

7)Develop a program to visualize the image using various color.<br>
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


8)Write a program to display the image attributes.<br>
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

9)Resize the original image<br>
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

10)Convert the original image to gray scale and then to binary....<br>
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
![image](https://user-images.githubusercontent.com/97940475/178947800-c12cafe7-93e0-4f71-96b2-996687b32a50.png)<br>
![image](https://user-images.githubusercontent.com/97940475/178947918-f91cce49-a8e6-4588-adbb-ff7adf419339.png)<br>
![image](https://user-images.githubusercontent.com/97940475/178948004-9e07fd78-ffe9-4b64-bd8f-6ed2188bcf09.png)<br>

11)Develop a program to readimage using URL.<br>
from skimage import io<br>
import matplotlib.pyplot as plt<br>
url='https://cdn.theatlantic.com/thumbor/viW9N1IQLbCrJ0HMtPRvXPXShkU=/0x131:2555x1568/976x549/media/img/mt/2017/06/shutterstock_319985324/original.jpg'<br>
image=io.imread(url)<br>
plt.imshow(image)<br>
plt.show()<br>

OUTPUT:<br>
![download](https://user-images.githubusercontent.com/97940475/175264633-b00283fc-ac7c-4374-a52b-eb8ef2da7f94.png)<br>

12)Write a program to mask and blur the image.<br>
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

13)Write a program to mask and blur the image.<br>
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

14)Write a program to perform arithmatic operations on images<br>
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

15)Develop the program to change the image to different color spaces.<br>
import cv2 <br>
img=cv2.imread("PLANT5.jpg")<br>
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)<br>
hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)<br>
lab=cv2.cvtColor(img,cv2.COLOR_BGR2LAB)<br>
hls=cv2.cvtColor(img,cv2.COLOR_BGR2HLS)<br>
yuv=cv2.cvtColor(img,cv2.COLOR_BGR2YUV)<br>
cv2.imshow("GRAY image",gray)<br>
cv2.imshow("HSV image",hsv)<br>
cv2.imshow("LAB image",lab)<br>
cv2.imshow("HLS image",hls)<br>
cv2.imshow("YUV image",yuv)<br>
cv2.waitKey(0)<br>
cv2.destroyAllWindows()<br>

OUTPUT:<br>
![image](https://user-images.githubusercontent.com/97940475/178949251-6c19da34-415a-43bd-a24f-53c4571051be.png)<br>
![image](https://user-images.githubusercontent.com/97940475/178949325-b9d21f76-4656-428a-a6c0-46b85923cca7.png)<br>
![image](https://user-images.githubusercontent.com/97940475/178949395-bbc19379-624b-49e1-ba18-c9958e99b485.png)<br>
![image](https://user-images.githubusercontent.com/97940475/178949451-3d9dbede-ed10-4451-8e60-fa79e4524ebd.png)<br>
![image](https://user-images.githubusercontent.com/97940475/178949498-45d33b7e-84c1-4625-812f-11e0e66a332a.png)<br>


16)Program to create an image using 2D array<br>
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

17)
import cv2<br>
import matplotlib.pyplot as plt<br>
image1=cv2.imread('BUTTERFLY1.png',1)<br>
image2=cv2.imread('BUTTERFLY1.png')<br>
ax=plt.subplots(figsize=(15,10))<br>
bitwiseAnd=cv2.bitwise_and(image1,image2)<br>
bitwiseOr=cv2.bitwise_or(image1,image2)<br>
bitwiseXor=cv2.bitwise_xor(image1,image2)<br>
bitwiseNot_img1=cv2.bitwise_not(image1)<br>
bitwiseNot_img2=cv2.bitwise_not(image2)<br>
plt.subplot(151)<br>
plt.imshow(bitwiseAnd)<br>
plt.subplot(152)<br>
plt.imshow(bitwiseOr)<br>
plt.subplot(153)<br>
plt.imshow(bitwiseXor)<br>
plt.subplot(154)<br>
plt.imshow(bitwiseNot_img1)<br>
plt.subplot(155)<br>
plt.imshow(bitwiseNot_img2)<br>
cv2.waitKey(0)<br>

OUTPUT:<br>
![download](https://user-images.githubusercontent.com/97940475/176416570-60218c0f-0cea-4c47-9e84-be20fa1b0146.png)<br>

18)
#importing libraries<br>
import cv2<br>
import numpy as np<br>
image=cv2.imread('BUTTERFLY1.png')<br>
cv2.imshow('Original Image',image)<br>
cv2.waitKey(0)<br>

#Gussian Blur<br>
Gaussian=cv2.GaussianBlur(image,(7,7),0)<br>
cv2.imshow('Gaussian Blurring',Gaussian)<br>
cv2.waitKey(0)<br>

#Median Blur<br>
median=cv2.medianBlur(image,5)<br>
cv2.imshow('Median Blurring',median)<br>
cv2.waitKey(0)<br>

#Bilateral Blur<br>
bilateral=cv2.bilateralFilter(image,9,75,75)<br>
cv2.imshow('Bilateral blurring',bilateral)<br>
cv2.waitKey(0)<br>
cv2.destroyAllWindows()<br>

OUTPUT:<br>
![image](https://user-images.githubusercontent.com/97940475/178950941-e517f39b-db6f-468f-9d26-0fb3b8791946.png)<br>
![image](https://user-images.githubusercontent.com/97940475/178950994-a00beb94-2392-4dae-b442-0d1b5572d252.png)<br>
![image](https://user-images.githubusercontent.com/97940475/178951093-50dc27a8-56ad-42bd-b8c7-c76c1c0a1e43.png)<br>
![image](https://user-images.githubusercontent.com/97940475/178951178-83a2a34d-dbf9-4db9-a6e0-3a47d111dc6e.png)<br>


19)
from PIL import Image<br>
from PIL import ImageEnhance<br>
image=Image.open('BUTTERFLY2.jpg')<br>
image.show()<br>
enh_bri=ImageEnhance.Brightness(image)<br>
brightness=1.5<br>
image_brightened=enh_bri.enhance(brightness)<br>
image_brightened.show()<br>
enh_col=ImageEnhance.Color(image)<br>
color=1.5<br>
image_colored=enh_col.enhance(color)<br>
image_colored.show()<br>
enh_con=ImageEnhance.Contrast(image)<br>
contrast=1.5<br>
image_contrasted=enh_con.enhance(contrast)<br>
image_contrasted.show()<br>
enh_sha=ImageEnhance.Sharpness(image)<br>
sharpness=3.0<br>
image_sharped=enh_sha.enhance(sharpness)<br>
image_sharped.show()<br>

OUTPUT:<br>
![image](https://user-images.githubusercontent.com/97940475/178953621-413ea939-662c-4194-8989-a445e1d83017.png)<br>
![image](https://user-images.githubusercontent.com/97940475/178953687-0348755a-4ea3-4659-af15-8ab0293d9149.png)<br>
![image](https://user-images.githubusercontent.com/97940475/178953758-88faed1f-0538-44d0-bd7e-b202b861bf95.png)<br>
![image](https://user-images.githubusercontent.com/97940475/178953829-c87695c0-6354-46be-a5a1-040197e1095e.png)<br>
![image](https://user-images.githubusercontent.com/97940475/178953891-3fe1cbaa-bb1c-4a0d-94ce-b53be2cadde0.png)<br>


20)
import cv2<br>
import numpy as np<br>
#from matplotlib import pyplt as plt<br>
import matplotlib.pyplot as plt<br>
from PIL import Image,ImageEnhance<br>
img=cv2.imread('FLOWER1.JPG',0)<br>
ax=plt.subplots(figsize=(20,10))<br>
kernel=np.ones((5,5),np.uint8)<br>
opening=cv2.morphologyEx(img,cv2.MORPH_OPEN,kernel)<br>
closing=cv2.morphologyEx(img,cv2.MORPH_CLOSE,kernel)<br>
erosion=cv2.erode(img,kernel,iterations=1)<br>
dilation=cv2.dilate(img,kernel,iterations=1)<br>
gradient=cv2.morphologyEx(img,cv2.MORPH_GRADIENT,kernel)<br>
plt.subplot(151)<br>
plt.imshow(opening)<br>
plt.subplot(152)<br>
plt.imshow(closing)<br>
plt.subplot(153)<br>
plt.imshow(erosion)<br>
plt.subplot(154)<br>
plt.imshow(dilation)<br>
plt.subplot(155)<br>
plt.imshow(gradient)<br>
cv2.waitKey(0)<br>

OUTPUT:<br>
![download](https://user-images.githubusercontent.com/97940475/176422302-8961beeb-659a-4d07-a742-d790f30c0861.png)<br>

21)Develop a program to<br>
i)Read the image,convert it into grayscale image<br>
ii)Write(save) the grayscale image and<br>
iii)Display the original image and grayscale image<br>
import cv2<br>
OriginalImg=cv2.imread('FLOWER1.jpg')<br>
GrayImg=cv2.imread('FLOWER1.jpg',0)<br>
isSaved=cv2.imwrite('C:/thash/th.jpg',GrayImg)<br>
cv2.imshow('Display Original Image',OriginalImg)<br>
cv2.imshow('Display GrayScale Image',GrayImg)<br>
cv2.waitKey(0)<br>
cv2.destroyAllWindows()<br>
if isSaved:<br>
    print('The image is succesfully saved.')<br>

OUTPUT:<br>
![image](https://user-images.githubusercontent.com/97940475/178954684-5d3fb0ec-f4b7-4e98-b665-e6e29fc1d527.png)<br>


22)
import cv2<br>
import numpy as np<br>
from matplotlib import pyplot as plt<br>
image=cv2.imread('CAT1.jpg',0)<br>
x,y=image.shape<br>
z=np.zeros((x,y))<br>
for i in range(0,x):<br>
    for j in range(0,y):<br>
        if(image[i][j]>50 and image[i][j]<150):<br>
            z[i][j]=255<br>
        else:<br>
            z[i][j]=image[i][j]<br>
equ=np.hstack((image,z))<br>
plt.title('Graylevel slicing with background')<br>
plt.imshow(equ,'gray')<br>
plt.show()<br>

OUTPUT:<br>
![download](https://user-images.githubusercontent.com/97940475/178706167-badef28f-b034-438e-a384-df738d518413.png)<br>

23)
import cv2<br>
import numpy as np<br>
from matplotlib import pyplot as plt<br>
image=cv2.imread('CAT1.jpg',0)<br>
x,y=image.shape<br>
z=np.zeros((x,y))<br>
for i in range(0,x):<br>
    for j in range(0,y):<br>
        if(image[i][j]>50 and image[i][j]<150):<br>
            z[i][j]=255<br>
        else:<br>
            z[i][j]=0<br>
equ=np.hstack((image,z))<br>
plt.title('Graylevel slicing without background')<br>
plt.imshow(equ,'gray')<br>
plt.show()<br>

OUTPUT:<br>
![download](https://user-images.githubusercontent.com/97940475/178706345-ec3389d2-4597-4452-a32f-16601e9bee4e.png)<br>

24)Analyze the image using Histogram<br>
import numpy as np<br>
import skimage.color<br>
import skimage.io<br>
import matplotlib.pyplot as plt<br>

#read the image of a plant seedling as grayscale from the outset<br>
image = skimage.io.imread(fname="DOG1.jpg",as_gray=True)<br>
image1 = skimage.io.imread(fname="DOG1.jpg") <br>

#display the image<br>
fig, ax = plt.subplots()<br>
plt.imshow(image, cmap="gray")<br>
plt.show()<br>

fig, ax = plt.subplots()<br>
plt.imshow(image1,cmap="gray")<br>
plt.show()<br>

#create the histogram<br>
histogram, bin_edges = np.histogram(image, bins=256, range=(0, 1))<br>

#configure and draw the histogram figure<br>
plt.figure()<br>
plt.title("Grayscale Histogram")<br>
plt.xlabel("grayscale value")<br>
plt.ylabel("pixel count")<br>
plt.xlim([0.0, 1.0])  # <- named arguments do not work here<br>

plt.plot(bin_edges[0:-1], histogram)  # <- or here<br>
plt.show()<br>

OUTPUT:<br>
![download](https://user-images.githubusercontent.com/97940475/178964765-f7066d16-27c1-4018-9d63-44d7c76ae84c.png)<br>
![download](https://user-images.githubusercontent.com/97940475/178964775-8eb09887-2297-4f1b-8dc9-6a06672855a2.png)<br>
![download](https://user-images.githubusercontent.com/97940475/178964810-41c45a12-8c56-4bed-bb6c-73e898a4e5df.png)<br>

25)Program to perform basic image data analysis using intensity transformation:<br>
a) Image negative<br>
b) Log transformation<br>
c) Gamma correction<br>
%matplotlib inline<br>
import imageio<br>
import matplotlib.pyplot as plt<br>
import warnings<br>
import matplotlib.cbook<br>
warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)<br>
pic=imageio.imread('PARROT1.jpg')<br>
plt.figure(figsize=(6,6))<br>
plt.imshow(pic);<br>
plt.axis('off');<br>

![download](https://user-images.githubusercontent.com/97940475/180177056-370f4b50-ab9e-45e4-bcd2-8c28bc9a3b90.png)<br>

negative=255-pic #neg=(l-1)-img<br>
plt.figure(figsize=(6,6))<br>
plt.imshow(negative);<br>
plt.axis('off');<br>

![download](https://user-images.githubusercontent.com/97940475/180177135-248c58e8-ee68-4585-86dd-a928d1799409.png)<br>

%matplotlib inline<br>

import imageio<br>
import numpy as np<br>
import matplotlib.pyplot as plt<br>

pic=imageio.imread('PARROT1.jpg')<br>
gray=lambda rgb:np.dot(rgb[...,:3],[0.299,0.587,0.114])<br>
gray=gray(pic)<br>

max_=np.max(gray)<br>

def log_transform():<br>
    return(255/np.log(1+max_))*np.log(1+gray)<br>
plt.figure(figsize=(5,5))<br>
plt.imshow(log_transform(),cmap=plt.get_cmap(name='gray'))<br>
plt.axis('off');<br>

![download](https://user-images.githubusercontent.com/97940475/180177227-11d94183-69b6-4431-9c6d-19167e9bd1da.png)<br>

import imageio<br>
import matplotlib.pyplot as plt<br>

#Gamma encoding<br>
pic=imageio.imread('PARROT1.jpg')<br>
gamma=2.2 #Gamma<1~Dark;Gamma >~Bright<br>

gamma_correction=((pic/255)**(1/gamma))<br>
plt.figure(figsize=(5,5))<br>
plt.imshow(gamma_correction)<br>
plt.axis('off');<br>

![download](https://user-images.githubusercontent.com/97940475/180177303-672cb0d1-5b3f-4544-adbf-7121f2dbc72b.png)<br>

26)Program to perform basic image manipulation:<br>
a) Sharpness<br>
b) Flipping<br>
c) Cropping<br>
#Image sharpen<br>
from PIL import Image<br>
from PIL import ImageFilter<br>
import matplotlib.pyplot as plt<br>

#Load the image<br>
my_image=Image.open('FISH1.jpg')<br>
plt.imshow(my_image)<br>
plt.show()<br>
#Use sharpen function<br>
sharp=my_image.filter(ImageFilter.SHARPEN)<br>
#Save the image<br>
sharp.save('C:/thash/Image_sharpen.jpg')<br>
sharp.show()<br>
plt.imshow(sharp)<br>
plt.show()<br>

![download](https://user-images.githubusercontent.com/97940475/180177456-ba983cbe-fc62-40ab-8e59-c997dcbcf87f.png)<br>
![download](https://user-images.githubusercontent.com/97940475/180177480-b8edbc59-9004-4ab0-8578-6074747ab04a.png)<br>

#Image flip<br>
import matplotlib.pyplot as plt<br>
#load the image<br>
img=Image.open('FISH1.jpg')<br>
plt.imshow(img)<br>
plt.show()<br>
#use the flip function<br>
flip=img.transpose(Image.FLIP_LEFT_RIGHT)<br>

#save the image<br>
flip.save('C:/thash/Image_flip.jpg')<br>
plt.imshow(flip)<br>
plt.show()<br>

![download](https://user-images.githubusercontent.com/97940475/180177565-6598cb13-7c0c-438a-997e-9659b4a7c028.png)<br>
![download](https://user-images.githubusercontent.com/97940475/180177585-39f46015-876f-4a8b-932e-47a26bf4bac5.png)<br>

#Image Crop<br>

#Importing Image class from PIL module<br>
from PIL import Image<br>
import matplotlib.pyplot as plt<br>
#Opens a image in RGB mode<br>
im=Image.open('FISH1.jpg')<br>

#Size of the image in pixels(size of original image)<br>
#(This is not mandatory)<br>
width,height=im.size<br>

#Cropped image of above dimension<br>
#(It will not Change original image)<br>
im1=im.crop((50,200,3000,1600))<br>

#Shows the image in image viewer<br>
im1.show()<br>
plt.imshow(im1)<br>
plt.show()<br>

![download](https://user-images.githubusercontent.com/97940475/180177663-5e56e468-ee07-4b71-9bbf-5493e7d935fb.png)<br>
