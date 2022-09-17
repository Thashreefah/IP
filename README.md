# IP

**1)Python program to explain cv2.imshow() method.<br>**
import cv2<br>
path='BUTTERFLY3.jpg'<br>
i=cv2.imread(path,1)<br>
cv2.imshow('image',i)<br>
cv2.waitKey(0)<br>
cv2.destroyAllWindows()<br>

**OUTPUT:<br>**
![image](https://user-images.githubusercontent.com/97940475/173816997-24596b5d-4e42-46bb-855d-6d5be00da6ca.png)<br>

**2)Develop a program to display grey scale image using read and write operations.<br>**
import cv2<br>
img=cv2.imread('BUTTERFLY1.jpg',0)<br>
cv2.imshow('image',img)<br>
cv2.waitKey(0)<br>
cv2.destroyAllWindows()<br>

**OUPUT:<br>**
![image](https://user-images.githubusercontent.com/97940475/178718561-fb71e4fb-17e9-43e4-9a29-8ac8e21cdb35.png)<br>

**3)Develop a program to display the image using matplotlib.<br>**
import matplotlib.image as mping<br>
import matplotlib.pyplot as plt<br>
img=mping.imread('FLOWER1.jpg')<br>
plt.imshow(img)<br>

**OUTPUT:<br>**
![download](https://user-images.githubusercontent.com/97940475/173810784-5e5b6688-e5b2-4d5b-8f63-42c4a06d4c3f.png)<br>

**4)Develop a program to perform linear transformation.<br>
1-Rotation<br>
2-Scalling<br>**
from PIL import Image<br>
img=Image.open("LEAF1.jpg")<br>
img=img.rotate(60)<br>
img.show()<br>
cv2.waitKey(0)<br>
cv2.destroyAllWindows()<br>

**OUTPUT:<br>**
![image](https://user-images.githubusercontent.com/97940475/173812270-14803676-cc20-45ca-bf79-59e5875ce08e.png)<br>

**5)Develop a program to convert color string to RGB color values.<br>**
from PIL import ImageColor<br>
img1=ImageColor.getrgb("pink")<br>
print(img1)<br>
img2=ImageColor.getrgb("blue")<br>
print(img2)<br>

**OUTPUT:<br>**
(255, 192, 203)<br>
(0, 0, 255)<br>

**6)Write a program to create image using colors spaces.<br>**
from PIL import Image<br>
img=Image.new('RGB',(200,400),(255,255,0))<br>
img.show()<br>

**OUTPUT:<br>**
![image](https://user-images.githubusercontent.com/97940475/173813213-de5acae2-6ef6-4202-94fa-0da206139145.png)<br>

**7)Develop a program to visualize the image using various color.<br>**
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

**OUTPUT:<br>**
![download](https://user-images.githubusercontent.com/97940475/173812693-fee609fd-1ec3-43c4-9a9a-5586ccae2b41.png)<br>
![download](https://user-images.githubusercontent.com/97940475/173812723-c5b43eb9-e2af-4f52-809c-855512fdd217.png)<br>
![download](https://user-images.githubusercontent.com/97940475/173812756-1df9fc22-364c-4e6d-a6da-ba9cf24af554.png)<br>


**8)Write a program to display the image attributes.<br>**
from PIL import Image<br>
image=Image.open('BUTTERFLY3.jpg')<br>
print("Filename:",image.filename)<br>
print("Format:",image.format)<br>
print("Mode:",image.mode)<br>
print("size:",image.size)<br>
print("Width:",image.width)<br>
print("Height:",image.height)<br>
image.close()<br>

**OUTPUT:<br>**
Filename: BUTTERFLY3.jpg<br>
Format: JPEG<br>
Mode: RGB<br>
size: (770, 662)<br>
Width: 770<br>
Height: 662<br>

**9)Resize the original image<br>**
import cv2<br>
img=cv2.imread('FLOWER2.jpg')<br>
print('Original image length width',img.shape)<br>
cv2.imshow('Original image',img)<br>
cv2.waitKey(0)<br>
imgresize=cv2.resize(img,(150,160))<br>
cv2.imshow('Resized image',imgresize)<br>
print('Resized image length width',imgresize.shape)<br>
cv2.waitKey(0)<br>

**OUTPUT:<br>**
![image](https://user-images.githubusercontent.com/97940475/174043738-b1cffb6b-41c9-4ce5-bcbf-d8413411edd2.png)<br>
![image](https://user-images.githubusercontent.com/97940475/174043822-d56fdf6b-7f5a-4584-abdb-144f70eb612e.png)<br>
Original image length width (668, 800, 3)<br>
Resized image length width (160, 150, 3)<br>

**10)Convert the original image to gray scale and then to binary....<br>**
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

**OUTPUT:<br>**
![image](https://user-images.githubusercontent.com/97940475/178947800-c12cafe7-93e0-4f71-96b2-996687b32a50.png)<br>
![image](https://user-images.githubusercontent.com/97940475/178947918-f91cce49-a8e6-4588-adbb-ff7adf419339.png)<br>
![image](https://user-images.githubusercontent.com/97940475/178948004-9e07fd78-ffe9-4b64-bd8f-6ed2188bcf09.png)<br>

**11)Develop a program to readimage using URL.<br>**
from skimage import io<br>
import matplotlib.pyplot as plt<br>
url='https://cdn.theatlantic.com/thumbor/viW9N1IQLbCrJ0HMtPRvXPXShkU=/0x131:2555x1568/976x549/media/img/mt/2017/06/shutterstock_319985324/original.jpg'<br>
image=io.imread(url)<br>
plt.imshow(image)<br>
plt.show()<br>

**OUTPUT:<br>**
![download](https://user-images.githubusercontent.com/97940475/175264633-b00283fc-ac7c-4374-a52b-eb8ef2da7f94.png)<br>

**12)Write a program to mask and blur the image.<br>**
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

**12.1(EXTRA)Write a program to mask and blur the image.<br>**
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

**13)Write a program to perform arithmatic operations on images<br>**
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

**OUTPUT:<br>**
![download](https://user-images.githubusercontent.com/97940475/175266340-77562e3d-46e1-47a3-bce0-06e1115edc7b.png)<br>
![download](https://user-images.githubusercontent.com/97940475/175266390-7b4b0af4-0ecb-4b60-bb4e-f67a7b53f3c6.png)<br>
![download](https://user-images.githubusercontent.com/97940475/175266441-6fc61c5e-4eb5-4540-909d-d8f05efe33c5.png)<br>
![download](https://user-images.githubusercontent.com/97940475/175266508-9e240717-910e-439b-9719-beb6a97b96b9.png)<br>

**14)Develop the program to change the image to different color spaces.<br>**
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

**OUTPUT:<br>**
![image](https://user-images.githubusercontent.com/97940475/178949251-6c19da34-415a-43bd-a24f-53c4571051be.png)<br>
![image](https://user-images.githubusercontent.com/97940475/178949325-b9d21f76-4656-428a-a6c0-46b85923cca7.png)<br>
![image](https://user-images.githubusercontent.com/97940475/178949395-bbc19379-624b-49e1-ba18-c9958e99b485.png)<br>
![image](https://user-images.githubusercontent.com/97940475/178949451-3d9dbede-ed10-4451-8e60-fa79e4524ebd.png)<br>
![image](https://user-images.githubusercontent.com/97940475/178949498-45d33b7e-84c1-4625-812f-11e0e66a332a.png)<br>


**15)Program to create an image using 2D array<br>**
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

**OUTPUT:<br>**
![image](https://user-images.githubusercontent.com/97940475/175268623-22cefe1a-fb53-46e1-9e3d-fce04be704b5.png)<br>

**16)Bitwise operation<br>**
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

**OUTPUT:<br>**
![download](https://user-images.githubusercontent.com/97940475/176416570-60218c0f-0cea-4c47-9e84-be20fa1b0146.png)<br>

**17)Blurring image<br>**
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

**OUTPUT:<br>**
![image](https://user-images.githubusercontent.com/97940475/178950941-e517f39b-db6f-468f-9d26-0fb3b8791946.png)<br>
![image](https://user-images.githubusercontent.com/97940475/178950994-a00beb94-2392-4dae-b442-0d1b5572d252.png)<br>
![image](https://user-images.githubusercontent.com/97940475/178951093-50dc27a8-56ad-42bd-b8c7-c76c1c0a1e43.png)<br>
![image](https://user-images.githubusercontent.com/97940475/178951178-83a2a34d-dbf9-4db9-a6e0-3a47d111dc6e.png)<br>


**18)Image Enhancement<br>**
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

**OUTPUT:<br>**
![image](https://user-images.githubusercontent.com/97940475/178953621-413ea939-662c-4194-8989-a445e1d83017.png)<br>
![image](https://user-images.githubusercontent.com/97940475/178953687-0348755a-4ea3-4659-af15-8ab0293d9149.png)<br>
![image](https://user-images.githubusercontent.com/97940475/178953758-88faed1f-0538-44d0-bd7e-b202b861bf95.png)<br>
![image](https://user-images.githubusercontent.com/97940475/178953829-c87695c0-6354-46be-a5a1-040197e1095e.png)<br>
![image](https://user-images.githubusercontent.com/97940475/178953891-3fe1cbaa-bb1c-4a0d-94ce-b53be2cadde0.png)<br>


**19)Morpholigical operation<br>**
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

**OUTPUT:<br>**
![download](https://user-images.githubusercontent.com/97940475/176422302-8961beeb-659a-4d07-a742-d790f30c0861.png)<br>

**20)Develop a program to<br>
i)Read the image,convert it into grayscale image<br>
ii)Write(save) the grayscale image and<br>
iii)Display the original image and grayscale image<br>**
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

**OUTPUT:<br>**
![image](https://user-images.githubusercontent.com/97940475/178954684-5d3fb0ec-f4b7-4e98-b665-e6e29fc1d527.png)<br>
The Image Is Successfully saved

**21)Slicing with background<br>**
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

**OUTPUT:<br>**
![download](https://user-images.githubusercontent.com/97940475/178706167-badef28f-b034-438e-a384-df738d518413.png)<br>

**22)Slicing without background<br>**
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

**OUTPUT:<br>**
![download](https://user-images.githubusercontent.com/97940475/178706345-ec3389d2-4597-4452-a32f-16601e9bee4e.png)<br>

**23)Analyze the image using Histogram<br>**
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

**OUTPUT:<br>**
![download](https://user-images.githubusercontent.com/97940475/178964765-f7066d16-27c1-4018-9d63-44d7c76ae84c.png)<br>
![download](https://user-images.githubusercontent.com/97940475/178964775-8eb09887-2297-4f1b-8dc9-6a06672855a2.png)<br>
![download](https://user-images.githubusercontent.com/97940475/178964810-41c45a12-8c56-4bed-bb6c-73e898a4e5df.png)<br>

**24)Program to perform basic image data analysis using intensity transformation:<br>
a) Image negative<br>
b) Log transformation<br>
c) Gamma correction<br>**
#%matplotlib inline<br>
import imageio<br>
import matplotlib.pyplot as plt<br>
#import warnings<br>
#import matplotlib.cbook<br>
#warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)<br>
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

**25)Program to perform basic image manipulation:<br>
a) Sharpness<br>
b) Flipping<br>
c) Cropping<br>**
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
![image](https://user-images.githubusercontent.com/97940475/180179226-0595a272-f726-48f2-80be-409e41086e25.png)

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
![image](https://user-images.githubusercontent.com/97940475/180179027-cda688c9-0795-4f1f-a47f-53a43b5795a3.png)

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

#d) Roberts Edge Detection- Roberts cross operator
#Roberts Edge Detection- Roberts cross operator
import cv2
import numpy as np
from scipy import ndimage
from matplotlib import pyplot as plt
roberts_cross_v=np.array([[1,0],[0,-1]])
roberts_cross_h=np.array([[0,1],[-1,0]])

img=cv2.imread("color.jpg",0).astype('float64')
img/=255.0
vertical=ndimage.convolve(img,roberts_cross_v)
horizontal=ndimage.convolve(img,roberts_cross_h)

edged_img=np.sqrt(np.square(horizontal)+np.square(vertical))
edged_img*=255
cv2.imwrite("Output.jpg",edged_img)
cv2.imshow("OutputImage",edged_img)
cv2.waitKey()
cv2.destroyAllWindows()


26)
from PIL import Image,ImageChops,ImageFilter<br>
from matplotlib import pyplot as plt <br>

#Create a PIL Image object<br>
x=Image.open("x.png")<br>
o=Image.open("o.png")<br>

#Find out attributes of Image Objects<br>
print('size of the image:',x.size, 'color mode:',x.mode)<br>
print('size of the image:',o.size, 'color mode:',o.mode)<br>

#plot 2 images one besides the other<br>
plt.subplot(121),plt.imshow(x)<br>
plt.axis('off')<br>
plt.subplot(122),plt.imshow(o)<br>
plt.axis('off')<br>

#multiple images<br>
merged=ImageChops.multiply(x,o)<br>

#adding 2 images<br>
add=ImageChops.add(x,o)<br>

#convert colour mode<br>
greyscale=merged.convert('L')<br>
greyscale<br>
size of the image: (256, 256) color mode: RGB<br>
size of the image: (256, 256) color mode: RGB<br>

OUTPUT:<br>
![download](https://user-images.githubusercontent.com/97940475/190844338-1385a547-bce6-4812-8b12-1081fe9aa3bb.png)<br>

#more attributes<br>
image=merged<br>

print('image size: ',image.size,<br>
      '\ncolor mode: ',image.mode,<br>
     '\nimage width: ',image.width,'| also represented by: ',image.size[0],<br>
     '\nimage height: ',image.height,'| also represented by: ',image.size[1],)<br>

OUTPUT:<br>
image size:  (256, 256) <br>
color mode:  RGB <br>
image width:  256 | also represented by:  256 <br>
image height:  256 | also represented by:  256<br>

#mapping the pixels of the image so we can use them as coordinates<br>
pixel=greyscale.load()<br>

#a nested loop to parse through all the pixels in the image<br>
for row in range(greyscale.size[0]):<br>
  for column in range(greyscale.size[1]):<br>
    if pixel[row,column]!=(255):<br>
     pixel[row,column]=(0)<br>

greyscale<br>
![download](https://user-images.githubusercontent.com/97940475/190844377-825d5cd2-d11d-4297-ae2b-48dd7204a775.png)<br>

#1.invert image<br>
invert=ImageChops.invert(greyscale)<br>

#2.invert by substraction<br>
bg=Image.new('L',(256,256),color=(255))#create a new image with a solid white background<br>
subt=ImageChops.subtract(bg,greyscale)#substract image from background<br>

#3.rotate<br>
rotate=subt.rotate(45)<br>
rotate<br>
![download](https://user-images.githubusercontent.com/97940475/190844387-00e21dce-111a-4b8c-98fb-d160f34198e4.png)<br>

#gaussian blur<br>
blur=greyscale.filter(ImageFilter.GaussianBlur(radius=1))<br>

#edge detection <br>
edge=blur.filter (ImageFilter.FIND_EDGES)<br>
edge<br>
![download](https://user-images.githubusercontent.com/97940475/190844400-efcef671-95c2-4b4e-be45-65f71c3e0ca6.png)<br>

#Change edge colours<br>
edge=edge.convert('RGB')<br>
bg_red=Image.new('RGB',(256,256),color=(255,0,0))<br>

filled_edge=ImageChops.darker(bg_red,edge)<br>
filled_edge<br>
![download](https://user-images.githubusercontent.com/97940475/190844427-6a3b47fe-0fb0-41a9-8764-a5f32bd44f47.png)<br>
#save image in the directory<br>
edge.save('processed.png')<br>


**Implement a program to perform various edge detection techniques<br>**
**a) Canny Edge detection **<br>

#Canny Edge detection<br> 
import cv2<br> 
import numpy as np<br> 
import matplotlib.pyplot as plt<br> 
plt.style.use('seaborn')<br> 

loaded_image=cv2.imread("c7.jpg")<br> 
loaded_image=cv2.cvtColor(loaded_image,cv2.COLOR_BGR2RGB)<br> 

gray_image=cv2.cvtColor(loaded_image,cv2.COLOR_BGR2GRAY)<br> 

edged_image=cv2.Canny(gray_image,threshold1=30,threshold2=100)<br> 

plt.figure(figsize=(20,20))<br> 
plt.subplot(1,3,1)<br> 
plt.imshow(loaded_image,cmap="gray")<br> 
plt.title("Original Image")<br> 
plt.axis("off")<br> 
plt.subplot(1,3,2)<br> 
plt.imshow(gray_image,cmap="gray")<br> 
plt.axis("off")<br> 
plt.title("GrayScale Image")<br> 
plt.subplot(1,3,3)<br> 
plt.imshow(edged_image,cmap="gray")<br> 
plt.axis("off")<br> 
plt.title("Canny Edge Detected Image")<br> 
plt.show<br> 
OUTPUT:
![image](https://user-images.githubusercontent.com/97940468/187900875-c39f6a7f-bc66-4941-aa47-71ab064b1b1b.png)
![image](https://user-images.githubusercontent.com/97940468/187900970-8e1c2f70-0c6b-404c-80f3-8b010065d594.png)
![image](https://user-images.githubusercontent.com/97940468/187901100-16c6025b-b985-478c-b012-f3fcfee976f4.png)

**b) Edge detection schemas-the gradient(Sobel-first order derivatives)based edge detector and the Laplacian(2nd order derivative,so it is extremely sensitive to noise)based edge detector**<br>
#LapLacian and Sobel Edge detecting methods<br> 
import cv2<br> 
import numpy as np<br> 
from matplotlib import pyplot as plt<br> 

#Loading image<br> 
#img0=cv2.imread('sanFrancisco.jpg',)<br> 
img0=cv2.imread("c7.jpg")<br> 

#Converting to gray scale<br> 
gray=cv2.cvtColor(img0,cv2.COLOR_BGR2GRAY)<br> 

#remove noise<br> 
img=cv2.GaussianBlur(gray,(3,3),0)<br> 

#covolute with proper kernels<br> 
laplacian=cv2.Laplacian(img,cv2.CV_64F)<br> 
sobelx=cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5) #X<br> 
sobely=cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5) #y<br> 

plt.subplot(2,2,1),plt.imshow(img,cmap='gray')<br> 
plt.title('Original'),plt.xticks([]),plt.yticks([])<br> 
plt.subplot(2,2,2),plt.imshow(laplacian,cmap='gray')<br> 
plt.title('Laplacian'),plt.xticks([]),plt.yticks([])<br> 
plt.subplot(2,2,3),plt.imshow(sobelx,cmap='gray')<br> 
plt.title('Sobel X'),plt.xticks([]),plt.yticks([])<br> 
plt.subplot(2,2,4),plt.imshow(sobely,cmap='gray')<br> 
plt.title('Sobel Y'),plt.xticks([]),plt.yticks([])<br> 

plt.show()<br> 
OUTPUT:<br> 
![image](https://user-images.githubusercontent.com/97940468/187901330-37d06e5d-bf06-4388-bba6-26d860bdc9f1.png)
<br> 
**c) Edge detection using Prewitt Operator** <br>
#Edge detection using Prewitt operator<br> 
import cv2<br> 
import numpy as np<br> 
from matplotlib import pyplot as plt<br> 
img=cv2.imread('c7.jpg')<br> 
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)<br> 
img_gaussian=cv2.GaussianBlur(gray,(3,3),0)<br> 

#prewitt<br> 
kernelx=np.array([[1,1,1],[0,0,0],[-1,-1,-1]])<br> 
kernely=np.array([[-1,0,1],[-1,0,1],[-1,0,1]])<br> 
img_prewittx=cv2.filter2D(img_gaussian,-1,kernelx)<br> 
img_prewitty=cv2.filter2D(img_gaussian,-1,kernely)<br> 

cv2.imshow("Original Image",img)<br> 
cv2.imshow("Prewitt X", img_prewittx)<br> 
cv2.imshow("Prewitt Y", img_prewitty)<br> 
cv2.imshow("Prewitt",img_prewittx+img_prewitty)<br> 
cv2.waitKey()<br> 
cv2.destroyAllWindows()<br> 

OUTPUT:<br>
![image](https://user-images.githubusercontent.com/97940468/187901965-a0affd3c-8ac2-45be-8ab7-14a2fdc45be2.png)
![image](https://user-images.githubusercontent.com/97940468/187902089-214468a1-dbfc-4297-ae11-df171fbfdab8.png)
![image](https://user-images.githubusercontent.com/97940468/187902173-5b443cf9-35cd-406a-b229-be31ec0d4bcd.png)
![image](https://user-images.githubusercontent.com/97940468/187902241-abd81635-2785-47e7-a23b-a158522e8ea6.png)

<br>

**d) Roberts Edge Detection-Roberts cross operator<br>**

#Roberts Edge Detection- Roberts cross operator<br>
import cv2<br>
import numpy as np<br>
from scipy import ndimage<br>
from matplotlib import pyplot as plt<br>
roberts_cross_v=np.array([[1,0],[0,-1]])<br>
roberts_cross_h=np.array([[0,1],[-1,0]])<br>

img=cv2.imread("c7.jpg",0).astype('float64')<br>
img/=255.0<br>
vertical=ndimage.convolve(img,roberts_cross_v)<br>
horizontal=ndimage.convolve(img,roberts_cross_h)<br>

edged_img=np.sqrt(np.square(horizontal)+np.square(vertical))<br>
edged_img*=255<br>
cv2.imwrite("Output.jpg",edged_img)<br>
cv2.imshow("OutputImage",edged_img)<br>
cv2.waitKey()<br>
cv2.destroyAllWindows()<br>

OUTPUT:<br>
![image](https://user-images.githubusercontent.com/97940468/187902436-ebadfd95-7045-42c8-9459-bedc01bc5fab.png)<br>


import numpy as np<br>
import cv2<br>
import matplotlib.pyplot as plt<br>

#Open the image<br>
img = cv2.imread('dimage_damaged.png')<br>
plt.imshow(img)<br>
plt.show()<br>

#load the mask<br>
mask=cv2.imread('dimage_mask.png',0)<br>
plt.imshow(mask)<br>
plt.show()<br>

#Inpaint<br>
dst = cv2.inpaint(img,mask,3,cv2.INPAINT_TELEA)<br>

#Write the output.<br>
cv2.imwrite('dimage_inpainted.png',dst)<br>
plt.imshow(dst)<br>
plt.show()<br>

OUTPUT:<br>
![download](https://user-images.githubusercontent.com/97940468/190844556-a8a96309-8acf-4622-9c14-e84ae77009c4.png)<br>
![download](https://user-images.githubusercontent.com/97940468/190844562-892844e6-4f61-4f72-9809-70be879ecaac.png)<br>
![download](https://user-images.githubusercontent.com/97940468/190844567-6469c739-004a-40a6-bb9a-338e98890738.png)<br>

import numpy as np<br>
import matplotlib.pyplot as plt<br>
import pandas as pd<br>
plt.rcParams['figure.figsize'] =(10,8)<br>

def show_image(image,title='Image',cmap_type='gray'):<br>
    plt.imshow(image,cmap=cmap_type)<br>
    plt.title(title)<br>
    plt.axis('off')<br>
    
def plot_comparison(img_original,img_filtered,img_title_filtered):<br>
    fig,(ax1,ax2)=plt.subplots(ncols=2, figsize=(10,8), sharex=True, sharey=True)<br>
    ax1.imshow(img_original,cmap=plt.cm.gray)<br>
    ax1.set_title('Original')<br>
    ax1.axis('off')<br>
    ax2.imshow(img_filtered,cmap=plt.cm.gray)<br>
    ax2.set_title('img_title_filtered')<br>
    ax2.axis('off')<br>
    
from skimage.restoration import inpaint<br>
from skimage.transform import resize<br>
from skimage import color<br>

image_with_logo=plt.imread('imlogo.png')<br>

#Initialize the mask<br>
mask=np.zeros(image_with_logo.shape[:-1])<br>

#Set the pixels where the Logo is to  1<br>
mask[210:272, 360:425] = 1<br>

#Apply inpainting to remove the Logo<br>
image_logo_removed = inpaint.inpaint_biharmonic(image_with_logo,<br>
                                              mask,<br>
                                              multichannel=True)<br>

#show the original and Logo removed images<br>
plot_comparison(image_with_logo,image_logo_removed,'Image with logo removed')<br>


OUTPUT:<br>
![download](https://user-images.githubusercontent.com/97940468/190844700-c90a55d5-4ce6-4cb2-a5ad-861d100ec16d.png)<br>


from skimage.util import random_noise<br>

fruit_image=plt.imread('fruits.jpg')<br>

#add noise to the image<br>
noisy_image=random_noise(fruit_image)<br>

#Show the original and resulting image<br>
plot_comparison(fruit_image, noisy_image, 'Noisy image')<br>

OUTPUT:<br>
![download](https://user-images.githubusercontent.com/97940468/190844774-1ec064ca-5715-4078-8db9-e4215fb8085c.png)<br>

from skimage.restoration import denoise_tv_chambolle<br>

noisy_image=plt.imread('noisy.jpg')<br>

#Apply total variation filter denoising<br>
denoised_image=denoise_tv_chambolle(noisy_image,multichannel=True)<br>

#show the noisy and denopised image<br>
plot_comparison(noisy_image,denoised_image,'Denoised Image')<br>

OUTPUT:<br>
![download](https://user-images.githubusercontent.com/97940468/190844810-ceefaf3a-12d9-4385-8175-a198d5188e2b.png)


from skimage.restoration import denoise_bilateral<br>

landscape_image=plt.imread('noisy.jpg')<br>

#Apply bilateral filter denoising<br>
denoised_image=denoise_bilateral(landscape_image,multichannel=True)<br>

#Show original and resulting images<br>
plot_comparison(landscape_image, denoised_image,'Denoised Image')<br>


OUTPUT:<br>
![download](https://user-images.githubusercontent.com/97940468/190844844-dd994cf8-ed1a-4e74-9065-8fb3d4661fa3.png)<br>

from skimage.segmentation import slic<br>
from skimage.color import label2rgb<br>
import matplotlib.pyplot as plt<br>
import numpy as np<br>
face_image = plt.imread('face.jpg')<br>
segments = slic(face_image, n_segments=400)<br>
segmented_image=label2rgb(segments,face_image,kind='avg')<br>
plt.imshow(face_image)<br>
plt.show()<br>
plt.imshow((segmented_image * 1).astype(np.uint8))<br>
plt.show()<br>

OUTPUT:<br>
![download](https://user-images.githubusercontent.com/97940468/190844878-1ac3ef79-fe8c-4a5c-8a56-973d312c137e.png)<br>
![download](https://user-images.githubusercontent.com/97940468/190844889-094d27e1-24e7-4ca5-a1fb-6fa05b08ffbc.png)<br>

def show_image_contour(image,contours):<br>
    plt.figure()<br>
    for n, contour in enumerate(contours):<br>
        plt.plot(contour[:,1], contour[:,0],linewidth=3)<br>
    plt.imshow(image,interpolation='nearest',cmap='gray_r')<br>
    plt.title('Contours')<br>
    plt.axis('off')<br>
    
    
from skimage import measure, data<br>

#obtain the horse image<br>
horse_image=data.horse()<br>

#Find the contours with a constant level value of 0.8<br>
contours=measure.find_contours(horse_image,level=0.8)<br>

#Shows the image with contours found<br>
show_image_contour(horse_image,contours)  <br>
    
OUTPUT:<br>
![download](https://user-images.githubusercontent.com/97940468/190844948-454e22ad-b0e2-4603-b89f-ff569d8ee07e.png)<br>

from skimage.io import imread <br>
from skimage.filters import threshold_otsu <br>

image_dices=imread('diceimg.png') <br>

#make the image grayscale <br>
image_dices=color.rgb2gray(image_dices) <br>

#Obtain the optimal thresh value <br>
thresh=threshold_otsu(image_dices) <br>

#Apply threshholding <br>
binary=image_dices > thresh <br>

#Find contours at aconstant value of 0.8 <br>
contours=measure.find_contours(binary,level=0.8) <br>

#Show the image <br>
show_image_contour(image_dices, contours) <br>

OUTPUT:<br>
![download](https://user-images.githubusercontent.com/97940468/190844985-9c486bca-8a2b-438d-b6c3-844bc628adac.png)<br>

#Create list with the shape of each contour<br>
shape_contours=[cnt.shape[0] for cnt in contours]<br>

#Set 50 as the maximum sixe of the dots shape<br>
max_dots_shape=50<br>

#Count dots in contours excluding bigger then dots size<br>
dots_contours=[cnt for cnt in contours if np.shape(cnt)[0]<max_dots_shape]<br>

#Shows all contours found<br>
show_image_contour(binary, contours)<br>

#Print the dice's number<br>
print('Dices dots number:{}.'.format(len(dots_contours)))<br>

OUTPUT:<br>
Dices dots number:21.<br>
![download](https://user-images.githubusercontent.com/97940468/190845017-25e8abb4-32b9-447f-86de-17b8a4cd28ed.png)<br>


27)
28)a) Canny Edge detection
#Canny Edge detection
import cv2
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn')

loaded_image=cv2.imread("color.jpg")
loaded_image=cv2.cvtColor(loaded_image,cv2.COLOR_BGR2RGB)

gray_image=cv2.cvtColor(loaded_image,cv2.COLOR_BGR2GRAY)

edged_image=cv2.Canny(gray_image,threshold1=30,threshold2=100)

plt.figure(figsize=(20,20))
plt.subplot(1,3,1)
plt.imshow(loaded_image,cmap="gray")
plt.title("Original Image")
plt.axis("off")
plt.subplot(1,3,2)
plt.imshow(gray_image,cmap="gray")
plt.axis("off")
plt.title("GrayScale Image")
plt.subplot(1,3,3)
plt.imshow(edged_image,cmap="gray")
plt.axis("off")
plt.title("Canny Edge Detected Image")
plt.show
![download](https://user-images.githubusercontent.com/97940475/190844669-15aa867b-f29d-44ae-b671-b6bd3b71c8c6.png)
#b) Edge detection schemes - the gradient (Sobel - first order derivatives)
#based edge detector and the Laplacian (2nd order derivative, so it is
#extremely sensitive to noise) based edge detector.
#LapLacian and Sobel Edge detecting methods
import cv2
import numpy as np
from matplotlib import pyplot as plt

#Loading image
#img0=cv2.imread('sanFrancisco.jpg',)
img0=cv2.imread("color.jpg")

#Converting to gray scale
gray=cv2.cvtColor(img0,cv2.COLOR_BGR2GRAY)

#remove noise
img=cv2.GaussianBlur(gray,(3,3),0)

#covolute with proper kernels
laplacian=cv2.Laplacian(img,cv2.CV_64F)
sobelx=cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5) #X
sobely=cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5) #y

plt.subplot(2,2,1),plt.imshow(img,cmap='gray')
plt.title('Original'),plt.xticks([]),plt.yticks([])
plt.subplot(2,2,2),plt.imshow(laplacian,cmap='gray')
plt.title('Laplacian'),plt.xticks([]),plt.yticks([])
plt.subplot(2,2,3),plt.imshow(sobelx,cmap='gray')
plt.title('Sobel X'),plt.xticks([]),plt.yticks([])
plt.subplot(2,2,4),plt.imshow(sobely,cmap='gray')
plt.title('Sobel Y'),plt.xticks([]),plt.yticks([])

plt.show()
![download](https://user-images.githubusercontent.com/97940475/190844681-750e6d16-c38b-49bc-a1af-a7fa6203c5d6.png)
#c) Edge detection using Prewitt Operator
#Edge detection using Prewitt operator
import cv2
import numpy as np
from matplotlib import pyplot as plt
img=cv2.imread('color.jpg')
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
img_gaussian=cv2.GaussianBlur(gray,(3,3),0)

#prewitt
kernelx=np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
kernely=np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
img_prewittx=cv2.filter2D(img_gaussian,-1,kernelx)
img_prewitty=cv2.filter2D(img_gaussian,-1,kernely)

cv2.imshow("Original Image",img)
cv2.imshow("Prewitt X", img_prewittx)
cv2.imshow("Prewitt Y", img_prewitty)
cv2.imshow("Prewitt",img_prewittx+img_prewitty)
cv2.waitKey()
cv2.destroyAllWindows()


#d) Roberts Edge Detection- Roberts cross operator
#Roberts Edge Detection- Roberts cross operator
import cv2
import numpy as np
from scipy import ndimage
from matplotlib import pyplot as plt
roberts_cross_v=np.array([[1,0],[0,-1]])
roberts_cross_h=np.array([[0,1],[-1,0]])

img=cv2.imread("color.jpg",0).astype('float64')
img/=255.0
vertical=ndimage.convolve(img,roberts_cross_v)
horizontal=ndimage.convolve(img,roberts_cross_h)

edged_img=np.sqrt(np.square(horizontal)+np.square(vertical))
edged_img*=255
cv2.imwrite("Output.jpg",edged_img)
cv2.imshow("OutputImage",edged_img)
cv2.waitKey()
cv2.destroyAllWindows()
