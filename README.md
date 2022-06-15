# IP
http://localhost:8890/notebooks/Thashreefah/program1.ipynb<br>

import cv2<br>
path='BUTTERFLY3.jpg'<br>
i=cv2.imread(path,1)<br>
cv2.imshow('image',i)<br>
cv2.waitKey(0)<br>
cv2.destroyAllWindows()<br>

OUTPUT:<br>
![image](https://user-images.githubusercontent.com/97940475/173816997-24596b5d-4e42-46bb-855d-6d5be00da6ca.png)<br>


import cv2<br>
img=cv2.imread('BUTTERFLY1.jpg',0)<br>
cv2.imshow('image',img)<br>
cv2.waitKey(0)<br>
cv2.destroyAllWindows()<br>

OUPUT:<br>
![image](https://user-images.githubusercontent.com/97940475/173811858-b3894fbb-9298-4d18-8e4b-6b3e23ca7ec8.png)<br>


import matplotlib.image as mping<br>
import matplotlib.pyplot as plt<br>
img=mping.imread('FLOWER1.jpg')<br>
plt.imshow(img)<br>

OUTPUT:<br>
![download](https://user-images.githubusercontent.com/97940475/173810784-5e5b6688-e5b2-4d5b-8f63-42c4a06d4c3f.png)<br>

from PIL import Image<br>
img=Image.open("LEAF1.jpg")<br>
img=img.rotate(60)<br>
img.show()<br>
cv2.waitKey(0)<br>
cv2.destroyAllWindows()<br>

OUTPUT:<br>
![image](https://user-images.githubusercontent.com/97940475/173812270-14803676-cc20-45ca-bf79-59e5875ce08e.png)<br>


from PIL import ImageColor<br>
img1=ImageColor.getrgb("pink")<br>
print(img1)<br>
img2=ImageColor.getrgb("blue")<br>
print(img2)<br>

OUTPUT:<br>
(255, 192, 203)<br>
(0, 0, 255)<br>


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

from PIL import Image<br>
img=Image.new('RGB',(200,400),(255,255,0))<br>
img.show()<br>

OUTPUT:<br>
![image](https://user-images.githubusercontent.com/97940475/173813213-de5acae2-6ef6-4202-94fa-0da206139145.png)<br>


