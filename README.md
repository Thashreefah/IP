# IP
http://localhost:8890/notebooks/Thashreefah/program1.ipynb


import cv2
img=cv2.imread('BUTTERFLY1.jpg',0)
cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

OUPUT:
![image](https://user-images.githubusercontent.com/97940475/173811858-b3894fbb-9298-4d18-8e4b-6b3e23ca7ec8.png)



import matplotlib.image as mping
import matplotlib.pyplot as plt
img=mping.imread('FLOWER1.jpg')
plt.imshow(img)

OUTPUT:

![download](https://user-images.githubusercontent.com/97940475/173810784-5e5b6688-e5b2-4d5b-8f63-42c4a06d4c3f.png)

from PIL import Image
img=Image.open("LEAF1.jpg")
img=img.rotate(60)
img.show()
cv2.waitKey(0)
cv2.destroyAllWindows()

OUTPUT:
![image](https://user-images.githubusercontent.com/97940475/173812270-14803676-cc20-45ca-bf79-59e5875ce08e.png)


from PIL import ImageColor
img1=ImageColor.getrgb("pink")
print(img1)
img2=ImageColor.getrgb("blue")
print(img2)

OUTPUT:
(255, 192, 203)
(0, 0, 255)


import cv2
import matplotlib.pyplot as plt
import numpy as np
img=cv2.imread('PLANT1.jpeg')
plt.imshow(img)
plt.show()
img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
plt.imshow(img)
plt.show()
img=cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
plt.imshow(img)
plt.show()


OUTPUT:

![download](https://user-images.githubusercontent.com/97940475/173812693-fee609fd-1ec3-43c4-9a9a-5586ccae2b41.png)
![download](https://user-images.githubusercontent.com/97940475/173812723-c5b43eb9-e2af-4f52-809c-855512fdd217.png)
![download](https://user-images.githubusercontent.com/97940475/173812756-1df9fc22-364c-4e6d-a6da-ba9cf24af554.png)

from PIL import Image
image=Image.open('BUTTERFLY3.jpg')
print("Filename:",image.filename)
print("Format:",image.format)
print("Mode:",image.mode)
print("size:",image.size)
print("Width:",image.width)
print("Height:",image.height)
image.close()

OUTPUT:
Filename: BUTTERFLY3.jpg
Format: JPEG
Mode: RGB
size: (770, 662)
Width: 770
Height: 662

from PIL import Image
img=Image.new('RGB',(200,400),(255,255,0))
img.show()

OUTPUT:
![image](https://user-images.githubusercontent.com/97940475/173813213-de5acae2-6ef6-4202-94fa-0da206139145.png)


