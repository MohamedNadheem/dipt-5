# Implementation-of-filter
## Aim:
To implement filters for smoothing and sharpening the images in the spatial domain.
## Software Required:
Anaconda - Python 3.7
## Algorithm:
### Step1
</br>
Import the required libraries.
</br> 

### Step2
</br>
Convert the image from BGR to RGB.
</br> 

### Step3
</br>
Apply the required filters for the image separately.
</br> 

### Step4
</br>
Plot the original and filtered image by using matplotlib.pyplot.
</br> 

### Step5
</br>
End the program.
</br> 

## Program
### Developed By : MOHAMED NADHEEM N
### Register Number :212223240091
</br>

## 1. Smoothing Filters

### i) Using Averaging Filter
```
import cv2
import numpy as np
import matplotlib.pyplot as plt
# Read the input image
image = cv2.imread('rithika.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# Define a Kernel
kernel = np.ones((5,5), dtype = np.float32) / 5**2

print (kernel)

image = cv2.imread('rithika.jpg')
dst = cv2.filter2D(image, ddepth = -1, kernel = kernel)

plt.figure(figsize = (18, 6))
plt.subplot(121); plt.imshow(image [:, :, ::-1]);
plt.title('Input Image')
plt.subplot(122); plt.imshow(average_filter[:, :, ::-1]);
plt.title('Output Image ( Average Filter)')

```
### ii) Using Weighted Averaging Filter
```
kernel = np.array([[1,2,1],
                   [2,4,2],
                   [1,2,1]])/16
weighted_average_filter = cv2.filter2D(image, -1, kernel)
# Display the images.
plt.figure(figsize = (18, 6))
plt.subplot(121);plt.subplot(121); plt.imshow(image [:, :, ::-1]);
plt.title('Input Image')
plt.subplot(122);plt.imshow(weighted_average_filter[:, :, ::-1]);
plt.title('Output Image(weighted_average_filter)');plt.show()

```
### iii) Using Gaussian Filter
```
# Apply Gaussian blur.
gaussian_filter = cv2.GaussianBlur(image, (29,29), 0, 0)
# Display the images.

plt.figure(figsize = (18, 6))
plt.subplot(121); plt.imshow(image [:, :, ::-1]); plt.title('Input Image')
plt.subplot(122); plt.imshow(gaussian_filter[:, :, ::-1]); plt.title('Output Image ( Gaussian Filter)')
```
### iv)Using Median Filter
```


median_filter = cv2.medianBlur(image, 19)
# Display the images.

plt.figure(figsize = (18, 6))
plt.subplot(121); plt.imshow(image [:, :, ::-1]); plt.title('Input Image')
plt.subplot(122); plt.imshow(median_filter[:, :, ::-1]); plt.title('Output Image ( Median_filter)')
```

## 2. Sharpening Filters
### i) Using Laplacian Linear Kernal
```
# i) Using Laplacian Kernel (Manual Kernel)
laplacian_kernel = np.array([[0, -1, 0],
                             [-1, 5, -1],
                             [0, -1, 0]])
sharpened_laplacian_kernel = cv2.filter2D(image, -1, kernel = laplacian_kernel)
# Display the images.

plt.figure(figsize = (18, 6))
plt.subplot(121); plt.imshow(image [:, :, ::-1]); plt.title('Input Image')
plt.subplot(122); plt.imshow(sharpened_laplacian_kernel[:, :, ::-1]); plt.title('Output Image ( Laplacian_filter)')

```
### ii) Using Laplacian Operator
```
# ii) Using Laplacian Operator (OpenCV built-in)
gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
laplacian_operator = cv2.Laplacian(gray_image, cv2.CV_64F)
laplacian_operator = np.uint8(np.absolute(laplacian_operator))
# Display the images.

plt.figure(figsize = (18, 6))
plt.subplot(131); plt.imshow(image [:, :, ::-1]); plt.title('Input Image')
plt.subplot(132); plt.imshow(gray_image, cmap='gray'); plt.title('Gray_image')
plt.subplot(133); plt.imshow(laplacian_operator,cmap='gray'); plt.title('Output Image ( Laplacian_filter)')

```

## OUTPUT:
## 1. Smoothing Filters


### i) Using Averaging Filter
<img width="1233" height="596" alt="{BD68A338-8CD8-4722-BCFF-4B5218193238}" src="https://github.com/user-attachments/assets/aa2c6ae2-5bbf-46bf-8e72-fafaf1ff677b" />




### ii)Using Weighted Averaging Filter

<img width="1231" height="557" alt="{3B5883C3-7D9A-4DB3-A241-E530BDA5E9C8}" src="https://github.com/user-attachments/assets/694cf79f-04be-4646-bae9-e61d678e32bb" />



### iii)Using Gaussian Filter

<img width="1238" height="603" alt="{B4F2E1DA-9E83-46E0-B21C-F575A3FF7E0F}" src="https://github.com/user-attachments/assets/fa60c724-1fa3-491f-8b29-806e829e3a8a" />



### iv) Using Median Filter

<img width="1223" height="600" alt="{351C2DFA-00C3-410F-AAEB-B835FCCB8B98}" src="https://github.com/user-attachments/assets/08ca8f08-ae8d-4315-b3a5-30dc682eeaac" />


## 2. Sharpening Filters


### i) Using Laplacian Kernal
<img width="1229" height="611" alt="{5CC4DD80-567F-4D68-9C7E-5409F6DA2CBC}" src="https://github.com/user-attachments/assets/5b94227b-b84e-4789-9d00-8d4a3e24584c" />



### ii) Using Laplacian Operator
<img width="1233" height="504" alt="{0FDE4F1F-E640-4252-ACEB-12D3C1012D10}" src="https://github.com/user-attachments/assets/ac0ef476-39c2-4425-807d-9f4f58943b85" />



## Result:
Thus the filters are designed for smoothing and sharpening the images in the spatial domain.
