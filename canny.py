import cv2
import numpy as np
from skimage import io
import matplotlib.pyplot as plt
import time 
from numba import jit, cuda

start = time.time()
def gaussian_blur(size, sigma=1):
    size = int(size) // 2
    x, y = np.mgrid[-size:size+1, -size:size+1]
    normal = 1 / (2.0 * np.pi * sigma**2)
    g =  np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal
    blur_img = cv2.filter2D(original_image, ddepth=-1,kernel=g )
    return blur_img

@jit(target_backend='cuda') 
def sobel(blur_img):
  sobely=np.array([[1,2,1],
                  [0,0,0],
                  [-1,-2,-1]])
  
  sobelx=np.array([[1,0,-1],
                  [2,0,-2],
                  [1,0,-1]])
  s = blur_img.shape
  img_h = s[0]
  img_w = s[1]
  
  fil_h = sobelx.shape[0]
  fil_w = sobelx.shape[1]
  
  h = fil_h //2 #gives an intger
  w = fil_w //2
  
  ix = np.zeros(blur_img.shape)
  iy = np.zeros(blur_img.shape)
  for i in range(h,img_h-h):
      for j in range(w,img_w-w):
          sum=0
          for m in range(fil_h):
              for n in range(fil_w):
                  sum=sum + sobelx[m][n]*blur_img[i-h+m][j-w+n]
          ix[i][j]=sum

  for i in range(h,img_h-h):
      for j in range(w,img_w-w):
          sum=0
          for m in range(fil_h):
              for n in range(fil_w):
                  sum=sum + sobely[m][n]*blur_img[i-h+m][j-w+n]
          iy[i][j]=sum

  img_sobel= np.hypot(ix,iy)
  img_sobel = img_sobel / img_sobel.max() * 255
  theta = np.arctan2(iy, ix)

  return (img_sobel, theta)


def nom_max_supression(img_sobel,theta):

  m,n = img_sobel.shape
  z = np.zeros((m,n))
  angle = theta * 180 / np.pi
  angle[angle < 0] += 180

  q=255
  r=255

  for i in range(m-1):
    for j in range(n-1):

      # 0
      if (0 <= angle[i,j] <22.5 or 157.5 < angle[i,j] <=180 ):
        q = img_sobel[i, j+1]
        r = img_sobel[i, j-1]

       # 45 
      elif (22.5 <= angle[i,j] < 67.5 ):  
        q = img_sobel[i+1, j-1]
        r = img_sobel[i-1, j+1]

        # 90
      elif (67.5 <= angle[i,j] < 112.5 ):  
        q = img_sobel[i+1, j]
        r = img_sobel[i-1, j]

        # 135

      elif (112.5 <= angle[i,j] < 157.5 ):  
        q = img_sobel[i-1, j-1]
        r = img_sobel[i+1, j+1]   

      if ( img_sobel[i,j] >= q ) and img_sobel[i,j] >= r:
        z[i,j] = img_sobel[i,j]

      else:
        z[i,j]=0     

  return z

@jit(target_backend='cuda')
def double_threshold(img):
    highThreshold = 30
    lowThreshold = 15
    
    M, N = img.shape
    res = np.zeros((M,N))
    
    weak = 150
    strong = 255
    
    for i in range(M):
          for j in range(N):
                if(img[i,j]>=highThreshold):
                    res[i,j]=strong
                elif(lowThreshold <= img[i,j] <=highThreshold):
                    res[i,j]=weak
                else:
                    res[i,j]=0
     
    return (res, weak, strong)
   
@jit(target_backend='cuda')
def hysteresis(img, low):
    M, N = img.shape 
    strong = img.max()
    for i in range(1, M-1):
        for j in range(1, N-1):
            if (img[i,j]==low):
                  if ((img[i+1, j-1] == strong) or (img[i+1, j] == strong) or (img[i+1, j+1] == strong)
                      or (img[i, j-1] == strong) or (img[i, j+1] == strong)
                      or (img[i-1, j-1] == strong) or (img[i-1, j] == strong) or (img[i-1, j+1] == strong)):
                      img[i, j] = strong
                  else:
                      img[i, j] = 0

    return img       

def rgb2gray(rgb):

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray


original_image = io.imread('D:/DIP3E_Original_Images_CH02/car.png')
fig1 = plt.figure(1)
# plt.imshow(original_image)

original_image =  rgb2gray(original_image)   
stop = time.time()
print(stop - start, 's', 'no_noise')       
# plt.imshow(original_image, cmap ="gray")


blur_img =gaussian_blur(5)
stop = time.time()
print(stop - start, 's', 'blur_img')
# plt.imshow(blur_img, cmap ="gray")

sobel , theta = sobel(blur_img)
stop = time.time()
print(stop - start, 's', 'sobel,theta')
# plt.imshow(sobel, cmap ="gray")

nms = nom_max_supression(sobel , theta)
stop = time.time()
print(stop - start, 's', 'nms')
# plt.imshow(nms, cmap ="gray")

img , weak , strong =  double_threshold(nms)
stop = time.time()
print(stop - start, 's', 'img,weak,strong')
# plt.imshow(img, cmap ="gray")

output = hysteresis(img , weak)
stop = time.time()
print(stop - start, 's', 'output')
plt.imshow(output, cmap ="gray")
plt.show()
