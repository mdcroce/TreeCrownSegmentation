"""
Tree Crown Segmentation with Watershed and Random Walker


Usage: python2 watershed_randomwalker.py original_image ground_truth soil_removed_image(optional)

Packages required: python2, opencv2, numpy, scipy, matplotlib, scikit-image 0.11.3
For images 512x512 use mode=cg_mg for Random Walker

"""
import sys
import numpy as np
from skimage.morphology import watershed
from skimage.feature import peak_local_max
from skimage import morphology
from skimage.segmentation import random_walker
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage import io
from matplotlib.image import imsave
import cv2
import cv2.cv as cv
from skimage.filter import threshold_otsu,rank
import time
from skimage.morphology import rectangle

if len(sys.argv) > 4 or len(sys.argv) < 3:
    print "Usage: python watershed_randomwalker.py original_image ground_truth soil_removed_image(optional)"
    print "\n"
    exit(0)

#loading images
image = io.imread(str(sys.argv[1]))
image_color = image
ground_truth = cv2.imread(str(sys.argv[2]),cv2.CV_LOAD_IMAGE_GRAYSCALE)

if (image==None) or (ground_truth==None):
    print "Images not found!"
    exit(0)

if (len(sys.argv) == 4):
	soil_removed =  True
else:
	soil_removed =  False

#2046x1265
#1023x632
#define region of interest
roi = False
if roi == True:
    roi_x1 = 3000
    roi_x2 = 2400
    roi_y1 = 1600
    roi_y2 = 1000
else:
    roi_x1 = image.shape[0]
    roi_x2 = 0
    roi_y1 = image.shape[1]
    roi_y2 = 0
    
image = image[roi_x2:roi_x1, roi_y2:roi_y1]
ground_truth = ground_truth[roi_x2:roi_x1, roi_y2:roi_y1]


if image.ndim == 3:
    print "Extract NIR channel"
    image = image[:,:,0] 
else:
    print "One dimension image"

print "OTSU Threshold with sliding window"

selem = rectangle(100,100)
local_otsu = rank.otsu(image, selem)


#
if soil_removed:
	soil_removed_image=io.imread(str(sys.argv[3]),as_grey=True)
	if (roi == True):
		soil_removed_image = soil_removed_image[roi_x2:roi_x1, roi_y2:roi_y1]
	ii, jj = np.where(soil_removed_image==0)
	image[ii, jj] = 0
 
binary_image = image > local_otsu


print "Distance Transform"
distance = ndimage.distance_transform_edt(binary_image)


print "Extract local maxima"
local_maxi = peak_local_max(distance, indices=False, footprint=np.ones((55,55)),threshold_abs = 10, labels=binary_image)

print "Markers for WS"
markers = morphology.label(local_maxi)

print "Watershed"
start_ws = time.clock()
labels_ws = watershed(-distance, markers, mask=binary_image)
end_ws = time.clock()
time_ws = end_ws - start_ws

imsave("labelsws.png",labels_ws)

print "Threshold to extract circles"
rest, ground_truth = cv2.threshold(ground_truth,7,255,cv2.THRESH_BINARY)

print "HoughCircles"
circles = cv2.HoughCircles(ground_truth,cv.CV_HOUGH_GRADIENT,1,20,
                            param1=35,param2=15,minRadius=5,maxRadius=35)

if circles == None:
    print "No circles found... exiting"
    sys.exit(0)

print "Creating image with circles for ground truth"
circle_image_ws = np.zeros(shape=(roi_x1-roi_x2,roi_y1-roi_y2,3),dtype=np.uint8)
circle_image_ws[:]=254
circle_image_rw = np.zeros(shape=(roi_x1-roi_x2,roi_y1-roi_y2,3),dtype=np.uint8)
circle_image_rw[:]=254

for i in circles[0,:]:
    # draw the outer circle
    cv2.circle(circle_image_ws,(i[0],i[1]),i[2],(0,255,0),2)
    cv2.circle(circle_image_rw,(i[0],i[1]),i[2],(0,255,0),2)


print "Markers for RW"
markers[~binary_image.astype(bool)] = -1 

print "Random walker"
start_rw = time.clock()
labels_rw = random_walker(binary_image.astype(bool), markers,copy=False,beta=10,mode='bf')
end_rw = time.clock()
tempo_rw = end_rw - start_rw

imsave("labelsrw.png",labels_rw)

print "Find detected trees in watershed"
slices = ndimage.find_objects(labels_ws)

print "Find detected trees in random walker"
slices_rw = ndimage.find_objects(labels_rw)

print "Creating central points in image for watershed segmentation"
imagem_pontos = np.zeros(shape=(roi_x1-roi_x2,roi_y1-roi_y2,1),dtype=np.uint8)
for i in range(len(slices)):
    x,y = [(side.start+side.stop)/2. for side in slices[i]]
    cv2.circle(circle_image_ws,(int(y),int(x)),5,(0,0,255),3)
    imagem_pontos[x,y]=255
    #circle_image_ws[x,y]=(255,255,255)

imsave("circle_image_ws.png",circle_image_ws)

print "Creating central points in image for random walker segmentation"

imagem_pontos_rw = np.zeros(shape=(roi_x1-roi_x2,roi_y1-roi_y2,1),dtype=np.uint8)
for i in range(len(slices_rw)):
    x,y = [(side.start+side.stop)/2. for side in slices_rw[i]]
    cv2.circle(circle_image_rw,(int(y),int(x)),5,(0,0,255),3)
    imagem_pontos_rw[x,y]=255
    circle_image_rw[x,y]=(255,255,255)

imsave("circle_image_rw.png",circle_image_rw)

print "\n"
print "Region of interest roix1,roix2,roiy1,roiy2-> ", roi_x1, " ", roi_x2, " ", roi_y1, " ", roi_y2

quantidade_arvores_gd = len(circles[0,])
print "Number of trees in ground truth image", quantidade_arvores_gd

lista_pontos_ws = np.where(imagem_pontos==255)
lista_pontos_rw = np.where(imagem_pontos_rw==255)


print "\n"
################ WATHERSHED
quantidade_arvores_detectadas = len(slices)
print "Number of detected trees in watershed: ", quantidade_arvores_detectadas

pm = np.ones(len(slices)).astype(int)
acertos = 0
falso_negativo = 0
pontos_no_circulo = 0 #quantidade de pontos dentro do circulo
for central_point in circles[0,]:
    vv= False
    for i in range(len(slices)):
        eps=(pow(lista_pontos_ws[0][i]-central_point[1],2) + pow(lista_pontos_ws[1][i]-central_point[0],2))
        eps = float(eps)
        if((eps<pow(float(central_point[2]),2)) and pm[i] == 1):
            acertos=acertos+1;
            pm[i] = 0
            vv=True
            break
            
    if vv == False:
        falso_negativo += 1
            
tp = acertos
fp = np.sum(pm)

print "Results watershed:"

print "Ns: ", tp

print "No: ", falso_negativo

print "Nc:", fp

print "Number of errors: ", falso_negativo + fp

acuracia = float(tp)/(tp + fp + falso_negativo)
print "Score: ", acuracia

print "Time watershed: ", time_ws




#print "Salva imagens das arvores individuais"
#for i in range(len(slices)):
#    imsave(str(i)+".jpg",image_color[slices[i]],cmap="gray")

########## RANDOM WALK
print "\n"
quantidade_arvores_detectadas = len(slices_rw)
print "Number of detect trees in random waker: ", quantidade_arvores_detectadas

pm = np.ones(len(slices)).astype(int)

acertos = 0
falso_negativo = 0
pontos_no_circulo = 0 #quantidade de pontos dentro do circulo
for central_point in circles[0,]:
    vv= False
    for i in range(len(slices_rw)):
        eps=(pow(lista_pontos_rw[0][i]-central_point[1],2) + pow(lista_pontos_rw[1][i]-central_point[0],2))
        eps = float(eps)
        if((eps<pow(float(central_point[2]),2)) and pm[i] == 1):
            acertos=acertos+1;
            pm[i] = 0
            vv=True
            break
            
    if vv == False:
        falso_negativo += 1
            
tp = acertos
fp = np.sum(pm)

print "Results for RandomWalk:"

print "Ns: ", tp

print "No: ", falso_negativo

print "Nc:", fp

print "Number of errors: ", falso_negativo + fp

acuracia = float(tp)/(tp + fp + falso_negativo)
print "Score: ", acuracia

print "Time randomwalk: ", tempo_rw
#print "Salva imagens das arvores individuais"
#for i in range(len(slices_rw)):
#    imsave(str(i)+".jpg",image_color[slices_rw[i]])


################################################
sys.exit(0)
print "Creating plot"
plt.figure(figsize=(12, 3.5))
plt.subplot(141)
plt.imshow(image, cmap='gray', interpolation='nearest')
plt.axis('off')
plt.title('image')
plt.subplot(142)
#plt.imshow(-distance, interpolation='nearest')
plt.imshow(labels_ws, cmap='spectral', interpolation='nearest')
plt.axis('off')
plt.title('watershed')
plt.subplot(143)
plt.imshow(ground_truth, cmap='spectral', interpolation='nearest')
plt.axis('off')
plt.title('bin')
plt.subplot(144)
plt.imshow(circle_image_ws)
plt.axis('off')
plt.title('circle')


plt.tight_layout()


exit()
