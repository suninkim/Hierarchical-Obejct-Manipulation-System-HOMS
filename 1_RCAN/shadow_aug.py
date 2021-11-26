import sys 
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
import numpy as np
import random

def generate_shadow_coordinates(imshape, no_of_shadows=1):
    vertices_list=[]
    for index in range(no_of_shadows):
        vertex=[]        # imshape[0]//3+  np.random.randint(3,7)
        for dimensions in range(3): ## Dimensionality of the shadow polygon
            vertex.append(( imshape[1]*np.random.uniform(),imshape[0]*np.random.uniform()))
            vertices = np.array([vertex], dtype=np.int32) ## single shadow vertices         
            vertices_list.append(vertices)    

    return vertices_list ## List of shadow vertices

def add_shadow(image,no_of_shadows=1):
    image_HLS = cv2.cvtColor(image,cv2.COLOR_RGB2HLS) ## Conversion to HLS   
    mask = np.zeros_like(image)
    imshape = image.shape
    vertices_list= generate_shadow_coordinates(imshape, no_of_shadows) #3 getting list of shadow vertices    
    for vertices in vertices_list:
        cv2.fillPoly(mask, vertices, 255) ## adding all shadow polygons on empty mask, single 255 denotes only red channel
        degree = np.random.uniform(0.6,0.9)

        image_HLS[:,:,1][mask[:,:,0]==255] = image_HLS[:,:,1][mask[:,:,0]==255]*degree   ## if red channel is hot, image's "Lightness" channel's brightness is lowered
        image_RGB = cv2.cvtColor(image_HLS,cv2.COLOR_HLS2RGB) ## Conversion to RGB 
    
    return image_RGB

def add_circle(image):
    image_HLS = cv2.cvtColor(image,cv2.COLOR_RGB2HLS)
    mask = np.zeros_like(image)
    imshape = image.shape

    max_size = 0.3*imshape[0]*np.random.uniform(0.4,1.2,2)
    max_size = np.array(max_size, dtype=np.int32)
    
    center_position = 0.1*imshape[0] + 0.8*imshape[0]*np.random.uniform(0,1,2)
    center_position = np.array(center_position, dtype=np.int32)

    angle = 180*np.random.uniform()

    degree = np.random.uniform(0.6,0.9)

    mask = cv2.ellipse(mask,(center_position, max_size, angle),(255,0,0),-1)
    image_HLS[:,:,1][mask[:,:,0]==255] = image_HLS[:,:,1][mask[:,:,0]==255]*degree
    image_RGB = cv2.cvtColor(image_HLS,cv2.COLOR_HLS2RGB)
    return image_RGB

def add_noise(image):
    eps = random.uniform(0,1)

    if eps < 0.5 :
        row,col,ch= image.shape
        mean = 0
        var = 15
        sigma = var**0.5
        gauss = np.random.normal(mean,sigma,(row,col,ch))
        gauss = gauss.reshape(row,col,ch)
        noisy = image + gauss
        noisy = np.clip(noisy,0,255)
        return noisy.astype(np.uint8)

    else:
        return image
