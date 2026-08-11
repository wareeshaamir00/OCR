import cv2
import numpy as np

def resize(image):
    image = cv2.resize(image, None, fx = 2, fy = 2, interpolation=cv2.INTER_CUBIC)
    return image

def bw(image):
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
    return grey

def threshold(image):
    thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    return thresh

def otsut(image):
    _, otsu = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return otsu

def adaptive(image):
    ad = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 5)
    return ad

def denoise(image):
    denoised = cv2.fastNlMeansDenoising(image)
    return denoised

def blur(image):
    blurred =cv2.GaussianBlur(image, (5, 5), 0)
    return blurred

def contrast(image):
    contrasted = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = contrasted.apply(image)
    return enhanced

def firstpipe(image):
    return otsut(blur(bw(image)))

def secondpipe(image):
    return adaptive(bw(image))

def thirdpipe(image):
    return threshold(blur(bw(image)))

def fourthpipe(image):
    return otsut(denoise(bw(image)))

def fifthpipe(image):
    return adaptive(contrast(bw(image)))