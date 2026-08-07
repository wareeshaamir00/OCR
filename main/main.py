import cv2
import numpy as np

def resize(image):
    image = cv2.resize(image, None, fx = 2, fy = 2, interpolation=cv2.INTER_CUBIC)
    return image

def bw(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 

def condition(image):
    bright = np.mean(image)
    if bright < 180:
        return True
    else:
        return False
    
def threshold(image):
    thresh = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 7)
    return thresh

def order_points(pts):
    rect = np.zeros((4,2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  
    rect[2] = pts[np.argmax(s)]   
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)] 
    rect[3] = pts[np.argmax(diff)]
    return rect

def contour(image):
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image
    else:      
        largest = max(contours, key=cv2.contourArea)
        contourarea = cv2.contourArea(largest)
        area = image.shape[0] * image.shape[1]
        percentage = contourarea / area * 100
        if 20 < percentage:
                if cv2.isContourConvex(largest):
                    perimeter = cv2.arcLength(largest, True)
                    approx = cv2.approxPolyDP(largest, 0.02 * perimeter,True) 
                    if len(approx)==4:
                        pts = approx.reshape(4, 2)
                        rect = order_points(pts)
                        (tl, tr, br, bl) = rect
                        width_top = np.linalg.norm(tr - tl)
                        width_bottom = np.linalg.norm(br - bl)
                        width = int(max(width_top, width_bottom))
                        height_left = np.linalg.norm(bl - tl)
                        height_right = np.linalg.norm(br - tr)
                        height = int(max(height_left, height_right))
                        dst = np.array([
                            [0, 0],
                            [width - 1, 0],
                            [width - 1, height - 1],
                            [0, height - 1]
                        ], dtype="float32")
                        matrix = cv2.getPerspectiveTransform(rect, dst)
                        warped = cv2.warpPerspective(image, matrix, (width, height))
                        return warped
    return image

def preprocess(image):
    height, width = image.shape[:2]
    if height<600 and width<600:
        image = resize(image)
    if len(image.shape) == 3:
        image = bw(image)
    if condition(image) == True:
        image = threshold(image)
    image = contour(image)
    return image
