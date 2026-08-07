import pytesseract
from main import preprocess
import cv2

input = input("Enter the path of the image: ")
image = cv2.imread(input)
def psm(image):
    height,width = image.shape[:2]
    area = height*width
    if width > height * 1.5:
        return 4
    elif area < 300000:
        return 11
    else:
        return 6

psm = psm(image)
text = pytesseract.image_to_string(image, config=f"--oem 3 --psm {psm}")
with open("output/text.txt", "w", encoding="utf-8") as f:
    f.write(text)