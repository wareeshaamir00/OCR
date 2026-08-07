import pytesseract
from main import preprocess
import cv2
input = input("Enter the path of the image: ")
image = cv2.imread(input)
image = preprocess(image)
text = pytesseract.image_to_string(image, config="--oem 3 --psm 6")
with open("output/text.txt", "w", encoding="utf-8") as f:
    f.write(text)