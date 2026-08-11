import pytesseract
import cv2
import os
from main import firstpipe, secondpipe, thirdpipe, fourthpipe, fifthpipe
import xml.etree.ElementTree as ET
import jiwer

pipelines = {
    "otsu": firstpipe,
    "adaptive": secondpipe,
    "threshold": thirdpipe,
    "denoise": fourthpipe,
    "contrast": fifthpipe
}

tree = ET.parse("annotation/annotations.xml")
root = tree.getroot()
ground = { }
for image in root.findall(".//image"):
    name = image.get("name")
    fname = os.path.splitext(os.path.basename(name))[0]
    texts = []
    for attribute in image.findall(".//attribute[@name='text']"):
        texts.append(attribute.text)

    ground[fname] = " ".join(texts)

folder = "test subjects"
dict = {
    "otsu": [],
    "adaptive": [],
    "threshold" : [],
    "denoise" : [],
    "contrast" : []
 }
bestp = {}
for current, _, files in os.walk(folder):
    for filename in files:
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            path = os.path.join(current, filename)
            image = cv2.imread(path)
            img = {}
            for name, pipeline in pipelines.items():
                processed = pipeline(image)
                id = os.path.splitext(filename)[0]
                text = pytesseract.image_to_string(processed)
                reference = ground[id]
                result = jiwer.process_words(reference, text)
                dict[name].append(result.wer)
                img[name] = result.wer
                minimum = min(img, key=img.get)
                bestp[filename] = minimum
                bestpipeline = pipelines[bestp[filename]]
                image2 = bestpipeline(image)
                texts = pytesseract.image_to_string(image2)
                output = os.path.join("output", os.path.splitext(filename)[0] + ".txt")
                with open(output, "w", encoding = "utf-8") as f:
                    f.write(texts)
                    