import pandas as pd
import matplotlib.pyplot as plt
import cv2
import easyocr
import os



#Reading_Image
img_path = r'C:\Users\Faizul Robin\Downloads\CV_Projects\Text_Detection\Street_sign.jpg'
img = cv2.imread(img_path)

#Instance_Text_Detector
reader = easyocr.Reader(['en'], gpu=False )

#Detect Text on Image
text_ = reader.readtext(img)


# Draw Bounding Box over Text
for bbox, text, score in text_:
    # Extracting coordinates
    x1, y1 = bbox[0]
    x2, y2 = bbox[2]
    # Convert coordinates to integers
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    # Draw rectangle
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 5)
    # Add text
    cv2.putText(img, text, (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 0.65, (255, 0, 0), 2)


plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()
cv2.waitKey(0)

# Path to save the modified image
output_path = os.path.join(os.path.dirname(img_path), "Detected_Text.jpg")

# Save the modified image
cv2.imwrite(output_path, img)