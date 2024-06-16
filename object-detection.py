#Python Code for OpenCV

import numpy as np
import cv2
import serial

# Setup the serial connection
ser = serial.Serial('COM5', 9600)  # replace 'COM3' with your Arduino's port

dist = 0
focal = 642
pixels = 107
width = 5
tolerance = 20  # pixels
yellow_detected = False

def get_dist(rectange_params, image, width, focal, font):
    pixels = rectange_params[1][0]
    print(pixels)
    dist = (width * focal) / pixels
    center = (image.shape[1] // 2, image.shape[0] // 2)
    image = cv2.putText(image, 'Distance from Camera in CM :', org, font,  
       1, color, 2, cv2.LINE_AA)

    image = cv2.putText(image, str(dist), (110, 50), font,  
       fontScale, color, 1, cv2.LINE_AA)

    return image, center, dist

cap = cv2.VideoCapture(0)
kernel = np.ones((3, 3), 'uint8')
font = cv2.FONT_HERSHEY_SIMPLEX 
org = (0, 20)  
fontScale = 0.6 
color = (0, 0, 255) 
thickness = 2

cv2.namedWindow('Object Dist Measure ', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Object Dist Measure ', 700, 600)

# Loop to capture video frames
while True:
    ret, img = cap.read()
    img = cv2.flip(img, 1)
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Predefined mask for yellow color detection
    lower = np.array([20, 100, 100])
    upper = np.array([30, 255, 255])
    mask = cv2.inRange(hsv_img, lower, upper)

    # Remove extra garbage from image
    d_img = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations = 5)

    # Find the histogram
    cont, hei = cv2.findContours(d_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cont = sorted(cont, key = cv2.contourArea, reverse = True)[:1]

    if len(cont) == 0:  # No obstacle detected
        img = cv2.putText(img, 'Move forward', (img.shape[1] // 2, img.shape[0] // 2), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
        ser.write(b'1')  # Send signal to Arduino to turn on all bulbs
    else:
        for cnt in cont:
            # Check for contour area
            if (cv2.contourArea(cnt) > 100 and cv2.contourArea(cnt) < 306000):

                # Draw a rectangle on the contour
                rect = cv2.minAreaRect(cnt)
                box = cv2.boxPoints(rect) 
                box = np.int0(box)
                cv2.drawContours(img, [box], -1, (255, 0, 0), 3)
                
                img, img_center, dist = get_dist(rect, img, width, focal, font)
                obstacle_center = np.mean(box, axis=0)
                cv2.rectangle(img, (int(obstacle_center[0]), int(obstacle_center[1])), (img_center[0], img_center[1]), (0, 255, 0), 2)
                tolerance_region = (img_center[0] - tolerance, img_center[0] + tolerance, img_center[1] - tolerance, img_center[1] + tolerance)  
            if dist >30:
                if tolerance_region[0] <= obstacle_center[0] <= tolerance_region[1] and tolerance_region[2] <= obstacle_center[1] <= tolerance_region[3]:
                    img = cv2.putText(img, 'Move forward', (img_center[0] + 50, img_center[1]), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
                    ser.write(b'1')  # Send signal to Arduino to turn on all bulbs
                else:
                    if obstacle_center[0] > img_center[0]:
                        img = cv2.putText(img, 'Move Right', (img_center[0] + 50, img_center[1]), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
                        ser.write(b'2')  # Send signal to Arduino to turn on one bulb
                    else:
                        img = cv2.putText(img, 'Move Left', (img_center[0] - 50, img_center[1]), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
                        ser.write(b'3')  # Send signal to Arduino to turn on another bulb
                    if obstacle_center[1] > img_center[1]:
                        img = cv2.putText(img, 'Move Down', (img_center[0], img_center[1] + 50), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
                        ser.write(b'4')  # Send signal to Arduino to turn on another bulb
                    else:
                        img = cv2.putText(img, 'Move Up', (img_center[0], img_center[1] - 50), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
                        ser.write(b'5')  # Send signal to Arduino to turn on another bulb
            elif dist <= 30:
                yellow_detected = True
                while yellow_detected:
                    ret, img = cap.read()  # display camera feed
                    img = cv2.flip(img, 1)
                    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                    mask = cv2.inRange(hsv_img, lower, upper)  # update mask
                    img = cv2.putText(img, 'Move right', (img_center[0], img_center[1] - 50), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
                    ser.write(b'2')  # Send signal to Arduino to turn on one bulb
                    cv2.imshow('Object Dist Measure ', img)  # display image in screen
                    # check if there is any yellow in the screen if no set yellow_detected to false
                    if cv2.countNonZero(mask) == 0:
                        yellow_detected = False
                    if cv2.waitKey(1) & 0xFF == ord('q'):  # add this line
                        break
    cv2.imshow('Object Dist Measure ', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
