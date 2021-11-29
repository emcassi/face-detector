import cv2

# Imports the pre-trained data I downloaded from OpenCV that was trained using the Haar cascade model
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


# imports the image we are using
# ////// CHANGE "EXAMPLE.JPEG" TO FILE YOU WANT TO USE \\\\\\\\\\\\\
img = cv2.imread('example.jpeg')

# converts the image to black and white in order to better detect faces
grayscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# Returns a set of rectangles where faces were found
# detectMultiScale allows us to detect faces of different sizes (different distances)
face_coordinates = trained_face_data.detectMultiScale(grayscaled_img, 1.1, 50)


# Draws rectangles on the screen using the coordinates we just got
for (x, y, w, h) in face_coordinates:
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 255, 0), 2)

# Displays the image in a window
cv2.imshow("Face Detector", img)

# Waits for the user to press a key then exits the program
key = cv2.waitKey()