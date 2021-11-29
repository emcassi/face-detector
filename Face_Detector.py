import cv2

# Imports the pre-trained data I downloaded from OpenCV that was trained using the Haar cascade model
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


# Sets up the webcame for use in project
webcam = cv2.VideoCapture(0)

while True:

    # Begins using the video from the webcam
    successful_frame_read, frame = webcam.read()

    # converts the data to black and white in order to better detect faces
    grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


    # Returns a set of rectangles where faces were found
    # detectMultiScale allows us to detect faces of different sizes (different distances)
    face_coordinates = trained_face_data.detectMultiScale(grayscaled_img, 1.1, 50)


    # Draws rectangles on the screen using the coordinates we just got
    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 0), 2)


    # Displays the image in a window
    cv2.imshow("Face Detector", frame)


    # refreshes every millisecond to get webcam data
    key = cv2.waitKey(1)

    # quits the program if 'q' or 'Q' is pressed
    if key==81 or key==113:
        break