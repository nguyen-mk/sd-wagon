import cv2
import numpy as np

camera = cv2.VideoCapture(1)

images = []

img_counter = 0

while True:
    ret, frame = camera.read()
    frame = cv2.flip(frame, -1)
    cv2.imshow("Capture Image", frame)
    if not ret:
        break
    k = cv2.waitKey(1)

    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k%256 == 32:
        # SPACE pressed
        img_name = "opencv_frame_{}.png".format(img_counter)
        #cv2.imwrite(img_name, frame)
        images.append(frame)
        print("Captured Image! ".format(img_name))
        img_counter += 1

camera.release()
cv2.destroyAllWindows()

imgpoints = []
objpoints = []
nx = 7
ny = 5
objp = np.zeros((nx*ny,3), np.float32)
objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2)


for image in images:
    #cv2.namedWindow("window")
    #cv2.imshow('Lane Lines Show', image)
        
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
    
    # If found
    if ret == True:
        # add object points, image points
        imgpoints.append(corners)
        objpoints.append(objp)
        image = cv2.drawChessboardCorners(image, (nx, ny), corners, ret)

    cv2.imshow('Grid Image', image)

    cv2.waitKey(0)
    #print(image)

#cv2.destroyAllWindows()