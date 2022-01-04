import cv2
import numpy as np
import pickle

#undistort the images
def undistort(gray_img,mtx,dist):
    undst = cv2.undistort(gray_img, mtx, dist, None, mtx)
    return undst

#Capture images by using the spacebar to get new images, ESC to finish capturing
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

# find the checkerboard pattern in the images
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


#Calibrate the camera
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

#Show the indistorted images
for image in images:
    test_img_undist = undistort(np.copy(image),mtx,dist)

    cv2.imshow('Undistorted Image', test_img_undist)

    cv2.waitKey(0)

#Save the calibration values for usage with the project
with open('cameraCals.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump([ret, mtx, dist, rvecs, tvecs], f)

# Getting back the objects:
#with open('objs.pkl') as f:  # Python 3: open(..., 'rb')
#    obj0, obj1, obj2 = pickle.load(f)