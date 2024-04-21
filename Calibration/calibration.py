# import the necessary packages
import cv2
import argparse
import numpy as np
import os

# Pqrser les arguments
parser = argparse.ArgumentParser()
parser.add_argument("-v", "--videosource", type=int, default=0, help="Custom video source, otherwise 0")
parser.add_argument("-X", "--numberX", type=int, default=1, help="Custom x directional squares")
parser.add_argument("-Y", "--numberY", type=int, default=1, help="Custom y directionnal squares")
parser.add_argument("-S", "--squareSize", type=float, default=0.1, help="Custom square size")
parser.add_argument("-o", "--output", type=str, default=".", help="Custom result path, default \".\"")
args = vars(parser.parse_args())

# Récpérer le flux video
print("[INFO] Starting video stream...")
videostream = cv2.VideoCapture(args["videosource"])
if not videostream.isOpened():
    print("[ERROR] Cannot open camera")
    exit()

squaresX = args["numberX"]

if squaresX <= 0:
    exit()

squaresY = args["numberY"]

if squaresY <= 0:
    exit();

squareSize = args["squareSize"]

outputFile = args["output"]

if not os.path.exists(outputFile):
    print("Path must be valid")
    exit()

CHECKERBOARD = (squaresX,squaresY)
imageSize = None

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Create a 3D vector of chess table
objpoints = []

# Create a 2D vector of Chess table
imgpoints = []


# Define 3D positions
objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp = objp * squareSize;
prev_img_shape = None

# Capture image
capture = False
number_capture = 0

while videostream.grab():
    ret, image = videostream.retrieve()

    if not imageSize:
        imageSize = (image.shape[0], image.shape[1])

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key == ord('c'):
        capture = True

    if corners is not None and len(corners[0]) > 0:
        image = cv2.drawChessboardCorners(image, CHECKERBOARD, corners, ret)

    cv2.imshow("Calibration", image)

    # save image
    if corners is not None and len(corners[0]) > 0 and capture == True:
        capture = False
        objpoints.append(objp)
        imgpoints.append(corners)
        number_capture += 1
        print("Numbers:", number_capture)


videostream.release()
cv2.destroyAllWindows()

if len(imgpoints) < 4:
    print("Not enough corners for calibration")
    exit(0)

calibration_flags = 0
aspect_ratio = 1.0
camera_matrix = np.eye(3, 3, dtype=np.float64)

if calibration_flags & cv2.CALIB_FIX_ASPECT_RATIO:
    camera_matrix[0, 0] = aspect_ratio

dist_coeffs = np.zeros((5, 1), np.float64)

# Calibrate camera
ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, imageSize, camera_matrix, dist_coeffs, flags=calibration_flags)

# Save calirage
fs = cv2.FileStorage(outputFile + "/" + "calibration_params.yml", cv2.FILE_STORAGE_WRITE)
fs.write("CameraMatrix", camera_matrix)
fs.write("DistortionCoeffs", dist_coeffs)
fs.release()

# Display results
print("Reprojection Error:", ret)
print("Camera Matrix:\n", camera_matrix)
print("Distortion Coefficients:\n", dist_coeffs)
