import cv2
import argparse
import numpy as np
import os

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument("-t", "--type", type=str, default="DICT_ARUCO_ORIGINAL", help="dictionnary used to generate the ArUco tag")
parser.add_argument("-o", "--output", default="./", help="path to outputdirectory to save the ArUco tag")
parser.add_argument("-l", "--length", type=float, default=0.0, help="marken length")
parser.add_argument("-v", "--videosource", type=int, default=0, help="Customvideo source, otherwise 0")
args = vars(parser.parse_args())

# LEs dictionnaires
ARUCO_DICT = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
    "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,

    "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
    "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
    "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
    "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
    "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
    "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
    "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
    "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
    "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
    "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
    "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
    "DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
    "DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
    "DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
    "DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11
}

# Veérification du dictionaire
if ARUCO_DICT.get(args["type"], None) is None:
    print("[ERROR] Dictionnary ’{}’ of ArUco tags is not supported".format(args["type"]))
    exit(0)

# Charger la dictionnaire ArUco
dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT[args["type"]])

markerLength = args["length"]

if markerLength == 0.0:
    print("[ERROR] length mush be positive")
    exit()

# Récpérer le flux video
print("[INFO] Starting video stream...")
videostream = cv2.VideoCapture(args["videosource"])
if not videostream.isOpened():
    print("[ERROR] Cannot open camera")
    exit()

outputFile = args["output"]

if not os.path.exists(outputFile):
    print("[ERROR] Path must be valid")
    exit()

fs = cv2.FileStorage(outputFile, cv2.FILE_STORAGE_READ)

if not fs.isOpened():
    print("[ERROR] Unable to open the calibration file")
    exit(-1)

# Récupérer le calibrage
cameraMatrix = fs.getNode("CameraMatrix").mat()
distCoeffs = fs.getNode("DistortionCoeffs").mat()

print("Camera Matrix:\n", cameraMatrix)
print("Distortion Coefficients:\n", distCoeffs)

fs.release()


wait_time = 10

rvecs, tvecs = [], []

objPoints = np.array([[-markerLength/2, markerLength/2, 0],
                      [markerLength/2, markerLength/2, 0],
                      [markerLength/2, -markerLength/2, 0],
                      [-markerLength/2, -markerLength/2, 0]], dtype=np.float32).reshape(-1, 1, 3)

while videostream.grab():
    ret, image = videostream.retrieve()
    imageCopy = image.copy()

    # Detect aruco markers
    corners, ids, _ = cv2.aruco.detectMarkers(image, dictionary)

    # If at least one marker detected
    if ids is not None and len(ids) > 0:
        cv2.aruco.drawDetectedMarkers(imageCopy, corners, ids)
        nMarkers = len(corners)

        # Get markers position
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, markerLength, cameraMatrix, distCoeffs)

        # Display markers axes
        for i in range(nMarkers):
            cv2.drawFrameAxes(imageCopy, cameraMatrix, distCoeffs, rvecs[i], tvecs[i], 0.1)

        for i in range(nMarkers):
            x, y, z = tvecs[i][0]
            print(f"Marker {ids[i]} - 3D Coordinates: ({x}, {y}, {z})")
            print("Distance:", np.linalg.norm(tvecs[i]))

    cv2.imshow("Position", imageCopy)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

print("[INFO] Closing video stream...")
videostream.release()
cv2.destroyAllWindows()
