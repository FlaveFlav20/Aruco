import cv2
import argparse
import numpy as np
import os
import csv
# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument("-t", "--type", type=str, default="DICT_ARUCO_ORIGINAL", help="dictionnary used to generate the ArUco tag")
parser.add_argument("-o", "--output", default="./", help="path to outputdirectory to save the ArUco tag")
parser.add_argument("-l", "--length", type=float, default=0.0, help="marken length")
parser.add_argument("-c", "--csv", type=str, default=".", help="csv output directory (file will be named all_pos.csv)")
parser.add_argument("-i", "--imageDir", type=str, default=".", help="Image output directory")
parser.add_argument("-b", "--beginImage", type=int, default=0, help="The begin index of image")
parser.add_argument("-n", "--numberMarkers", type=int, default=1, help="The number of marker")
parser.add_argument("-img", "--image", type=str, default=".", help="The imput image")
args = vars(parser.parse_args())

# Dictionaries
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

# Check dictionary
if ARUCO_DICT.get(args["type"], None) is None:
    print("[ERROR] Dictionnary ’{}’ of ArUco tags is not supported".format(args["type"]))
    sys.exit(0)

# Load dictionary
dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT[args["type"]])

markerLength = args["length"]

if markerLength == 0.0:
    print("[ERROR] length mush be positive")
    exit()

csvPath = args["csv"]
if not os.path.exists(csvPath):
    print("[ERROR] CSV file  (-c/-csv) must be valid")
    exist()

fscsv = open(csvPath + "/" + "all_pos.csv", "a")

csvWriter = csv.writer(fscsv, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

outputFile = args["output"]

if not os.path.exists(outputFile):
    print("[ERROR] Calibration path (-o/--output) must be valid")
    exit()

imgPath = args["output"]

if not os.path.exists(imgPath):
    if not os.path.isfile(imgPath):
        print("[ERROR] Image (-img/--image) must be a file")
        exit()
    print("[ERROR] Image path (-img/--image) must be valid")
    exit()

fs = cv2.FileStorage(outputFile, cv2.FILE_STORAGE_READ)

if not fs.isOpened():
    print("[ERROR] Unable to open the calibration file (-o/--output)")
    exit(-1)

# Get calibration result
cameraMatrix = fs.getNode("CameraMatrix").mat()
distCoeffs = fs.getNode("DistortionCoeffs").mat()

print("Camera Matrix:\n", cameraMatrix)
print("Distortion Coefficients:\n", distCoeffs)

fs.release()

ImgDir = args["imageDir"]

if not os.path.exists(ImgDir):
    print("[ERROR] Destination dir must be valid (-i/--imageDir)")
    exit()

wait_time = 10

rvecs, tvecs = [], []

objPoints = np.array([[-markerLength/2, markerLength/2, 0],
                      [markerLength/2, markerLength/2, 0],
                      [markerLength/2, -markerLength/2, 0],
                      [-markerLength/2, -markerLength/2, 0]], dtype=np.float32).reshape(-1, 1, 3)

numberMarkers = args["numberMarkers"]

capture = True
count = args["beginImage"]


listPos = ["marker_id (dict =" + args["type"] + ")", "x", "y", "z", "pitch", "yaw", "roll"] * numberMarkers
#           number_capture      id          x   y    z    x component rotation | y component rotation | z component rotation
listPos.insert(0, "number_capture")
csvWriter.writerow(listPos)

savedVideoOriginal = None

## Let's do it

image = cv2.read(imgPath)

height, width, _ = image.shape

if savedVideoOriginal == None:
    savedVideoOriginal = cv2.VideoWriter(ImgDir + "/saved_video.avi", cv2.VideoWriter_fourcc(*'MJPG'),  10, (width, height))

imageCopy = image.copy()

# Detect aruco markers
corners, ids, _ = cv2.aruco.detectMarkers(image, dictionary)

def mouse_callback_top_left(event, x, y, flags, param):
    global corners
    if event == cv2.EVENT_LBUTTONDOWN:
        top_left = [x, y]

cv2.setMouseCallback("Position top left", mouse_callback_top_left)

def mouse_callback_top_right(event, x, y, flags, param):
    global corners
    if event == cv2.EVENT_LBUTTONDOWN:
        top_right = [x, y]

cv2.setMouseCallback("Position top right", mouse_callback_top_right)

def mouse_callback_down_right(event, x, y, flags, param):
    global corners
    if event == cv2.EVENT_LBUTTONDOWN:
        down_right = [x, y]

cv2.setMouseCallback("Position down right", mouse_callback_down_right)

def mouse_callback_down_left(event, x, y, flags, param):
    global corners
    if event == cv2.EVENT_LBUTTONDOWN:
        fown_left = [x, y]

cv2.setMouseCallback("Position down left", mouse_callback_down_left)

# If at least one marker detected
if ids is not None and len(ids) > 0:
    nMarkers = len(corners)

    current_marker_corners = corners[i][0]
    top_left = current_marker_corners[0]
    top_right = current_marker_corners[1]
    bottom_right = current_marker_corners[2]
    bottom_left = current_marker_corners[3]

    cv2.imshow("Position top left", imageCopy)
    cv2.waitkey()
    cv2.imshow("Position top right", imageCopy)
    cv2.waitkey()
    cv2.imshow("Position down right", imageCopy)
    cv2.waitkey()
    cv2.imshow("Position down left", imageCopy)
    cv2.waitkey()

    # Get markers positions
    rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, markerLength, cameraMatrix, distCoeffs)

    listPos = []

    # Display markers axes
    for i in range(nMarkers):
        cv2.drawFrameAxes(imageCopy, cameraMatrix, distCoeffs, rvecs[i], tvecs[i], 0.1)

    for i in range(nMarkers):
        x, y, z = tvecs[i][0]
        pitch, yaw, roll = rvecs[i][0]

        listPos.append(ids[i][0])
        listPos.append(x)
        listPos.append(y)
        listPos.append(z)
        listPos.append(pitch)
        listPos.append(yaw)
        listPos.append(roll)



    if numberMarkers != nMarkers and capture == True:
        print("Not the same number of markers. Got: ",nMarkers, ";Expected: ", numberMarkers)
        exit()

    if capture == True:
        listPos.insert(0, count)
        print("Captured")
        csvWriter.writerow(listPos)
        cv2.imwrite(ImgDir + "/Original_" + str(count) + ".jpg", image)
        cv2.imwrite(ImgDir + "/Detected_" + str(count) + ".jpg", imageCopy)
        count+=1

iret = savedVideoOriginal.write(imageCopy)

cv2.imshow("Position", imageCopy)


print("[INFO] Closing video stream...")
videostream.release()
cv2.destroyAllWindows()
fscsv.close()
