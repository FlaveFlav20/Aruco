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
parser.add_argument("-v", "--videosource", type=int, default=0, help="Customvideo source, otherwise 0")
parser.add_argument("-c", "--csv", type=str, default=".", help="csv output directory (file will be named all_pos.csv)")
parser.add_argument("-i", "--imageDir", type=str, default=".", help="Image output directory")
parser.add_argument("-b", "--beginImage", type=int, default=0, help="The begin index of image")
parser.add_argument("-n", "--numberMarkers", type=int, default=1, help="The number of marker")
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

# Get video camera
print("[INFO] Starting video stream...")
videostream = cv2.VideoCapture(args["videosource"])
if not videostream.isOpened():
    print("[ERROR] Cannot open camera")
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

capture = False
count = args["beginImage"]


listPos = ["marker_id (dict =" + args["type"] + ")", "x", "y", "z", "pitch", "yaw", "roll"] * numberMarkers
#           number_capture      id          x   y    z    x component rotation | y component rotation | z component rotation
listPos.insert(0, "number_capture")
csvWriter.writerow(listPos)

print("Press c to capture (or cancel a capture that didn't work)")

savedVideoOriginal = None

while videostream.grab():
    ret, image = videostream.retrieve()


    if ret == False:
        continue

    height, width, _ = image.shape

    if savedVideoOriginal == None:
        savedVideoOriginal = cv2.VideoWriter(ImgDir + "/saved_video.avi", cv2.VideoWriter_fourcc(*'MJPG'),  10, (width, height))

    imageCopy = image.copy()


    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

    if key == ord("c"):
        if capture == False:
            capture = True
        else:
            capture = False

    if key == ord('p'):
        print(listPos)


    # Detect aruco markers
    corners, ids, _ = cv2.aruco.detectMarkers(image, dictionary)

    # If at least one marker detected
    if ids is not None and len(ids) > 0:
        nMarkers = len(corners)

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
            continue

        if capture == True:
            listPos.insert(0, count)
            print("Captured")
            csvWriter.writerow(listPos)
            cv2.imwrite(ImgDir + "/Original_" + str(count) + ".jpg", image)
            cv2.imwrite(ImgDir + "/Detected_" + str(count) + ".jpg", imageCopy)
            count+=1

    ret = savedVideoOriginal.write(imageCopy)

    cv2.imshow("Position", imageCopy)
    capture = False


print("[INFO] Closing video stream...")
videostream.release()
cv2.destroyAllWindows()
fscsv.close()
