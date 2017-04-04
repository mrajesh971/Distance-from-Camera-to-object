import numpy as np
import cv2

def eulerAnglesToRotationMatrix(theta) :
     
    R_x = np.array([[1,         0,                  0                   ],
                    [0,         math.cos(theta[0]), -math.sin(theta[0]) ],
                    [0,         math.sin(theta[0]), math.cos(theta[0])  ]
                    ])
         
         
                     
    R_y = np.array([[math.cos(theta[1]),    0,      math.sin(theta[1])  ],
                    [0,                     1,      0                   ],
                    [-math.sin(theta[1]),   0,      math.cos(theta[1])  ]
                    ])
                 
    R_z = np.array([[math.cos(theta[2]),    -math.sin(theta[2]),    0],
                    [math.sin(theta[2]),    math.cos(theta[2]),     0],
                    [0,                     0,                      1]
                    ])
                     
                     
    R = np.dot(R_z, np.dot( R_y, R_x ))
 
    return R
def isRotationMatrix(R) :
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6
 
 
# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).
def rotationMatrixToEulerAngles(R) :
 
    assert(isRotationMatrix(R))
     
    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
     
    singular = sy < 1e-6
 
    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0
 
    return np.array([x, y, z])

def find_marker(image):
    # convert the image to grayscale and blur to detect edges
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 35, 125)
 
    # find the contours in the edged image
 
    (_,cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    c = max(cnts, key = cv2.contourArea)
 
    return cv2.minAreaRect(c)
 
def distance_to_camera(knownWidth, focalLength, perWidth):
    # compute and return the distance from the image to the camera
    return (focalLength * knownWidth) / perWidth
 
# initialize the known distance from the camera to the object
KNOWN_DISTANCE = 8.8
 
# initialize the known object width, which in this case, the piece of
# paper is 21 cm wide
KNOWN_WIDTH = 8.8
 
# initialize the list of images that we'll be using
#(the first image is the one we know the distance from camera.)
IMAGE_PATHS = ["/Users/Rakesh/Desktop/Images/IMG_6719.jpg","/Users/Rakesh/Desktop/Images/IMG_6720.jpg","/Users/Rakesh/Desktop/Images/IMG_6721.jpg"]
# load the image that contains an object that is KNOWN TO BE from our camera
image = cv2.imread(IMAGE_PATHS[0])
marker = find_marker(image)
print ("hi",marker[1][0])
focalLength = (marker[1][0] * KNOWN_DISTANCE) / KNOWN_WIDTH
print (focalLength)
 
# loop over the image
for imagePath in IMAGE_PATHS:
    # compute the distance to the paper from the camera
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    marker = find_marker(image)
    CM = distance_to_camera(KNOWN_WIDTH, focalLength, marker[1][0])

    print ("This is ",CM)
    # draw a circle around the image and display it which is also circle
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1.2, 100)
    # ensure at least some circles were found
    if circles is not None:
        #convert the (x, y) coordinates and radius of the circles to integers
        circles = np.round(circles[0, :]).astype("int")
 
    # loop over the (x, y) coordinates and radius of the circles
        for (x, y, r) in circles:
            cv2.circle(image, (x, y), r, (0, 0, 0), 5)
 
    cv2.putText(image, "%.2fcm" % CM ,
        (image.shape[1] - 350, image.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX,
        2.0, (255,0,0), 3)
    cv2.imshow("image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()