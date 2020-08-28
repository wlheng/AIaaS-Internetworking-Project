# import the necessary packages
from imutils import paths
import time
import os
import cv2

#read image
def get_image(imageFilePath):
    image = cv2.imread(imageFilePath)
    return image

#write image to folder
def output_image(option, fileName, output):
    output_path = 'output_images'
    cv2.imwrite(os.path.join(output_path, fileName), output)
    cv2.waitKey(0)
    print('[INFO] Image is saved to: %s' % output_path)
    
#method for detecting blur
def variance_of_laplacian(image):
	# compute the Laplacian of the image and then return the focus
	# measure, which is simply the variance of the Laplacian
	return cv2.Laplacian(image, cv2.CV_64F).var()

#dectecing blur
def detect_blur(imageFilePath, imageFileName):
    start = time.time()
    fileName = imageFileName.split(sep='.')[0]
    # load the image, convert it to grayscale, and compute the
    # focus measure of the image using the Variance of Laplacian
    image = get_image(imageFilePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fm = variance_of_laplacian(gray)
    text = "Not Blurry"
    threshold = 100.0

	# if the focus measure is less than the supplied threshold (100),
	# then the image should be considered "blurry"
    if fm < threshold:
        text = "Blurry"
    
    cv2.putText(image, "{}: {:.2f}".format(text, fm), (10, 30), 
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
    
    if fm < threshold:
        outputFileName = fileName + '_Blurry_' + '.jpg'
        output_image(7, outputFileName, image)
    else:  
        outputFileName = fileName + '_Not Blurry_' + '.jpg'
        output_image(7, outputFileName, image)

    end = time.time()
    return start, end