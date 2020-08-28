# import the necessary packages
import numpy as np
import cv2
import os
import time

#read image
def get_image(imageFilePath):
    # input_image = Image.open(imageFilePath).convert("RGB")
    image = cv2.imread(imageFilePath)
    return image

#write image to folder
def output_image(option, fileName, output):
    output_path = 'output_images'
    cv2.imwrite(os.path.join(output_path, fileName), output)
    cv2.waitKey(0)
    print('[INFO] Image is saved to: %s' % output_path)

def blur_face(imageFilePath, imageFileName, face, method):
    start = time.time()
    face_confidence = 0.5
    # load our serialized face detector model from disk
    print("[INFO] loading face blur model...")
    prototxtPath = os.path.sep.join([face, "deploy.prototxt"])
    weightsPath = os.path.sep.join([face,
        "res10_300x300_ssd_iter_140000.caffemodel"])
    net = cv2.dnn.readNet(prototxtPath, weightsPath)

    # load the input image from disk, clone it, and grab the image spatial
    # dimensions
    image = get_image(imageFilePath)
    orig = image.copy()
    (h, w) = image.shape[:2]

    # construct a blob from the image
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300),
        (104.0, 177.0, 123.0))

    # pass the blob through the network and obtain the face detections
    print("[INFO] computing face blur model...")
    net.setInput(blob)
    detections = net.forward()
    
    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the
        # detection
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the confidence is greater
        # than the minimum confidence
        if confidence > face_confidence:
            # compute the (x, y)-coordinates of the bounding box for the
            # object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # extract the face ROI
            face = image[startY:endY, startX:endX]

            # check to see if we are applying the "simple" face blurring
            # method
            if method == "simple":
                face = anonymize_face_simple(face, 3.0)

            # otherwise, we must be applying the "pixelated" face
            # anonymization method
            else:
                face = anonymize_face_pixelate(face, 20)

            # store the blurred face in the output image
            image[startY:endY, startX:endX] = face

    fileName = imageFileName.split(sep='.')[0]
    outputFileName = '_faceblurred_' + fileName + '.jpg'
    output_image(8, outputFileName, image)
    print('[INFO] Image: ' + outputFileName + ' saved to output_images')

    end = time.time()
    return start, end

def anonymize_face_simple(image, factor=3.0):
    # automatically determine the size of the blurring kernel based
	# on the spatial dimensions of the input image
	(h, w) = image.shape[:2]
	kW = int(w / factor)
	kH = int(h / factor)
	# ensure the width of the kernel is odd
	if kW % 2 == 0:
		kW -= 1
	# ensure the height of the kernel is odd
	if kH % 2 == 0:
		kH -= 1
	# apply a Gaussian blur to the input image using our computed
	# kernel size

	return cv2.GaussianBlur(image, (kW, kH), 0)

def anonymize_face_pixelate(image, blocks=3):
	# divide the input image into NxN blocks
	(h, w) = image.shape[:2]
	xSteps = np.linspace(0, w, blocks + 1, dtype="int")
	ySteps = np.linspace(0, h, blocks + 1, dtype="int")

	# loop over the blocks in both the x and y direction
	for i in range(1, len(ySteps)):
		for j in range(1, len(xSteps)):
			# compute the starting and ending (x, y)-coordinates
			# for the current block
			startX = xSteps[j - 1]
			startY = ySteps[i - 1]
			endX = xSteps[j]
			endY = ySteps[i]
			# extract the ROI using NumPy array slicing, compute the
			# mean of the ROI, and then draw a rectangle with the
			# mean RGB values over the ROI in the original image
			roi = image[startY:endY, startX:endX]
			(B, G, R) = [int(x) for x in cv2.mean(roi)[:3]]
			cv2.rectangle(image, (startX, startY), (endX, endY),
				(B, G, R), -1)

	# return the pixelated blurred image
	return image