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
    outputFileName = fileName
    cv2.imwrite(os.path.join(output_path, outputFileName), output)
    cv2.waitKey(0)
    print('[INFO] Image is saved to: %s' % output_path)

#detecting age
def detect_age(imageFilePath, imageFileName, face, age, face_confidence):
    AGE_BUCKETS = ["(0-2)", "(4-6)", "(8-12)", "(15-20)", "(25-32)",
        "(38-43)", "(48-53)", "(60-100)"]

    # load our serialized face detector model from disk
    print("[INFO] loading face detector model...")
    prototxtPath = os.path.sep.join([face, "deploy.prototxt"])
    weightsPath = os.path.sep.join([face,
        "res10_300x300_ssd_iter_140000.caffemodel"])
    faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

    # load our serialized age detector model from disk
    print("[INFO] loading age detector model...")
    prototxtPath = os.path.sep.join([age, "age_deploy.prototxt"])
    weightsPath = os.path.sep.join([age, "age_net.caffemodel"])
    ageNet = cv2.dnn.readNet(prototxtPath, weightsPath)

    # load the input image and construct an input blob for the image
    image = get_image(imageFilePath)
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300),
        (104.0, 177.0, 123.0))

    # pass the blob through the network and obtain the face detections
    print("[INFO] computing face detections...")
    faceNet.setInput(blob)
    start = time.time()
    detections = faceNet.forward()
    end = time.time()


    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the prediction
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the confidence is
        # greater than the minimum confidence
        if confidence > face_confidence:
            # compute the (x, y)-coordinates of the bounding box for the
            # object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # extract the ROI of the face and then construct a blob from
            # *only* the face ROI
            face = image[startY:endY, startX:endX]
            faceBlob = cv2.dnn.blobFromImage(face, 1.0, (227, 227),
                (78.4263377603, 87.7689143744, 114.895847746),
                swapRB=False)

            # make predictions on the age and find the age bucket with
            # the largest corresponding probability
            ageNet.setInput(faceBlob)
            preds = ageNet.forward()
            i = preds[0].argmax()
            age = AGE_BUCKETS[i]
            ageConfidence = preds[0][i]

            # display the predicted age to our terminal
            text = "{}: {:.2f}%".format(age, ageConfidence * 100)
            print("[INFO] {}".format(text))

            # draw the bounding box of the face along with the associated
            # predicted age
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(image, (startX, startY), (endX, endY),
                (0, 0, 255), 2)
            cv2.putText(image, text, (startX, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

            i += 1
            fileName = imageFileName.split(sep='.')[0]
            outputFileName = fileName + '_iterations_' + str(i) + '.jpg'
            output_image(3, outputFileName, image)
            print('[INFO] Iterations ' + str(i) + ' saved to output_images, age: %s' % text)

    return start, end