import numpy as np
import random
import time
import cv2
import os

def get_image(imageFilePath):
    # input_image = Image.open(imageFilePath).convert("RGB")
    image = cv2.imread(imageFilePath)
    return image

def output_image(option, fileName, output):
    output_path = 'output_images'
    outputFileName = fileName
    cv2.imwrite(os.path.join(output_path, outputFileName), output)
    cv2.waitKey(0)
    print('[INFO] Image is saved to: %s' % output_path)

def mask_r_cnn(imageFilePath, imageFileName, mask_rcnn, confidences, visualize, threshold):
    labelsPath = os.path.sep.join([mask_rcnn, 
        'object_detection_classes_coco.txt'])
    LABELS = open(labelsPath).read().strip().split('\n')

    # load the set of colors that will be used when visualizing a given
    # instance segmentation
    colorsPath = os.path.sep.join([mask_rcnn, 'colors.txt'])
    COLORS = open(colorsPath).read().strip().split("\n")
    COLORS = [np.array(c.split(",")).astype("int") for c in COLORS]
    COLORS = np.array(COLORS, dtype="uint8")

    # derive the paths to the Mask R-CNN weights and model configuration
    weightsPath = os.path.sep.join([mask_rcnn,
        'frozen_inference_graph.pb'])
    configPath = os.path.sep.join([mask_rcnn,
        'mask_rcnn_inception_v2_coco_2018_01_28.pbtxt'])

    # load our Mask R-CNN trained on the COCO dataset (90 classes)
    # from disk
    print("[INFO] loading Mask R-CNN from disk...")
    net = cv2.dnn.readNetFromTensorflow(weightsPath, configPath)

    image = get_image(imageFilePath)
    (H, W) = image.shape[:2]

    # construct a blob from the input image and then perform a forward
    # pass of the Mask R-CNN, giving us (1) the bounding box  coordinates
    # of the objects in the image along with (2) the pixel-wise segmentation
    # for each specific object
    blob = cv2.dnn.blobFromImage(image, swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    (boxes, masks) = net.forward(["detection_out_final", "detection_masks"])
    end = time.time()

    objects = 0
    # loop over the number of detected objects
    for i in range(0, boxes.shape[2]):
        # extract the class ID of the detection along with the confidence
        # (i.e., probability) associated with the prediction
        classID = int(boxes[0, 0, i, 1])
        confidence = boxes[0, 0, i, 2]
        # filter out weak predictions by ensuring the detected probability
        # is greater than the minimum probability
        if confidence > confidences:
            # clone our original image so we can draw on it
            clone = image.copy()
            # scale the bounding box coordinates back relative to the
            # size of the image and then compute the width and the height
            # of the bounding box
            box = boxes[0, 0, i, 3:7] * np.array([W, H, W, H])
            (startX, startY, endX, endY) = box.astype("int")
            boxW = endX - startX
            boxH = endY - startY

            # extract the pixel-wise segmentation for the object, resize
            # the mask such that it's the same dimensions of the bounding
            # box, and then finally threshold to create a *binary* mask
            mask = masks[i, classID]
            mask = cv2.resize(mask, (boxW, boxH),
                interpolation=cv2.INTER_NEAREST)
            mask = (mask > threshold)
            # extract the ROI of the image
            roi = clone[startY:endY, startX:endX]
            # check to see if are going to visualize how to extract the
            # masked region itself
            if visualize > 0:
                # convert the mask from a boolean to an integer mask with
                # to values: 0 or 255, then apply the mask
                visMask = (mask * 255).astype("uint8")
                instance = cv2.bitwise_and(roi, roi, mask=visMask)

            # now, extract *only* the masked region of the ROI by passing
            # in the boolean mask array as our slice condition
            roi = roi[mask]

            # randomly select a color that will be used to visualize this
            # particular instance segmentation then create a transparent
            # overlay by blending the randomly selected color with the ROI
            color = random.choice(COLORS)
            blended = ((0.4 * color) + (0.6 * roi)).astype("uint8")

            # store the blended ROI in the original image
            clone[startY:endY, startX:endX][mask] = blended

            # draw the bounding box of the instance on the image
            color = [int(c) for c in color]
            cv2.rectangle(clone, (startX, startY), (endX, endY), color, 2)

            # draw the predicted label and associated probability of the
            # instance segmentation on the image
            text = "{}: {:.4f}".format(LABELS[classID], confidence)
            cv2.putText(clone, text, (startX, startY - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            i += 1
            fileName = imageFileName.split(sep='.')[0]
            outputFileName = fileName + '_Object_' + str(i) + '.jpg'
            output_image(3, outputFileName, clone)
            print('[INFO] Object ' + str(i) + ' saved to output_images, classfication: %s' % text)
            objects += 1 #determine the number of object detected.
            
    return objects, start, end