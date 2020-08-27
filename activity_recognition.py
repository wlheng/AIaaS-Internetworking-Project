# from collections import deque
import numpy as np
import argparse
import imutils
import sys
import cv2
import time
import os

def count_frames(videoFilePath, override=False):
    video = cv2.VideoCapture(videoFilePath)
    total = 0
    if override:
        total = count_frames_manual(video)
    else:
        try:
            total = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        except:
            total = count_frames_manual(video)
    video.release()

    return total

def count_frames_manual(video):
    # initialize the total number of frames read
	total = 0
	# loop over the frames of the video
	while True:
		# grab the current frame
		(grabbed, frame) = video.read()
	 
		# check to see if we have reached the end of the
		# video
		if not grabbed:
			break
		# increment the total number of frames read
		total += 1
	# return the total number of frames in the video file
	return total

def activity_recognition(videoFilePath, videoFileName):
    video_frame = count_frames(videoFilePath)
    print("[INFO] Video contain %d frames" % video_frame)

    #classes file path
    classes = 'models/activity_recognition/action_recognition_kinetics.txt'
    # load the contents of the class labels file, then define the sample
    # duration (i.e., # of frames for classification) and sample size
    # (i.e., the spatial dimensions of the frame)
    CLASSES = open(classes).read().strip().split("\n")
    SAMPLE_DURATION = 16
    SAMPLE_SIZE = 112

    #model file path
    model = 'models/activity_recognition/resnet-34_kinetics.onnx'
    # load the human activity recognition model
    print("[INFO] loading human activity recognition model...")
    net = cv2.dnn.readNet(model)

    # grab a pointer to the input video stream
    print("[INFO] accessing video...")
    cap = cv2.VideoCapture(videoFilePath)

    start = time.time()
    process_no = 0 
    i = 0
    output_frames = []
    while i < video_frame:
        frames = []
        count = 0
        # loop over the number of required sample frames
        for i in range(0, SAMPLE_DURATION):
            # read a frame from the video stream
            (grabbed, frame) = cap.read()
            # if the frame was not grabbed then we've reached the end of
            # the video stream so exit the script
            if not grabbed:
                print("[INFO] no frame read from video - exiting")
                break
            # otherwise, the frame was read so resize it and add it to
            # our frames list
            # frame = imutils.resize(frame, width=400)
            frames.append(frame)
            i += 1
            count += 1
        
        if count == 16:
            # now that our frames array is filled we can construct our blob
            blob = cv2.dnn.blobFromImages(frames, 1.0,
                (SAMPLE_SIZE, SAMPLE_SIZE), (114.7748, 107.7354, 99.4750),
                swapRB=True, crop=True)
            blob = np.transpose(blob, (1, 0, 2, 3))
            blob = np.expand_dims(blob, axis=0)

            # pass the blob through the network to obtain our human activity
            # recognition predictions
            net.setInput(blob)
            outputs = net.forward()
            label = CLASSES[np.argmax(outputs)]

            # loop over our frames
            for frame in frames:
                # draw the predicted activity on the frame
                cv2.rectangle(frame, (0, 0), (300, 40), (0, 0, 0), -1)
                cv2.putText(frame, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 255, 255), 2)
                output_frames.append(frame)
                process_no += 1
            print("[INFO] Processed %d frames" % process_no)

        else:
            break
    #write processed video to file
    write_video(cap, output_frames, videoFileName)
    end = time.time()

    return start, end

#write processed video to file    
def write_video(cap, frames, videoFileName):
    sframe = 0
    out = None
    for output in frames:
        (h, w) = output.shape[:2]
        if out == None:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            videoOutputPath = os.path.sep.join(['output_videos', videoFileName])
            out = cv2.VideoWriter(videoOutputPath, fourcc, 30.0, 
                (w, h))
        out.write(output)
        print("[INFO] Writing frame[%d] to video" % sframe)
        sframe += 1

    #Release resources
    cap.release()
    out.release()
    print('[INFO] Video saved to: %s' % videoOutputPath)
    

    