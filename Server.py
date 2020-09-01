import cv2
import time
import os
from mask_rcnn import mask_r_cnn
from EAST_textDetector import detect_text
from age_detection import detect_age
from neural_style_transfer import apply_style
from activity_recognition import activity_recognition
# from Handwriting_Recognition import handwriting_recognition
from blur_detection import detect_blur
from blur_face import blur_face
from imutils import paths
from flask import request, Flask
app = Flask(__name__)

#write image to folder
def output_image(option, fileName, output):
    output_path = 'output_images'
    outputFileName = fileName
    if option == 1:
        #converting normalisation before writing image to folder
        output -= output.min()
        output /= output.max()
        output *= 255   
    cv2.imwrite(os.path.join(output_path, outputFileName), output)
    cv2.waitKey(0)
    print('[INFO] Image is saved to: %s' % output_path)

#select and load the model selected by the user
def load_style(input_model):
    modelPaths = paths.list_files('models', validExts=(".t7",))
    modelPaths = sorted(list(modelPaths))
    inputLowered = input_model.lower()
    current_style = inputLowered
    for model in modelPaths:
        if inputLowered in model:
            modelPath = model
            return modelPath
    #default     
    print('No model found! Using default model: Composition.t7')
    modelPath =  "models/eccv16/composition_vii.t7"

    return modelPath

#process file received from the client
def process_file(received_file):
    # received_file = request.files['file']
    startTime = time.time()
    imageFileName = received_file.filename
    received_dirPath = 'received_images'
    if not os.path.isdir(received_dirPath):
        os.makedirs(received_dirPath)
    imageFilePath = os.path.join(received_dirPath, imageFileName)
    received_file.save(imageFilePath)
    print('image is saved to: %s' % imageFilePath)
    usedTime = time.time() - startTime
    print('received and saved, time:%.2f second' % usedTime)
    startTime = time.time()
    print(imageFilePath)
    return  imageFileName, imageFilePath, startTime

#define callback function to receive /post request, and reply with result
@app.route("/", methods=['POST'])
def return_result():
    option = int(request.values['option'])
    print('option selected: %d' % option)
    #Option 2. Neural Style Transfer   
    if option == 1:
        print('[INFO] This is option 1. Neural Style Transfer') 
        received_file = request.files['file']
        if received_file:
            imageFileName, imageFilePath, startTime = process_file(received_file)

            style_option = request.values['model']
            print('style applied is %s' % style_option)
            #commence style transfer
            output, process_start, process_end = apply_style(imageFilePath, 
                load_style(style_option))
            print("[INFO] neural style transfer took {:.4f} seconds".format(
                process_end - process_start))

            #output processed image to folder
            output_image(option, imageFileName, output)
            usedTime = time.time() - startTime
            print('Server process completed, time:%.2f second' % usedTime)
            result = 'Server process completed, time:%.2f second' % usedTime
            return result
        else:
            return '[INFO] Option 1 process failed'

    #Option 2. EAST Text Detector       
    elif option == 2:
        print('[INFO] This is option 2. EAST text detector')
        received_file = request.files['file']
        if received_file:
            imageFileName, imageFilePath, startTime = process_file(received_file)
            width = int(request.values['width'])
            height = int(request.values['height'])
            east = request.values['east']
            min_confidence = float(request.values['min_confidence'])
            original, process_start, process_end = detect_text(imageFilePath, width,
                height, east,min_confidence)
            print("[INFO] text detection took {:.6f} seconds".format(process_end - process_start))
            output_image(option, imageFileName, original)  
            usedTime = time.time() - startTime
            print('Server process completed, time:%.2f second' % usedTime)
            result = 'Server process completed, time:%.2f second' % usedTime
            return result
        else:
            return '[INFO] Option 2 process failed'
    
    #Option 3. Mask_RCNN Object detection  
    elif option == 3:
        print('[INFO] This is option 3. Mask R-CNN')
        received_file = request.files['file']
        if received_file:
            imageFileName, imageFilePath, startTime = process_file(received_file)
            mask_rcnn = request.values['mask_rcnn']
            min_confidence = float(request.values['min_confidence'])
            visualize = int(request.values['visualize'])
            threshold = float(request.values['threshold'])
            objects, process_start, process_end = mask_r_cnn(imageFilePath, imageFileName, 
                mask_rcnn, min_confidence, visualize, threshold)
            print("[INFO] mask detection took {:.6f} seconds".format(process_end - process_start))
            print('[INFO] Number of object(s) detected: %d' % objects)
            usedTime = time.time() - startTime
            result = 'Server process completed, time:%.2f second' % usedTime
            return result
        else:
            return '[INFO] Option 3 process failed'
    
    #Option 4. Age detection
    elif option == 4:
        print('[INFO] This is option 4. Age detector')
        received_file = request.files['file']
        if received_file:
            imageFileName, imageFilePath, startTime = process_file(received_file)
            face = request.values['face']
            age = request.values['age']
            face_confidence = float(request.values['min_confidence'])
            process_start, process_end = detect_age(imageFilePath, imageFileName, face, age, face_confidence)
            print("[INFO] face detection took {:.6f} seconds".format(process_end - process_start))
            usedTime = time.time() - startTime
            print('Server process completed, time:%.2f second' % usedTime)
            result = 'Server process completed, time:%.2f second' % usedTime
            return result
        else:
            return '[INFO] Option 4 process failed'

    #Option 5. Activity Recognition
    elif option == 5:
        print('[INFO] This is option 5. Activity Recognition')
        received_file = request.files['file']
        if received_file:
            videoFileName, videoFilePath, startTime = process_file(received_file)
            print('[INFO] Video filename is: %s' % videoFileName)
            process_start, process_end = activity_recognition(videoFilePath, videoFileName)
            usedTime = time.time() - startTime
            print('Server process completed, time:%.2f second' % usedTime)
            result = 'Server process completed, time:%.2f second' % usedTime
            return result
        else:
            return '[INFO] Option 5 process failed'
    
    # #Option 6. Handwriting Recognition
    # elif option == 6:
    #     print('[INFO] This is option 6. Handwriting Recognition')
    #     received_file = request.files['file']
    #     if received_file:
    #         imageFileName, imageFilePath, startTime = process_file(received_file)
    #         #main function goes here
    #         output, process_start, process_end = handwriting_recognition(imageFilePath)
    #         output_image(option, imageFileName, output)  
    #         usedTime = time.time() - startTime
    #         print('Server process completed, time:%.2f second' % usedTime)
    #         result = 'Server process completed, time:%.2f second' % usedTime
    #         return result
    #     else:
    #         return '[INFO] Option 6 process failed'

    #Option 6. Blur detection
    elif option == 6:
        print('[INFO] This is option 6. Blur detection')
        received_file = request.files['file']
        if received_file:
            imageFileName, imageFilePath, startTime = process_file(received_file)
            process_start, process_end = detect_blur(imageFilePath, imageFileName)
            print("[INFO] blur detection took {:.6f} seconds".format(process_end - process_start))
            usedTime = time.time() - startTime
            print('Server process completed, time:%.2f second' % usedTime)
            result = 'Server process completed, time:%.2f second' % usedTime
            return result
        else:
            return '[INFO] Option 6 process failed'

    #Option 7. Blur face
    elif option == 7:
        print('[INFO] This is option 7. Blur face')
        received_file = request.files['file']
        if received_file:
            imageFileName, imageFilePath, startTime = process_file(received_file)
            face = request.values['face']
            method = request.values['method']
            process_start, process_end = blur_face(imageFilePath, imageFileName, face, 
                method)
            print("[INFO] blur face took {:.6f} seconds".format(process_end - process_start))
            usedTime = time.time() - startTime
            print('Server process completed, time:%.2f second' % usedTime)
            result = 'Server process completed, time:%.2f second' % usedTime
            return result
        else:
            return '[INFO] Option 7 process failed'
    else:
        print('No such option')
        return 'No such option'

if __name__ == "__main__":
    #print('before start C/S mode, lets test locally')
    # result = predict_image(model_loaded, 'cat.jpg')
    # print(result)
    #test C/S in the same machine
    app.run("127.0.0.1", port=5000)
    #test C/S in different machines with the server ip here
    # app.run("xxx.xxx.xxx.xxx", port=5000)
