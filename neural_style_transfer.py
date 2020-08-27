import imutils
import time
import cv2

#read image
def get_image(imageFilePath):
    # input_image = Image.open(imageFilePath).convert("RGB")
    image = cv2.imread(imageFilePath)
    return image
    
#image pre-process (resize) 
def resize_image(input_image, width):
    resized_image = imutils.resize(input_image, width=width)
    return resized_image

#write image to folder
def output_image(option, fileName, output):
    output_path = 'output_images'
    outputFileName = fileName
    output -= output.min()
    output /= output.max()
    output *= 255   
    cv2.imwrite(os.path.join(output_path, outputFileName), output)
    cv2.waitKey(0)
    print('[INFO] Image is saved to: %s' % output_path)

#apply style (Neural Style Transfer)
def apply_style(imageFilePath, modelPath):
    print("[INFO] loading style transfer model...")
    net = cv2.dnn.readNetFromTorch(modelPath) 

    input_image = get_image(imageFilePath)
    resized_image = resize_image(input_image, 600)
    (h, w) = resized_image.shape[:2]
    print('Resize completed!')
    # For debugging purposes.
    # resized_path = '/Users/wlheng/Desktop/Internetworking Project/AIaaS Project/resized_image'
    # outputFileName = 'resized.jpg'
    # cv2.imwrite(os.path.join(resized_path, outputFileName), resized_image)
    # cv2.waitKey(0)
    blob = cv2.dnn.blobFromImage(resized_image, 1.0, (w, h),
	    (103.939, 116.779, 123.680), swapRB=False, crop=False)
    net.setInput(blob)
    start = time.time()
    output = net.forward()
    end = time.time()
    print('model applied and processed!')
    # reshape the output tensor, add back in the mean subtraction, and
    # then swap the channel ordering
    output = output.reshape((3, output.shape[2], output.shape[3]))
    output[0] += 103.939
    output[1] += 116.779
    output[2] += 123.680
    output /= 255.0
    output = output.transpose(1, 2, 0)
    
    return output, start, end