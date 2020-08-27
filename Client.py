import requests
import os
import argparse

ap = argparse.ArgumentParser()
ap.add_argument('-o', '--option', default='1', required=False,
    help='select action..')
ap.add_argument('-i', '--image', default='images/jurassic_park.jpg', required=False,
    help='upload image to apply neural style transfer to')
ap.add_argument("-m", "--model", default='composition', required=False,
	help="neural style transfer model")
ap.add_argument("-east", "--east", type=str, default='models/frozen_east_text_detection.pb', required=False,
	help="path to input EAST text detector")
ap.add_argument("-c", "--min-confidence", type=float, default=0.5, required=False,
	help="minimum probability required to inspect a region")
ap.add_argument("-w", "--width", type=int, default=320, required=False,
	help="resized image width (should be multiple of 32)")
ap.add_argument("-e", "--height", type=int, default=320, required=False,
	help="resized image height (should be multiple of 32)")
ap.add_argument("-mrcnn", "--mask-rcnn", default='models/mask-rcnn-coco', required=False,
	help="base path to mask-rcnn directory")    
ap.add_argument("-v", "--visualize", type=int, default=0, required=False,
	help="whether or not we are going to visualize each instance")
ap.add_argument("-t", "--threshold", type=float, default=0.3, required=False,
	help="minimum threshold for pixel-wise mask segmentation")
ap.add_argument("-f", "--face", default='models/face_detector', required=False,
	help="path to face detector model directory")
ap.add_argument("-a", "--age", default='models/age_detector', required=False,
	help="path to age detector model directory")              
args = vars(ap.parse_args())

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":
    url = "http://127.0.0.1:5000" #if you test C/S in the same machine
    # url = "http://138.25.63.194:5000" #different machine, set server ip here
    #while True:
        # input_content = input('input image file here, cat.jpg") ')
        # if input_content.strip() == "":
        #     input_content = 'cat.jpg'
    option = int(args['option'])
    input_content = args['image']
    model = args['model']
    east = args['east']
    min_confidence = args['min_confidence']
    width = args['width']
    height = args['height']
    mask_rcnn = args['mask_rcnn']
    visualize = args['visualize']
    threshold = args['threshold']
    face = args['face']
    age = args['age']
    if input_content.strip() == "-1":
        print('Wrong image')
    elif not os.path.exists(input_content.strip()):
        print('wrong image file, retry')
    else:
        imageFilePath = input_content.strip()
        imageFileName = os.path.split(imageFilePath)[1]
        file_dict = {
            'file':(imageFileName,
                open(imageFilePath,'rb'),
                'image/jpg'),
            # 'model': (None, model)
            }
        values = {
            'option': option,
            'model': model,
            'east': east,
            'min_confidence': min_confidence,
            'width': width,
            'height': height,
            'mask_rcnn': mask_rcnn,
            'visualize': visualize,
            'threshold': threshold,
            'face': face,
            'age': age
        }       
        #testing
        result = requests.post(url, files=file_dict, data=values)
        outcome = result.text
        print('image path:%s, result:%s\n' %(imageFilePath, outcome))
