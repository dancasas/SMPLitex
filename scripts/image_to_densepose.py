import torch
import numpy as np
import sys
import os
from PIL import Image
import argparse


class RGB2DensePose:

    def __init__(self) -> None:
        pass

    '''
    Computes densepose to all images in input_folder/ and saves .png and .pkl
    '''
    def folder2densepose(self, input_folder):

        # gets parent directory of input folder (WARNING: input_folder should not end with '/')
        parent_directory = os.path.dirname(input_folder)
        print ("Image dir " + input_folder)
        print("Parent: " + parent_directory)

        # creates densepose/ directory
        densepose_image_directory = os.path.join(parent_directory, 'densepose')
        isExist = os.path.exists(densepose_image_directory)
        if not isExist:
            os.makedirs(densepose_image_directory)

        # creates densepose_pkl/ directory
        densepose_pkl_directory = os.path.join(parent_directory, 'densepose_pkl')
        isExist = os.path.exists(densepose_pkl_directory)
        if not isExist:
            os.makedirs(densepose_pkl_directory)

        files = os.listdir(input_folder)

        for current_filename in files:
            print('Pocessing ' + current_filename.lower())

            current_filepath = os.path.join(input_folder, current_filename)

            # checks if it's a valid file
            if os.path.isfile(current_filepath):
                
                image_filename, extension = os.path.splitext(current_filename)
                
                # if file is image file
                if extension in ('.jpg','.JPG', '.png', '.PNG', '.jpeg', '.JPEG'):
                    
                    densepose_image_filepath = os.path.join(densepose_image_directory, image_filename +  "_densepose.png")
                    densepose_pkl_filepath = os.path.join(densepose_pkl_directory, image_filename + "_densepose.pkl") 
                    
                    self.img2densepose(current_filepath, densepose_pkl_filepath, densepose_image_filepath)

    def img2densepose(self, input_image, output_pkl, output_image):
        
        os.system("python apply_net.py dump configs/densepose_rcnn_R_101_FPN_s1x.yaml https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_101_FPN_s1x/165712084/model_final_c6ab63.pkl " + input_image + " --output " + output_pkl + " -v")

        img          = Image.open(input_image)
        img_w ,img_h = img.size

        # loads .pkl with dense pose data
        with open(output_pkl, 'rb') as f:
            data = torch.load(f)

        i       = data[0]['pred_densepose'][0].labels.cpu().numpy()
        uv      = data[0]['pred_densepose'][0].uv.cpu().numpy()
        iuv     = np.stack((uv[1,:,:], uv[0,:,:], i))
        iuv     = np.transpose(iuv, (1,2,0))
        iuv_img = Image.fromarray(1 - np.uint8(iuv*255),"RGB")

        #iuv_img.show() #It shows only the croped person

        box     = data[0]["pred_boxes_XYXY"][0]
        box[2]  = box[2]-box[0]
        box[3]  = box[3]-box[1]
        x,y,w,h = [int(v) for v in box]
        bg      = np.zeros((img_h,img_w,3))
        bg[y:y+h,x:x+w,:] = iuv
        bg_img  = Image.fromarray(1 - np.uint8(bg*255),"RGB")

        bg_img.save(output_image)


parser = argparse.ArgumentParser(description= 'Computes DensePose from RGB image')
parser.add_argument('--detectron2', type=str, help='Path to your detectron2/projects/Densepose directory', required=True)
parser.add_argument('--input_folder', type=str, help='Folder with images', required=True)

args = parser.parse_args()

DETECTRON_PATH = args.detectron2
INPUT_IMAGE_PATH = args.input_folder

# removes backslash in case it's the last character of the input path
if INPUT_IMAGE_PATH.endswith('/'):
    INPUT_IMAGE_PATH = INPUT_IMAGE_PATH[:-1]

# Change path to your dir of detectron2/projects/DensePose
os.chdir(os.path.join(DETECTRON_PATH,"projects/DensePose"))        

# Adds DensePose to PYTHONPATH
sys.path.append(os.path.join(DETECTRON_PATH,"projects/DensePose")) 

densepose = RGB2DensePose()
densepose.folder2densepose(INPUT_IMAGE_PATH)


