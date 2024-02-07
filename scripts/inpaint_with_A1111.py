import os
import argparse
import numpy as np
from datetime import datetime
from PIL import Image, ImageFilter, ImageOps
from stablediffusion_wrapper import StableDiffusionAPI

class InpaintWithA1111:

    def __init__(self, partial_texture_folder, masks_folder, output_folder) -> None:
        self.partial_texture_folder = partial_texture_folder
        self.masks_folder = masks_folder
        self.output_folder = output_folder

        # WARNING: You have to activate Automatic1111 API before:
        #   ~/stable-diffusion-webui$ ./webui.sh --disable-safe-unpickle --api     
        self.sd_api = StableDiffusionAPI()

        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder) 

    def inpaint_texture(self, partial_texture_path, mask_path, cfg = 2.0, denoising_strength = 0.8):


            current_texture_filename = os.path.basename(partial_texture_path)
            now = datetime.now() # current date and time
            date_time = now.strftime("%m%d%Y-%H%M%S")

            #   loads image 
            try:
                im = Image.open(partial_texture_path)
            except IOError:
                print("ERROR: could not open ", partial_texture_path)

            #   loads mask 
            try:    
                im_mask = Image.open(mask_path)
            except IOError:
                print("ERROR: could not open ", mask_path)


            # creates a new texture by filling some of black pixels
            im_eroded = im.filter(ImageFilter.MaxFilter(3))
            im = Image.composite(im, im_eroded, ImageOps.invert(im_mask))

            # updates the mask for the new filled texture
            im_eroded_np = np.array(im_eroded)
            updated_mask_np = (im_eroded_np[:, :, 0] < 2) & (im_eroded_np[:, :, 1] < 2) & (im_eroded_np[:, :, 1] < 2)
            im_mask = Image.fromarray(updated_mask_np)

            # erodes the mask (to shrink the texture area, aims to get rid of artefacts at the border)
            im_mask = im_mask.convert("L").filter(ImageFilter.MaxFilter(5)).convert("P")

            # inpaints intput image (notice it will generate #batch_size versions)
            inpainted_result = self.sd_api.img2img("a sks texturemap", 
                                    images = [im], 
                                    batch_size = 8,
                                    mask_image = im_mask, 
                                    cfg_scale = cfg,
                                    denoising_strength = denoising_strength,
                                    restore_faces = True,  
                                    width = 512,            
                                    height = 512,           
                                    )
            
            # for each inpainted sample (i.e. as many as the batch size) we do an img2img pass to improve quality
            for idx, inpainted_image in enumerate(inpainted_result.images):

                # saves raw inpainted result
                # output_filename = current_texture_filename.replace(".png", "_inpaint-" + str(idx).zfill(3) + "_cfg" + str(cfg) + "_" + date_time  + ".png")
                # output_filename_path = os.path.join(self.output_folder, output_filename)
                # inpainted_image.save(output_filename_path)
                
                # img2img 
                img2img_result = self.sd_api.img2img("a sks texturemap", 
                                    images = [inpainted_image],
                                    steps=20, 
                                    cfg_scale = 2, # 1 or 2
                                    batch_size = 1,
                                    denoising_strength = 0.05, # 0.05 or 0.1
                                    restore_faces = True, 
                                    width = 512,            
                                    height = 512,           
                                    ) 
            
                # saves each img2img-ed inpainted result
                for idx2, img2img_current_result in enumerate(img2img_result.images):
                    img2img_filename = current_texture_filename.replace(".png", "_inpaint-" + str(idx).zfill(3) + "_img2img-" + str(idx2).zfill(3) + "_cfg" + str(cfg)+ "_" + date_time +".png")
                    output_img2img_filename_path = os.path.join(self.output_folder, img2img_filename)
                    img2img_current_result.save(output_img2img_filename_path)
                
    def inpaint_folder(self):
        
        #   extract list of texture files
        if os.path.exists(self.partial_texture_folder):
            files = os.listdir(self.partial_texture_folder)
        else:
            print("WARNING: ", self.partial_texture_folder, " does not exit")

        total_textures = 0
        texture_file_paths = []

        #   iterates over all files in partial_textures_folder/ stores full paths sorted
        for current_file in files:
            total_textures += 1
            current_texture_path = os.path.join(self.partial_texture_folder, current_file)
            texture_file_paths.append(current_texture_path)

        texture_file_paths.sort()

        for idx, current_texture_file_path in enumerate(texture_file_paths):
            
            print(idx+1, '/', len(texture_file_paths) ,'   Processing image ', current_texture_file_path )
            
            # current mask filename
            mask_filename = os.path.basename(current_texture_file_path).replace(".png", "_mask.png") 

            # current mask filepath
            current_mask_file_path = os.path.join(self.masks_folder, mask_filename)
         
            # inpaints current texture   
            self.inpaint_texture(current_texture_file_path, current_mask_file_path)

parser = argparse.ArgumentParser(description= 'Inpaints partial texturemaps')
parser.add_argument('--partial_textures', type=str, help='Input folder with partial textures', required=True)
parser.add_argument('--masks', type=str, help='Input folder with UV masks (i.e. pixels to inpaint)', required=True)
parser.add_argument('--inpainted_textures', type=str, help='Output folder for inpainted textures', required=True)

args = parser.parse_args()

INPUT_FOLDER = args.partial_textures
OUTPUT_FOLDER = args.inpainted_textures
MASKS_FOLDER = args.masks

# removes backslash in case it's the last character of the input path
if INPUT_FOLDER.endswith('/'):
    INPUT_FOLDER = INPUT_FOLDER[:-1]

if OUTPUT_FOLDER.endswith('/'):
    OUTPUT_FOLDER = OUTPUT_FOLDER[:-1]

if MASKS_FOLDER.endswith('/'):
    MASKS_FOLDER = MASKS_FOLDER[:-1]

inpaint = InpaintWithA1111(INPUT_FOLDER, MASKS_FOLDER, OUTPUT_FOLDER)
inpaint.inpaint_folder()