
from UVTextureConverter import UVConverter
from UVTextureConverter import Normal2Atlas
from UVTextureConverter import Atlas2Normal
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import numpy as np
import argparse
import skimage
import re
import os

class RGB2Texture:

    def __init__(self, dataset_root_path) -> None:
        self.dataset_root_path = dataset_root_path
        
        
    def apply_mask_to_iuv_images(self):
        dataset_iuv_path = os.path.join(self.dataset_root_path, 'densepose')
        dataset_mask_path = os.path.join(self.dataset_root_path, 'images-seg')

        if not os.path.exists(dataset_iuv_path):
            print("ERROR: ", dataset_iuv_path, " does not exist")
            return

        if not os.path.exists(dataset_mask_path):
            print("ERROR: ", dataset_mask_path, " does not exist")
            return

        # output path for masked iuv
        output_iuv_masked_folder = "densepose-masked"
        output_iuv_masked_folder_path = os.path.join(self.dataset_root_path, output_iuv_masked_folder)
        isExist = os.path.exists(output_iuv_masked_folder_path)
        if not isExist:
            os.makedirs(output_iuv_masked_folder_path)

        files = os.listdir(dataset_iuv_path)

        num_images = 0

        # reads all the images and stores full paths in list
        for path in files:

            num_images += 1
            current_iuv_path = os.path.join(dataset_iuv_path, path)
            current_mask_path = path.replace("_densepose.png", ".png")
            current_mask_path = os.path.join(dataset_mask_path, current_mask_path)

            if os.path.isfile(current_iuv_path):
                if os.path.isfile(current_mask_path):

                    with Image.open(current_iuv_path) as im_iuv:
                        with Image.open(current_mask_path) as im_mask:

                            print('\nSegmenting image ', num_images, '/', len(files))
                            #print('Loading      ', current_iuv_path)
                            #print('Loading      ', current_mask_path)

                            iuv_w, iuv_h = im_iuv.size
                            mask_w, mask_h = im_mask.size

                            if (iuv_w == mask_w) and (iuv_h == mask_h):
                                
                                threshold = 250
                                im_mask = im_mask.point(lambda x: 255 if x > threshold else 0)
                                blank = im_iuv.point(lambda _: 0)
                                masked_iuv_image = Image.composite(im_iuv, blank, im_mask)
                                
                                print('Writing image ', os.path.join(output_iuv_masked_folder_path, path))
                                masked_iuv_image.save(os.path.join(output_iuv_masked_folder_path, path), "PNG")
                            else:

                                print('Discarding images because densepose and RGB do not match. Probably densepose failed?')
                                
            
                else:
                    print(current_mask_path, 'does not exist')
            else:
                print(current_iuv_path, 'does not exist')


    def generate_uv_texture(self):
    
        # paths
        dataset_image_path = os.path.join(self.dataset_root_path, 'images')
        dataset_iuv_path = os.path.join(self.dataset_root_path, 'densepose-masked')
        
        # output path for UV textures
        output_textures_folder = "uv-textures"
        output_textures_folder_path = os.path.join(self.dataset_root_path, output_textures_folder)
        isExist = os.path.exists(output_textures_folder_path)
        if not isExist:
            os.makedirs(output_textures_folder_path)

        # output path for debug figure
        output_debug_folder = "debug"
        output_debug_folder_path = os.path.join(self.dataset_root_path, output_debug_folder)
        isExist = os.path.exists(output_debug_folder_path)
        if not isExist:
            os.makedirs(output_debug_folder_path)

        # list to store files
        images_file_paths = []
        images_iuv_paths = []

        num_images = 0

        # extract list of image files
        files = os.listdir(dataset_image_path)

        # size (in pixels) of each part in texture
        # WARNING: for best results, update this value depending on the input image size
        parts_size = 120 

        # reads all the images and stores full paths in list
        for path in files:
            num_images += 1
            current_image_path = os.path.join(dataset_image_path, path)
            current_iuv_path = os.path.join(dataset_iuv_path, path.replace(".jpg", "_densepose.png"))

            # check if both image and iuv exists
            if os.path.isfile(current_image_path):
                if os.path.isfile(current_iuv_path):
                    images_file_paths.append(current_image_path)
                    images_iuv_paths.append(current_iuv_path)
                    # print('\nProcessing image ', num_images, '/', len(files))
                    # print(current_image_path)
                    # print(current_iuv_path)

                else:
                    print(current_iuv_path, ' does not exist')
            else:
                print(current_image_path, ' does not exist')

        num_images = 0

        # sorts filenames alphabetically
        images_iuv_paths.sort()
        images_file_paths.sort()

        images_iuv_paths_filtered = images_iuv_paths.copy()
        images_file_paths_filtered = images_file_paths.copy()

        num_images = 0
        previous_tex = 0

        for current_image_path, current_iuv_path in zip(images_file_paths_filtered, images_iuv_paths_filtered):
            
            num_images += 1
            
            print('\nComputing UV texture ', num_images, '/', len(images_file_paths_filtered))
            #print('Loading        ', current_image_path)
            #print('Loading        ', current_iuv_path)

            tokenized_file_name = re.split(r'_|-',os.path.basename(current_image_path))

            # create texture by densepose in atlas style
            # tex_trans: (24, 200, 200, 3), mask_trans: (24, 200, 200)
            tex_trans, mask_trans = UVConverter.create_texture(current_image_path, current_iuv_path,
            parts_size=parts_size, concat=False)

            # for display
            # tex = UVConverter.concat_atlas_tex(tex_trans)  # 800 x 1200 x 3
            # mask = UVConverter.concat_atlas_tex(mask_trans)  # 800 x 1200

            # convert from atlas to normal
            texture_size = 512 

            converter = Atlas2Normal(atlas_size=parts_size, normal_size=texture_size)
            normal_tex, normal_ex = converter.convert((tex_trans*255).astype('int'), mask=mask_trans)

            # shows result
            fig, (ax1, ax2, ax3) = plt.subplots(1,3,figsize=(13,4))
            ax1.imshow(normal_tex)
            
            ax2.set_title(os.path.basename(current_image_path), fontsize=8)
            ax2.imshow(mpimg.imread(current_image_path))
            
            ax3.set_title(os.path.basename(current_iuv_path), fontsize=8)
            ax3.imshow(mpimg.imread(current_iuv_path))
            
            # plt.show()

            # file names for debug and output texture            
            output_debug_filename = os.path.basename(current_image_path).replace(".jpg", "_debug.png")
            output_debug_file_path = os.path.join(output_debug_folder_path, output_debug_filename)

            output_texture_filename = os.path.basename(current_image_path).replace(".jpg", "_texture.png")
            output_textures_file_path = os.path.join(output_textures_folder_path, output_texture_filename)

            # save debug image
            print('Saving         ', output_debug_file_path)
            plt.savefig(output_debug_file_path, dpi=150)
            plt.close(fig) 

            # save output uv texture
            normal_tex = (normal_tex * 255).round().astype(np.uint8)
            im = Image.fromarray(normal_tex, 'RGB')
            im.save(output_textures_file_path)
            print('Saving         ', output_textures_file_path)
            
            previous_tex = normal_tex

    def compute_mask_of_partial_uv_textures(self):

        # folder where the uv textures are. Ideally, these are textures generated with masked images 
        # (e.g, the IUV images where segmented using masks computed on the rendered images)
        uv_textures_path = os.path.join(self.dataset_root_path, 'uv-textures')
        
        files = os.listdir(uv_textures_path)

        # output path for UV masks
        output_masks_folder = "uv-textures-masks"
        output_masks_folder_path = os.path.join(self.dataset_root_path, output_masks_folder)
        isExist = os.path.exists(output_masks_folder_path)
        if not isExist:
            os.makedirs(output_masks_folder_path)

        num_images = 0

        # reads all the images and computes UV mask
        for image_filename in files:
            num_images += 1
            print('\nComputing UV mask ', num_images, '/', len(files))

            current_image = skimage.io.imread(os.path.join(uv_textures_path, image_filename))

            mask = (current_image[:, :, 0] < 2) & (current_image[:, :, 1] < 2) & (current_image[:, :, 1] < 2)

            mask_filename = image_filename.replace(".png", "_mask.png")
            print('Writing image ', os.path.join(output_masks_folder_path, mask_filename))
            skimage.io.imsave(os.path.join(output_masks_folder_path, mask_filename), skimage.img_as_ubyte(mask))


parser = argparse.ArgumentParser(description= 'Computes partial texturemap from RGB image')
parser.add_argument('--input_folder', type=str, help='Root folder with subfolders images/ and densepose/', required=True)

args = parser.parse_args()

INPUT_FOLDER = args.input_folder

# removes backslash in case it's the last character of the input path
if INPUT_FOLDER.endswith('/'):
    INPUT_FOLDER = INPUT_FOLDER[:-1]

densepose = RGB2Texture(INPUT_FOLDER)
densepose.apply_mask_to_iuv_images()
densepose.generate_uv_texture()
densepose.compute_mask_of_partial_uv_textures()