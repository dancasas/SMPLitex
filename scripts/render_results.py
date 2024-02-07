import os
import torch
import argparse
import numpy as np
import smplx as SMPL
from PIL import Image
from pytorch3d.io import load_obj
from utils.smpl_helpers import render_360_gif


class Render360:

    def __init__(self) -> None:

        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else: 
            self.device = "cpu"

        #   creates a SMPL instance, and samples it T pose
        smpl_path = os.path.abspath(os.path.join(__file__ ,"../../sample-data/SMPL/models/"))
        print(smpl_path)
        self.smpl = SMPL.create(smpl_path, model_type='smpl', gender='male')

        self.body_pose = torch.zeros(1,69)
        self.betas = torch.zeros(1,10)
        self.body_pose[0,47] = -1.35   # for A pose
        self.body_pose[0,50] =  1.30   # for A pose

        self.smpl_output = self.smpl( betas=self.betas, 
                            body_pose=self.body_pose, 
                            return_verts=True)

        self.verts = self.smpl_output.vertices[0]
        self.faces = self.smpl.faces_tensor

        #   loads the SMPL template mesh to extract UV coordinates for the textures
        mesh_filename = os.path.join(__file__,"../../sample-data/smpl_uv_20200910/smpl_uv.obj")
        _, self.faces_verts, aux = load_obj(mesh_filename)
        self.verts_uvs = aux.verts_uvs[None, ...]        # (1, F, 3)
        self.faces_uvs = self.faces_verts.textures_idx[None, ...]   # (1, F, 3)

    def render_textures(self, textures_folder):

        #   extract list of texture files
        if os.path.exists(textures_folder):
            files = os.listdir(textures_folder)
        else:
            print("ERROR: ", textures_folder, " does not exit")

        for idx, current_file in enumerate(files):
            current_texture_path = os.path.join(textures_folder, current_file)
            print('\nProcessing image ', current_texture_path)

            if ".jpg" in current_texture_path or ".png" in current_texture_path:
                with Image.open(current_texture_path) as image:
                    current_image_np = np.asarray(image.convert("RGB")).astype(np.float32)

                render_360_gif(self.device, self.verts, 
                        current_image_np, self.verts_uvs, 
                        self.faces_uvs, self.faces_verts.verts_idx, 
                        current_texture_path.replace(".png", "-360.gif"))
                
parser = argparse.ArgumentParser(description= 'Renders SMPL 360 gifs given input textures')
parser.add_argument('--textures', type=str, help='Folder with textures', required=True)

args = parser.parse_args()

INPUT_FOLDER = args.textures

render = Render360()
render.render_textures(INPUT_FOLDER)