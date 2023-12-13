import argparse
import os

import numpy as np
import torch
from PIL import Image
from diffusers import StableDiffusionInpaintPipeline
from diffusers.utils import load_image

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_path",
        default="simplitex-trained-model",
        type=str,
        help="Path to the model to use.",
    )
    parser.add_argument(
        "--guidance_scale", type=float, default=2, help="Value of guidance step"
    )

    parser.add_argument(
        "--guidance_scale_refinement",
        type=float,
        default=1,
        help="Value of guidance step for refining steps. Not used when --refine is False.",
    )
    parser.add_argument(
        "--inference_steps", type=int, default=200, help="Number of inference steps"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="a sks texturemap",
        help="Prompt to use. Use sks texture map as part of your prompt for best results",
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        default="output",
        help="Folder onto which to save the results.",
    )

    parser.add_argument(
        "--mask_path",
        type=str,
        default="data_inpainting/mask_example.png",
        help="Path to mask image.",
    )

    parser.add_argument(
        "--image_path",
        type=str,
        default="data_inpainting/input_example.png",
        help="Path to input image.",
    )

    parser.add_argument(
        "--refine",
        type=bool,
        default=False,
        help="Set to True if you want to refine the results after inpainting",
    )

    parser.add_argument(
        "--render",
        type=bool,
        default=False,
        help="Set to True if you want to render the results",
    )

    args = parser.parse_args()

    assert args.guidance_scale >= 0.0, "Invalid guidance scale value"
    assert args.inference_steps > 0, "Invalid inference steps number"

    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        args.model_path, safety_checker=None
    )
    pipe.to("cuda")

    image = load_image(args.image_path)
    mask_image = load_image(args.mask_path)

    image = pipe(
        prompt=args.prompt,
        image=image,
        mask_image=mask_image,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.inference_steps,
        strength=1,
    ).images[0]

    if args.refine:
        assert args.guidance_scale_refinement >= 0.0, "Invalid guidance scale value"
        print("Refining in two steps")
        from diffusers import StableDiffusionImg2ImgPipeline

        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            args.model_path, safety_checker=None
        )
        pipe.to("cuda")
        image = pipe(
            prompt=args.prompt,
            image=image,
            guidance_scale=args.guidance_scale_refinement,
            num_inference_steps=args.inference_steps * 4,
            strength=0.05,
        ).images[0]

        image = pipe(
            prompt=args.prompt,
            image=image,
            guidance_scale=args.guidance_scale_refinement,
            num_inference_steps=args.inference_steps * 20,
            strength=0.01,
        ).images[0]

    path_save = os.path.join(os.getcwd(), args.output_folder)

    if not os.path.exists(path_save):
        os.mkdir(path_save)

    image.save(os.path.join(path_save, "texture.png"))

    if args.render:
        import smplx as SMPL
        from pytorch3d.io import load_obj
        from utils.smpl_helpers import render_mesh_textured

        with Image.open(os.path.join(path_save, "texture.png")) as image:
            np_image = np.asarray(image.convert("RGB")).astype(np.float32)

        smpl = SMPL.create(
            "../sample-data/SMPL/models/", model_type="smpl", gender="male"
        )
        betas, expression = None, None

        smpl_output = smpl(betas=betas, expression=expression, return_verts=True)

        #   loads the SMPL template mesh to extract UV coordinates for the textures
        mesh_filename = "../sample-data/smpl_uv_20200910/smpl_uv.obj"
        _, faces_verts, aux = load_obj(mesh_filename)
        verts_uvs = aux.verts_uvs[None, ...]  # (1, F, 3)
        faces_uvs = faces_verts.textures_idx[None, ...]  # (1, F, 3)

        #   debug: saves sampled SMPL mesh
        # save_obj('/tmp/hello_smpl3.obj', smpl_output.vertices[0], smpl.faces_tensor)
        # verts, faces_idx, _ = load_obj('/tmp/hello_smpl3.obj')

        verts = smpl_output.vertices[0]
        faces = smpl.faces_tensor

        verts_T_pose = verts

        render_mesh_textured(
            "cpu",
            verts,
            np_image,
            verts_uvs,
            faces_uvs,
            faces_verts.verts_idx,
            image_size=1024,  # image resolution
            cam_pos=torch.tensor([2.0, 0.35, 0]),  # camera position
            mesh_rot=0,  # mesh rotation in Y axis in degrees
            output_path=path_save,
            output_filename="front_render.png",
        )

        # back render
        render_mesh_textured(
            "cpu",
            verts,
            np_image,
            verts_uvs,
            faces_uvs,
            faces_verts.verts_idx,
            image_size=1024,  # image resolution
            cam_pos=torch.tensor([2.0, 0.35, 0]),  # camera position
            mesh_rot=180,  # mesh rotation in Y axis in degrees
            output_path=path_save,
            output_filename="back_render.png",
        )
