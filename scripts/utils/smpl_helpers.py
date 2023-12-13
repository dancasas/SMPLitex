import os
import sys

import imageio
import numpy as np
import torch

file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)

from renderer.pytorch3d_renderer import render_mesh_textured


def render_amass_motion(
    motion_filepath,
    smpl,
    betas,
    expression,
    mp4_output_filename,
    device,
    texture_np,
    verts_uvs,
    faces_uvs,
    faces_verts,
    fps=30,
):
    body_poses, global_orient = load_motion(motion_filepath)

    smpl_output = smpl(
        betas=betas,
        expression=expression,
        body_pose=body_poses,
        global_orient=global_orient,
        return_verts=True,
    )

    writer = imageio.get_writer(mp4_output_filename, fps=60)

    for frame_number, current_vertices in enumerate(smpl_output.vertices):
        # renders only the first 500 frames
        if frame_number < 500:
            # output_mesh_filepath = '/tmp/hello_smpl' + str(frame_number).zfill(5) +'.obj'
            # save_obj(output_mesh_filepath, current_vertices, smpl.faces_tensor)

            frame = render_mesh_textured(
                device,
                current_vertices,
                texture_np,
                verts_uvs,
                faces_uvs,
                faces_verts.verts_idx,
                image_size=512,  # image resolution
                cam_pos=torch.tensor([2.0, 0.0, 0]),  # camera position
                azimut=90,
                mesh_rot=0,  # mesh rotation in Y axis in degrees
            )

            writer.append_data(np.asarray(frame))

    writer.close()

    # to save PIL images as saves mp4
    # mp4_output_filename = output_filename_path.replace(".png", "-preview.mp4")
    # writer = imageio.get_writer(mp4_output_filename, fps=15)
    # for i, frame in enumerate(images_for_gif):
    #    writer.append_data(np.asarray(frame))
    # writer.close()


def render_360_gif(
    device,
    verts,
    current_image_np,
    verts_uvs,
    faces_uvs,
    verts_idx,
    output_filename_path,
):
    """
    renders a 360 video in T pose using the texture from current_image_np
    """

    rotation_offset = 10
    images_for_gif = []

    for mesh_rot in np.arange(0, 360, rotation_offset):
        current_im = render_mesh_textured(
            device,
            verts,
            current_image_np,
            verts_uvs,
            faces_uvs,
            verts_idx,
            image_size=512,  # image resolution
            cam_pos=torch.tensor([2.0, 0.35, 0]),  # camera position
            mesh_rot=mesh_rot,  # mesh rotation in Y axis in degrees
        )
        images_for_gif.append(current_im)

    images_for_gif[0].save(
        output_filename_path,
        save_all=True,
        append_images=images_for_gif[1:],
        optimize=False,
        duration=40,
        loop=0,
    )


def load_motion(motion_filepath):
    """

    loads a SMPL+H motion file from .npz
    returns body_poses[#frames][69] and global_orient[#frames][3]

    """
    # open motion file
    motion = np.load(motion_filepath, allow_pickle=True)
    _motion = {}
    for k, v in motion.items():
        if isinstance(v, np.ndarray):
            print(k, motion[k].shape, motion[k].dtype)
            if motion[k].dtype in ("<U7", "<U5", "<U4", "object", "|S7"):
                _motion[k] = str(motion[k])
            else:
                _motion[k] = torch.from_numpy(motion[k]).float()
        else:
            print(k, v)
            _motion[k] = v
    motion = _motion

    body_poses = motion["poses"][:, 3:72]
    global_orient = motion["poses"][:, 0:3]

    return body_poses, global_orient
