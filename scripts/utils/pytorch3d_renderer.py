import torch
import torch.nn as nn
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from PIL import Image
import os

# from configs import paths
uv_path = "./densepose_uv_data/UV_Processed.mat"

# Data structures and functions for rendering
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    OrthographicCameras,
    PointLights,
    RasterizationSettings,
    MeshRasterizer,
    MeshRenderer,
    SoftPhongShader,
    TexturesUV,
    TexturesVertex,
    BlendParams,
)


def render_mesh_plain_color(device, verts, faces):
    # Initialize each vertex to be white in color.
    verts_rgb = torch.ones_like(verts)[None]  # (1, V, 3)
    textures = TexturesVertex(verts_features=verts_rgb.to(device))

    # Create a Meshes object for the teapot. Here we have only one mesh in the batch.
    mesh = Meshes(verts=[verts.to(device)], faces=[faces.to(device)], textures=textures)

    # Place a point light in front of the object. As mentioned above, the front of the cow is facing the
    # -z direction.
    lights = PointLights(device=device, location=[[0.0, 0.0, -3.0]])

    # Initialize a camera.
    # With world coordinates +Y up, +X left and +Z in, the front of the cow is facing the -Z direction.
    # So we move the camera by 180 in the azimuth direction so it is facing the front of the cow.
    R, T = look_at_view_transform(2.7, 0, 0)
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T)

    # Define the settings for rasterization and shading. Here we set the output image to be of size
    # 512x512. As we are rendering images for visualization purposes only we will set faces_per_pixel=1
    # and blur_radius=0.0. We also set bin_size and max_faces_per_bin to None which ensure that
    # the faster coarse-to-fine rasterization method is used. Refer to rasterize_meshes.py for
    # explanations of these parameters. Refer to docs/notes/renderer.md for an explanation of
    # the difference between naive and coarse-to-fine rasterization.
    raster_settings = RasterizationSettings(
        image_size=512,
        blur_radius=0.0,
        faces_per_pixel=1,
    )

    background_color = (0.0, 0.0, 0.0)
    blend_params = BlendParams(background_color=background_color)

    # Create a Phong renderer by composing a rasterizer and a shader. The textured Phong shader will
    # interpolate the texture uv coordinates for each vertex, sample from a texture image and
    # apply the Phong lighting model
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
        shader=SoftPhongShader(
            device=device, cameras=cameras, lights=lights, blend_params=blend_params
        ),
    )

    images = renderer(mesh)
    R_channel = images[0, :, :, 0].detach().cpu().numpy()
    G_channel = images[0, :, :, 1].detach().cpu().numpy()
    B_channel = images[0, :, :, 2].detach().cpu().numpy()
    rgbArray = np.zeros((512, 512, 3), "uint8")
    rgbArray[..., 0] = R_channel * 255
    rgbArray[..., 1] = G_channel * 255
    rgbArray[..., 2] = B_channel * 255
    img = Image.fromarray(rgbArray)
    img.save("/tmp/myimg_white.png")

    # plt.figure(figsize=(10, 10))
    # plt.imshow(images[0, ..., :3].cpu().numpy())
    # plt.axis("off")
    # plt.pause(1)


def render_mesh_iuv_color(
    device,
    verts,
    faces,
    image_size=None,
    cam_pos=None,
    mesh_rot=None,
    background_color=None,
    output_path="/tmp",
    output_filename="myimage_iuv.png",
):
    batch_size = 1

    # default image size
    if image_size is None:
        image_size = 512

    # default camera position
    if cam_pos is None:
        cam_pos = torch.tensor([2.0, 0.35, 0])

    # default mesh rotation
    if mesh_rot is None:
        mesh_rot = 0

    # default background color
    if background_color is None:
        background_color = (0.0, 0.0, 0.0)

    verts_uv_offset, verts_iuv, verts_map, faces_densepose = preprocess_densepose_UV(
        uv_path=uv_path, batch_size=batch_size
    )

    vertices = verts[
        verts_map, :
    ]  # From SMPL verts indexing (0 to 6889) to DP verts indexing (0 to 7828), verts shape is (B, 7829, 3)
    textures = TexturesVertex(verts_features=verts_iuv.to(device))
    mesh = Meshes(
        verts=[vertices.to(device)],
        faces=[faces_densepose[0].to(device)],
        textures=textures,
    )

    lights = PointLights(
        device=device,
        ambient_color=[[1, 1, 1]],
        diffuse_color=[[0, 0, 0]],
        specular_color=[[0, 0, 0]],
    )

    # Initialize a camera.
    # With world coordinates +Y up, +X left and +Z in, the front of the mesh is facing the +Z direction.
    # So we move the camera by mesh_rotation in the azimuth direction.
    R, T = look_at_view_transform(cam_pos[0], 0, mesh_rot)
    T[0, 1] += cam_pos[1]
    # cameras = FoVPerspectiveCameras(device=device, R=R, T=T)
    cameras = OrthographicCameras(device=device, T=T, R=R)

    # Define the settings for rasterization and shading. Here we set the output image to be of size
    # 512x512. As we are rendering images for visualization purposes only we will set faces_per_pixel=1
    # and blur_radius=0.0. We also set bin_size and max_faces_per_bin to None which ensure that
    # the faster coarse-to-fine rasterization method is used. Refer to rasterize_meshes.py for
    # explanations of these parameters. Refer to docs/notes/renderer.md for an explanation of
    # the difference between naive and coarse-to-fine rasterization.
    raster_settings = RasterizationSettings(
        image_size=image_size,
        blur_radius=0.0,
        faces_per_pixel=1,
    )

    blend_params = BlendParams(background_color=background_color)

    # Create a Phong renderer by composing a rasterizer and a shader. The textured Phong shader will
    # interpolate the texture uv coordinates for each vertex, sample from a texture image and
    # apply the Phong lighting model
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
        shader=SoftPhongShader(
            device=device, cameras=cameras, lights=lights, blend_params=blend_params
        ),
    )

    images = renderer(mesh)
    I_channel = images[0, :, :, 0].detach().cpu().numpy().round()
    U_channel = images[0, :, :, 1].detach().cpu().numpy()
    V_channel = images[0, :, :, 2].detach().cpu().numpy()
    rgbArray = np.zeros((image_size, image_size, 3), "uint8")

    # WARNING: CHECK THIS ASSIGNMENT IF UV VALUES DO NOT MATCH WITH TENSEPOSE OUTPUT. Perhaps swap channels?
    rgbArray[..., 0] = (V_channel * 255).astype(int)
    rgbArray[..., 1] = (U_channel * 255).astype(int)
    rgbArray[..., 2] = I_channel.astype(int)
    img = Image.fromarray(rgbArray, mode="RGB")
    img.save(os.path.join(output_path, output_filename))

    # plt.figure(figsize=(10, 10))
    # plt.imshow(images[0, ..., :3].cpu().numpy())
    # plt.axis("off")
    # plt.pause(1)


def render_mesh_textured(
    device,
    verts,
    textures,
    verts_uvs,
    faces_uvs,
    faces_vertices,
    image_size=None,
    cam_pos=None,
    azimut=0,
    mesh_rot=None,
    background_color=None,
    output_path=None,
    output_filename=None,
):
    batch_size = 1

    # default image size
    if image_size is None:
        image_size = 512

    # default camera position
    if cam_pos is None:
        cam_pos = torch.tensor([2.0, 0.35, 0])

    # default mesh rotation
    if mesh_rot is None:
        mesh_rot = 0

    # default background color
    if background_color is None:
        background_color = (1.0, 1.0, 1.0)

    tex = torch.from_numpy(textures / 255.0)[None].to(device)
    textures_rgb = TexturesUV(
        maps=tex, faces_uvs=faces_uvs.to(device), verts_uvs=verts_uvs.to(device)
    )

    mesh = Meshes(
        verts=[verts.to(device)],
        faces=[faces_vertices.to(device)],
        textures=textures_rgb,
    )

    lights = PointLights(
        device=device,
        ambient_color=[[1, 1, 1]],
        diffuse_color=[[0, 0, 0]],
        specular_color=[[0, 0, 0]],
    )

    # Initialize a camera.
    # With world coordinates +Y up, +X left and +Z in, the front of the mesh is facing the +Z direction.
    # So we move the camera by mesh_rotation in the azimuth direction.
    R, T = look_at_view_transform(cam_pos[0], azimut, mesh_rot)
    T[0, 1] += cam_pos[1]
    # cameras = FoVPerspectiveCameras(device=device, R=R, T=T)
    cameras = OrthographicCameras(device=device, T=T, R=R)

    # Define the settings for rasterization and shading. Here we set the output image to be of size
    # image_size_inximage_size_in. As we are rendering images for visualization purposes only we
    # will set faces_per_pixel=1 and blur_radius=0.0. We also set bin_size and max_faces_per_bin to
    # None which ensure that the faster coarse-to-fine rasterization method is used. Refer to
    # rasterize_meshes.py for explanations of these parameters. Refer to docs/notes/renderer.md for an
    # explanation of the difference between naive and coarse-to-fine rasterization.
    raster_settings = RasterizationSettings(
        image_size=image_size,
        blur_radius=0.0,
        faces_per_pixel=1,
    )

    blend_params = BlendParams(background_color=background_color)

    # Create a Phong renderer by composing a rasterizer and a shader. The textured Phong shader will
    # interpolate the texture uv coordinates for each vertex, sample from a texture image and
    # apply the Phong lighting model
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
        shader=SoftPhongShader(
            device=device, cameras=cameras, lights=lights, blend_params=blend_params
        ),
    )

    images = renderer(mesh)
    R_channel = images[0, :, :, 0].detach().cpu().numpy()
    G_channel = images[0, :, :, 1].detach().cpu().numpy()
    B_channel = images[0, :, :, 2].detach().cpu().numpy()
    rgbArray = np.zeros((image_size, image_size, 3), "uint8")
    rgbArray[..., 0] = (R_channel * 255).astype(int)
    rgbArray[..., 1] = (G_channel * 255).astype(int)
    rgbArray[..., 2] = (B_channel * 255).astype(int)
    img = Image.fromarray(rgbArray, mode="RGB")

    if output_filename is not None:
        print("Saving ", os.path.join(output_path, output_filename), "\n")
        img.save(os.path.join(output_path, output_filename))
    else:
        return img
    # plt.figure(figsize=(10, 10))
    # plt.imshow(images[0, ..., :3].cpu().numpy())
    # plt.axis("off")
    # plt.pause(1)


def preprocess_densepose_UV(uv_path, batch_size):
    DP_UV = loadmat(uv_path)
    faces_bodyparts = torch.Tensor(
        DP_UV["All_FaceIndices"]
    ).squeeze()  # (13774,) face to DensePose body part mapping
    faces_densepose = torch.from_numpy(
        (DP_UV["All_Faces"] - 1).astype(np.int64)
    )  # (13774, 3) face to vertices indices mapping
    verts_map = (
        torch.from_numpy(DP_UV["All_vertices"][0].astype(np.int64)) - 1
    )  # (7829,) DensePose vertex to SMPL vertex mapping
    u_norm = torch.Tensor(
        DP_UV["All_U_norm"]
    )  # (7829, 1)  # Normalised U coordinates for each vertex
    v_norm = torch.Tensor(
        DP_UV["All_V_norm"]
    )  # (7829, 1)  # Normalised V coordinates for each vertex

    # RGB texture images/maps are processed into a 6 x 4 grid (atlas) of 24 textures.
    # Atlas is ordered by DensePose body parts (down rows then across columns).
    # UV coordinates for vertices need to be offset to match the texture image grid.
    offset_per_part = {}
    already_offset = set()
    cols, rows = 4, 6
    for i, u in enumerate(np.linspace(0, 1, cols, endpoint=False)):
        for j, v in enumerate(np.linspace(0, 1, rows, endpoint=False)):
            part = rows * i + j + 1  # parts are 1-indexed in face_indices
            offset_per_part[part] = (u, v)
    u_norm_offset = u_norm.clone()
    v_norm_offset = v_norm.clone()
    vertex_parts = torch.zeros(
        u_norm.shape[0]
    )  # Also want to get a mapping between vertices and their corresponding DP body parts (technically one-to-many but ignoring that here).
    for i in range(len(faces_densepose)):
        face_vert_idxs = faces_densepose[i]
        part = faces_bodyparts[i]
        offset_u, offset_v = offset_per_part[int(part.item())]
        for vert_idx in face_vert_idxs:
            # vertices are reused (at DensePose part boundaries), but we don't want to offset multiple times
            if vert_idx.item() not in already_offset:
                # offset u value
                u_norm_offset[vert_idx] = u_norm_offset[vert_idx] / cols + offset_u
                # offset v value
                # this also flips each part locally, as each part is upside down
                v_norm_offset[vert_idx] = (
                    1 - v_norm_offset[vert_idx]
                ) / rows + offset_v
                # add vertex to our set tracking offsetted vertices
                already_offset.add(vert_idx.item())
        vertex_parts[face_vert_idxs] = part

    # WARNING: CHECK THIS IF UV VALUES DO NOT MATCH WITH TENSEPOSE OUTPUT
    # invert V values
    # v_norm = 1 - v_norm
    # v_norm_offset = 1 - v_norm_offset

    # Combine body part indices (I), and UV coordinates
    verts_uv_offset = torch.cat(
        [u_norm_offset[None], v_norm_offset[None]], dim=2
    ).expand(
        batch_size, -1, -1
    )  # (batch_size, 7829, 2)
    verts_iuv = torch.cat(
        [vertex_parts[None, :, None], u_norm[None], v_norm[None]], dim=2
    ).expand(
        batch_size, -1, -1
    )  # (batch_size, 7829, 3)

    # Add a batch dimension to faces
    faces_densepose = faces_densepose[None].expand(batch_size, -1, -1)

    return verts_uv_offset, verts_iuv, verts_map, faces_densepose
