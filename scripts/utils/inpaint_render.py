"""
step 1. inpainting

0. interrogate CLIP
1. upload mask
2. check 'inpaint not masked'
3. check 'mask content' --> fill
4. check 'inpaint at full resolution'
5. check 'resze and fill'
6. batch count 1
7. batch size 4
8. cfg scale 17
9. denoising step 0.5
10 settings:
    Steps: 75, CFG scale: 17, Seed: 2419560507, Size: 512x512,
    Model hash: 7460a6fa, Batch size: 4, Batch pos: 0,
    Denoising strength: 0.5, Mask blur: 0


step 2.
0. download inpainted image
1. downscale to 256x256
2. img2img using as prompt the description of dataset + photorealisc high-quality dlsr. negative prompt: drawing
3. use script loopback. loops 4. denoising strength: 0.9
4. settings:
        steps: 75, CFG scale: 27, Seed: 196686560, Size: 512x512,
        Model hash: 7460a6fa, Denoising strength: 0.3,
        Denoising strength change factor: 0.9

"""
import webuiapi
from PIL import Image
import PIL


class StableDiffusionAPI:
    def __init__(self):
        # create API client
        self.api = webuiapi.WebUIApi()

        # create API client with custom host, port
        self.api = webuiapi.WebUIApi(host="127.0.0.1", port=7860)

        # create API client with default sampler, steps.
        self.api = webuiapi.WebUIApi(sampler="Euler a", steps=20)

    def img2img(
        self,
        prompt,
        negative_prompt="",
        batch_size=1,
        steps=20,
        cfg_scale=17,
        seed=-1,
        denoising_strength=0.8,
        mask_image=None,
        images=None,
    ):
        result = self.api.img2img(
            prompt=prompt,
            negative_prompt=negative_prompt,
            images=images,
            mask_image=mask_image,
            seed=seed,
            inpainting_mask_invert=0,
            cfg_scale=cfg_scale,
            mask_blur=0,
            steps=steps,
            batch_size=batch_size,
            denoising_strength=denoising_strength,
            inpainting_fill=0,  # 0 --> fill; 1--> original
            restore_faces=True,
        )

        return result

    def txt2img(
        self,
        prompt,
        steps=30,
        cfg_scale=2.5,
        batch_size=1,
        negative_prompt="",
        styles=[],
        seed=-1,
    ):
        result = self.api.txt2img(
            prompt=prompt,
            negative_prompt=negative_prompt,
            seed=seed,
            batch_size=batch_size,
            styles=styles,
            cfg_scale=cfg_scale,
            restore_faces=True,
            # sampler_index='DDIM',
            steps=steps,
        )
        # images contains the returned images (PIL images)
        # result.images

        # image is shorthand for images[0]
        # image = result.image
        # image.save('/tmp/webuiap.png')

        # info contains text info about the api call
        # result.info

        # info contains paramteres of the api call
        # result.parameters

        return result
