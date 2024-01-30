import webuiapi
from PIL import Image 
import PIL 

class StableDiffusionAPI:
    def __init__(self):
        
        # create API client
        self.api = webuiapi.WebUIApi()

        # create API client with custom host, port
        self.api = webuiapi.WebUIApi(host='127.0.0.1', port=7860)

        # create API client with default sampler, steps.
        self.api = webuiapi.WebUIApi(sampler='Euler a', steps=20)

    def img2img(self, prompt, negative_prompt="", batch_size=1, steps=20, cfg_scale=17, seed=-1, denoising_strength=0.8, mask_image=None, images=None, width=512, height=512, restore_faces=True, inpainting_fill=0):
        result = self.api.img2img(prompt=prompt,
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
                            inpainting_fill=inpainting_fill, # 0 --> fill; 1--> original; 2-->nose
                            restore_faces=restore_faces,
                            width=width,
                            height=height

                            )

        return result


    def txt2img(self, prompt, steps=20, cfg_scale=2.5, batch_size=1, negative_prompt="", styles=[], seed=-1):

        result = self.api.txt2img(prompt=prompt,
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
        #result.images

        # image is shorthand for images[0]
        #image = result.image
        #image.save('/tmp/webuiap.png')

        # info contains text info about the api call
        #result.info

        # info contains paramteres of the api call
        #result.parameters

        return result




