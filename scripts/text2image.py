import argparse

from diffusers import StableDiffusionPipeline

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", default="simplitex-trained-model", type=str, help="Path to the model to use.")
    parser.add_argument("--guidance_scale", type=float, default=2.5, help="Value of guidance step")
    parser.add_argument("--inference_steps", type=int, default=100, help="Numver of inference steps")
    parser.add_argument("--prompt", type=str, default="a sks texturemap of an astronaut",
                        help="Prompt to use. Use sks texture map as part of your prompt for best results")
    parser.add_argument("--output_file", type=str, default="output.png", help="File onto which to save the results.")

    args = parser.parse_args()

    assert args.guidance_scale >= 0., "Invalid guidance scale value"
    assert args.inference_steps > 0, "Invalid inference steps number"

    pipe = StableDiffusionPipeline.from_pretrained(args.model_path, safety_checker = None)
    pipe.to("cuda")

    image = pipe(args.prompt, guidance_scale=args.guidance_scale, num_inference_steps=args.inference_steps).images[0]

    image.save(args.output_file)
