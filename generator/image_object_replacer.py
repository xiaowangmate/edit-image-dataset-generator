from diffusers import AutoPipelineForInpainting
from diffusers.utils import load_image
import torch
import time


class ImageObjectReplacer:
    def __init__(self, output_folder):
        self.pipe = AutoPipelineForInpainting.from_pretrained("diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
                                                              torch_dtype=torch.float16, variant="fp16").to("cuda")
        self.output_folder = output_folder

    def replace_object(self, source_image, mask_image, prompt):
        generator = torch.Generator(device="cuda").manual_seed(0)
        image = self.pipe(
            prompt=prompt,
            image=source_image,
            mask_image=mask_image,
            guidance_scale=8.0,
            num_inference_steps=20,  # steps between 15 and 30 work well for us
            strength=0.99,  # make sure to use `strength` below 1.0
            generator=generator,
        ).images[0]
        # image_name = prompt.replace(" ", "_").replace("/", "_")
        target_image_path = f"{self.output_folder}/edit.jpg"
        image.save(target_image_path)
        return target_image_path


if __name__ == '__main__':
    img_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo.png"
    mask_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo_mask.png"
    source_image = load_image(img_url).resize((1024, 1024))
    mask_image = load_image(mask_url).resize((1024, 1024))

    ior = ImageObjectReplacer(".")
    ior.replace_object(source_image, mask_image, "a robot sitting on a park bench")
