import json
import base64
import requests
from PIL import Image


class EditImageGenerator:
    def __init__(self, host, port):
        self.imi_url = f"http://{host}:{port}/sdapi/v1/img2img"

    def image2base64(self, image_path):
        with open(image_path, mode="rb") as r:
            image_base64 = base64.b64encode(r.read()).decode("utf-8")
            return image_base64

    def gen_edit_image(self, target_prompt, original_image, original_image_name):
        print("\nedit image...")
        if type(original_image) == bytes:
            init_images = base64.b64encode(original_image).decode("utf-8")
        else:
            with open(image_path, mode="rb") as r:
                init_images = base64.b64encode(r.read()).decode("utf-8")

        if not os.path.exists(f"./output/images/{original_image_name}"):
            img = Image.open(original_image)
            img.save(f"./output/images/{original_image_name}")

        parameter = {
            "prompt": target_prompt + "photo",
            "negative_prompt": "nsfw, bad_quality",
            "seed": -1,
            "init_images": [
                init_images
            ],
            "mask": self.image2base64("./output/mask/sam_mask1.png"),
            "inpainting_fill": 2,
            "width": "1024",
            "height": "1024",
            "denoising_strength": 0.8,
            "cfg_scale": 10,
            "steps": 28,
            "sampler_index": "DPM++ 2M Karras",
        }

        response = requests.post(self.imi_url, data=json.dumps(parameter))
        # print(response.content)
        output_image_name = "edit_" + original_image_name
        with open(f"./output/images/{output_image_name}", mode="wb") as w:
            w.write(base64.b64decode(response.json()['images'][0]))
        print("edit image successful.")


if __name__ == '__main__':
    edit = EditImageGenerator(host="127.0.0.1", port="7860")
    import os
    os.chdir('..')

    image_path = "./input/586877.jpg"
    image_name = image_path.split("/")[-1]
    # with open(image_path, mode="rb") as r:
    #     image_base64 = r.read()

    edit.gen_edit_image("a bag", image_path, image_name)
