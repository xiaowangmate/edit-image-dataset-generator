import json
import base64
import requests


class EditImageGenerator:
    def __init__(self, host, port):
        self.imi_url = f"http://{host}:{port}/sdapi/v1/img2img"

    def img2base64(self, image_path):
        with open(image_path, mode="rb") as r:
            image_base64 = base64.b64encode(r.read()).decode("utf-8")
            return image_base64

    def gen_edit_image(self, target_prompt, image_path):
        print("\nedit image...")
        parameter = {
            "prompt": target_prompt,
            "negative_prompt": "nsfw, bad_quality",
            "seed": -1,
            "init_images": [
                self.img2base64(image_path)
            ],
            "mask": self.img2base64("./output/mask/sam_mask1.png"),
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
        output_image_name = "edit_" + image_path.split("/")[-1]
        with open(f"./output/{output_image_name}", mode="wb") as w:
            w.write(base64.b64decode(response.json()['images'][0]))
        print("edit image successful.")


if __name__ == '__main__':
    edit = EditImageGenerator(host="127.0.0.1", port="7860")
    image_path = "../input/586877.jpg"
    mask_path = "../input/mask.png"
    edit.gen_edit_image("pocket", image_path)
