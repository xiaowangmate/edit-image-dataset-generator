import os
import json
import base64
import requests
from PIL import Image
from io import BytesIO


class EditImageGenerator:
    def __init__(self, host, port, output_folder_path="./output/images"):
        self.imi_url = f"http://{host}:{port}/sdapi/v1/img2img"
        self.output_folder_path = output_folder_path
        self.task_type_list = ["replace_object", "replace_background", "change_color",
                               "change_style", "remove_object", "add_object"]
        self.controlnet_model_list_url = f"http://{host}:{port}/controlnet/model_list?update=true"
        self.depth_model_name = self.get_controlnet_model_name("depth")

    def get_controlnet_model_name(self, module_name):
        model_list = json.loads(requests.get(self.controlnet_model_list_url).content)['model_list']
        if model_list:
            for model in model_list:
                if module_name in model:
                    return model
        else:
            raise f"not found model, please download the {module_name} model."

    def image2base64(self, image_path):
        with open(image_path, mode="rb") as r:
            image_base64 = base64.b64encode(r.read()).decode("utf-8")
            return image_base64

    def gen_edit_image(self, task_type, target_prompt, original_image, original_image_name):
        print("\nedit image...")
        if type(original_image) == bytes:
            init_images = base64.b64encode(original_image).decode("utf-8")
            img = Image.open(BytesIO(original_image))
        else:
            with open(original_image, mode="rb") as r:
                image_file = r.read()
                init_images = base64.b64encode(image_file).decode("utf-8")
                img = Image.open(BytesIO(image_file))

        width, height = img.size

        source_image_path = f"{self.output_folder_path}/{original_image_name}"
        if not os.path.exists(source_image_path):
            img.save(source_image_path)

        if task_type == "replace_object":
            parameter = self.task_replace_object(target_prompt)
        elif task_type == "replace_background":
            parameter = self.task_replace_background(target_prompt, init_images)
        elif task_type == "change_style":
            parameter = self.task_change_style(target_prompt)
        else:
            parameter = self.task_replace_object(target_prompt)

        parameter["width"] = str(width)
        parameter["height"] = str(height)
        parameter["init_images"] = [init_images]

        response = requests.post(self.imi_url, data=json.dumps(parameter))
        # print(response.content)
        image_type = original_image_name.split(".")[-1]
        output_image_name = original_image_name.replace(f".{image_type}", "_edit.jpg")
        target_image_path = f"{self.output_folder_path}/{output_image_name}"
        with open(target_image_path, mode="wb") as w:
            w.write(base64.b64decode(response.json()['images'][0]))
        print("edit image successful.")
        return source_image_path, target_image_path

    def task_replace_object(self, target_prompt):
        parameter = {
            "prompt": target_prompt + ",real, photograph",
            "negative_prompt": "nsfw, bad_quality, broken",
            "seed": -1,
            "mask": self.image2base64("./output/mask/sam_mask1.png"),
            "inpainting_fill": 3,
            "denoising_strength": 0.5,
            "cfg_scale": 6,
            "steps": 48,
            "sampler_index": "DPM++ 2M SDE Heun Karras",
        }
        return parameter

    def task_replace_object2(self, target_prompt):
        parameter = {
            "prompt": target_prompt + ",real, photograph",
            "negative_prompt": "nsfw, bad_quality, broken",
            "seed": -1,
            "mask": self.image2base64("./output/mask/sam_mask1.png"),
            # "inpainting_mask_invert": 1,
            "inpainting_fill": 2,
            "denoising_strength": 0.8,
            "cfg_scale": 10,
            "steps": 28,
            "sampler_index": "DPM++ 2M Karras",
        }
        return parameter

    def task_replace_background(self, target_prompt, init_images):
        parameter = {
            "prompt": target_prompt,
            "negative_prompt": "nsfw,bad_quality,broken,cartoon,anime,manga,comic",
            "seed": -1,
            "mask": self.image2base64("./output/mask/sam_mask1.png"),
            "inpainting_mask_invert": 1,
            "inpainting_fill": 1,
            "denoising_strength": 0.8,
            "cfg_scale": 10,
            "steps": 48,
            "sampler_index": "DPM++ 2M SDE Heun Karras",
            "alwayson_scripts": {
                "controlnet": {
                    "args": [
                        {
                            "input_image": init_images,
                            "module": "depth",
                            "model": self.depth_model_name,
                            "pixel_perfect": True
                        }
                    ]
                }
            }
        }
        return parameter

    def task_change_color(self, target_prompt):
        parameter = {
            "prompt": target_prompt,
            "negative_prompt": "nsfw, bad_quality",
            "seed": -1,
            "mask": self.image2base64("./output/mask/sam_mask1.png"),
            "inpainting_fill": 2,
            "denoising_strength": 0.8,
            "cfg_scale": 10,
            "steps": 28,
            "sampler_index": "DPM++ 2M Karras",
        }
        return parameter

    def task_change_style(self, target_prompt):
        parameter = {
            "prompt": target_prompt,
            "negative_prompt": "nsfw, bad_quality",
            "seed": -1,
            "denoising_strength": 0.5,
            "cfg_scale": 10,
            "steps": 28,
            "sampler_index": "DPM++ 2M Karras",
        }
        return parameter

    def task_remove_object(self, target_prompt):
        parameter = {
            "prompt": target_prompt,
            "negative_prompt": "nsfw, bad_quality",
            "seed": -1,
            "mask": self.image2base64("./output/mask/sam_mask1.png"),
            "inpainting_fill": 2,
            "denoising_strength": 1,
            "cfg_scale": 10,
            "steps": 24,
            "sampler_index": "DPM++ 2M Karras",
        }
        return parameter

    def task_add_object(self, target_prompt):
        parameter = {
            "prompt": target_prompt,
            "negative_prompt": "nsfw, bad_quality",
            "seed": -1,
            "mask": self.image2base64("./output/mask/sam_mask1.png"),
            "inpainting_fill": 2,
            "denoising_strength": 1,
            "cfg_scale": 10,
            "steps": 24,
            "sampler_index": "DPM++ 2M Karras",
        }
        return parameter


if __name__ == '__main__':
    edit = EditImageGenerator(host="127.0.0.1", port="7860", output_folder_path="./output")
    os.chdir('..')

    image_path = "./input/R.jpg"
    image_name = image_path.split("/")[-1]
    # with open(image_path, mode="rb") as r:
    #     image_base64 = r.read()

    edit.gen_edit_image("replace_background", "a river", image_path, image_name)
