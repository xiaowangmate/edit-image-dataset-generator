import os
import json
import base64
import requests
from PIL import Image
from io import BytesIO


class SAMPredictor:
    def __init__(self, host, port):
        self.sam_url = f"http://{host}:{port}/sam/sam-predict"
        self.sam_model_url = f"http://{host}:{port}/sam/sam-model"
        self.sam_expand_url = f"http://{host}:{port}/sam/dilate-mask"
        self.mask_save_path = f"./output/mask/sam_mask.png"
        self.preview_grid = Image.new('RGBA', (3 * 512, 3 * 512))
        self.sam_model = self.get_sam_model_name()

    def get_sam_model_name(self):
        response = requests.get(self.sam_model_url)
        model_list = json.loads(response.content)
        if model_list:
            return model_list[0]
        else:
            raise "not found model, please download a sam model."

    def image2base64(self, image_path):
        with open(image_path, mode="rb") as r:
            image_base64 = base64.b64encode(r.read()).decode("utf-8")
            return image_base64

    def gen_mask(self, mask_target, original_image, dilate_amount=0):
        print("\ngen mask...")
        if type(original_image) == bytes:
            input_image = base64.b64encode(original_image).decode()
        else:
            input_image = self.image2base64(original_image)

        payload = {
            "sam_model_name": self.sam_model,
            "input_image": input_image,
            "dino_enabled": True,
            "dino_text_prompt": mask_target,
            "dino_preview_checkbox": False,
        }
        response = requests.post(self.sam_url, json=payload)
        reply = response.json()

        # print(reply, reply.keys())
        if 'errors' in reply.keys():
            print(f"segment error: {reply}")
            raise reply['errors']
        else:
            print(f"segment info: {reply['msg']}")

            mask = reply["masks"][0]

            if dilate_amount == 0:
                self.write_mask(mask)
            else:
                self.expand_mask(input_image, mask, dilate_amount)

    def reset_preview_grid(self):
        self.preview_grid = Image.new('RGBA', (3 * 512, 3 * 512))

    def preview_mask(self, images, row):
        for idx, img in enumerate(images):
            img_pil = Image.open(BytesIO(base64.b64decode(img))).resize((512, 512))
            self.preview_grid.paste(img_pil, (idx * 512, row * 512))
            self.preview_grid.show()

    def expand_mask(self, input_image, mask, dilate_amount):
        payload = {
            "input_image": input_image,
            "mask": mask,
            "dilate_amount": dilate_amount
        }
        response = requests.post(self.sam_expand_url, json=payload)
        reply = response.json()
        # print(reply, reply.keys())

        self.write_mask(reply["mask"])
        print("expand mask successful.")

    def write_mask(self, mask):
        file = open(self.mask_save_path, "wb")
        file.write(base64.b64decode(mask))
        file.close()


def test(input_path, mask_target, dilate_amount):
    os.chdir('..')
    with open("configs/config.json", mode="r", encoding="utf-8") as r:
        config = json.loads(r.read())
        host = config['base_config']['host']
        port = config['base_config']['port']

    predictor = SAMPredictor(host=host, port=port)
    predictor.gen_mask(mask_target, input_path, dilate_amount)


if __name__ == '__main__':
    input_path = "./input/R.jpg"

    dilate_amount = 0
    mask_target = "puppy"

    test(input_path, mask_target, dilate_amount)
