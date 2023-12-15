import os
import json
import base64
import requests
from PIL import Image
from io import BytesIO


class SAMPredictor:
    def __init__(self, host, port):
        self.sam_url = f"http://{host}:{port}/sam/sam-predict"
        self.sam_model = self.get_sam_model_name()

    def get_sam_model_name(self):
        response = requests.get(self.sam_url.replace("predict", "model"))
        model_list = json.loads(response.content)
        if model_list:
            return model_list[0]
        else:
            raise "not found model, please download a sam model."

    def image2base64(self, image_path):
        with open(image_path, mode="rb") as r:
            image_base64 = base64.b64encode(r.read()).decode("utf-8")
            return image_base64

    def gen_mask(self, mask_target, original_image):
        print("\ngen mask...")
        if type(original_image) == bytes:
            input_image = base64.b64encode(original_image).decode()
        else:
            input_image = self.image2base64(original_image)

        payload = {
            "sam_model_name": self.sam_model,
            "input_image": input_image,
            "dino_enabled": True,
            # "sam_positive_points": [[0, 0]],
            "dino_text_prompt": mask_target,
            "dino_preview_checkbox": False,
        }
        response = requests.post(self.sam_url, json=payload)
        reply = response.json()

        # print(reply)
        print(f"segment info: {reply['msg']}")

        grid = Image.new('RGBA', (3 * 512, 3 * 512))
        self.paste(reply["masks"], 1, grid)
        self.paste(reply["masked_images"], 2, grid)

    def paste(self, images, row, grid):
        for idx, img in enumerate(images):
            img_pil = Image.open(BytesIO(base64.b64decode(img))).resize((512, 512))
            grid.paste(img_pil, (idx * 512, row * 512))

            # grid.show()
            # img_pil.save("output/sam_predict.png")

            if idx == 1 and (row == 1 or row == 2):
                file = open(f"./output/mask/sam_mask{row}.png", "wb")
                file.write(base64.b64decode(img))
                file.close()


if __name__ == '__main__':
    predictor = SAMPredictor(host="127.0.0.1", port="7860")
    os.chdir('..')

    image_path = "./input/R.jpg"
    # with open(image_path, mode="rb") as r:
    #     image_base64 = r.read()

    predictor.gen_mask("puppy", image_path)
