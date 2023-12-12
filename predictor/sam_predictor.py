import base64
import requests
from PIL import Image
from io import BytesIO


class SAMPredictor:
    def __init__(self, host, port):
        self.sam_url = f"http://{host}:{port}/sam/sam-predict"

    def image2base64(self, image_path):
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode()

    def gen_mask(self, original_image_path, mask_target):
        print("\ngen mask...")
        payload = {
            "input_image": self.image2base64(original_image_path),
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
    image_path = "../input/inpainting.png"
    predictor.gen_mask(image_path, "tree")
