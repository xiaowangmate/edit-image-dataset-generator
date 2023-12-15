import os
import json
import base64
from PIL import Image
from io import BytesIO
from discriminator.gpt_discriminator import GPTDiscriminator
from predictor.sam_predictor import SAMPredictor
from generator.edit_image_generator import EditImageGenerator

os.chdir('..')
with open("configs/config.json", mode="r", encoding="utf-8") as r:
    config = json.loads(r.read())
    host = config['base_config']['host']
    port = config['base_config']['port']
    output_image_folder_path = config['io_config']['output_image_folder_path']
    gpt_key = config['api_key']['gpt']

gpt = GPTDiscriminator(gpt_key)
predictor = SAMPredictor(host=host, port=port)
edit = EditImageGenerator(host=host, port=port, output_folder_path=output_image_folder_path)


def pixels_meeting(img_obj):
    pixels_meeting_value = False
    width, height = img_obj.size
    if width >= 400 or height >= 400:
        width_height_ratio = height / width if width > height else width / height
        if width_height_ratio >= 0.6:
            pixels_meeting_value = True
    return pixels_meeting_value


def test(task_type, mask_object, edit_content, input_image_path):
    with open(input_image_path, mode="rb") as r:
        image_file = r.read()
        img = Image.open(BytesIO(image_file))

        if pixels_meeting(img):
            input_image_name = input_image_path.split("/")[-1]
            predictor.gen_mask(mask_object, image_file)
            source_image_path, target_image_path = edit.gen_edit_image(task_type, edit_content,
                                                                       image_file, input_image_name)
            print(f"task completed-\nsource_image_path: {source_image_path}\ntarget_image_path: {target_image_path}")
        else:
            print("image pixels not meeting.")


if __name__ == '__main__':
    input_image_path = "input/R.jpg"

    task_type = "replace_object"
    mask_object = "puppy"  # if it is change_style, you can fill in the blanks
    edit_content = "cat"

    test(task_type, mask_object, edit_content, input_image_path)
