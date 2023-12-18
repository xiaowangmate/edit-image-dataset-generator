import glob
import os
import json
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

            if "\\" in input_image_path:
                input_image_name = input_image_path.split("\\")[-1]
            else:
                input_image_name = input_image_path.split("/")[-1]
            predictor.gen_mask(mask_object, image_file, 13)
            source_image_path, target_image_path = edit.gen_edit_image(task_type, edit_content,
                                                                       image_file, input_image_name)
            print(f"task completed-\nsource_image_path: {source_image_path}\ntarget_image_path: {target_image_path}")
        else:
            print("image pixels not meeting.")


def batch_test_img(test_mapping_path):
    with open(test_mapping_path, mode="r", encoding="utf-8") as mp:
        json_lines = mp.readlines()
        for index, json_line in enumerate(json_lines):
            json_line = json.loads(json_line)

            input_image_path = json_line['source_image']

            if index + 1 in [8]:
                task_type = json_line['task_type']
                mask_object = json_line['mask_object']
                edit_content = json_line['edit_content']

                test(task_type, mask_object, edit_content, input_image_path)


if __name__ == '__main__':
    batch_test_img("output/jsonl/test_mapping.jsonl")


    # input_image_path = "input/2bfe8b36806940d2863841fddf7db67a.jpg"
    #
    # task_type = "replace_object"
    # mask_object = "gun"  # if it is change_style, you can fill in the blanks
    # edit_content = "water bottle"
    #
    # test(task_type, mask_object, edit_content, input_image_path)
