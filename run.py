import os
import glob
import json
import tarfile
from PIL import Image
from io import BytesIO
from multiprocessing import Pool
from discriminator.gpt_discriminator import GPTDiscriminator
from predictor.sam_predictor import SAMPredictor
from generator.edit_image_generator import EditImageGenerator

with open("configs/config.json", mode="r", encoding="utf-8") as r:
    config = json.loads(r.read())
    host = config['base_config']['host']
    port = config['base_config']['port']
    gpt_key = config['api_key']['gpt']

gpt = GPTDiscriminator(gpt_key)
predictor = SAMPredictor(host=host, port=port)
edit = EditImageGenerator(host=host, port=port)


def find_all_tar(tar_base_path):
    for folder, sub_folders, files in os.walk(tar_base_path):
        for tar_path in glob.glob(os.path.join(folder, "*.tar")):
            read_tar(tar_path)
            break


def read_tar(tar_path):
    with tarfile.open(tar_path, 'r') as tar:
        members = tar.getmembers()

        for member in members:
            img_f = tar.extractfile(member)

            if ".jpg" in member.name:
                image = img_f.read()
                img_obj = Image.open(BytesIO(image))
                if pixels_meeting(img_obj):
                    image_name = member.name
                    txt_f = tar.extractfile(image_name.replace('.jpg', '.txt'))
                    caption = txt_f.read().decode("utf-8")
                    # print(caption)
                    instruction = gpt.gen_instruction(caption)
                    gen_edit_image(instruction, image, image_name)


def pixels_meeting(img_obj):
    pixels_meeting_value = False
    width, height = img_obj.size
    if width >= 400 or height >= 400:
        width_height_ratio = height / width if width > height else width / height
        if width_height_ratio >= 0.6:
            pixels_meeting_value = True
    return pixels_meeting_value


def gen_edit_image(instruction, image, image_name):
    predict_json = gpt.gen_predict_json(instruction)
    predictor.gen_mask(predict_json['mask_prompt'], image)
    edit.gen_edit_image(predict_json['target_prompt'], image, image_name)


def append2jsonl(json_line):
    with open("output/jsonl/edit-image-dataset.jsonl", mode="a+", encoding="utf-8") as a:
        a.write(f"{json_line}\n")


if __name__ == '__main__':
    # image_path = "./input/inpainting.png"
    # instruction = "Turn the mountain into a dragon"
    # gen_edit_image(image_path, instruction)
    find_all_tar(r"C:\unsplash")
