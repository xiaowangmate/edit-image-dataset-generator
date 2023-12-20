import os
import glob
import json
import random
import tarfile
from PIL import Image
from io import BytesIO
from discriminator.gpt_discriminator import GPTDiscriminator
from predictor.sam_predictor import SAMPredictor
from generator.edit_image_generator import EditImageGenerator

with open("configs/config.json", mode="r", encoding="utf-8") as r:
    config = json.loads(r.read())
    host = config['base_config']['host']
    port = config['base_config']['port']
    input_tar_folder_path = config['io_config']['input_tar_folder_path']
    output_image_folder_path = config['io_config']['output_image_folder_path']
    output_jsonl_folder_path = config['io_config']['output_jsonl_folder_path']
    gpt_key = config['api_key']['gpt']

gpt = GPTDiscriminator(gpt_key)
predictor = SAMPredictor(host=host, port=port)
edit = EditImageGenerator(host=host, port=port, output_folder_path=output_image_folder_path)


def process_all_tar(tar_base_path):
    for folder, sub_folders, files in os.walk(tar_base_path):
        for tar_path in glob.glob(os.path.join(folder, "*.tar")):
            read_tar(tar_path)
            break


def read_tar(tar_path):
    with tarfile.open(tar_path, 'r') as tar:
        members = tar.getmembers()

        change_style_instructions = read_local_instructions()
        for member in members:
            try:
                img_f = tar.extractfile(member)

                if ".jpg" in member.name:
                    if not os.path.exists(f"{output_image_folder_path}/{member.name}"):
                        image = img_f.read()
                        img_obj = Image.open(BytesIO(image))
                        if pixels_meeting(img_obj):
                            image_name = member.name
                            txt_f = tar.extractfile(image_name.replace('.jpg', '.txt'))
                            caption = txt_f.read().decode("utf-8")
                            # print(caption)
                            task_type = random.choice(['replace_object', 'change_style'])
                            if task_type == 'replace_object':
                                instruction = gpt.gen_instruction(caption)
                                predict_json = gpt.gen_predict_json(instruction)
                                predictor.gen_mask(predict_json['mask_prompt'], image)
                                source_image_path, target_image_path = edit.gen_edit_image(task_type, predict_json[
                                    'target_prompt'], image, image_name)
                            else:
                                instruction = random.choice(change_style_instructions)
                                source_image_path, target_image_path = edit.gen_edit_image(task_type, instruction,
                                                                                           image, image_name)

                            json_line = {"source_image": source_image_path, "target_image": target_image_path,
                                         "instruction": instruction}
                            append2jsonl(json_line)
            except Exception as e:
                print(f"error: {member.name}, {str(e)}")


def pixels_meeting(img_obj):
    pixels_meeting_value = False
    width, height = img_obj.size
    if width >= 400 or height >= 400:
        width_height_ratio = height / width if width > height else width / height
        if width_height_ratio >= 0.6:
            pixels_meeting_value = True
    return pixels_meeting_value


def read_local_instructions():
    with open("discriminator/instructions/change_style.txt", mode="r", encoding="utf-8") as r:
        change_style_instructions = r.read().split("\n")
        return change_style_instructions


def append2jsonl(json_line):
    with open(f"{output_jsonl_folder_path}/edit-image-dataset.jsonl", mode="a+", encoding="utf-8") as a:
        a.write(f"{json_line}\n")


if __name__ == '__main__':
    # image_path = "./input/inpainting.png"
    # instruction = "Turn the mountain into a dragon"
    # gen_edit_image(image_path, instruction)
    process_all_tar(input_tar_folder_path)
