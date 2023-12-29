import glob
import os
import json
from PIL import Image
from io import BytesIO
from diffusers.utils import load_image
from predictor.sam_predictor import SAMPredictor
from generator.image_object_replacer import ImageObjectReplacer

os.chdir('..')
with open("configs/config.json", mode="r", encoding="utf-8") as r:
    config = json.loads(r.read())
    host = config['base_config']['host']
    port = config['base_config']['port']
    output_image_folder_path = config['io_config']['output_image_folder_path']

predictor = SAMPredictor(host=host, port=port)
replacer = ImageObjectReplacer(output_image_folder_path)


def auto_batch_test():
    with open("output/jsonl/test_mapping.jsonl", mode="r", encoding="utf-8") as mp:
        json_lines = mp.readlines()
        for index, json_line in enumerate(json_lines):
            json_line = json.loads(json_line)

            input_image_path = json_line['source_image']

            if index + 1 in [3, 6, 7]:
                mask_object = json_line['mask_object']
                edit_content = json_line['edit_content']

                test(input_image_path, mask_object, edit_content)


def test(source_image, replace_object, target_object):
    predictor.gen_mask(replace_object, source_image, 50)
    replacer.replace_object(Image.open(source_image), Image.open("output/mask/sam_mask.png"), target_object)


if __name__ == '__main__':
    auto_batch_test()
