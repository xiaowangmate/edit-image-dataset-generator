import os
import glob
import json
import random
import tarfile
from predictor.sam_predictor import SAMPredictor
from generator.edit_image_generator import EditImageGenerator

os.chdir('..')
with open("configs/config.json", mode="r", encoding="utf-8") as r:
    config = json.loads(r.read())
    host = config['base_config']['host']
    port = config['base_config']['port']
    input_tar_folder_path = config['io_config']['input_tar_folder_path']
    output_image_folder_path = config['io_config']['output_image_folder_path']
    output_jsonl_folder_path = config['io_config']['output_jsonl_folder_path']

predictor = SAMPredictor(host=host, port=port)
edit = EditImageGenerator(host=host, port=port, output_folder_path=output_image_folder_path)


class StyleConverter:
    def __init__(self, input_image_tar_folder_path):
        self.total_styles = [
            'Cartoon style',
            'Van Gogh style',
            'Impressionism style',
            'Abstract style',
            'Pop art style',
            'Surrealism style',
            'Cubism style',
            'Watercolor style',
            'Pixel Art style',
            'Graffiti style',
            'Noir style',
            'Vintage/Retro style',
            'Abstract style',
            'Futuristic style',
            'Gothic style',
            'Oil Painting style',
            'Charcoal Drawing style',
            'Pastel style',
            'Manga style',
            'Anime style',
            'Tourism illustration',
            'Japanese Ukiyo-e',
            'Hayao Miyazaki style',
            'American anime',
            'Kyoto animation',
            'Blockprint style',
            'Chinese ink and wash style',
            'Fairy tale style',
            'Colorink on paper style',
            'Doodle style',
            'Sketch',
            'Traditional Chinese Ink painting',
            'Montage style',
            'Hyperrealism style',
            'Cyberpunk style',
            'Leonardo da Vinci style',
            'Acrylic painting',
            'Game scenes',
            'Pointillism style',
            'Renaissance style',
            'Makoto Shinkai style',
            'Ghibli Studio style',
            'Pixar style',
        ]
        self.input_image_tar_folder_path = input_image_tar_folder_path

    def process_all_tar(self):
        for folder, sub_folders, files in os.walk(self.input_image_tar_folder_path):
            for tar_path in glob.glob(os.path.join(folder, "*.tar")):
                self.read_tar(tar_path)

    def read_tar(self, tar_path):
        with tarfile.open(tar_path, 'r') as tar:
            members = tar.getmembers()

            for member in members:
                img_f = tar.extractfile(member)

                if ".jpg" in member.name:
                    if not os.path.exists(f"{output_image_folder_path}/{member.name}"):
                        image = img_f.read()
                        image_name = member.name
                        txt_f = tar.extractfile(image_name.replace('.jpg', '.txt'))
                        caption = txt_f.read().decode("utf-8")
                        convert_style = random.choice(self.total_styles)
                        edit_prompt = ",".join([caption, convert_style])
                        image_type = image_name.split(".")[-1]
                        convert_image_name = image_name.replace(f".{image_type}",
                                                                f"_{image_type.replace(' ', '_')}.jpg")
                        source_image_path, target_image_path = edit.gen_edit_image("change_style", edit_prompt,
                                                                                   image, image_name,
                                                                                   convert_image_name)

                        json_line = {"source_image": source_image_path, "target_image": target_image_path,
                                     "style": convert_style}
                        self.append2jsonl(json_line)

    def append2jsonl(self, json_line):
        with open(f"{output_jsonl_folder_path}/convert-style-images.jsonl", mode="a+", encoding="utf-8") as a:
            a.write(f"{json_line}\n")


if __name__ == '__main__':
    sc = StyleConverter(input_tar_folder_path)
    sc.process_all_tar()
