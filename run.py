import json
from discriminator.gpt_discriminator import GPTDiscriminator
from predictor.sam_predictor import SAMPredictor
from generator.edit_image_generator import EditImageGenerator

if __name__ == '__main__':
    with open("configs/config.json", mode="r", encoding="utf-8") as r:
        config = json.loads(r.read())
        host = config['base_config']['host']
        port = config['base_config']['port']
        gpt_key = config['api_key']['gpt']

    instruction = "Turn the mountain into a dragon"
    image_path = "./input/inpainting.png"

    gpt = GPTDiscriminator(gpt_key)
    predict_json = gpt.gen_predict_json(instruction)

    predictor = SAMPredictor(host=host, port=port)
    predictor.gen_mask(image_path, predict_json['mask_prompt'])

    edit = EditImageGenerator(host=host, port=port)
    edit.gen_edit_image(predict_json['target_prompt'], image_path)

