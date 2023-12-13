import os
import glob
import re
import time
import openai
import tarfile
from PIL import Image
from io import BytesIO
from multiprocessing import Pool

openai.api_base = "https://gptproxy.llmpaas.tencent.com/v1"
openai.api_key = "XJj2rwwM8uzYXJ4J86UdPXb42M6kgGNy"
gen_instruction_prompt = "this is an image captions, please generate instructions that edit the image by replacing the main object with a different object. For example, if the caption is 'A refrigerator filled with food in the room', you can generate an instruction like: 'Turn the refrigerator into a bookshelf with books' or 'Make the refrigerator an apple' etc, instruction should be short, no exceed to 15 words."
respond_format_prompt = 'then, according the instruction give me the respond json such as input: Turn the refrigerator into a bookshelf with books. so the output like this format: { "mask_prompt": "refrigerator", "target_prompt": "a bookshelf with books"}'


def get_completions(msg):
    completions = openai.ChatCompletion.create(
        model="gpt-35-turbo",
        messages=msg,
        max_tokens=1024,
        temperature=0.3,
    )
    return completions


def gen_instruction(prompt):
    try:
        print("gen instruction...")
        requests = [{"role": "user", "content": prompt + gen_instruction_prompt + respond_format_prompt}]
        response = get_completions(requests)['choices'][0]['message']['content']
        print(response)
        if not response:
            gen_instruction(prompt)
        else:
            return response
    except Exception as e:
        if "Please retry after" in str(e):
            print("error:", str(e), "wait 5 second.")
            time.sleep(5)
        elif "exceed request rate limit" in str(e):
            print("error:", str(e), "wait 5 second.")
            time.sleep(5)
        elif "This model's maximum context length is 4096 tokens." in str(e):
            print("error:", str(e), "cut some tokens.")
            prompt = prompt[:1024]
        else:
            print("error:", str(e), "change gpt model.")
        gen_instruction(prompt)


def find_all_tar(tar_base_path):
    tar_list = []
    for folder, sub_folders, files in os.walk(tar_base_path):
        for tar_path in glob.glob(os.path.join(folder, "*.tar")):
            tar_list.append(tar_path)
    return tar_list


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
                    response = gen_instruction(caption)
                    instruction,edit_prompt = extract_instruction_and_edit_prompt(response)
                    origin_image_path = os.path.join(tar_path, image_name)
                    gen_jsonl_image_instruction_mapping(origin_image_path, instruction, edit_prompt)


def extract_instruction_and_edit_prompt(respond_content):
    instruction = re.findall("[Input|input|Instruction|instruction]: (.*)", respond_content)[0]
    edit_prompt = re.findall('(\{.*?\})', respond_content, re.S)[0]
    # print(instruction, edit_prompt)
    return instruction, edit_prompt


def pixels_meeting(img_obj):
    pixels_meeting_value = False
    width, height = img_obj.size
    if width >= 400 or height >= 400:
        width_height_ratio = height / width if width > height else width / height
        if width_height_ratio >= 0.6:
            pixels_meeting_value = True
    return pixels_meeting_value


def gen_jsonl_image_instruction_mapping(origin_image_path, instruction, edit_prompt):
    json_line = {"image": origin_image_path, "instruction": instruction, "edit_prompt": edit_prompt}

    with open("./image-instruction-mapping.jsonl", mode="a+", encoding="utf-8") as a:
        a.write(f"{json_line}\n")


if __name__ == '__main__':
    tar_base_path = r"C:\unsplash"
    tar_list = find_all_tar(tar_base_path)

    processes_num = 1

    with Pool(processes=processes_num) as pool:
        for tar_path in pool.imap_unordered(read_tar, tar_list):
            print(tar_path)