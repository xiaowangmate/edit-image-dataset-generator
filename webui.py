import json
import base64
import gradio as gr
from predictor.sam_predictor import SAMPredictor
from generator.edit_image_generator import EditImageGenerator

with open("configs/config.json", mode="r", encoding="utf-8") as r:
    config = json.loads(r.read())
    host = config['base_config']['host']
    port = config['base_config']['port']
    output_image_folder_path = config['io_config']['output_image_folder_path']

predictor = SAMPredictor(host=host, port=port)
edit = EditImageGenerator(host=host, port=port, output_folder_path=output_image_folder_path)


def image2base64(image_path):
    with open(image_path, mode="rb") as r:
        image_base64 = base64.b64encode(r.read()).decode("utf-8")
        return image_base64


def mask_preview(image, mask_prompt, expand):
    predictor.gen_mask(mask_prompt, image, expand)
    return ["output/mask/sam_mask.png"]


def gen_edit_image(image, task, edit_prompt):
    if "\\" in image:
        input_image_name = image.split("\\")[-1]
    else:
        input_image_name = image.split("/")[-1]
    source_image_path, target_image_path = edit.gen_edit_image(task.replace(" ", "_"), edit_prompt,
                                                               image, input_image_name)
    return [target_image_path]


def oneclick_generation(image, mask_prompt, expand, task, edit_prompt):
    predictor.gen_mask(mask_prompt, image, expand)
    if "\\" in image:
        input_image_name = image.split("\\")[-1]
    else:
        input_image_name = image.split("/")[-1]
    source_image_path, target_image_path = edit.gen_edit_image(task, edit_prompt,
                                                               image, input_image_name)
    return [target_image_path, "output/mask/sam_mask.png"]


def start_ui():
    with gr.Blocks(title="编辑图像生成器") as edit_image_dataset_generator:
        gr.HTML("<h1 style='text-align:center;'>编辑图像生成器</h1>")
        with gr.Row():
            with gr.Column():
                image_input = gr.Image(
                    label="source image",
                    type="filepath"
                )
                with gr.Row():
                    mask_object = gr.Text(label="mask object")
                    expand_value = gr.Slider(
                        label="expand value",
                        value=0
                    )
                gen_mask_button = gr.Button(value="mask preview")
                with gr.Row():
                    task_type = gr.Dropdown(
                        label="task type",
                        choices=[
                            "replace object",
                            "replace background",
                            "change style"
                        ],
                        value="replace object"
                    )
                    edit_content = gr.Text(label="edit content")
                    gen_edit_image_button = gr.Button(value="edit image generation")
            with gr.Row():
                edit_image_output = gr.Gallery(
                    label="edit image",
                    allow_preview=True,
                    preview=True,
                    selected_index=0
                )

        one_click_generate_button = gr.Button(
            value="one-click generation",
            variant="primary"
        )

        gen_mask_button.click(
            fn=mask_preview,
            inputs=[
                image_input,
                mask_object,
                expand_value
            ],
            outputs=[
                edit_image_output
            ]
        )

        gen_edit_image_button.click(
            fn=gen_edit_image,
            inputs=[
                image_input,
                task_type,
                edit_content
            ],
            outputs=[
                edit_image_output
            ]
        )

        one_click_generate_button.click(
            fn=oneclick_generation,
            inputs=[
                image_input,
                mask_object,
                expand_value,
                task_type,
                edit_content
            ],
            outputs=[
                edit_image_output
            ]
        )

    edit_image_dataset_generator.launch(share=True)


if __name__ == '__main__':
    start_ui()
