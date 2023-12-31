import re
import time
import json
import openai


class GPTDiscriminator:
    def __init__(self, api_key):
        openai.api_base = "https://gptproxy.llmpaas.tencent.com/v1"
        openai.api_key = api_key
        # self.gpt_model = "gpt-4"
        self.gpt_model = "gpt-35-turbo"
        self.gen_instruction_prompt = "this is an image captions, please generate instructions that edit the image by replacing the main object with a different object. For example, if the caption is 'A refrigerator filled with food in the room', you can generate an instruction like: 'Turn the refrigerator into a bookshelf with books' or 'Make the refrigerator an apple' etc."
        self.respond_format_prompt = 'please give me the respond json such as input: Turn the refrigerator into a bookshelf with books. so the output like this format: { "mask_prompt": "refrigerator", "target_prompt": "a bookshelf with books"}'

    def get_completions(self, msg):
        completions = openai.ChatCompletion.create(
            model=self.gpt_model,
            messages=msg,
            max_tokens=1024,
            temperature=0.3,
        )
        return completions

    def gen_instruction(self, prompt):
        try:
            print("gen instruction...")
            requests = [{"role": "user", "content": prompt + self.gen_instruction_prompt}]
            response = self.get_completions(requests)['choices'][0]['message']['content']
            print(response)
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
                self.gpt_model = "gpt-35-turbo"
            self.gen_instruction(prompt)

    def gen_predict_json(self, prompt):
        try:
            print("gen prompt json...")
            requests = [{"role": "user", "content": prompt + self.respond_format_prompt}]
            response = self.get_completions(requests)['choices'][0]['message']['content']
            try:
                print(response)
                response = re.findall('(\{.*?\})', response, re.S)[0]
                response = json.loads(response)
                return response
            except Exception as e0:
                print("error:", str(e0), "respond format error, try again")
                self.gen_predict_json(prompt)
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
                self.gpt_model = "gpt-35-turbo"
            self.gen_predict_json(prompt)


if __name__ == '__main__':
    with open("../configs/config.json", mode="r", encoding="utf-8") as f:
        config = json.loads(f.read())
    gpt = GPTDiscriminator(config["api_key"]["gpt"])

    instruction = "change the mountain to a big chicken."
    print(gpt.gen_predict_json(instruction))

    # caption = "steps leading up to a stone wall with graffiti on it"
    # gpt.gen_instruction(caption)
