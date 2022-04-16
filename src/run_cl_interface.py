import os

import torch
import transformers

from src.gptj_model import GPTJForCausalLM, add_adapters
from src.utils import create_text_from_summary_and_dialogue

_device = "cuda"
_path = os.path.dirname(__file__)
config = transformers.GPTJConfig.from_pretrained("EleutherAI/gpt-j-6B")
tokenizer = transformers.AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")

#gpt = GPTJForCausalLM.from_pretrained("hivemind/gpt-j-6B-8bit", low_cpu_mem_usage=True)

gpt = GPTJForCausalLM(config)
add_adapters(gpt)
gpt.load_state_dict(torch.load(os.path.join(_path, "../models/gptj-1")))
gpt.to(_device)

summary = """
Alberto is a customer. Alberto ordered pizza and lasagne at Dominos one hour ago. He did not order anything else.
Alberto wants to know where his pizza is. Alberto is calling Dominos to know where his orders are.
John works at Dominos. John says that his pizza is almost ready. It will be at Alberto's home in 10 minutes. 
The lasagne is already on its way with the delivery person. Unfortunately the driver had an accident and is now in the hospital.
If Alberto says "rabbit" then John replies "horse". If Alberto says "tree" John says "red".
""".strip().replace(
    "\n", " "
)

dialogue = """
John: Hello, how can I help?
""".strip()


if __name__ == "__main__":
    print(dialogue)

    while True:
        user_input = input()
        if user_input[-1] not in ['?', '!', '.']:
            user_input += '.'

        dialogue += "\nAlberto: " + user_input + "\nJohn: "
        text = create_text_from_summary_and_dialogue(summary, dialogue)
        prompt = tokenizer(text, return_tensors="pt")
        prompt = {key: value.to(_device) for key, value in prompt.items()}
        out = gpt.generate(
            **prompt, max_length=prompt["input_ids"].shape[1] + 20, do_sample=False

        )
        out = out[0][prompt["input_ids"].shape[1]:]
        answer = tokenizer.decode(out)
        answer = answer[: answer.find("\n")].strip()
        print(answer)
        dialogue += answer
