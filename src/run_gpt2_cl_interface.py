import os

import torch
import transformers

from transformers import GPT2LMHeadModel
from src.utils import create_text_from_summary_and_dialogue

_device = "cuda"
_path = os.path.dirname(__file__)
config = transformers.GPTJConfig.from_pretrained("EleutherAI/gpt-j-6B")
tokenizer = transformers.AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")

gpt = GPT2LMHeadModel.from_pretrained("gpt2")
checkpoint = torch.load(os.path.join(_path, "../models/save_small" + str(1)))
gpt.load_state_dict(checkpoint["model_state_dict"])
gpt.to(_device)

bad_words_list = [tokenizer.encode(x, add_special_tokens=False) for x in ["<file_other>"]]

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


def get_answer_for_speaker(speaker, answer):
    for line in answer.split("\n"):
        if line.lower().find(speaker.lower() + ":") != 0:
            continue

        return line

    return "I don't understand."

if __name__ == "__main__":
    print(dialogue)

    while True:
        user_input = input("Alberto: ")
        if user_input[-1] not in ["?", "!", "."]:
            user_input += "."

        dialogue += "\nAlberto: " + user_input + "\n"
        text = create_text_from_summary_and_dialogue(summary, dialogue)
        prompt = tokenizer(text, return_tensors="pt")
        prompt = {key: value.to(_device) for key, value in prompt.items()}
        out = gpt.generate(
            **prompt,
            max_length=prompt["input_ids"].shape[1] + 20,
            num_beams=50,
            top_k=100,
            num_return_sequences=50,
            remove_invalid_values=True,
            bad_words_ids=bad_words_list,
        )
        print(f"{tokenizer.batch_decode(out)}")
        for item in out:
            tokens = item[prompt["input_ids"].shape[1] :]
            answer = tokenizer.decode(tokens)
            if "John: " not in answer:
                continue

            if "<file_" in answer:
                continue

            answer = get_answer_for_speaker("John", answer)
            print(answer)
            dialogue += answer
            break
