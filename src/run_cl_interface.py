import os

import time
import transformers

from src.gptj_model import GPTJForCausalLM
from src.utils import create_text_from_summary_and_dialogue

_device = "cuda"
_path = os.path.dirname(__file__)
config = transformers.GPTJConfig.from_pretrained("EleutherAI/gpt-j-6B")
tokenizer = transformers.AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")

gpt = GPTJForCausalLM.from_pretrained("fractalego/samsumbot")
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


def generate_reply(model, dialogue, query):
    dialogue += "\nAlberto: " + query + "\nJohn: "
    answer = ""

    terminal_characters = [".", "!", "?", "\n"]

    while all(item not in answer for item in terminal_characters):
        text = create_text_from_summary_and_dialogue(summary, dialogue + answer)
        prompt = tokenizer(text, return_tensors="pt")
        prompt = {key: value.to(_device) for key, value in prompt.items()}
        out = model.generate(
            **prompt,
            max_length=prompt["input_ids"].shape[1] + 5,
            num_beams=2,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id,
        )
        out = out[0][prompt["input_ids"].shape[1] :]
        answer += tokenizer.decode(out)

    end = min(
        [answer.find(item) for item in terminal_characters if answer.find(item) > 0]
    )
    answer = answer[:end].strip()

    return answer, dialogue + answer


if __name__ == "__main__":
    print(dialogue)

    while True:
        user_input = ""
        while not user_input:
            user_input = input("Alberto: ")

        if user_input[-1] not in ["?", "!", "."]:
            user_input += "."

        start = time.time()
        answer, dialogue = generate_reply(gpt, dialogue, user_input)
        end = time.time()
        print(f"John: {answer} [{end - start}s]")
