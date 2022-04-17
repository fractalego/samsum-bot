import json
import os
import torch
import transformers

from src.gptj_model import GPTJForCausalLM, add_adapters
from src.utils import tokenize_data, batchify, test

_device = "cuda"
_path = os.path.dirname(__file__)
config = transformers.GPTJConfig.from_pretrained("EleutherAI/gpt-j-6B")
tokenizer = transformers.AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")

test_set = json.load(open(os.path.join(_path, "../data/samsum-val.json")))

num_epochs = 1

if __name__ == "__main__":
    tokenized_test_data = tokenize_data(test_set, tokenizer, max_length=90)
    test_batches = batchify(tokenized_test_data, 1)

    for epoch in range(num_epochs):
        gpt = GPTJForCausalLM(config)
        add_adapters(gpt)
        gpt.load_state_dict(torch.load(os.path.join(_path, f"../models/gptj-{epoch}")))
        gpt.to(_device)
        print(f"Epoch {epoch}", test(gpt, test_batches))
        del gpt
