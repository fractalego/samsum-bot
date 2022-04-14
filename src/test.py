import json
import os
import transformers

from src.gptj_model import GPTJForCausalLM
from src.utils import tokenize_data, batchify, test

_device = "cuda"
_path = os.path.dirname(__file__)
config = transformers.GPTJConfig.from_pretrained("EleutherAI/gpt-j-6B")
tokenizer = transformers.AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
gpt = GPTJForCausalLM.from_pretrained(
    os.path.join(_path, "../models/gptj"), low_cpu_mem_usage=True
)
gpt.to(_device)

test_set = json.load(open(os.path.join(_path, "../data/samsum-val.json")))

if __name__ == "__main__":
    tokenized_test_data = tokenize_data(test_set, tokenizer, max_length=100)
    test_batches = batchify(tokenized_test_data, 1)
    print(test(gpt, test_batches))
