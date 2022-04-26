import os
import torch
import transformers

from src.gptj_model import GPTJForCausalLM, add_adapters

_path = os.path.dirname(__file__)

if __name__ == "__main__":
    config = transformers.GPTJConfig.from_pretrained("EleutherAI/gpt-j-6B")
    gpt = GPTJForCausalLM(config)
    add_adapters(gpt)
    gpt.load_state_dict(torch.load(os.path.join(_path, "../models/gptj-0")))
    gpt.push_to_hub("samsum_bot")
