import os
import torch
import transformers

from src.gptj_model import GPTJForCausalLM

_device = "cuda"
_path = os.path.dirname(__file__)
_save_path = os.path.join(_path, "../models/gptj_model")
config = transformers.GPTJConfig.from_pretrained("EleutherAI/gpt-j-6B")
tokenizer = transformers.AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")

if __name__ == "__main__":
    gpt = GPTJForCausalLM(config)
    gpt = gpt.from_pretrained("fractalego/samsumbot")
    torch.save(gpt.state_dict(), _save_path)
