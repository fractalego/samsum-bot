import os
import json
import joblib
import transformers
import torch

from tqdm import tqdm
from src.gptj_model import GPTJForCausalLM, add_adapters
from src.utils import create_text_from_summary_and_dialogue

_device = "cuda"
_path = os.path.dirname(__file__)
config = transformers.GPTJConfig.from_pretrained("EleutherAI/gpt-j-6B")
tokenizer = transformers.AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
gpt = GPTJForCausalLM(config)
add_adapters(gpt)
gpt.load_state_dict(torch.load(os.path.join(_path, "../models/gptj-0")))
gpt.to(_device)

train_set = json.load(open(os.path.join(_path, "../data/train.json")))


def select_best_probs(logits, num_best=400):
    logits, indices = torch.sort(logits, descending=True)
    logits = logits[:, :, :num_best]
    indices = indices[:, :, :num_best]

    return {"logits": logits.tolist(), "indices": indices.tolist()}


if __name__ == "__main__":
    _limit = 110
    train_embeddings = []
    for item in tqdm(train_set):
        text = create_text_from_summary_and_dialogue(item["summary"], item["dialogue"])
        tokens = tokenizer.encode(
            text, return_tensors="pt", truncation=True, max_length=_limit
        )
        output = gpt(tokens.cuda())

        embeddings_dict = {}
        embeddings_dict["logits_and_indices"] = select_best_probs(
            output.logits.cpu().detach()
        )
        embeddings_dict["input_ids"] = tokens.cpu().detach().tolist()
        train_embeddings.append(embeddings_dict)

        del tokens
        del output

    joblib.dump(
        train_embeddings,
        os.path.join(_path, "../data/train_embeddings_logits_only.joblib"),
    )
