import json
import os
import torch
import transformers

from src.gptj_model import GPTJForCausalLM
from src.utils import (
    tokenize_data,
    batchify,
    test,
    get_n_params,
    make_only_attentions_as_trainable,
)
from tqdm import tqdm
from torch.nn import functional as F
from bitsandbytes.optim import Adam8bit

_device = "cuda"
_path = os.path.dirname(__file__)
config = transformers.GPTJConfig.from_pretrained("EleutherAI/gpt-j-6B")
tokenizer = transformers.AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
gpt = GPTJForCausalLM.from_pretrained("hivemind/gpt-j-6B-8bit", low_cpu_mem_usage=True)

train_set = json.load(open(os.path.join(_path, "../data/train.json")))
test_set = json.load(open(os.path.join(_path, "../data/samsum-val.json")))

num_epochs = 1
lr = 1e-6

if __name__ == "__main__":
    tokenized_train_data = tokenize_data(train_set, tokenizer)
    train_batches = batchify(tokenized_train_data, 1)
    tokenized_test_data = tokenize_data(test_set, tokenizer, max_length=90)
    test_batches = batchify(tokenized_test_data, 1)

    optimizer = Adam8bit(gpt.parameters(), lr=lr)

    gpt.train()
    gpt.to(_device)

    gpt = make_only_attentions_as_trainable(gpt)
    print("Trainable parameters:", get_n_params(gpt))

    with torch.cuda.amp.autocast():

        for epoch in range(num_epochs):
            print("Epoch", epoch)

            for batch in tqdm(train_batches):
                out = gpt.forward(batch.cuda())

                loss = F.cross_entropy(
                    out.logits[:, :-1, :].flatten(0, -2),
                    batch[:, 1:].flatten().cuda(),
                    reduction="mean",
                )

                loss.backward()

                optimizer.step()
                optimizer.zero_grad()
                del batch

            torch.save(gpt.state_dict(), os.path.join(_path, f"../models/gptj-{epoch}"))
            print("Val loss:", test(gpt, test_batches))
