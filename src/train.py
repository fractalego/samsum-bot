import json
import os
import torch
import transformers

from src.gptj_model import GPTJForCausalLM
from src.utils import make_only_adapters_as_trainable, tokenize_data, batchify
from tqdm import tqdm
from torch.nn import functional as F

from bitsandbytes.optim import Adam8bit

_device = "cuda"
_path = os.path.dirname(__file__)
config = transformers.GPTJConfig.from_pretrained("EleutherAI/gpt-j-6B")
tokenizer = transformers.AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
gpt = GPTJForCausalLM.from_pretrained("hivemind/gpt-j-6B-8bit", low_cpu_mem_usage=True)
gpt.to(_device)

train_set = json.load(open(os.path.join(_path, "../data/train.json")))

num_epochs = 2

if __name__ == "__main__":
    make_only_adapters_as_trainable(gpt)
    tokenized_train_data = tokenize_data(train_set, tokenizer)
    train_batches = batchify(tokenized_train_data, 1)

    gpt.gradient_checkpointing_enable()
    optimizer = Adam8bit(gpt.parameters(), lr=1e-6)

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

    gpt.save_pretrained(os.path.join(_path, "../models/gptj"))
