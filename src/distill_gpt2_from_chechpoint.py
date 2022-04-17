import os
import random
import json
import joblib
import torch

from tqdm import tqdm
from torch.nn import functional as F
from transformers import GPT2LMHeadModel, AutoTokenizer
from src.utils import batchify, tokenize_data, test
from torch.optim.lr_scheduler import StepLR

_device = "cuda"
_path = os.path.dirname(__file__)
tokenizer = AutoTokenizer.from_pretrained("gpt2")
gpt_small = GPT2LMHeadModel.from_pretrained("gpt2")
checkpoint = torch.load(os.path.join(_path, "../models/save_small" + str(1)))
gpt_small.load_state_dict(checkpoint["model_state_dict"])
gpt_small.cuda()

dev_set = json.load(open(os.path.join(_path, "../data/samsum-val.json")))
train_embeddings = joblib.load(
    os.path.join(_path, "../data/train_embeddings_logits_only.joblib")
)


def get_probability_vector(log_prob_dict, temp):
    _vocab_size = 50257

    logits = torch.tensor(log_prob_dict["logits"])
    indices = torch.tensor(log_prob_dict["indices"])
    vectors = []

    for index_set, logs in zip(indices[0], logits[0]):
        v = (
            torch.sparse_coo_tensor([index_set.tolist()], logs, (_vocab_size,))
            .to_dense()
            .float()
        )
        v[v == 0] = torch.tensor(float("-inf"))
        vectors.append(v)

    vectors = torch.stack(vectors, dim=0)
    return F.softmax(vectors / temp, dim=-1)


if __name__ == "__main__":
    tokenized_test_data = tokenize_data(dev_set, tokenizer, max_length=1024)
    dev_batches = batchify(tokenized_test_data, 1)

    lr = 1e-5
    optimizer = torch.optim.Adam(gpt_small.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=2, gamma=0.9)
    epochs = 20

    steps = 0
    for epoch_num in range(epochs):
        gpt_small.train()
        temp = 2
        random.shuffle(train_embeddings)

        for item in tqdm(train_embeddings):
            input_ids = torch.tensor([item["input_ids"]]).cuda()
            label_p = get_probability_vector(
                item["logits_and_indices"], temp=temp
            ).cuda()
            out_logits = gpt_small.forward(input_ids).logits
            out_p = F.softmax(out_logits / temp, dim=-1)

            loss = -torch.mean(torch.mul(torch.log(out_p).flatten(), label_p.flatten()))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            steps += 1

            if steps % 10000 == 0:
                print("steps", steps)
                print("Dev loss:", test(gpt_small, dev_batches))
                torch.save(
                    gpt_small.state_dict(),
                    os.path.join(
                        _path, f"../models/distilled_gptj-onto-gpt2-small-{steps}"
                    ),
                )

        scheduler.step()
