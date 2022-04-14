import torch
from torch.nn import functional as F

from tqdm import tqdm


def get_n_params(model):
    pp = 0
    for p in list(model.parameters()):
        if not p.requires_grad:
            continue

        nn = 1
        for s in list(p.size()):
            nn = nn * s
        pp += nn
    return pp


def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def batchify(data, n):
    len_dict = {}
    for item in data:
        length = item.shape[1]
        try:
            len_dict[length].append(item)
        except:
            len_dict[length] = [item]

    batch_chunks = []
    for k in len_dict.keys():
        vectors = len_dict[k]
        batch_chunks += chunks(vectors, n)

    batches = []
    for chunk in batch_chunks:
        inputs = torch.stack([item[0] for item in chunk])
        batches.append((inputs))

    return batches


def create_text_from_summary_and_dialogue(summary, dialogue):
    text = f"""
A partial summary of the conversation is:
{summary}

With the dialogue being:
{dialogue}
    """.strip()

    return text.replace("\r\n", "\n")


def tokenize_data(data, tokenizer, max_length=110):
    _limit = 1024
    tokenized_data = []
    total_skipped = 0
    for item in data:
        text = create_text_from_summary_and_dialogue(item["summary"], item["dialogue"])
        tokens = tokenizer.encode(
            text, return_tensors="pt", truncation=True, max_length=max_length
        )
        if tokens.shape[1] > _limit:
            tokens = tokens[:, :_limit]
        tokenized_data.append(tokens)

    print(f"Skipped {total_skipped} out of {len(data)}")
    return tokenized_data


def test(test_model, batches):
    test_model.eval()
    total_loss = 0.0
    index = 0
    for batch in tqdm(batches):
        try:
            out = test_model.forward(batch.cuda())
            index += 1

        except RuntimeError:
            break

        loss = F.cross_entropy(
            out.logits[:, :-1, :].flatten(0, -2),
            batch[:, 1:].flatten().cuda(),
            reduction="mean",
        )
        total_loss += float(loss)
        del batch

    return total_loss / index
