import os

import torch
import transformers

from src.gptj_model import GPTJForCausalLM

_path = os.path.dirname(__file__)
_device = "cpu"
_save_path_onnx = os.path.join(_path, "../samsumbot_onnx/gptj_model.onnx")
config = transformers.GPTJConfig.from_pretrained("EleutherAI/gpt-j-6B")
tokenizer = transformers.AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")


if __name__ == "__main__":
    gpt = GPTJForCausalLM(config)
    gpt.load_state_dict(
        torch.load(os.path.join(_path, f"../models/gptj-0"), map_location=_device)
    )
    x = tokenizer.encode(
        "This is a test", return_tensors="pt", truncation=True, max_length=110
    )
    gpt.to(_device)
    x.to(_device)
    torch.onnx.export(
        gpt,  # model being run
        x,  # model input (or a tuple for multiple inputs)
        _save_path_onnx,  # where to save the model (can be a file or file-like object)
        export_params=True,  # store the trained parameter weights inside the model file
        opset_version=15,  # the ONNX version to export the model to
        do_constant_folding=False,  # whether to execute constant folding for optimization
        input_names=["input"],  # the model's input names
        output_names=["output"],  # the model's output names
        dynamic_axes={
            "input": {0: "batch_size"},  # variable length axes
            "output": {0: "batch_size"},
        },
    )
