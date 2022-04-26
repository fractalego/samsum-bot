import logging
import os
import torch
import transformers

from ts.torch_handler.base_handler import BaseHandler
from src.gptj_model import GPTJForCausalLM, add_adapters

_path = os.path.dirname(__file__)
_logger = logging.getLogger(__file__)


class TransformersClassifierHandler(BaseHandler):
    def __init__(self):
        super(TransformersClassifierHandler, self).__init__()
        self.initialized = False

    def initialize(self, ctx):
        self.manifest = ctx.manifest

        properties = ctx.system_properties
        model_dir = properties.get("model_dir")
        self.device = torch.device("cuda:" + str(properties.get("gpu_id")) if torch.cuda.is_available() else "cpu")

        self.tokenizer = transformers.AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
        config = transformers.GPTJConfig.from_pretrained("EleutherAI/gpt-j-6B")
        gpt = GPTJForCausalLM(config)
        add_adapters(gpt)
        gpt.load_state_dict(torch.load(os.path.join(_path, "../models/gptj-0")))
        self.model = gpt

        self.model.to(self.device)
        self.model.eval()

        _logger.debug('Transformer model from path {0} loaded successfully'.format(model_dir))
        self.initialized = True

    def preprocess(self, data):
        text = data[0].get("data")
        if text is None:
            text = data[0].get("body")
        sentences = text.decode('utf-8')
        _logger.info("Received text: '%s'", sentences)

        prompt = self.tokenizer(text, return_tensors="pt")
        prompt = {key: value.to(self.device) for key, value in prompt.items()}
        return prompt

    def inference(self, prompt):
        out = self.model.generate(
            **prompt,
            max_length=prompt["input_ids"].shape[1] + 5,
            num_beams=2,
            num_return_sequences=1,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        out = out[0][prompt["input_ids"].shape[1] :]
        return [out]

    def postprocess(self, inference_output):
        return inference_output

