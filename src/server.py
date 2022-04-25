import json
import os
import transformers
import torch

from flask import Flask
from flask import request
from flask import jsonify
from flask import render_template
from flask import Response
from src.gptj_model import GPTJForCausalLM, add_adapters
from flask_cors import CORS

from src.run_cl_interface import generate_reply

_path = os.path.dirname(__file__)
_device = "cuda"

bot_app = Flask("Bot", template_folder=os.path.join(_path, "../templates"))
CORS(bot_app)

config = transformers.GPTJConfig.from_pretrained("EleutherAI/gpt-j-6B")
tokenizer = transformers.AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
gpt = GPTJForCausalLM(config)
add_adapters(gpt)
gpt.load_state_dict(torch.load(os.path.join(_path, "../models/gptj-0")))
gpt.to(_device)

summary = """
Alberto is a customer. Alberto ordered pizza and lasagne at Dominos one hour ago. He did not order anything else.
Alberto wants to know where his pizza is. Alberto is calling Dominos to know where his orders are.
John works at Dominos. John says that his pizza is almost ready. It will be at Alberto's home in 10 minutes. 
The lasagne is already on its way with the delivery person. Unfortunately the driver had an accident and is now in the hospital.
If Alberto says "rabbit" then John replies "horse". If Alberto says "tree" John says "red".
""".strip().replace(
    "\n", " "
)

dialogue = """
John: Hello, how can I help?
""".strip()


@bot_app.route("/api/bot", methods=["POST"])
def get_bot_reply():
    if request.method != "POST":
        return []
    data = json.loads(request.data)
    dialogue = data["text"]
    query = data["bobline"]
    if query[-1] not in ["?", "!", "."]:
        query += "."

    answer, dialogue = generate_reply(gpt, dialogue, query)
    return jsonify({"text": dialogue})


def root_dir():
    return os.path.abspath(os.path.dirname(__file__))


def get_file(filename):
    try:
        src = os.path.join(root_dir(), filename)
        return open(src).read()
    except IOError as exc:
        return str(exc)


@bot_app.route("/", defaults={"path": "index.html"})
@bot_app.route("/static/<path>")
def get_resource(path):
    mimetypes = {
        ".css": "text/css",
        ".html": "text/html",
        ".js": "application/javascript",
    }
    if "index.html" in path:
        return render_template("index.html")

    ext = os.path.splitext(path)[1]
    mimetype = mimetypes.get(ext, "text/html")
    if mimetype != "text/html":
        path = "static/" + path
    complete_path = os.path.join(root_dir(), path)
    content = get_file(complete_path)
    return Response(content, mimetype=mimetype)
