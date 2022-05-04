import json
import requests
import transformers

from src.utils import create_text_from_summary_and_dialogue

_server_url = f"https://127.0.0.1:8080/predictions/bot"

_tokenizer = transformers.AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")

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

dialogue += "\nAlberto: "


def predict_answer(question: str):
    payload = {"data": question}
    r = requests.post(_server_url, json=payload, verify=False)
    answer = json.loads(r.content.decode("utf-8"))
    print(_tokenizer.decode(answer))


if __name__ == "__main__":
    predict_answer(create_text_from_summary_and_dialogue(summary, dialogue))
