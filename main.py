import torch
import os
import numpy as np

from transformers import BertModel, BertConfig, BertTokenizer

from flask import Flask, request, jsonify


flask_app = Flask(__name__)


# model_dir = 'C:/Users/ddale/Downloads/NLP/rubert_deeppavlov/rubert_cased_L-12_H-768_A-12_v2'
model_dir = 'models/rubert_cased_L-12_H-768_A-12_v2'

bert_tokenizer = BertTokenizer.from_pretrained(os.path.join(model_dir, 'vocab.txt'))
inv_voc = {idx: tok for tok, idx in bert_tokenizer.vocab.items()}

bert_config = BertConfig.from_pretrained(os.path.join(model_dir, 'bert_config.json'))
bert_model = BertModel.from_pretrained(os.path.join(model_dir, 'bert_model.bin'), config=bert_config)


def normalize(v):
    return v / sum(v**2)**0.5


def bert_emb_cls(word):
    with torch.no_grad():
        raw, pooled = bert_model(torch.tensor([bert_tokenizer.encode(word)]))
    return normalize(pooled[0].numpy())


def bert_emb_pool(word):
    with torch.no_grad():
        raw, pooled = bert_model(torch.tensor([bert_tokenizer.encode(word)]))
    return normalize(raw[0].numpy().mean(axis=0))


def bert_cos(w1, w2, cls=False):
    if cls:
        bert_emb = bert_emb_cls
    else:
        bert_emb = bert_emb_pool
    e1 = bert_emb(w1)
    e2 = bert_emb(w2)
    return np.dot(e1, e2) / np.sqrt(np.dot(e1, e1) * np.dot(e2, e2))


@flask_app.route('/', methods=['POST', 'GET'])
def main_api(text='привет'):
    if request.json:
        text = request.json['text']
    return jsonify({'text': text, 'result': bert_emb_pool(text).tolist()})


@flask_app.route('/<word>', methods=['GET'])
def main_api_word(word='привет'):
    return main_api(text=word)


if __name__ == '__main__':
    host = "0.0.0.0"
    port = int(os.environ.get('PORT', 5000))
    flask_app.run(host=host, port=port)
