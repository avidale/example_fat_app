import os
import gensim

from flask import Flask, request, jsonify

app = Flask(__name__)


path_to_model = 'C:/Users/ddale/Downloads/NLP/rusvectores/model.model'
print('loading ft')
ft = gensim.models.fasttext.FastTextKeyedVectors.load(path_to_model)
print('adjusting ft')
ft.adjust_vectors()


@app.route('/', methods=['POST', 'GET'])
def main_api():
    if not request.json:
        text = 'привет'
    else:
        text = request.json['text']
    return jsonify({'text': text, 'result': ft[text].tolist()})


if __name__ == '__main__':
    host = "0.0.0.0"
    port = int(os.environ.get('PORT', 5000))
    app.run(host=host, port=port)
