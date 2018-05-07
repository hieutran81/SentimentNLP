#!/usr/bin/python3
from flask import Flask, request, jsonify
from flask_restful import Resource, Api
from SentimentClassify import predict_sentiment, predict_cnn
from sqlalchemy import create_engine
from json import dumps

#db_connect = create_engine('sqlite:///chinook.db')
app = Flask(__name__)
@app.route('/')
def index():
    return "hello, hieu"

@app.route('/sentiment', methods = ["GET", "POST"])
def sentiment():
    #print("hieuvodoi")
    #print(request.method)
    if (request.method == "POST"):
        text = request.form['text']
        print("cnn")
        print(text)
        label, prob = predict_cnn(text)
        proba = []
        for num in prob:
            proba.append(float(num))
        print(proba)
        label = str(label+1)
        print(label)
        return jsonify({'status': 'success', 'text': text, 'label' : label, 'proba': proba})
    return "Sentiment classifier"

if __name__ == '__main__':
    app.run(host="0.0.0.0",debug=True)