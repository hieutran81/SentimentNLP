#!/usr/bin/python3
from flask import Flask, request, jsonify
from flask_restful import Resource, Api
from SentimentClassify import predict_sentiment
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
        print(text)
        label, prob = predict_sentiment(text)
        print(prob.shape)
        label = str(label+1)
        print(label)
        return jsonify({'status': 'success', 'text': text, 'label' : label, 'proba': list(prob[0])})
    return "Sentiment classifier"

if __name__ == '__main__':
    app.run(debug=True)