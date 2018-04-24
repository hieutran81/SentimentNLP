import json
from flask import Flask, request, jsonify, Blueprint, abort
from flask.views import MethodView
from SentimentClassify import predict_sentiment

sentiment = Blueprint('sentiment', __name__)
app = Flask(__name__)
app.register_blueprint(sentiment)

@sentiment.route('/')
@sentiment.route('/home')
def home():
    return "Welcome to the Catalog Home."


class SentimentView(MethodView):
    def get(self, id=None, page=1):
        return "ha ha"

    def post(self):
        text = request.form.get('text')
        label = predict_sentiment(text)
        return jsonify({ 'text': text,'label': label})

    def put(self, id):
        # Update the record for the provided id
        # with the details provided.
        return

    def delete(self, id):
        # Delete the record for the provided id.
        return


sentiment_view = SentimentView.as_view('sentiment_view')
app.add_url_rule(
    '/sentiment/', view_func=sentiment, methods=['GET', 'POST']
)
if __name__ == "__main__":
    app.run(debug=True)
