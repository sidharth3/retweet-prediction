from flask import Flask, jsonify, request
import json
from flask_cors import CORS
from NN_Tweet import *

app = Flask(__name__)
CORS(app)


@app.route("/predict", methods=["POST"])
def predict():
    response_tweetID = request.get_json()["tweet"]
    response_username = request.get_json()["username"]
    response_followers = request.get_json()["followers"]
    response_friends = request.get_json()["friends"]
    response_favorites = request.get_json()["favorites"]
    response_entities = request.get_json()["entities"]
    response_POSsentiment = request.get_json()["POSsentiment"]
    response_NEGsentiment = request.get_json()["NEGsentiment"]
    response_mentions = request.get_json()["mentions"]
    response_hashtags = request.get_json()["hashtags"]
    response = call_model(response_tweetID, response_username, response_followers, response_friends, response_favorites, response_entities, response_POSsentiment, response_NEGsentiment, response_mentions, response_hashtags)
    print(str(response))
    # return json.dumps(response)
    return str(response)


if __name__ == "__main__":
    app.run()
