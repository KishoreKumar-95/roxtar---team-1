import json
from src import *
from flask import Flask, request

app = Flask(__name__)

with open("config.json", "r") as fread:
    config = json.load(fread)

bot = MerlinBot(config)
bot._init_conversation_bot()

@app.route('/query', methods=["POST"])
def perform_query():
    return json.dumps(
        {
            "result":bot.perform_query(request.get_json()["query"])
        }
    )

if __name__=="__main__":
    app.run(debug=True)
