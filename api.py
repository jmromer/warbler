from flask import Flask
from flask_restful import Api

from content_recommender import Recommendation

app = Flask(__name__)
api = Api(app)

api.add_resource(Recommendation, "/")

if __name__ == "__main__":
    app.run(debug=True)
