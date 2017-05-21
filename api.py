from flask import Flask
from flask_restful import Api, Resource, reqparse

from content_recommender import Recommender

app = Flask(__name__)
api = Api(app)


class Recommendation(Resource):
    def put(self):
        recommender = Recommender()
        recommender.train("./data/squawks.csv")
        return {"success": True}

    def get(self):
        parser = reqparse.RequestParser()
        parser.add_argument("squawk_id", type=int, required=True)
        args = parser.parse_args()
        item_id = args['squawk_id']
        num_recommendations = 5

        recommender = Recommender()
        prediction = recommender.predict(item_id, num_recommendations)
        ids = list(map(lambda b: int(b.decode("utf-8")), prediction))
        return {"squawk_ids": ids}


api.add_resource(Recommendation, "/recommendation")

if __name__ == "__main__":
    app.run(debug=True)
