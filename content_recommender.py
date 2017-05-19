import time

from flask_restful import Resource


# import numpy as np
# import pandas as pd
# import redis
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics_pairwise import linear_kernel


class Recommendation(Resource):
    def get(self):
        return {"hello": "dave"}
