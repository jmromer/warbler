import os
import time

import pandas as pd
import redis
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


class Recommender(object):
    SIMKEY = "p:smlr:%s"
    REDIS_URL = os.environ.get("REDIS_URL") or "redis://localhost:6379"

    def __init__(self):
        self._redis = redis.StrictRedis.from_url(self.REDIS_URL)

    def train(self, csv_path):
        start = time.time()
        df = pd.read_csv(csv_path)
        print("Loading data: %s secs." % (time.time() - start))
        start = time.time()
        self._train_engine(df)
        print("Training model: %s secs." % (time.time() - start))

    def predict(self, item_id, num):
        return self._redis.zrange(
            self.SIMKEY % item_id, 0, num - 1, withscores=False, desc=True)

    def _train_engine(self, data_frame):
        tf = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 3),
            min_df=0,
            stop_words="english")

        content = data_frame["content"]
        tfidf_matrix = tf.fit_transform(content)
        cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix)

        for idx, row in data_frame.iterrows():
            similar_indices = cosine_similarities[idx].argsort()[:-100:-1]
            similar_items = [(cosine_similarities[idx][i], data_frame["id"][i])
                             for i in similar_indices]
            flattened = sum(similar_items[1:], ())
            self._redis.zadd(self.SIMKEY % row["id"], *flattened)
