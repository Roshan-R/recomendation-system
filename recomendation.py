#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2022 rosh <rosh@parippvada>
#
# Distributed under terms of the MIT license.

from typing import Optional

from fastapi import FastAPI


import pandas as pd
from scipy import sparse
from sklearn.metrics.pairwise  import cosine_similarity

ratings = pd.read_csv('data.csv', index_col=0)
# fill n/a with 0
ratings = ratings.fillna(0)

def pup(row):
    n_row = (row - row.mean())/(row.max()-row.min())
    return n_row

p_ratings = ratings.apply(pup)
similarity = cosine_similarity(p_ratings.T)

similarity_df = pd.DataFrame(similarity, index=ratings.columns, columns=ratings.columns)

def get_similar_ratings(material, rating):
    similar_score = similarity_df[material]*rating
    similar_score = similar_score.sort_values(ascending=False)

    return similar_score

app = FastAPI()

@app.get("/items/{material}/{item_id}")
def read_item(item_id: int, material: str):
    return {'list':get_similar_ratings(material, item_id)}
