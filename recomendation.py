#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2022 rosh <rosh@parippvada>
#
# Distributed under terms of the MIT license.

import pandas as pd
from scipy import sparse
from sklearn.metrics.pairwise  import cosine_similarity


class Recomendator:

    def __init__(self):
        self.ratings = pd.read_csv('data.csv', index_col=0)
        self.ratings = self.ratings.fillna(0)

    def pup(self, row):
        n_row = (row - row.mean())/(row.max()-row.min())
        return n_row

    def ml(self):

        self.p_ratings = self.ratings.apply(self.pup)
        self.similarity = cosine_similarity(self.p_ratings.T)
        self.similarity_df = pd.DataFrame(self.similarity, index=self.ratings.columns, columns=self.ratings.columns)

    def get_similar_ratings(self,materialNo, rating):
        material = 'Material ' + str(materialNo)
        similar_score = self.similarity_df[material]*rating
        similar_score = similar_score.sort_values(ascending=False)

        return similar_score

# FastAPI stuff

from typing import Optional
from fastapi import FastAPI
from pydantic import BaseModel

class Rating(BaseModel):
    materialNo: int
    rating: int

app = FastAPI()
recomendator = Recomendator()
recomendator.ml()

@app.post("/api/getRecomendations")
def read_item(rating:Rating):
    return recomendator.get_similar_ratings(rating.materialNo, rating.rating)
