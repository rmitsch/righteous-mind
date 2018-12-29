"""
Data and utilities related to the moral matrix weighting scheme introduced in TRM.
"""

import os
import pickle

import pandas as pd
from typing import Tuple
import logging
import numpy as np
from tqdm import tqdm
from bert_serving.client import BertClient
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import classification_report
import xgboost as xgb
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score


class MoralMatrix:
    """
    Responsible for parsing and processing moral value information data.
    Sources:
        - V2: http://www.jeremyfrimer.com/research-downloads.html
        - V1: https://www.moralfoundations.org/othermaterials
    """

    def __init__(self, path_to_file: str, logger: logging.Logger):
        """
        Initializes moral matrix value data.
        :param path_to_file: Path to .dic file with terms per moral value.
        """
        logger.info("Initializing moral matrix.")

        self._logger = logger

        # Define moral value weighting; read and process moral dictionary.
        # Note: Stored as pickle and generate only in case of pickle not being available.
        self._moral_matrix_weights = MoralMatrix._define_moral_matrix_weights()
        self._morals_to_phrases_df, self._phrase_embeddings = self._read_moral_dictionary(path_to_file)
        self._mv_predictor = self._train_moral_value_classifier()

    def _train_moral_value_classifier(self):
        """
        Trains model classifying phrases into sentiments.
        :return: 
        """

        mv_model_path = "data/mv_predictor.pkl"
        mv_predictor = None

        # Build model predicting moral values for word.
        if not os.path.isfile(mv_model_path):
            df = self._phrase_embeddings
            x = np.asarray([np.asarray(x) for x in df.embeddings.values])
            le = preprocessing.LabelEncoder()
            le.fit(self._morals_to_phrases_df.index.values)
            y = le.transform(df.moral_values.values)

            self._logger.info("Training moral value predictor.")
            for train_index, test_index in StratifiedShuffleSplit(n_splits=1, test_size=0.25).split(x, y):
                x_train, x_test = x[train_index], x[test_index]
                y_train, y_test = y[train_index], y[test_index]

                mv_predictor = xgb.XGBClassifier(
                    objective='binary:logistic',
                    colsample_bytree=0.7,
                    learning_rate=0.05,
                    n_estimators=3000,
                    n_jobs=0,
                    nthread=0
                )

                mv_predictor.fit(x_train, y_train)
                pickle.dump(mv_predictor, open(mv_model_path, "wb"))
                # y_pred = mv_predictor.predict(x_test)
                # print(classification_report(y_test, y_pred, target_names=self._morals_to_phrases_df.index.values))

        # Load built model.
        else:
            mv_predictor = pd.read_pickle(path=mv_model_path)

        return mv_predictor

    def _read_moral_dictionary(self, path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Reads and parses moral dictionary.
        :param path: Path to dictionary file.
        :return: (1) Dataframe with morals-to-phrase mapping, (2) dataframe with phrase embedding.
        """
        morals_to_phrases_path = "data/morals-to-phrases.pkl"
        phrase_embeddings_path = "data/phrase-embeddings.pkl"

        # Create if files not available, otherwise read from file.
        if not os.path.isfile(morals_to_phrases_path) or not os.path.isfile(phrase_embeddings_path):
            dict_file = open(path, 'r')
            lines = dict_file.readlines()
            morals_to_phrases = {}

            # Get index -> value/polarity association in dict file.
            moral_value_lookup = {}
            for line in lines[1:11]:
                vals = line.replace("\n", "").split("\t")
                content_vals = vals[1].split(".")
                moral_value_lookup[int(vals[0])] = {"value": content_vals[0], "polarity": content_vals[1]}

                if content_vals[0] not in morals_to_phrases:
                    morals_to_phrases[content_vals[0]] = {}
                morals_to_phrases[content_vals[0]][content_vals[1]] = set()

            # Sort words by moral value and polarity.
            tokens = []
            moral_values = []
            for line in lines[12:]:
                vals = line.replace("\n", "").split("\t")
                moral_info = moral_value_lookup[int(vals[1])]
                morals_to_phrases[moral_info["value"]][moral_info["polarity"]].add(vals[0])

                # Collect tokens for batch inference with ELMo.
                tokens.append(vals[0])
                moral_values.append(moral_info["value"])

            self._logger.info("Inferring embeddings for moral dictionary.")
            bert_client = BertClient()
            pbar = tqdm(total=len(tokens))
            phrase_encodings = []
            for token in tokens:
                phrase_encodings.append(bert_client.encode([token])[0])
                pbar.update(1)
            pbar.close()

            moral_phrase_embeddings_df = pd.DataFrame(tokens).rename({0: "phrase"}, axis=1).set_index("phrase")
            moral_phrase_embeddings_df["embeddings"] = phrase_encodings
            moral_phrase_embeddings_df["moral_values"] = moral_values

            # Save encodings.
            morals_to_phrases_df = pd.DataFrame().from_dict(morals_to_phrases).T
            morals_to_phrases_df.to_pickle(path=morals_to_phrases_path)
            moral_phrase_embeddings_df.to_pickle(path=phrase_embeddings_path)

        else:
            morals_to_phrases_df = pd.read_pickle(path=morals_to_phrases_path)
            moral_phrase_embeddings_df = pd.read_pickle(path=phrase_embeddings_path)

        return morals_to_phrases_df, moral_phrase_embeddings_df

    @staticmethod
    def _define_moral_matrix_weights() -> dict:
        """
        Defines moral matrix per political affiliation with normalized weights.
        :return:
        """

        moral_matrix_weights = {
            "liberal": {
                "care": 4,
                "liberty": 3,
                "fairness": 1,
                "loyalty": 1.0 / 3,
                "authority": 1.0 / 3,
                "sanctity": 1.0 / 3
            },
            "libertarian": {
                "care": 1.0 / 3,
                "liberty": 5,
                "fairness": 1,
                "loyalty": 1.0 / 3,
                "authority": 1.0 / 3,
                "sanctity": 1.0 / 3
            },
            "conservative": {
                "care": 1,
                "liberty": 1,
                "fairness": 1,
                "loyalty": 1,
                "authority": 1,
                "sanctity": 1
            }
        }

        for affiliation in moral_matrix_weights:
            weight_sum = sum(list(moral_matrix_weights[affiliation].values()))
            for moral_value in moral_matrix_weights[affiliation]:
                moral_matrix_weights[affiliation][moral_value] /= weight_sum

        print(moral_matrix_weights)

        return moral_matrix_weights

    def predict_mv_probabilities(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Predicts probabilities for affiliation with defined moral values.
        :param embeddings:
        :return: Matrix with probabilities for affiliation with moral values for each embedding vector.
        """

        return self._mv_predictor.predict_proba(np.asarray([emb for emb in embeddings]))

    def get_moral_values(self):
        return self._morals_to_phrases_df.index.values
