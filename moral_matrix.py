"""
Data and utilities related to the moral matrix weighting scheme introduced in TRM.
"""
import os

import pandas as pd
from typing import Tuple
import tensorflow as tf
from tensorflow.python.client import session as tf_session
import tensorflow_hub as hub
import logging
from scipy import spatial


class MoralMatrix:
    """
    Responsible for parsing and processing moral value information data.
    Sources:
        - V2: http://www.jeremyfrimer.com/research-downloads.html
        - V1: https://www.moralfoundations.org/othermaterials
    """

    def __init__(self, path_to_file: str, elmo_cache_directory: str, logger: logging.Logger):
        """
        Initializes moral matrix value data.
        :param path_to_file: Path to .dic file with terms per moral value.
        :param elmo_cache_directory: Directory in which to store ELMO model.
        """
        logger.info("Initializing moral matrix and ELMO.")

        self._moral_matrix_weights = MoralMatrix._define_moral_matrix_weights()
        self._morals_to_words_df, self._words_to_morals_df = MoralMatrix.read_moral_dictionary(path_to_file)

        # Load elmo TF module and initialize session.
        os.environ["TFHUB_CACHE_DIR"] = elmo_cache_directory
        self._elmo, self._tf_session = MoralMatrix.initialize_elmo(elmo_cache_directory)

        embeddings = self._elmo(
            ["the cat is on the", "dog are in the fog"],
            signature="default",
            as_dict=True
        )["elmo"]

        print(1 - spatial.distance.cosine(self._tf_session.run(embeddings[0][1]), self._tf_session.run(embeddings[1][0])))

    @staticmethod
    def initialize_elmo(elmo_cache_directory: str) -> Tuple[hub.module.Module, tf_session.Session]:
        """
        Initializes ELMO and the corresponding TF session.
        :param elmo_cache_directory:
        :return:
        """

        os.environ["TFHUB_CACHE_DIR"] = elmo_cache_directory

        elmo = hub.Module("https://tfhub.dev/google/elmo/2")
        init = tf.initialize_all_variables()
        session = tf.Session()
        session.run(init)

        return elmo, session

    @staticmethod
    def read_moral_dictionary(path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Reads and parses moral dictionary.
        :param path: Path to dictionary file.
        :return: (1) Dataframe with morals-to-word mapping, (2) dataframe with word-to-morals mapping.
        """
        dict_file = open(path, 'r')
        lines = dict_file.readlines()
        morals_to_words = {}
        words_to_morals = {}

        # Get index -> value/polarity association in dict file.
        moral_value_lookup = {}
        for line in lines[1:11]:
            vals = line.replace("\n", "").split("\t")
            content_vals = vals[1].split(".")
            moral_value_lookup[int(vals[0])] = {"value": content_vals[0], "polarity": content_vals[1]}

            if content_vals[0] not in morals_to_words:
                morals_to_words[content_vals[0]] = {}
            morals_to_words[content_vals[0]][content_vals[1]] = set()

        # Sort words by moral value and polarity.
        for line in lines[12:]:
            vals = line.replace("\n", "").split("\t")
            moral_info = moral_value_lookup[int(vals[1])]

            morals_to_words[moral_info["value"]][moral_info["polarity"]].add(vals[0])
            words_to_morals[vals[0]] = moral_info

        return pd.DataFrame().from_dict(morals_to_words).T, pd.DataFrame().from_dict(words_to_morals).T

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
            "conservative": {
                "care": 1.0 / 3,
                "liberty": 5,
                "fairness": 1,
                "loyalty": 1.0 / 3,
                "authority": 1.0 / 3,
                "sanctity": 1.0 / 3
            },
            "libertarian": {
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

        return moral_matrix_weights
