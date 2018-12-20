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
from tqdm import tqdm
import numpy as np


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
        self._logger = logger
        logger.info("Initializing moral matrix.")

        # Load elmo TF module and initialize session.
        os.environ["TFHUB_CACHE_DIR"] = elmo_cache_directory
        self._elmo, self._tf_session = MoralMatrix._initialize_elmo(elmo_cache_directory)

        # Define moral value weighting; read and process moral dictionary.
        # Note: Stored as pickle and generate only in case of pickle not being available.
        self._moral_matrix_weights = MoralMatrix._define_moral_matrix_weights()
        morals_to_phrases_path = "data/morals-to-phrases.pkl"
        phrase_embeddings_path = "data/phrase-embeddings.pkl"
        if not os.path.isfile(morals_to_phrases_path) or not os.path.isfile(phrase_embeddings_path):
            self._morals_to_phrases_df, self._phrase_embeddings = self._read_moral_dictionary(path_to_file)
            self._morals_to_phrases_df.to_pickle(path=morals_to_phrases_path)
            self._phrase_embeddings.to_pickle(path=phrase_embeddings_path)
        else:
            self._morals_to_phrases_df = pd.read_pickle(path=morals_to_phrases_path)
            self._phrase_embeddings = pd.read_pickle(path=phrase_embeddings_path)

    @staticmethod
    def _initialize_elmo(elmo_cache_directory: str) -> Tuple[hub.module.Module, tf_session.Session]:
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

    def _read_moral_dictionary(self, path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Reads and parses moral dictionary.
        :param path: Path to dictionary file.
        :return: (1) Dataframe with morals-to-phrase mapping, (2) dataframe with phrase embedding.
        """
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
        token_set = set()
        tokens = []
        seq_lengths = []
        for line in lines[12:]:
            vals = line.replace("\n", "").split("\t")
            moral_info = moral_value_lookup[int(vals[1])]
            morals_to_phrases[moral_info["value"]][moral_info["polarity"]].add(vals[0])

            # Collect tokens for batch inference with ELMo.
            if vals[0] not in token_set:
                token_set.add(vals[0])
                tokens.append(vals[0])
                seq_lengths.append(len(vals[0].split()))

        self._logger.info("Inferring ELMo embeddings for moral dictionary.")
        embeddings = self._tf_session.run(self._elmo(tokens, signature="default", as_dict=True)["elmo"])
        moral_phrase_embeddings = pd.DataFrame(tokens).rename({0: "phrase"}, axis=1).set_index("phrase")
        moral_phrase_embeddings["embeddings"] = [embedding[:seq_lengths[i]] for i, embedding in enumerate(embeddings)]

        return pd.DataFrame().from_dict(morals_to_phrases).T, moral_phrase_embeddings

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

