import argparse
import logging
import os
import sys
from typing import Tuple
import tensorflow as tf
from tensorflow.python.client import session as tf_session
import tensorflow_hub as hub


def setup_custom_logger(name):
    """
    Set up logger.
    Source: https://stackoverflow.com/questions/28330317/print-timestamp-for-logging-in-python.
    :param name:
    :return:
    """
    formatter = logging.Formatter(fmt='%(asctime)s %(levelname)-8s %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')
    handler = logging.FileHandler('log.txt', mode='w')
    handler.setFormatter(formatter)
    screen_handler = logging.StreamHandler(stream=sys.stdout)
    screen_handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    logger.addHandler(screen_handler)

    return logger


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


def parse_args() -> argparse.Namespace:
    """
    Parses arguments.
    :return: Result of argparse.parse_args().
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("-t", dest="tweets_path", type=str)
    parser.add_argument("-u", dest="users_path", type=str)
    parser.add_argument("-d", dest="moral_dictionary_path", type=str)
    parser.add_argument("-e", dest="elmo_cache_directory", type=str)
    args = parser.parse_args()

    return args
