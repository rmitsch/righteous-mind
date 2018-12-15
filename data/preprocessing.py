# todo
#   - get tweeter affilication (liberal/libertarian/conservative).
#   - preprocess and clean tweet data.
#   - fix spelling errors
#   - get and preprocess reference documents for moral values.

import pandas as pd
import argparse
import json
import logging
from typing import Tuple
import requests
import bs4

import utils
import time


def query_for_political_party(politican_name: str, logger: logging.Logger):
    """
    Queries google for political party of specified politician.
    :param politican_name:
    :param logger:
    :return:
    """

    r = requests.get(
        'https://www.google.com/search?hl=en&q=' + politican_name + ' political party',
        headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/'
                          '64.0.3282.186 Safari/537.36'
        }
    )
    soup = bs4.BeautifulSoup(r.text, 'lxml')
    result = soup.find('div', class_='Z0LcW')
    logger.info(politican_name + ": " + (result.text if result is not None else "None"))
    time.sleep(1)

    return result.text if result is not None else None


def associate_political_parties(users_df: pd.DataFrame, users_path: str, logger: logging.Logger) -> pd.DataFrame:
    """
    Get user bias for politicans sending the respective tweets.
    :param users_df:
    :param users_path:
    :param logger:
    :return: Dataframe with politicians + party affiliation.
    """

    users_df["party"] = users_df.name.apply(lambda x: query_for_political_party(x, logger))
    # Complement party information for individuals for which it could not be captured automatically.
    users_df = refine_party_affiliation(users_df)

    users_df.to_pickle(path=users_path.split(".")[:-1][0] + ".pkl")

    return users_df


def add_tweets_to_df_from_line_buffer(df: pd.DataFrame, buffer: list) -> pd.DataFrame:
    """
    Adds tweets from line buffer to specified dataframe.
    :param df:
    :param buffer:
    :return:
    """

    buffer_df = pd.DataFrame(buffer)
    buffer_df = buffer_df.drop([
        "coordinates", "created_at", "display_text_range", "entities", "extended_entities", "geo",
        "in_reply_to_screen_name", "in_reply_to_status_id", "in_reply_to_status_id_str",
        "in_reply_to_user_id", "in_reply_to_user_id_str", "lang", "place", "retweeted",
        "quoted_status_id_str", "quoted_status_id", "is_quote_status", "favorited", "truncated", "id",
        "id_str", "contributors", "source", "possibly_sensitive"
    ], axis=1)

    return pd.concat([df, buffer_df], axis=0, sort=True)


def load_data(users_input_path: str, tweets_input_path: str, logger: logging.Logger) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads and parses input data (users and tweets).
    :param users_input_path:
    :param tweets_input_path:
    :param logger:
    :return:
    """
    _users_df = None
    _tweets_df = pd.DataFrame()

    # Read users.
    if users_input_path.endswith(".pkl"):
        logger.info("Reading " + users_input_path + ".")
        _users_df = pd.read_pickle(path=users_input_path)
    else:
        logger.info("Reading and parsing " + users_input_path + ".")
        with open(users_input_path) as f:
            _users_df = pd.DataFrame([json.loads(line) for line in f.readlines()])
            _users_df.name = _users_df.name.str.replace(
                r"Governor |Gov. | Rep. |\(|\)|U.S. Rep. |Cong. |Rep.|Senator |US Rep |Congresswoman |Congressman "
                r"|Leader |Sen. |Sen |Captain |Dr. ",
                ""
            ).str.lstrip().str.rstrip()

            _users_df.to_pickle(path=users_input_path.split(".")[:-1][0] + ".pkl")

    # Read tweets.
    if tweets_input_path.endswith(".pkl"):
        logger.info("Reading " + users_input_path + ".")
        _tweets_df = pd.read_pickle(path=tweets_input_path)
    else:
        logger.info("Reading and parsing " + tweets_input_path + ".")
        with open(tweets_input_path) as f:
            buffer = []
            buffer_size = 100000

            # Batch line processing due to memory constraints.
            for line_idx, line in enumerate(f.readlines()):
                buffer.append(json.loads(line))

                if line_idx > 0 and not line_idx % buffer_size:
                    _tweets_df = add_tweets_to_df_from_line_buffer(df=_tweets_df, buffer=buffer)
                    buffer = []

            # Add remaining lines, store as pickle.
            _tweets_df = add_tweets_to_df_from_line_buffer(df=_tweets_df, buffer=buffer)
            _tweets_df = _tweets_df.drop(["withheld_copyright", "withheld_in_countries", "withheld_scope"], axis=1)
            _tweets_df.to_pickle(path=tweets_input_path.split(".")[:-1][0] + ".pkl")

    return _users_df, _tweets_df


def refine_party_affiliation(users_df: pd.DataFrame) -> pd.DataFrame:
    """
    Manual refinement of party affiliations.
    :param users_df:
    :return:
    """
    democratic_names = {
        "Collin Peterson", "John Carney", "USRick Nolan", "Dan Malloy", "Mark Dayton", "USAl Lawson Jr",
        "Nanette D. Barragán", "Jared Huffman", "TeamMoulton"
    }
    republican_names = {
        "Bill Walker", "evinBrady", "Dr. Roger Marshall", "Roger Marshall", "Dr. Neal Dunn", "Steve Chabot 🇺🇸",
        "Asa Hutchinson", "Paul R. LePage", "Sam Brownback", "Jerry Morran", "Sensenbrenner Press",
        "Mac Thornberry Press", "Paul Gosar, DDS", "John Faso", "cottPerry", "Daniel Webster", "SenDanSullivan",
        "tiberipress"
    }
    libertarian_names = {
        "Judge Carter"
    }
    # List of accounts to drop due to a lack of stringend political affiliation.
    to_remove = {
        "arkAmodei", "Jasmine Coleman"
    }

    users_df.loc[users_df.name.isin(democratic_names), "party"] = "Democratic Party"
    users_df.loc[users_df.name.isin(republican_names), "party"] = "Republic Party"
    users_df.loc[users_df.name.isin(libertarian_names), "party"] = "Libertarians"
    users_df.drop(users_df.name.isin(to_remove).index, inplace=True)

    return users_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", dest="tweets_path", type=str)
    parser.add_argument("-u", dest="users_path", type=str)
    args = parser.parse_args()
    logger = utils.setup_custom_logger("preprocessing")

    logger.info("Loading data.")
    users_df, tweets_df = load_data(args.users_path, args.tweets_path, logger)

    logger.info("Getting party affiliation.")
    # users_df = associate_political_parties(users_df, args.users_path, logger)

    users_df = refine_party_affiliation(users_df)
    # users_df.to_pickle(path=args.users_path.split(".")[:-1][0] + ".pkl")

    # print(users_df[['name']].groupby(['party']).agg(['count']))
