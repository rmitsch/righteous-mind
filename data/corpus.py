import pandas as pd
import json
import logging
from typing import Tuple
import requests
import bs4
import time
from symspellpy.symspellpy import SymSpell, Verbosity
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


class Corpus:
    """
    Class handling all tweet- and tweeter-related data processing tasks.
    """

    symspell_config = {
        "initial_capacity": 83000,
        "max_edit_distance_dictionary": 2,
        "prefix_length": 7,
        "max_edit_distance_lookup": 2,
        "suggestion_verbosity": Verbosity.CLOSEST
    }

    def __init__(self, user_path: str, tweets_path: str, logger: logging.Logger):
        """
        Initializes corpus by loading data.
        :param user_path:
        :param tweets_path:
        :param logger:
        """
        self._logger = logger
        self._user_path = user_path
        self._tweets_path = tweets_path

        logger.info("Loading data.")
        self._users_df, self._tweets_df = self._load_data()

        logger.info("Getting party affiliation.")
        self._users_df = self._associate_political_parties()

        # Note: Spelling correction does not seem necessary right now.
        # logger.info("Correcting spelling errors.")
        # self._tweets_df = self._correct_spelling_errors()

        logger.info("Calculating intensity scores.")
        self._estimate_emotional_intensity()

    def _estimate_emotional_intensity(self):
        """
        Estimates emotional intensity in each tweet.
        Adds pandas series with "intensity" score calculated as 1 - neutral score as determined by VADER to tweet data-
        frame.
        :return
        """

        # Ignore, if "intensity" is already in dataframe.
        if "intensity" not in self._tweets_df.columns:
            analyzer = SentimentIntensityAnalyzer()
            self._tweets_df["intensity"] = self._tweets_df.text.apply(lambda x: 1 - analyzer.polarity_scores(x)["neu"])
            self._tweets_df.to_pickle(path=self._tweets_path.split(".")[:-1][0] + ".pkl")

    def _correct_spelling_errors(self):
        sym_spell = SymSpell(
            Corpus.symspell_config["initial_capacity"],
            Corpus.symspell_config["max_edit_distance_dictionary"],
            Corpus.symspell_config["prefix_length"]
        )
        config = Corpus.symspell_config

        self._tweets_df = self._tweets_df.sample(frac=1)
        for idx, record in self._tweets_df.iterrows():
            suggestions = sym_spell.lookup_compound(record.text, config["max_edit_distance_lookup"])
            for suggestion in suggestions:
                print("  {}, {}, {}".format(suggestion.term, suggestion.count, suggestion.distance))

        return self._tweets_df

    def _query_for_political_party(self, politican_name: str):
        """
        Queries google for political party of specified politician.
        :param politican_name:
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
        self._logger.info(politican_name + ": " + (result.text if result is not None else "None"))
        time.sleep(1)

        return result.text if result is not None else None

    def _associate_political_parties(self) -> pd.DataFrame:
        """
        Gets user bias for politicans sending the respective tweets.
        :return: Dataframe with politicians + party affiliation.
        """

        # Only if not done already: Get party affiliation.
        if "party" not in self._users_df.columns:
            self._users_df["party"] = self._users_df.name.apply(lambda x: self._query_for_political_party(x))
            # Complement party information for individuals for which it could not be captured automatically.
            users_df = Corpus._refine_party_affiliation(self._users_df)

            if len(users_df) > 0:
                users_df.to_pickle(path=self._user_path.split(".")[:-1][0] + ".pkl")

            return users_df

        return self._users_df

    @staticmethod
    def _add_tweets_to_df_from_line_buffer(df: pd.DataFrame, buffer: list) -> pd.DataFrame:
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

    def _load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Loads and parses input data (users and tweets).
        :return:
        """
        _users_df = None
        _tweets_df = pd.DataFrame()

        # Read users.
        if self._user_path.endswith(".pkl"):
            self._logger.info("Reading " + self._user_path + ".")
            _users_df = pd.read_pickle(path=self._user_path)
        else:
            self._logger.info("Reading and parsing " + self._user_path + ".")
            with open(self._user_path) as f:
                _users_df = pd.DataFrame([json.loads(line) for line in f.readlines()])
                _users_df.name = _users_df.name.str.replace(
                    r"Governor |Gov. | Rep. |\(|\)|U.S. Rep. |Cong. |Rep.|Senator |US Rep |Congresswoman |Congressman "
                    r"|Leader |Sen. |Sen |Captain |Dr. ",
                    ""
                ).str.lstrip().str.rstrip()

                _users_df.to_pickle(path=self._user_path.split(".")[:-1][0] + ".pkl")

        # Read tweets.
        if self._tweets_path.endswith(".pkl"):
            self._logger.info("Reading " + self._tweets_path + ".")
            _tweets_df = pd.read_pickle(path=self._tweets_path)
        else:
            self._logger.info("Reading and parsing " + self._tweets_path + ".")
            with open(self._tweets_path) as f:
                buffer = []
                buffer_size = 100000

                # Batch line processing due to memory constraints.
                for line_idx, line in enumerate(f.readlines()):
                    buffer.append(json.loads(line))

                    if line_idx > 0 and not line_idx % buffer_size:
                        _tweets_df = Corpus._add_tweets_to_df_from_line_buffer(df=_tweets_df, buffer=buffer)
                        buffer = []

                # Add remaining lines, store as pickle.
                _tweets_df = self._add_tweets_to_df_from_line_buffer(df=_tweets_df, buffer=buffer)
                _tweets_df = _tweets_df.drop(["withheld_copyright", "withheld_in_countries", "withheld_scope"], axis=1)
                _tweets_df.text = _tweets_df.str.replace("&amp;", "")
                _tweets_df.to_pickle(path=self._tweets_path.split(".")[:-1][0] + ".pkl")

        return _users_df, _tweets_df

    @staticmethod
    def _refine_party_affiliation(users_df: pd.DataFrame) -> pd.DataFrame:
        """
        Manual refinement of party affiliations.
        :param users_df:
        :return:
        """
        democratic_names = {
            "Collin Peterson", "John Carney", "USRick Nolan", "Dan Malloy", "Mark Dayton", "USAl Lawson Jr",
            "Nanette D. Barragán", "Jared Huffman", "TeamMoulton", "Bernie Sanders", "Al Franken"
        }
        republican_names = {
            "Bill Walker", "evinBrady", "Dr. Roger Marshall", "Roger Marshall", "Dr. Neal Dunn", "Steve Chabot 🇺🇸",
            "Asa Hutchinson", "Paul R. LePage", "Sam Brownback", "Jerry Morran", "Sensenbrenner Press",
            "Mac Thornberry Press", "Paul Gosar, DDS", "John Faso", "cottPerry", "Daniel Webster", "SenDanSullivan",
            "tiberipress", "Hatch Office", "Jerry Moran"
        }
        libertarian_names = {
            "Judge Carter"
        }
        # List of accounts to drop due to a lack of stringend political affiliation.
        to_remove = {
            "arkAmodei", "Jasmine Coleman", "Angus King"
        }

        users_df.loc[users_df.name.isin(democratic_names), "party"] = "Democratic Party"
        users_df.loc[users_df.name.isin(republican_names), "party"] = "Republic Party"
        users_df.loc[users_df.name.isin(libertarian_names), "party"] = "Libertarians"
        users_df.party = users_df.party. \
            str.replace("Democratic Party of Oregon", "Democratic Party"). \
            str.replace("Montana Republican Party", "Republican Party"). \
            str.replace("Republic Party", "Republican Party")
        users_df = users_df.loc[~users_df.name.isin(to_remove)]

        assert len(users_df.isnull()) != 0, "Records not assigned to any party in dataframe."

        return users_df
