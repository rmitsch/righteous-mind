import pandas as pd
import json
import logging
from typing import Tuple
import requests
import bs4
import time
import numpy as np
from bert_serving.client import BertClient
from symspellpy.symspellpy import SymSpell, Verbosity
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from tqdm import tqdm
import wordsegment
from spacy.lang.en.stop_words import STOP_WORDS

from moral_matrix import MoralMatrix


class Corpus:
    """
    Class handling all tweet- and tweeter-related data processing tasks.
    """

    symspell_config = {
        "initial_capacity": 83000,
        "max_edit_distance_dictionary": 3,
        "prefix_length": 7,
        "max_edit_distance_lookup": 3,
        "suggestion_verbosity": Verbosity.CLOSEST
    }

    def __init__(self, user_path: str, tweets_path: str, moral_matrix: MoralMatrix, logger: logging.Logger):
        """
        Initializes corpus by loading data.
        :param user_path:
        :param tweets_path:
        :param moral_matrix:
        :param logger:
        """
        self._logger = logger
        self._moral_matrix = moral_matrix
        self._user_path = user_path
        self._tweets_path = tweets_path
        self._stop_words = STOP_WORDS
        self._stop_words.update((
            "here", "its", "im", "the", "in", "w", "you", "i", "u", "r", "b", "tbt", "ut", "ive", "wknd", "said"
        ))

        # Load and preprocess data. Note that preprocessing steps only execute if they haven't been applied so far-
        self._users_df, self._tweets_df = self._load_data()
        self._users_df = self._associate_political_parties()
        # Note: Spelling correction does not seem necessary right now.
        # self._tweets_df = self._correct_spelling_errors()
        self._estimate_emotional_intensity()
        self._clean_tweets()
        self._infer_embeddings_for_tweets()

    @staticmethod
    def _transform_hashtags_to_words(text: str) -> str:
        """
        Tries to transform hashtags into words.
        :param text:
        :return:
        """

        tokens = text.split()
        text_refined = []

        for i, token in enumerate(tokens):
            if token.startswith("#"):
                text_refined.extend([word.lower() for word in wordsegment.segment(token[1:])])
            else:
                text_refined.append(token.lower())

        # Return concatenated text in lower case and w/o stop words.
        return " ".join(text_refined)

    def _clean_tweets(self):
        """
        Removes stopwords and non-interpretable symbols (URLs, hashtags, user handles, etc.) from tweets.
        :return:
        """

        if "clean_text" not in self._tweets_df:
            self._logger.info("Cleaning tweets.")

            # Try to transform hashtags into real words.
            wordsegment.load()
            self._tweets_df = self._tweets_df
            self._tweets_df["clean_text"] = self._tweets_df.text.apply(lambda x: Corpus._transform_hashtags_to_words(x))

            # Remove URLs, user handles, useless characters.
            self._tweets_df.clean_text = self._tweets_df.clean_text. \
                str.replace(r"https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+", ""). \
                str.replace(
                    r"rt |RT |&|[|]|\"|'|\\n|:|→|http…|ht…|,|\(|\)|🇺🇸|\.|⚡|“|!|\?|”|’|–|-|'|\+|gt;|…|\d|\\\w*|%|📷|#|"
                    r"👇🏽",
                    " "
                ). \
                str.replace(r"@\w*|/\w*", ""). \
                str.replace(r"\s+", " "). \
                str.strip()

            # Remove stopwords.
            self._tweets_df.clean_text = self._tweets_df.clean_text.apply(
                lambda x: " ".join([item for item in x.split() if item not in STOP_WORDS and len(item) > 1])
            )

            # Save updated dataframe.
            self._tweets_df.to_pickle(path=self._tweets_path.split(".")[:-1][0] + ".pkl")

    def _infer_embeddings_for_tweets(self):
        """
        Infers embeddings for tweets. Updates tweets dataframe and stores updated version on disk.
        :return:
        """
        self._logger.info("Inferring embeddings for corpus.")
        bert_client = BertClient()

        # Prepare dataframes if this is the first inference run.
        if "embeddings" not in self._tweets_df:
            self._tweets_df["num_words"] = self._tweets_df.clean_text.str.split().apply(len)
            self._tweets_df["embeddings"] = None
            self._users_df["mv_scores"] = [np.asarray([0] * len(self._moral_matrix.get_moral_values()))] * len(self._users_df)
            self._users_df["num_tweets"] = 0
            self._users_df["num_words"] = 0
            self._users_df = self._users_df.set_index("id")[[
                "description", "favourites_count", "friends_count", "id_str", "location", "name", "mv_scores",
                "num_tweets", "num_words", "screen_name", "party"
            ]]

        processed_tweets_count = self._tweets_df.embeddings.count()

        pbar = tqdm(total=(len(self._tweets_df) - processed_tweets_count))
        for i in range(self._tweets_df.embeddings.count(), len(self._tweets_df)):
            tweet = self._tweets_df.iloc[i]

            # Infer embeddings for each token in this tweet.
            self._tweets_df.iloc[i, self._tweets_df.columns.get_loc('embeddings')] = np.asarray([
                bert_client.encode([tweet.clean_text])[0][1:1 + len(tweet.clean_text.split())]
            ])

            # Get probabilities for moral values.
            self._users_df.at[tweet.user_id, "mv_scores"] = \
                self._users_df.at[tweet.user_id, "mv_scores"] + \
                self._moral_matrix.predict_mv_probabilities(self._tweets_df.iloc[i].embeddings)
            self._users_df.at[tweet.user_id, "num_tweets"] = self._users_df.at[tweet.user_id, "num_tweets"] + 1
            self._users_df.at[tweet.user_id, "num_words"] = \
                self._users_df.at[tweet.user_id, "num_words"] + len(tweet.clean_text)

            if i % 100 == 0 and i > 0:
                self._tweets_df.to_pickle(path=self._tweets_path.split(".")[:-1][0] + ".pkl")
                self._users_df.to_pickle(path=self._user_path.split(".")[:-1][0] + ".pkl")
            pbar.update(1)
        pbar.close()

        # Save updated dataframe.
        self._tweets_df.drop("num_words", axis=1)
        self._tweets_df.to_pickle(path=self._tweets_path.split(".")[:-1][0] + ".pkl")

    def _estimate_emotional_intensity(self):
        """
        Estimates emotional intensity in each tweet.
        Adds pandas series with "intensity" score calculated as 1 - neutral score as determined by VADER to tweet data-
        frame.
        :return
        """

        # Ignore if "intensity" is already in dataframe.
        if "intensity" not in self._tweets_df.columns:
            self._logger.info("Calculating intensity scores.")
            analyzer = SentimentIntensityAnalyzer()
            self._tweets_df["intensity"] = self._tweets_df.text.apply(lambda x: 1 - analyzer.polarity_scores(x)["neu"])
            self._tweets_df.to_pickle(path=self._tweets_path.split(".")[:-1][0] + ".pkl")

    def _correct_spelling_errors(self):
        """
        Corrects spelling errors in tweets using symspell.
        :return:
        """
        sym_spell = SymSpell(
            Corpus.symspell_config["initial_capacity"],
            Corpus.symspell_config["max_edit_distance_dictionary"],
            Corpus.symspell_config["prefix_length"]
        )
        config = Corpus.symspell_config

        # self._tweets_df = self._tweets_df.sample(frac=1)
        for idx, record in self._tweets_df.iterrows():
            suggestions = sym_spell.lookup_compound(record.text, config["max_edit_distance_lookup"])
            for suggestion in suggestions:
                print("  {}, {}, {}".format(suggestion.term, suggestion.count, suggestion.distance))

        return self._tweets_df

    def _query_for_political_party(self, politican_name: str, progress_bar: tqdm):
        """
        Queries google for political party of specified politician.
        :param politican_name:
        :param progress_bar:
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
        progress_bar.update(1)
        time.sleep(1)

        return result.text if result is not None else None

    def _associate_political_parties(self) -> pd.DataFrame:
        """
        Gets user bias for politicans sending the respective tweets.
        :return: Dataframe with politicians + party affiliation.
        """

        # Only if not done already: Get party affiliation.
        if "party" not in self._users_df.columns:
            self._logger.info("Getting party affiliation.")

            users_df = self._users_df
            pbar = tqdm(total=len(users_df))
            users_df["party"] = self._users_df.name.apply(lambda x: self._query_for_political_party(x, pbar))
            pbar.close()

            # Complement party information for individuals for which it could not be captured automatically.
            users_df = Corpus._refine_party_affiliation(users_df)

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

    @property
    def users(self):
        return self._users_df

    @property
    def tweets(self):
        return self._tweets_df
