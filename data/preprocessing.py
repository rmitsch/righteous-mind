# todo
#   - preprocess and clean tweet data.
#   - fix spelling errors
#   - get and preprocess reference documents for moral values.

"""
Workflow:
    - Assume data is ready.
    - Seed topic models.
    - Apply guided topic modeling.
    - Go through tweets for each person, sum up relevance of topics/moral values. Consider strength of topic AND
      emotional intensity.
    - Evaluation whether separation can be achieved.
        - Possible: Use sentence embeddings, average tweeter's coordinates, reduce to map.
"""

import argparse
import utils
import data.corpus

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", dest="tweets_path", type=str)
    parser.add_argument("-u", dest="users_path", type=str)
    args = parser.parse_args()

    logger = utils.setup_custom_logger("preprocessing")
    corpus = data.corpus.Corpus(args.users_path, args.tweets_path, logger)

