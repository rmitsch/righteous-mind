# todo
#   - set topic seeds for moral values.
#   - add sentiment/intensity analysis
#   - apply guided topic modeling
#   - test dataflow

"""
Workflow after preprocessing:
    - Seed topic models.
    - Apply guided topic modeling.
    - Go through tweets for each person, sum up relevance of topics/moral values. Consider strength of topic AND
      emotional intensity.
    - Evaluation whether separation can be achieved.
        - Possible: Use sentence embeddings, average tweeter's coordinates, reduce to map.
          Can be seen as alternative approach - more classic one would be n-grams. For both: We would only do that to
          compare with moral framework approach (which has a more direct theoretical footing).
"""

import argparse
import utils
import data.corpus
from moral_matrix import MoralMatrix

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", dest="tweets_path", type=str)
    parser.add_argument("-u", dest="users_path", type=str)
    parser.add_argument("-d", dest="moral_dictionary_path", type=str)
    args = parser.parse_args()

    logger = utils.setup_custom_logger("preprocessing")

    moral_matrix = MoralMatrix(path_to_file=args.moral_dictionary_path)
    # corpus = data.corpus.Corpus(args.users_path, args.tweets_path, logger)

