# todo
#   - test dataflow

"""
Workflow after preprocessing:
    - Apply guided topic modeling.
    - Go through tweets for each person, sum up relevance of topics/moral values. Consider strength of topic AND
      emotional intensity.
    - Evaluation whether separation can be achieved.
        - Possible: Use sentence embeddings, average tweeter's coordinates, reduce to map.
          Can be seen as alternative approach - more classic one would be n-grams. For both: We would only do that to
          compare with moral framework approach (which has a more direct theoretical footing).
"""

import utils
from corpus import Corpus
from moral_matrix import MoralMatrix

if __name__ == '__main__':
    args = utils.parse_args()
    logger = utils.setup_custom_logger("preprocessing")

    # Load elmo TF module and initialize session.
    logger.info("Initializing ELMo.")
    elmo, tf_session = utils.initialize_elmo(args.elmo_cache_directory)

    # Prepare corpus and moral value matrix.
    moral_matrix = MoralMatrix(args.moral_dictionary_path, elmo, tf_session, logger)
    corpus = Corpus(args.users_path, args.tweets_path, elmo, tf_session, logger)
