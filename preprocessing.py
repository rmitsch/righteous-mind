import utils
from corpus import Corpus
from moral_matrix import MoralMatrix

if __name__ == '__main__':
    args = utils.parse_args()
    logger = utils.setup_custom_logger("preprocessing")

    # Prepare corpus and moral value matrix.
    moral_matrix = MoralMatrix(args.moral_dictionary_path, logger)
    corpus = Corpus(args.users_path, args.tweets_path, moral_matrix, logger)
