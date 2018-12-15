import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import pandas as pd


def run_and_plot(session_, input_tensor_, messages_, encoding_tensor):
    message_embeddings_ = session_.run(encoding_tensor, feed_dict={input_tensor_: messages_})
    # plot_similarity(messages_, message_embeddings_, 90)
    corr = np.inner(message_embeddings_, message_embeddings_)
    print(corr)


if __name__ == '__main__':
    config = tf.ConfigProto(inter_op_parallelism_threads=4, intra_op_parallelism_threads=4)

    module_url = "https://tfhub.dev/google/universal-sentence-encoder-large/3"
    # @param ["https://tfhub.dev/google/universal-sentence-encoder/2", "https://tfhub.dev/google/universal-sentence-encoder-large/3"]

    # Import the Universal Sentence Encoder's TF Hub module
    embed = hub.Module(module_url)

    # Reduce logging output.
    tf.logging.set_verbosity(tf.logging.ERROR)

    messages = [
        # Smartphones
        "I like my phone",
        "My phone is not good.",
        "Your cellphone looks great.",
        "Will it snow tomorrow?",
        "Will it xxyaz tomorrow?",
        # Weatherf
        # "Will it snow tomorrow?",
        # "Recently a lot of hurricanes have hit the US",
        # "Global warming is real",

        # Food and health
        # "An apple a day, keeps the doctors away",
        # "Eating strawberries is healthy",
        # "Is paleo better than keto?",
        #
        # # Asking about age
        # "How old are you?",
        # "what is your age?",
    ]

    similarity_input_placeholder = tf.placeholder(tf.string, shape=(None))
    similarity_message_encodings = embed(similarity_input_placeholder)
    with tf.Session(config=config) as session:
      session.run(tf.global_variables_initializer())
      session.run(tf.tables_initializer())
      run_and_plot(session, similarity_input_placeholder, messages,
                   similarity_message_encodings)

