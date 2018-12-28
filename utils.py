import argparse
import logging
import os
import sys
from typing import Tuple
import tensorflow as tf
from tensorflow.python.client import session as tf_session
import tensorflow_hub as hub
from sklearn.utils.fixes import signature
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import itertools


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

    # todo set inter_op_parallelism_threads.
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
    parser.add_argument("-e", dest="model_cache_directory", type=str)
    args = parser.parse_args()

    return args


def plot_precision_recall_curve(y_test: np.ndarray, y_pred: np.ndarray):
        """
        Plots precision-recall curve.
        :param y_test:
        :param y_pred:
        :return:
        """
        average_precision = average_precision_score(y_test, y_pred)
        precision, recall, _ = precision_recall_curve(y_test, y_pred)

        step_kwargs = (
            {'step': 'post'}
            if 'step' in signature(plt.fill_between).parameters
            else {}
        )
        plt.step(recall, precision, color='b', alpha=0.2, where='post')
        plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)

        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(average_precision))
        plt.show()


def plot_roc_curve(y_test, y_score, n_classes):
    """
    Plot ROC curve.
    :param y_test:
    :param y_score:
    :param n_classes:
    :return:
    """

    y_test = np.asarray([y_test, np.abs(1 - y_test)]).T
    y_score = np.asarray([y_score, np.abs(1 - y_score)]).T

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Plot.
    plt.figure()
    lw = 2
    plt.plot(fpr[1], tpr[1], color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[1])
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()


def plot_confusion_matrix(
        y_test: np.ndarray,
        y_pred: np.ndarray,
        classes,
        normalize=False,
        title='Confusion matrix',
        cmap=plt.cm.Blues
):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    # Compute confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    np.set_printoptions(precision=2)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j, i, format(cm[i, j], fmt),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black"
    )

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()
