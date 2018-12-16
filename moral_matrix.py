"""
Data and utilities related to the moral matrix weighting scheme introduced in TRM.
"""
import copy

moral_matrix = {
    "weights": {
        "liberal": {
            "care:harm": 4,
            "liberty:oppression": 3,
            "fairness:cheating": 1,
            "loyalty:betrayal": 1.0 / 3,
            "authority:subversion": 1.0 / 3,
            "sanctity:degradation": 1.0 / 3
        },
        "conservative": {
            "care:harm": 1.0 / 3,
            "liberty:oppression": 5,
            "fairness:cheating": 1,
            "loyalty:betrayal": 1.0 / 3,
            "authority:subversion": 1.0 / 3,
            "sanctity:degradation": 1.0 / 3
        },
        "libertarian": {
            "care:harm": 1,
            "liberty:oppression": 1,
            "fairness:cheating": 1,
            "loyalty:betrayal": 1,
            "authority:subversion": 1,
            "sanctity:degradation": 1
        }
    },
    "topic_seeds": {
        "care:harm": [],
        "liberty:oppression": [],
        "fairness:cheating": [],
        "loyalty:betrayal": [],
        "authority:subversion": [],
        "sanctity:degradation": []
    }
}


def normalize_moral_matrix_weights():
    """
    Normalizes moral matrix weights per political affiliation.
    :return:
    """
    normalized_weights = copy.deepcopy(moral_matrix["weights"])

    for affiliation in normalized_weights:
        weight_sum = sum(list(normalized_weights[affiliation].values()))
        for moral_value in normalized_weights[affiliation]:
            normalized_weights[affiliation][moral_value] /= weight_sum

    return normalized_weights
