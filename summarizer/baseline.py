import copy

import numpy as np
from numpy.linalg import norm
from sklearn.metrics.pairwise import cosine_similarity

from summarizer.summaries_utils import create_centroid
from summarizer.utils import process_data_dicts


def lamsiyah_sum(multi_docs_lines, doc_features, budget):
    """Implements the summarization approach of Lamsiyah et al.
    (https://www.sciencedirect.com/science/article/pii/S0957417420308952)
    "The proposed method selects relevant sentences according to the final score
    obtained by combining three scores: sentence content relevance, sentence novelty,
    and sentence position scores. The sentence content relevance score is computed as
    the cosine similarity between the centroid embedding vector of the cluster of documents
    and the sentence embedding vectors. The sentence novelty metric is explicitly adopted to
    deal with redundancy. The sentence position metric assumes that the first sentences of a
    document are more relevant to the summary, and it assigns high scores to these
    sentences."


    Args:
        multi_docs_lines (list): List of lists where each sublist represents a document
        as a list of dictionaries.

        doc_features (list): A list containing the feature matrices for each
        document.

        budget (int): Constrains the number of words allowed
        in the summary.

    Returns:
       summary (list): List where each element is a dictionary
       with the selected sentence and its information.
    """
    # Create the unsupervised cluster centroid
    centroid = create_centroid(
        multi_docs_lines, doc_features, centroid_type="unsupervised"
    )
    centroid_aux = copy.deepcopy(centroid)

    # Check for duplicates or for sentences that exceed the summary
    # budget and remove them
    (multi_docs_lines, doc_features, build_summary) = process_data_dicts(
        multi_docs_lines, doc_features, budget
    )

    # Create a sentences_order list
    sentences_order = []
    for text in multi_docs_lines:
        sentences_order.append(np.arange(1, len(text) + 1))

    # Docs lengths
    docs_length = np.array([len(doc) for doc in multi_docs_lines])

    # Build the summary only if there are sentences in the cluster
    # that satisfy the budget constraint
    if build_summary:
        # Preprocess the correct sentences list and the correct features matrix
        # (Flattened to 1D)
        D_sentences = [
            item for sublist in multi_docs_lines for item in sublist
        ]
        try:
            D_features = np.vstack(doc_features)
        except Exception as e:
            print(e)

        # Sentence content relevance scores
        # Compute the similarity between the sentences and the centroid_aux
        score_relevance = np.dot(D_features, centroid_aux) / (
            norm(D_features, axis=1) * norm(centroid_aux)
        )

        sents_cos_sim = cosine_similarity(D_features, D_features)
        score_novelty = []
        tau = 0.95  # best values are between [0.5,0.95]

        # Sentence novelty scores
        if len(D_features) > 1:
            for i in range(len(D_features)):
                # Create a mask to exclude the entry
                mask = np.ones_like(sents_cos_sim[i], dtype=bool)
                mask[i] = False
                if np.max(sents_cos_sim[i][mask]) < tau:
                    score_novelty.append(1)
                elif (np.max(sents_cos_sim[i][mask]) > tau) and (
                    score_relevance[i]
                    > score_relevance[np.argmax(sents_cos_sim[i][mask])]
                ):
                    score_novelty.append(1)
                else:
                    score_novelty.append(1 - np.max(sents_cos_sim[i][mask]))
        else:
            score_novelty.append(0)

        score_novelty = np.array(score_novelty)

        # Sentence position scores
        score_position = []
        for i, doc_len in enumerate(docs_length):
            values = (-sentences_order[i]) / ((doc_len) ** (1 / 3))
            exp_values = np.exp(values)
            score_values = np.maximum(
                np.repeat(0.5, exp_values.shape[0]), exp_values
            )
            score_position.append(score_values)
        score_position = np.array(
            [item for subitem in score_position for item in subitem]
        )

        # Final score
        # alpha + beta + gamma = 1. alpha,beta,gamma in [0,1]

        # Hyperparameters obtained by the authors through grid search
        alpha = 0.6
        beta = 0.2
        gamma = 0.2

        # Combine the three scores to obtain the final
        # scores
        final_score = (
            alpha * score_relevance
            + beta * score_novelty
            + gamma * score_position
        )

        # Generate a summary, according to the final score
        # vector
        top_indices = np.argsort(final_score)[::-1]
        n_words = 0
        summary = []

        for idx in top_indices:
            if D_sentences[idx]["n_words"] + n_words <= budget:
                summary.append(D_sentences[idx])
                n_words = n_words + D_sentences[idx]["n_words"]
            else:
                break

        if len(summary) == 0:
            print("Something went wrong")
            exit()

    return summary
