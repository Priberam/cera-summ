import copy

import numpy as np
from numpy.linalg import norm

from summarizer.summaries_utils import fetch_best_sentence, summarizer
from summarizer.utils import process_data_dicts


def top_ranked(n, doc, multi_docs_lines, doc_features, features, scores):
    """Remove the sentences and their corresponding feature vectors from
    multi_docs_lines and doc_features respectively, leaving only the 'n'
    top ranked sentences and their corresponding feature vectors.

    Args:
        n (int): The number of 'n' sentences to consider from
            each document.

        doc (int): Document index.

        multi_docs_lines (list): List of lists where each sublist represents a document
        as a list of dictionaries.

        doc_features (list): A list containing the feature matrices for each
        document.

        features (numpy array): Matrix containing the feature vectors of the
            sentences in a document.

        scores (list): Score of each sentence in a document.

    Returns:

        multi_docs_lines (list of lists of str): List containing a
            pre-selection of sentences from a cluster per document

        doc_features (list of numpy arrays): List of matrices containing the
            feature vectors of the pre-selected candidate sentences per
            document
    """
    # Sentences indexes
    sentences_idxs = list(np.arange(features.shape[0]))

    # Compute the indexes of the n top ranked sentences
    if len(multi_docs_lines[doc]) > n:
        top_n_idxs = list(np.argpartition(scores, -n)[-n:])
    else:
        top_n_idxs = copy.deepcopy(sentences_idxs)

    # Keep only the top ranked sentences
    multi_docs_lines[doc] = [multi_docs_lines[doc][i] for i in top_n_idxs]
    doc_features[doc] = doc_features[doc][top_n_idxs]

    return multi_docs_lines, doc_features


def sentence_pre_selection(
    n, sentences, multi_docs_lines, doc_features, centroid
):
    """Pre-selects the sentences from each file in a
    cluster that should be considered has candidates
    for the summary.
    Types of pre-selection:

    "all" - Pre-selects all the sentences in the documents;

    "n_first" - Pre-selects the N first sentences of
        each document;

    "n_best" - Pre-selects the N sentences from each
        document which are more similar to the centroid.

    Args:
        n (int): The number of N sentences to consider from
            each document.

        sentences (str): Selects the type of pre-selection.

        multi_docs_lines (list): List of lists where each sublist represents a document
        as a list of dictionaries.

        doc_features (list): A list containing the feature matrices for each
        document.

        centroid (numpy array): Cluster centroid.

    Returns:

        multi_docs_lines (list): List of lists where each sublist represents a document
        as a list of dictionaries. There is a sentence-level pre-selection.

        doc_features (list): A list containing the feature matrices for each
        document. There is a sentence-level pre-selection.
    """
    if sentences == "all":
        pass

    elif sentences == "n_first":
        for i in range(len(multi_docs_lines)):
            multi_docs_lines[i] = multi_docs_lines[i][:n]
            doc_features[i] = doc_features[i][:n]

    elif sentences == "n_best":
        # Iterate through the features of each document
        for doc, features in enumerate(doc_features):
            # Compute the similarity between the sentences in
            # each document and the centroid
            output = np.dot(features, centroid) / (
                norm(features, axis=1) * norm(centroid)
            )

            # Update multi_docs_lines and doc_features
            multi_docs_lines, doc_features = top_ranked(
                n=n,
                doc=doc,
                multi_docs_lines=multi_docs_lines,
                doc_features=doc_features,
                features=features,
                scores=output,
            )

    return multi_docs_lines, doc_features


def sentence_selection(
    multi_docs_lines,
    doc_features,
    centroid,
    sentences="n_first",  # Best hyperparmeter found through Grid Search
    n=9,  # Best hyperparmeter found through Grid Search
    beam_width=5,  # Best hyperparmeter found through Grid Search
    counter_limit=9,  # Best hyperparmeter found through Grid Search
    budget=100,  # 100 for TAC2008, DUC2004, and CrossSum. 50 for WCEP-10. 230 for Multi-News
    R=False,  # Redundacy filter flag (set to false by default since redundancy is already mitigated)
    r=0.6,  # Similarity threshold for avoiding redundancy
    alpha=0,  # Trade-off between sentence length and candidate summary similarity to the centroid (default=0)
):
    """Implements the summarization approach of Gholipour Ghalandari, D. (2017) (https://doi.org/10.18653/v1/W17-4511).
        with Beam Search + Greedy Search.

    - The process begins by processing 'multi_docs_lines' and 'doc_features',
    removing duplicate sentences and sentences that surpass the 'budget'
    ('process_data_dicts' function).

    - If all the sentences are removed, the summary will be the sentence that is
     most similar to the centroid ('fetch_best_sentence' function)

    - There is also the possiblity of preselecting 'n' sentences from each document
     ('sentence_pre_selection' function)

    The summarization algorithm ('summarizer' function):

    - The beam search initiates by selecting the top 'beam_width' sentences with the highest
    similarity scores with the centroid. In each subsequent iteration, the algorithm finds the highest-scoring
    'beam_width' sentences on each beam, generating a total of 'beam_width'^2 candidates.
    Among these candidates, only the highest-ranked 'beam_width' sentences are retained. Suppose any of these
    sentences exceed the specified 'budget' length for the summary. In that case, we preserve the corresponding
    previous state, and no further exploration is conducted on that beam. The beam search concludes when all
    candidate beams have exceeded the 'budget' or when no more sentences are available.
    To exhaust the specified 'budget' and improve results, we add a greedy search of sentences that are allowed
    within the word budget. The top-scoring 'beam_width' states from the beam search are used as starting points
    for this greedy search. Then, for each state, we greedily select the highest scoring sentence that does not
    exceed the 'budget' among the top 'counter_budget' ranked sentences. This process
    iterates until either all of the top 'counter_budget' ranked sentences would exceed the 'budget' or there are no further
    sentences left for consideration.

    Args:
        multi_docs_lines (list): List of lists where each sublist represents a document
        as a list of dictionaries.

        doc_features (list): A list containing the feature matrices for each
        document.

        centroid (numpy array): Centroid of the feature vectors in a cluster.

        sentences (str): Constrains the candidate sentences for the summary.

        n (int): The number of 'n' sentences to consider from each document.

        beam_width (int): Controls the number of states selected from the candidate
        states to further generate more states.

        counter_limit (int): Number of iterations of the 'greedy algorithm'
          allowed after finding a sentence that cannot fit in the summary.

        budget (int): Constrains the number of words allowed
        in the summary.

        R (bool): Flag for the Redundancy filter.

        r (float): Similarity threshold for avoiding redundancy.

        alpha (float): Regularizes the cosine similarity and the number of
            words for the summaries.

    Returns:
       summary (list): List where each element is a dictionary
       with the selected sentence and its information.
    """

    # Create a copy of the centroid so that the original object doesn't
    # get modified while building the summaries
    centroid_aux = copy.deepcopy(centroid)

    # Save a copy of doc_features before changing it
    aux_doc_features = copy.deepcopy(doc_features)

    # Save a copy of the dictionary version of multi_docs_lines before
    # processing the data for the algorithm
    aux_multi_docs_lines = copy.deepcopy(multi_docs_lines)

    # Check for duplicates and for sentences that exceed the summary
    # budget and remove them
    (multi_docs_lines, doc_features, build_summary) = process_data_dicts(
        multi_docs_lines, doc_features, budget
    )

    # Build the summary only if there are sentences in the cluster
    # that satisfy the budget constraint
    if build_summary:
        # Sentences Pre-Selection
        multi_docs_lines, doc_features = sentence_pre_selection(
            n=n,
            sentences=sentences,
            multi_docs_lines=multi_docs_lines,
            doc_features=doc_features,
            centroid=centroid_aux,
        )

        # Preprocess the correct sentences list and the correct features matrix
        # (Flattened to 1D)
        D_sentences = [
            item for sublist in multi_docs_lines for item in sublist
        ]
        try:
            D_features = np.vstack(doc_features)
        except Exception as e:
            print(e)

        # Create copies of D_sentences, and D_features so that
        # the original objects don't get modified while building the summaries
        D_sentences_aux = copy.deepcopy(D_sentences)
        D_features_aux = copy.deepcopy(D_features)

        # Create a indices_aux list to help in the filtering of
        # D_features (filtered_D_features)
        indices_aux = np.arange(D_features.shape[0])

        # Compute the similarity between the sentences and the new centroid_aux
        output = np.dot(D_features_aux, centroid_aux) / (
            norm(D_features_aux, axis=1) * norm(centroid_aux)
        )

        # Generate a summary
        summary = summarizer(
            beam_width,
            budget,
            centroid_aux,
            D_features,
            D_sentences_aux,
            D_features_aux,
            output,
            indices_aux,
            alpha,
            counter_limit,
            R,
            r,
        )

    else:
        # If it was not possible to build a summary, return the
        # sentence that is most similar to the cluster centroid
        summary = fetch_best_sentence(
            centroid_aux, aux_multi_docs_lines, aux_doc_features
        )

    return summary
