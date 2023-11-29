import copy

import numpy as np
import torch
from numpy.linalg import norm

from centroid_attention.datasets_utils import (
    docs_no_sentences,
    pad_sentences_order,
    sequence_mask,
)


def create_centroid(
    multi_docs_lines,
    doc_features,
    centroid_type="unsupervised",
    centroid_model=None,
    target_features=None,
    device=None,
):
    """Create the cluster centroid according to the user selection for the
    'centroid_type'.

    - centroid_type = 'gold':  Compute the cluster centroid by averaging
    the sentence embeddings of the cluster target summary;
    - centroid_type = 'estimated':  Estimate the cluster centroid by using
    a CeRA or CeRAI model;
    - centroid_type = 'unsupervised': Compute the cluster centroid by averaging
    the sentence embeddings of the documents in the cluster.

    Args:
        multi_docs_lines (list): List of lists containing the documents' sentences.

        doc_features (list): List containing the sentence embeddings for each document.

        centroid_type (str): Select the type of cluster centroid. ('gold', 'estimated', 'unsupervised').

        centroid_model (obj): A CeRA or CeRAI model to estimate the centroid.

        target_features (list): List containing the sentence embeddings for the cluster target summary.

        device (str): Device used by the centroid_model (gpu or cpu).


    Returns:
        centroid (numpy array): Cluster centroid.
    """
    # Build the unsupervised centroid

    # Gold centroid
    if centroid_type == "gold":
        # Build the centroid
        centroid = np.zeros(doc_features[0].shape[1])
        n_samples = 0
        for features in target_features:
            centroid += features.sum(0)
            n_samples += features.shape[0]
        centroid = np.divide(centroid, n_samples)

    # Centroid estimated by CeRA or CeRAI
    elif centroid_type == "estimated":
        # Generate the CeRA or CeRAI model inputs
        (
            unsupervised_centroid,
            x,
            order,
            attention_mask,
            num_docs,
            docs_weights,
        ) = generate_att_model_data(
            centroid_model.hparams.num_positional_embeddings,
            multi_docs_lines,
            doc_features,
        )

        centroid_model = centroid_model.eval()

        # Predict the cluster centroid with the model
        with torch.no_grad():
            centroid = centroid_model(
                x=x.to(device),
                attention_mask=attention_mask.to(device),
                order=order.to(device),
                num_docs=num_docs.to(device),
                docs_weights=docs_weights.to(device),
                clusters_centroids=unsupervised_centroid.to(device),
            )
            centroid = centroid.detach().cpu().numpy().squeeze(0)

    else:
        # Build the unsupervised centroid
        centroid = np.zeros(doc_features[0].shape[1])
        n_samples = 0
        for features in doc_features:
            centroid += features.sum(0)
            n_samples += features.shape[0]

        centroid = np.divide(centroid, n_samples)

    return centroid


def generate_att_model_data(
    num_positional_embeddings, multi_docs_lines, doc_features
):
    """Generate the necessary inputs for the CeRA or CeRAI model
    to estimate the cluster centroid.

    Args:
        num_positional_embeddings (int): Number of positional embeddings used by the attention model.

        multi_docs_lines (list): List of lists containing the documents' sentences.

        doc_features (list): List containing the sentence embeddings for each document.

    Returns:
        unsupervised_centroid (torch tensor):  The unsupervised cluster centroid.

        x (numpy array): A matrix storing all of the sentence embeddings from the documents.

        order (numpy array): Contains the order of the senteces in each document.

        attention_mask (torch tensor): Attention mask over the cluster data (ignore padded data).

        no_docs (int): Number of documents in a cluster.

        docs_weights (torch tensor): Weight to assign to each sentence embedding. A sentence
        weight depends on its document's length (number of sentences in the document).
    """

    # Build the unsupervised centroid of the cluster
    unsupervised_centroid = np.zeros(doc_features[0].shape[1])
    n_samples = 0
    for features in doc_features:
        unsupervised_centroid += features.sum(0)
        n_samples += features.shape[0]

    unsupervised_centroid = np.divide(unsupervised_centroid, n_samples)
    unsupervised_centroid = (
        torch.from_numpy(unsupervised_centroid).unsqueeze(0).float()
    )

    # Stack the sentence embeddings
    x = np.vstack(doc_features)

    # Create an attention mask for the cluster
    lengths = torch.tensor([x.shape[0]])
    attention_mask = ~sequence_mask(lengths)

    # Create a sentences order list
    sentences_order = []
    for text in multi_docs_lines:
        sentences_order.append(np.arange(len(text)))
    order = np.concatenate([sublist for sublist in sentences_order])

    # Compute the number of documents and their lengths
    docs_length, no_docs = docs_no_sentences(doc_features, order)

    # Pad the sentences order and convert it to torch tensor
    order = torch.tensor(order).unsqueeze(0)
    order = torch.stack(
        pad_sentences_order(
            order,
            num_positional_embeddings,
            max(lengths),
        )
    )
    # Number of documents
    no_docs = torch.from_numpy(np.array(no_docs)).unsqueeze(0)

    # Documents lengths
    docs_length = np.array([len(doc) for doc in multi_docs_lines])

    # Weights
    docs_weights = (
        torch.tensor(np.repeat(np.divide(1, docs_length), docs_length))
        .float()
        .unsqueeze(0)
    )

    # Documents lengths to torch tensor
    docs_length = torch.tensor(docs_length).unsqueeze(0)

    # Documents' sentence embeddings to torch tensor
    x = torch.from_numpy(x).unsqueeze(0)

    return (
        unsupervised_centroid,
        x,
        order,
        attention_mask,
        no_docs,
        docs_weights,
    )


def fetch_best_sentence(centroid, multi_docs_lines, doc_features):
    """Fetch the sentence which is more similar to the centroid and
    choose it as the cluster's summary.

    Args:
        centroid (numpy array): Centroid of the feature vectors in a cluster.

        multi_docs_lines (list): List of lists where each sublist represents a document
        as a list of dictionaries.

        doc_features (list): A list containing the feature matrices for each
        document.

    Returns:
       summary (str): Sentence selected as the summary.
    """
    D_sentences = [item for sublist in multi_docs_lines for item in sublist]
    D_features = np.vstack(doc_features)

    # Compute the similarity between the sentences in
    # each document and the centroid
    output = np.dot(D_features, centroid) / (
        norm(D_features, axis=1) * norm(centroid)
    )

    best_sent_idx = np.argmax(output)

    summary = D_sentences[best_sent_idx]

    return summary


def redundancy_filter(summary_features, D_features_aux, output, r):
    """Applies the redundancy filter. The features present in
    "D_features_aux" are compared to the features of the
    sentences already in the summary. If the candidate
    sentences are too similar to sentences in the summary,
    the similarity of these candidate sentences with respect
    to the centroid are zeroed in the "output" vector so that
    they will no longer be selected for the predicted summary.

    Args:
        summary_features (numpy array): Matrix containing the feature vectors
            of the sentences present in the summary

        D_features_aux (numpy array): Matrix containing the feature vectors
            of the candidate sentences

        output (numpy array): Vector with the similarities of all the candidate
            sentences with respect to the centroid

        r (float): Similarity threshold value for the boolean masking procedure

    Returns:
        masked_output (numpy array): Masked vector with the similarities of
        all the candidate sentences with respect to the centroid
    """

    masked_output = output

    summary_features_rep = np.repeat(
        summary_features,
        np.array([D_features_aux.shape[0]] * summary_features.shape[0]),
        axis=0,
    )

    D_features_tile = np.tile(D_features_aux, (summary_features.shape[0], 1))

    # Similarities of the candidate sentences with respect to
    # the sentences already in the summary
    candidate_sims = np.sum(D_features_tile * summary_features_rep, axis=1) / (
        norm(D_features_tile, axis=1) * norm(summary_features_rep, axis=1)
    )

    # Create a boolean mask that marks as "True" the indexes of
    # the candidate sentences that are similar to the
    # ones already in the summary
    bool_mask = candidate_sims > r
    bool_mask = bool_mask.reshape(
        D_features_aux.shape[0], summary_features.shape[0]
    )
    bool_mask = np.any(bool_mask, axis=1)

    # Use the boolean mask to zero the similarity of the sentences
    # with respect to the centroid if they are similar to the ones
    # already in the summary
    masked_output[bool_mask] = 0

    return masked_output


def summarizer(
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
):
    """Build a summary for a cluster.
        Start of with an 'initial_state' that has available
        all of the sentences in the cluster. The 'initial_state'
        originates 'beam_width' different candidate states which hold
        'beam_width' distinct summaries. After this, each one of this states,
        which are now the 'current_states' will be the root to other states
        ('candidate_states') so, in a sense, given the sentences still
        available for each current state, new states are created, each one
        with a respective summary associated. From these new states, only the
        top 'beam_width' states that have summaries which do not exceed the
        budget constraint will be used to further generate more
        'candidate_states' and, consequently, augment the summaries. When the
        summary of a state in the 'current_states' has more than 'budget'
        words, the antecedent 'previous_states' are appended to a
        'final_states' list. The generation algorithm is stopped when all the
        states in the 'current_states' hold summaries that exceed the budget
        limit, resulting in no 'candidate_states', which is the stop condition
        for the algorithm. When that happens, the best 'beam_width' states are
        selected from the 'final_states' list. For each of these states, a
        "Gready Search" algorithm is run to check for sentences that can still
        fit the summaries of these states, in an attempt to increase the score
        of the summaries in relation to the centroid of the cluster. The best
        state is selected from the states that result from this "Greedy Search".
        The best state will hold the summary that is most similar to the
        centroid (the budget constraint is already assured by the
        "Greedy Search").

    Args:
        beam_width (int): Number of states selected from the 'candidate_states'
          to further generate more states.

        budget (int): Budget limit for the summaries.

        centroid_aux (np.array): Cluster centroid.

        D_features (list): List containing all the feature vectors
          for the sentences in the cluster.

        D_sentences_aux (list): List containing all the sentences
            in a cluster.

        D_features_aux (list): Copy of D_features. D_features_aux
          can be edited during summary updates.

        output (np.array): Vector containing the similarity of current
          summary union with each candidate sentence.

        indices_aux (list): To keep track of the indices of D_features
          that correspond to sentences that have not yet been analyzed.

        alpha (float): Regularizer for the score function responsible for
            the selection of sentences for the summaries.

        counter_limit (int): Number of iterations of the "Greedy search"
          algorithm allowed after finding a sentence that cannot fit in
          the summary.

        R (bool): Flag for the Redundancy filter.

        r (float): Similarity threshold for avoiding redundancy.

    Returns:
        summary (list): The summary generated for a specific cluster.
    """

    # Length of all the available first sentences for the summaries
    sentences_length = np.array([item["n_words"] for item in D_sentences_aux])

    # Scores to consider for the selection of candidate sentences
    scores = output / np.power(sentences_length, alpha)

    # Initialize the first state for the Beam Search
    initial_state = {}
    initial_state["summary"] = []
    initial_state["summary_features"] = np.array([])
    initial_state["D_sentences_aux"] = copy.deepcopy(D_sentences_aux)
    initial_state["D_features_aux"] = D_features_aux
    initial_state["output"] = output
    initial_state["indices_aux"] = indices_aux
    initial_state["similarity"] = 0.0
    initial_state["score"] = 0.0
    initial_state["n_words"] = 0

    current_states = [initial_state]
    n_words_list = [0 for _ in range(beam_width)]
    less_than_limit = False

    # List to store the previous states that originated states
    # which have summaries that exceeded the  budget limit
    final_states = []
    final_scores = []

    # Generate the candidate states from the root current
    # states that have summaries which do not exceed the budget limit.
    # Select the top 'beam_width' candidates to update the current states and
    # always keep track of the previous states so that if any state in the
    # current states holds a summary that surpasses the budget limit the
    # previous states which precede that state can be stored in the
    # 'final_states' list. No more states are generated when all the states
    # in the current states hold a summary that exceeds the 'n_words' limit
    stop = False
    while not stop:
        # Generate candidate solutions
        candidate_states = []
        candidate_scores = []
        for state in current_states:
            if state["n_words"] < budget:
                candidates, scores = generate_candidates(
                    state["summary"],
                    state["summary_features"],
                    state["D_sentences_aux"],
                    D_features,
                    state["D_features_aux"],
                    state["output"],
                    scores,
                    centroid_aux,
                    state["indices_aux"],
                    state["n_words"],
                    alpha,
                    beam_width,
                    R,
                    r,
                )

                candidate_states.extend(candidates)
                candidate_scores.extend(scores)

        # Save previous states
        prev_state = [state for state in current_states]
        prev_scores = [state["score"] for state in current_states]

        # If candidate states were generated, select the top
        # 'beam_width' candidate states and update the current
        # states. Also, update the 'n_words_list' which contain the
        # 'n_words' of each summary in the current states
        if len(candidate_states) > 0:
            top_indices = np.argsort(candidate_scores)[-beam_width:]
            current_states = [candidate_states[i] for i in top_indices]
            current_scores = [candidate_scores[i] for i in top_indices]

            n_words_list = [
                candidate_states[i]["n_words"] for i in top_indices
            ]

        # If no candidate states were generated stop the Beam Search
        else:
            break

        # Check wether any state in the current states contains a summary
        # that exceeds the budget limit or not. If any summary does exceed
        # the limit, store the the previous states that preceded this ocurrence
        # in a 'final_states' list
        for words in n_words_list:
            if words > budget:
                final_states.extend(prev_state)
                final_scores.extend(prev_scores)
                break
            elif words == budget:
                final_states.extend(current_states)
                final_scores.extend(current_scores)
                break

    final_indices = []

    # Search in the 'final_states' for the top 'beam_width' states
    # i.e. the states that hold the best summaries within the budget limit.
    # Store these states in the 'best_final_states' list
    if len(final_scores) == 0:
        final_states.extend(current_states)
        final_scores.extend(current_scores)
        less_than_limit = True

    for idx in np.argsort(final_scores).tolist():
        if final_states[idx]["n_words"] <= budget:
            final_indices.append(idx)
    best_indices = final_indices[-beam_width:]
    best_final_states = [final_states[i] for i in best_indices]

    best_states = []
    best_scores = []
    # For each state in the 'best_final_states' perform a "Greedy Search"
    # to check for sentences that could still be appended to these states'
    # summaries without exceeding the budget limit. This is performed in an
    # attempt to further improve on the summaries score in relation to the
    # centroid
    for state in best_final_states:
        if not less_than_limit and counter_limit > 0:
            candidate, score = greedy_search(
                state,
                counter_limit,
                budget,
                D_features,
                centroid_aux,
                alpha,
                R,
                r,
            )
        else:
            candidate = state
            score = state["score"]

        best_states.append(candidate)
        best_scores.append(score)

    try:
        best_state = best_states[np.argmax(best_scores)]
    # To check if everything is alright
    except Exception as e:
        print("FATAL ERROR:" + repr(e))

    best_summary = best_state["summary"]

    return best_summary


def generate_candidates(
    summary,
    summary_features,
    D_sentences_aux,
    D_features,
    D_features_aux,
    output,
    scores,
    centroid_aux,
    indices_aux,
    n_words,
    alpha,
    beam_width,
    R,
    r,
):
    """Given a state, check wether its' summary is still empty
    or not. If it is still empty there is no need to update the
    initial scores for each possible summary as the scores will coincide
    with the score of each sentence in thecluster. If the summary is not empty,
    the scores for each possible summary, given the current summary in the
    state must be computed. The 'scores' list will dictate what sentences will
    be appended to the summary of the given state in parallel, so that the
    different 'beam_width' summaries with the highest scores are generated.

    Args:
        summary (list): Summary of the given state.

        summary_features (np array): Features of the sentences which are
          already in the given state summary (empty array if there is still
          no summary).

        D_sentences_aux (list): List containing all the sentences
            available for the given state.

        D_features (list): List containing all the feature vectors
            for the sentences in the cluster.

        D_features_aux (list): List containing all the feature
            vectors for the sentences available for the given state.

        output (np.array): Vector containing the similarity of the union
          between the summary of the given state with each candidate sentence.

        scores (np.array): Vector containing the score of the possible
          summaries.

        centroid_aux (np.array): Cluster centroid.

        indices_aux (list): To keep track of the indices of D_features
            that correspond to sentences of the given state that have not yet
            been analyzed.

        n_words (int): To keep track of the number of words in the summary
            of the given state.

        alpha (float): Regularizer for the score function responsible for
            the selection of sentences for the summaries.

        beam_width (int): Number of states selected from the 'candidate_states'
          to further generate more states.

        R (bool): Flag for the Redundancy filter.

        r (float): Similarity threshold for avoiding redundancy.


    Returns:
        possible_states (list): List containing the possible states
          originated from the given state

        possible_scores (list): List containing the scores of the
          states originated from the give state.
    """

    # Check if for the given state, there are already sentences in
    # the summary
    if len(summary) > 0:
        # Compute the scores for the possible states' summaries which can
        # be generated from the current sate
        filtered_D_features = D_features[indices_aux]
        D_features_aux = filtered_D_features + summary_features.sum(
            axis=0
        ).reshape(1, summary_features.shape[1])
        summaries_length = np.array(
            [item["n_words"] + n_words for item in D_sentences_aux]
        )
        output = np.dot(D_features_aux, centroid_aux) / (
            norm(D_features_aux, axis=1) * norm(centroid_aux)
        )
        # Redundacy filter
        if R:
            output = redundancy_filter(
                summary_features, filtered_D_features, output, r
            )

        scores = output / np.power(summaries_length, alpha)

        # Rank the possible states by relevance of their summaries
        ind_list = np.argsort(-scores).tolist()

        # Generate the best 'beam_width' possible states from the
        # given state
        possible_states = []
        possible_scores = []
        for i, ind in enumerate(ind_list):
            # Initialize the possible state
            possible_dict = {}
            # Compute the summary_features for this possible state
            possible_dict["summary_features"] = np.concatenate(
                (
                    summary_features,
                    filtered_D_features[ind].reshape(1, -1),
                ),
                axis=0,
            )

            # Finish generating the possible state
            (
                possible_states,
                possible_scores,
            ) = generate_possibilities(
                possible_states,
                possible_scores,
                possible_dict,
                n_words,
                summary,
                D_sentences_aux,
                D_features_aux,
                indices_aux,
                output,
                ind,
                alpha,
            )

            # Check if 'beam_width' possible states have already been
            # generated and break from the loop
            if i == beam_width - 1:
                break

    # If the summary of the given state is still empty
    else:
        # Rank the possible states by relevance of their summaries
        ind_list = np.argsort(-scores).tolist()

        # Generate the best 'beam_width' possible states from the
        # given state
        possible_states = []
        possible_scores = []
        for i, ind in enumerate(ind_list):
            # Initialize the possible state
            possible_dict = {}
            # Compute the summary_features for the possible state
            possible_dict["summary_features"] = np.array([D_features_aux[ind]])

            # Finish generating the possible state
            (
                possible_states,
                possible_scores,
            ) = generate_possibilities(
                possible_states,
                possible_scores,
                possible_dict,
                n_words,
                summary,
                D_sentences_aux,
                D_features_aux,
                indices_aux,
                output,
                ind,
                alpha,
            )

            # Check if 'beam_width' possible states have already been
            # generated and break from the loop
            if i == beam_width - 1:
                break

    return possible_states, possible_scores


def generate_possibilities(
    possible_states,
    possible_scores,
    possible_dict,
    n_words,
    summary,
    D_sentences_aux,
    D_features_aux,
    indices_aux,
    output,
    ind,
    alpha,
):
    """Given a state and the index ('ind') of a sentence available for the
    summary, generate a possible state from this given state.

    Args:
        possible_states (list): List containing the possible states
          originated from the given state.

        possible_scores (list ): List containing the scores of the
          states originated from the give state.

        possible_dict (dict): State generated from the given state.

        n_words (int): To keep track of the number of words in the summary
            of the given state.

        summary (list): Summary of the given state.

        D_sentences_aux (list): List containing all the sentences
            available for the given state.

        D_features_aux (list): List containing all the feature
            vectors for the sentences available for the given state.

        indices_aux (list): To keep track of the indices of D_features
            that correspond to sentences of the given state that have not yet
            been analyzed.

        output (np.array): Vector containing the similarity of the union
            between the summary of the given state with each candidate sentence.

        beam_width (int): Number of states selected from the 'candidate_states'
            to further generate more states.

        ind (int): index of the sentence that was selected for the state that
            is generated from the given state.

        alpha (float): Regularizer for the score function responsible for
            the selection of sentences for the summaries.

    Returns:
        possible_states (list): Updated list containing the possible
          states originated from the given state

        possible_scores (list): Updated list containing the scores of
          the states originated from the give state
    """

    # Copy the 'n_words', the 'summary' list and the
    # 'D_sentences_aux' list from the given state to the
    # newly generated state
    possible_dict["n_words"] = n_words
    possible_dict["summary"] = copy.deepcopy(summary)
    possible_dict["D_sentences_aux"] = copy.deepcopy(D_sentences_aux)

    # Fetch the sentence that makes the summary the most
    # similar to the centroid as possible
    item = possible_dict["D_sentences_aux"].pop(ind)

    # Update the 'n_words' for this new state
    possible_dict["n_words"] += item["n_words"]
    # Update the summmary stored in the new state
    possible_dict["summary"].append(item)

    # Store the similarity of the summary in this new state.
    # Remove the 'similarity' and the 'feature vector' of the appended
    # sentence from "output" and "D_features_aux" respectively,
    # also, update "indices_aux". These updates are stored locally in the
    # new_state.
    possible_dict["similarity"] = output[ind]
    possible_dict["output"] = np.concatenate([output[:ind], output[ind + 1 :]])
    possible_dict["D_features_aux"] = np.concatenate(
        [D_features_aux[:ind], D_features_aux[ind + 1 :]], axis=0
    )
    possible_dict["indices_aux"] = np.concatenate(
        [indices_aux[:ind], indices_aux[ind + 1 :]]
    )
    # Compute the score for the new state
    possible_dict["score"] = score_candidate(possible_dict, alpha)

    # Append the newly generated state to the 'possible_states' list, which
    # is a list that contains new states generated from the given state. Also,
    # update the 'possible_scores' list.
    possible_states.append(possible_dict)
    possible_scores.append(possible_dict["score"])

    return possible_states, possible_scores


def score_candidate(state, alpha):
    """Compute the score for the summary of
    a state

    Args:
        state (dict): Possible new state.

        alpha (float): Regularizes the trade-off between the cosine similarity
            of the state's summary to the centroid_aux and the numer of words
            in the summary.


    Returns:
        score (float): Score for the state's summary.
    """
    score = state["similarity"] / np.power(state["n_words"], alpha)

    return score


def greedy_search(
    state, counter_limit, budget, D_features, centroid_aux, alpha, R, r
):
    """Receives a state found during Beam Search. This function has
    the objective of implementing a "Greedy Search" for the input state
    in an attempt to further improve on the score of it' summary.
    Starting with the input state, a maximum number of "counter_limit"
    iterations take place to try to find the sentence that augments
    the summary's score with respect to the centroid and does not
    cause the budget limit to be exceeded. If that sentence is
    not found, the algorithm stops and the original state and it' score
    are returned. If a sentence is found in "counter_limit" iterations,
    then the state is updated. If after updating the state it' summary
    has not yet exceeded the budget, then the "Greedy Search" runs again, to
    see if it's possible to append more sentences to the summary.

    Args:
        state (dict): A state that includes one of the best summaries.

        counter_limit (int): Number of iterations of the "Greedy search"
          algorithm allowed after finding a sentence that cannot fit in
          the summary.

        budget (int): Budget limit for the summaries.

        D_features (list of np.array): List containing all the feature vectors
          for the sentences in the cluster.

        centroid_aux (np.array): Cluster centroid.

        alpha (float): Regularizer for the score function responsible for
            the selection of sentences for the summaries.

        R (bool): Flag for the Redundancy filter.

        r (float): Similarity threshold for avoiding redundancy.

    Returns:
        state (dict): Updated state if it was possible to append more sentences
        to the summary or the original state as in the function's arguments

        state["score"] (float): The score for the summary of the updated state
          or the original state as in the function's arguments, depending if
          it was possible to append new sentences to the summary
    """

    # Compute the scores for the possible states' summaries which can
    # be generated from the current sate
    summaries_length = np.array(
        [
            item["n_words"] + state["n_words"]
            for item in state["D_sentences_aux"]
        ]
    )

    # If all the possible initial summaries exceed the budget,
    # return the current state
    if np.all(summaries_length > budget):
        return state, state["score"]

    # If there are still sentences available and the input state is not
    # the initial state, then prepare this state to generate more states
    if len(state["D_sentences_aux"]) > 0 and len(state["summary"]) > 0:
        # Compute the scores for the possible states' sumaries which can
        # be generated from the current sate
        filtered_D_features = D_features[state["indices_aux"]]
        try:
            state["D_features_aux"] = filtered_D_features + state[
                "summary_features"
            ].sum(axis=0).reshape(1, state["summary_features"].shape[1])

        except Exception as e:
            print("Exception occurred:" + repr(e))

        state["output"] = np.dot(state["D_features_aux"], centroid_aux) / (
            norm(state["D_features_aux"], axis=1) * norm(centroid_aux)
        )
        # Redundacy filter
        if R:
            state["output"] = redundancy_filter(
                state["summary_features"],
                filtered_D_features,
                state["output"],
                r,
            )

        scores = state["output"] / np.power(summaries_length, alpha)

    else:  # If the starting point for the Greedy Search is the initial state
        scores = state["output"]

    # GREEDY SEARCH
    # Search for the most similar sentence that fulfills the budget
    # constraint and add it to the summary
    count = 0
    update = False
    while count < counter_limit:
        # Check if there are no more sentences to consider for the
        # summary or if the search counter after finding a
        # sentence that exceeds the budget has reached the estabilished
        # threshold or if the summary is complete
        if len(state["D_sentences_aux"]) == 0 or state["n_words"] == budget:
            break

        ind = np.argmax(scores)

        # Fetch the candidate sentence for the summary
        n_words_aux = state["n_words"]
        item = state["D_sentences_aux"].pop(ind)
        n_words_aux += item["n_words"]

        # The sentence will enter the summary if its addition
        # does not result in exceeding the estabilished budget
        if n_words_aux <= budget:
            state["n_words"] = n_words_aux

            # Update the state's 'summary_features'
            if len(state["summary"]) > 0:
                filtered_D_features = D_features[state["indices_aux"]]

                state["summary_features"] = np.concatenate(
                    (
                        state["summary_features"],
                        filtered_D_features[ind].reshape(1, -1),
                    ),
                    axis=0,
                )
            else:
                state["summary_features"] = np.array(
                    [state["D_features_aux"][ind]]
                )

            # Store the 'similarity' of the summary
            # and it' 'score' in the state.
            state["similarity"] = state["output"][ind]
            state["score"] = score_candidate(state, alpha)
            # Append the sentence to the summary and
            # stop the while loop
            state["summary"].append(item)
            update = True

        # If adding the sentence to the summary results in exceeding
        # the budget, one must search if the next most similar sentence
        # satisfies the budget constraint
        else:
            count += 1

        # After analyzing one sentence
        # update the state's 'output','D_features_aux', and
        # 'indices_aux'. Also, update the 'scores'
        state["output"] = np.concatenate(
            [
                state["output"][:ind],
                state["output"][ind + 1 :],
            ]
        )

        scores = np.concatenate(
            [
                scores[:ind],
                scores[ind + 1 :],
            ]
        )
        state["D_features_aux"] = np.concatenate(
            [
                state["D_features_aux"][:ind],
                state["D_features_aux"][ind + 1 :],
            ],
            axis=0,
        )
        # Update "indices_aux" and the ind_list
        state["indices_aux"] = np.concatenate(
            [
                state["indices_aux"][:ind],
                state["indices_aux"][ind + 1 :],
            ]
        )

        # If a new sentence was added, one must compute the
        # updated 'scores' so that in case the state's summary
        # still has space for more words, the "Greedy Search"
        # may run again
        if update:
            # Update the state's 'D_features_aux'
            filtered_D_features = D_features[state["indices_aux"]]
            state["D_features_aux"] = filtered_D_features + state[
                "summary_features"
            ].sum(axis=0).reshape(1, state["summary_features"].shape[1])

            # Update the 'n_words' for the possible summaries
            current_n_words = np.sum(
                [item["n_words"] for item in state["summary"]]
            )
            summaries_length = np.array(
                [
                    item["n_words"] + current_n_words
                    for item in state["D_sentences_aux"]
                ]
            )

            # Update the cosine similarities ('output')
            state["output"] = np.dot(state["D_features_aux"], centroid_aux) / (
                norm(state["D_features_aux"], axis=1) * norm(centroid_aux)
            )
            # Redundacy filter
            if R:
                state["output"] = redundancy_filter(
                    state["summary_features"],
                    filtered_D_features,
                    state["output"],
                    r,
                )

            # Update the 'scores'
            scores = state["output"] / np.power(summaries_length, alpha)

            # Initialize the variables for a new run of the "Greedy Search"
            count = 0
            update = False

    return state, state["score"]
