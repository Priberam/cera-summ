import copy

import numpy as np
import torch
from nltk.tokenize import sent_tokenize

from summarizer.utils import fetch_original_targets


def sequence_mask(lengths):
    """
    Creates a boolean mask from sequence lengths.
    """
    batch_size = lengths.numel()
    max_len = lengths.max()
    try:
        return (
            torch.arange(0, max_len)
            .type_as(lengths)
            .repeat(batch_size, 1)
            .lt(lengths.unsqueeze(1))
        )
    except Exception as e:
        print(e)


def pad_sentences_order(z_samples, num_positional_embeddings, max_length):
    """Given a list with the sentences order in each cluster, limit
    each order array to 'num_positional_embeddings - 1'. Also, concatenate
    the order arrays that correspond to documents smaller than the
    largest document with padding vectors.

    Args:
        z_samples (list): A list of torch tensor arrays, containing the
        order of each sentence in the clusters

        num_positional_embeddings (int): The number of positional
        embeddings considered by the model

        max_length (int): Length of the largest cluster

    Returns:
        z_padded_samples: A padded list of torch tensor arrays
    """

    # Only allow positional encoding idxs up until
    # num_positional_embeddings-1. The idx corresponding to
    # num_positional_embeddings-1 identifies sentences that exceed the
    # average number of sentences per doc in the whole training data
    z_samples_new = []
    for order in z_samples:
        if torch.any(order > num_positional_embeddings - 1, dim=0):
            order[order > num_positional_embeddings - 1] = (
                num_positional_embeddings - 1
            )
        z_samples_new.append(order)

    # Pad z_samples_new using a pad_index = num_positional_embeddings
    z_padded_samples = [
        torch.cat(
            [
                z,
                torch.full(
                    (max_length - len(z),),
                    num_positional_embeddings,
                ),
            ]
        )
        for z in z_samples_new
    ]

    return z_padded_samples


def docs_no_sentences(cluster_features, order):
    """Compute an array that saves the size of
    each document in a cluster

    Args:
        cluster_features (np.array): A matrix containing the sentence
        embeddings of all the sentences in the cluster

        order (np.array): A numpy array containing the order of the
        sentences in the cluster

    Returns:
        docs_lenght (list): A list containing the size of each document in
        a cluster

        len(zero_list) (int): The total number of documents in a cluster
    """
    zero_list = find_zero_indexes(order)

    docs_length = []
    for z in range(len(zero_list)):
        if len(zero_list) > 1:
            if z + 1 < len(zero_list):
                first_zero = zero_list[z]
                second_zero = zero_list[z + 1]
                docs_length.append(
                    len(cluster_features[first_zero:second_zero])
                )
            else:
                first_zero = zero_list[z]
                docs_length.append(len(cluster_features[first_zero:]))
        else:
            first_zero = zero_list[z]
            docs_length.append(len(cluster_features[first_zero:]))

    return docs_length, len(zero_list)


def prepare_dataloader(
    datapoint,
    partition,
    budget=None,
    embeddings_model=None,
    dataset_name="",
):
    output_dict = {}

    target = datapoint["target"]
    multi_docs_lines = datapoint["multi_docs_lines"]
    multi_docs = datapoint["multi_docs"]

    if dataset_name == "CrossSum":
        if partition != "train":
            en_target = datapoint["en_target"]
            en_multi_docs_lines = datapoint["en_multi_docs_lines"]
            original_targets = datapoint["original_targets"]
        langs = datapoint["langs"]
        # Each CrossSum datapoint target is a list with the target
        # for each document
        target_list = copy.deepcopy(target)

    # Each Multi-News or WCEP10 datapoint target is a string
    elif dataset_name == "MultiNews" or dataset_name == "WCEP10":
        target_list = copy.deepcopy([target])

    # Each Multi-News or WCEP10 datapoint target is a string
    elif dataset_name == "DUC2004" or dataset_name == "TAC2008":
        target_list = copy.deepcopy(target)

    else:
        raise NotImplementedError

    # Skip iteration if the cluster is empty
    if dataset_name == "CrossSum" and partition != "train":
        if len(en_multi_docs_lines) <= 0:
            print("Skipped a cluster due to empty multi_docs_lines")
            return output_dict
    else:
        if len(multi_docs_lines) <= 0:
            print("Skipped a cluster due to empty multi_docs_lines")
            return output_dict

    # Vectorize every sentence in the documents
    doc_features = []
    for text in multi_docs_lines:
        # Contextual embeddings
        features = embeddings_model.encode(text)
        doc_features.append(features)

    # Compute the target summary embeddings
    # if considering the train partition
    if dataset_name == "CrossSum" and partition == "train":
        original_targets = fetch_original_targets(
            target_list,
            langs,
            budget,
        )

    if dataset_name == "CrossSum":
        # Fetch the CrossSum data target embeddings
        target_features = [embeddings_model.encode(original_targets)]

    # Fetch the Multi-News data target embeddings
    else:
        target_features = []
        for iter_target in target_list:
            target_sents = sent_tokenize(iter_target)
            fts = embeddings_model.encode(target_sents)
            target_features.append(fts)

    # Build the gold centroid
    gold_centroid = np.zeros(doc_features[0].shape[1])
    n_samples = 0
    for features in target_features:
        gold_centroid += features.sum(0)
        n_samples += features.shape[0]
    gold_centroid = np.divide(gold_centroid, n_samples)

    # Compute the sentences order
    if dataset_name == "CrossSum" and partition != "train":
        sentences_order = []
        for text in en_multi_docs_lines:
            sentences_order.append(np.arange(len(text)))
        order = np.concatenate([sublist for sublist in sentences_order])
    else:
        sentences_order = []
        for text in multi_docs_lines:
            sentences_order.append(np.arange(len(text)))
        order = np.concatenate([sublist for sublist in sentences_order])

    # Stack the document features
    stacked_doc_features = np.vstack(doc_features)

    # Documents length and number of documents
    docs_length, no_docs = docs_no_sentences(stacked_doc_features, order)

    # Update the dictionary
    output_dict["doc_features"] = doc_features
    output_dict["gold_centroid"] = gold_centroid
    output_dict["order"] = order
    output_dict["num_docs"] = no_docs
    output_dict["docs_length"] = docs_length
    output_dict["multi_docs"] = multi_docs
    output_dict["multi_docs_lines"] = multi_docs_lines
    output_dict["target"] = target

    if dataset_name == "CrossSum":
        output_dict["langs"] = langs
        if partition != "train":
            output_dict["en_multi_docs_lines"] = en_multi_docs_lines
            output_dict["en_target"] = en_target

    return output_dict


def find_zero_indexes(arr):
    """
    Return the indices where an array has entries equal to zero
    """
    return np.where(arr == 0)[0]
