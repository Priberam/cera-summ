import torch
import torch.nn.functional as F


def separate_per_document(clusters_fts, order, docs_lens):
    """Given feature vectors and order arrays in a cluster,
    separate this data per document

    Args:
        clusters_fts (tensor): Torch tensor containing all the sentence
        embeddings for a cluster

        order (tensor): Torch tensor containing the order of the sentences in a
        cluster

        docs_lens (tensor): Torch tensor containing the lengths of the
        documents in a cluster

    Returns:
        new_data (list): A list of tensors, where each element is the
        features of a document in the cluster

        new_order (list): A list of tensors, where each element is the
        order of the sentences in a document in the cluster
    """
    new_data = []
    new_order = []

    for enum, len in enumerate(docs_lens):
        new_data.append(clusters_fts[:len])
        new_order.append(order[:len])

        clusters_fts = clusters_fts[len:]
        order = order[len:]

    return new_data, new_order


def cosine_loss(output, target):
    """Given two tensors compute the cosine distance
    loss between them

    Args:
        output (tensor): Centroids predicted by the model (batch,embed_dim)
        target (tensor): Centroids from the gold summary (batch,embed_dim)

    Returns:
        loss (tensor): Cosine distance between the input tensors (batch,)
    """

    # Compute the cosine similarity between the normalized tensors
    cos_similarity = F.cosine_similarity(output, target, dim=1)

    # Calculate the loss as 1 - cosine similarity
    loss = 1 - cos_similarity

    # Take the mean of the loss
    loss = torch.mean(loss)

    return loss
