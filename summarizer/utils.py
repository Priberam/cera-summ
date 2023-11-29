import copy
import os
import re
from pathlib import Path

import numpy as np
import torch
from icu_tokenizer import SentSplitter
from lxml import etree
from nltk.tokenize import RegexpTokenizer, sent_tokenize


def dict_multi_docs_lines(multi_docs_lines, langs=None):
    """Transforms each sublist of multi_docs_lines
    into a list of dictionaries, where each dictionary
    stores a sentence in a document and also some
    additional information about the sentence.

    Args:
        multi_docs_lines (list): List of lists containing the documents' sentences.

        langs (list): List containing the ISO639-1 language code for each document.

    Returns:
        multi_docs_lines_new (list): List of lists where each sublist
        represents a document as a list of dictionaries. Each dictionary
        contains a sentence of the document ('sentence' key),
        the number of words of the sentence ('n_words' key), the number of
        the document from where the sentence comes from ('doc_num' key),
        and the sentence number in the document ('sentence_num' key). It can
        also include the language in which the document i written ('language' key),
        for the case of the CrossSum data.
    """

    multi_docs_lines_new = []
    word_count_tokenizer = RegexpTokenizer(r"\w+")

    if langs is None:
        for doc_num, doc in enumerate(multi_docs_lines):
            # Create the list of dictionaries
            # The list objects are dictionaries containing the sentence, its
            # number of words, the number of the document from where it comes
            # from and the sentence number in the document
            doc_split = []
            for sent_num, elem in enumerate(doc):
                doc_split.append(
                    {
                        "sentence": elem,
                        "doc_num": doc_num,
                        "sent_num": sent_num,
                        "n_words": len(word_count_tokenizer.tokenize(elem)),
                    }
                )
            # Update multi_docs_lines
            multi_docs_lines_new.append(doc_split)

    # Processing for CrossSum data
    else:
        for doc_num, (doc, lang) in enumerate(zip(multi_docs_lines, langs)):
            # Create the list of dictionaries
            # The list objects are dictionaries containing the sentence, its
            # number of words, the number of the document from where it comes
            # from and the sentence number in the document and also the ISO639-1
            # code of the language in which the document is written
            doc_split = []
            for sent_num, elem in enumerate(doc):
                doc_split.append(
                    {
                        "sentence": elem,
                        "doc_num": doc_num,
                        "sent_num": sent_num,
                        "n_words": len(word_count_tokenizer.tokenize(elem)),
                        "language": lang,
                    }
                )

            # Update multi_docs_lines
            multi_docs_lines_new.append(doc_split)

    return multi_docs_lines_new


def process_data_dicts(multi_docs_lines, doc_features, budget):
    """Before employing the summarization algorithm, remove the duplicate
    sentences from the dataset and the sentences that exceed the budget
    for the summary. Also, remove the corresponding sentence embeddings
    from doc_features.

    Args:
        multi_docs_lines (list): List of lists where each sublist
        represents a document as a list of dictionaries. Each dictionary
        contains a sentence of the document ('sentence' key),
        the number of words of the sentence ('n_words' key), the number of
        the document from where the sentence comes from ('doc_num' key),
        and the sentence number in the document ('sentence_num' key).
        It can also include the language in which the document is written
        ('language' key), for the case of the CrossSum data.

        doc_features (list): A list containing the feature matrices for each
        document.

        budget (int): Constrains the number of words allowed
        in the summary.

    Returns:
        new_multi_docs_lines (list): Processed multi_docs_lines.

        doc_features (list): Processed doc_features.

        build_summary (bool): Flag whether it will be possible to
        build a summary or not.

    """

    build_summary = True
    unique_strings = set()
    new_multi_docs_lines = []
    pop_idxs = []
    pop_idxs_doc_fts = []

    # Iterate through each document
    for doc_num, doc in enumerate(multi_docs_lines):
        unique_sublist = []
        # Iterate through each sentence
        for sent_num, elem in enumerate(doc):
            # Add sentence if it is not yet present in the set
            # and if it does not exceed the budget for the summary
            if (elem["sentence"] not in unique_strings) and elem[
                "n_words"
            ] <= budget:
                unique_strings.add(elem["sentence"])
                unique_sublist.append(elem)
            else:
                # Feature to remove from a document
                pop_idxs_doc_fts.append(sent_num)

        # Update new_multi_docs_lines
        if len(unique_sublist) != 0:
            new_multi_docs_lines.append(unique_sublist)
        # Add the index of documents to pop
        else:
            pop_idxs.append(doc_num)

        # Remove the features corresponding to sentences
        # that will not be considered by the summarization
        # algorithm
        for i, del_idx in enumerate(pop_idxs_doc_fts):
            true_idx = del_idx - i
            doc_features[doc_num] = np.concatenate(
                [
                    doc_features[doc_num][:true_idx],
                    doc_features[doc_num][true_idx + 1 :],
                ],
                axis=0,
            )
        pop_idxs_doc_fts.clear()

    # Eliminate empty documents data
    if len(pop_idxs) > 0:
        # Eliminate empty documents from doc_features
        for i, idx in enumerate(pop_idxs):
            true_idx = idx - i
            doc_features.pop(true_idx)

    if len(new_multi_docs_lines) == 0:
        build_summary = False

    return new_multi_docs_lines, doc_features, build_summary


def process_CrossSum(multi_docs_lines, langs, multi_docs, target_list):
    """Process multi_docs_lines for the CrossSum datasets.
    More specifically, eliminate empty sentences.
    Also, remove empty documents.

    Args:
        multi_docs_lines (list): List of lists containing the documents' sentences.

        langs (list): List containing the ISO639-1 code for the language of each document.

        multi_docs (list): List containing the documents.

        target_list (list): List containing the target summary of each document.

    Returns:
        multi_docs_lines (list): Processed multi_docs_lines.

        langs (list): Processed langs.

        multi_docs (list): Processed multi_docs.

        target_list (list): Processed target_list.
    """
    # Count the words in each sentence
    word_count_tokenizer = RegexpTokenizer(r"\w+")
    # Store the indexes of documents to eliminate
    pop_docs = []

    # Update multi_docs_lines
    # (no empty strings)
    for doc_num in range(len(multi_docs_lines)):
        remove_idxs = []
        for iter, sentence in enumerate(multi_docs_lines[doc_num]):
            if len(word_count_tokenizer.tokenize(sentence)) == 0:
                remove_idxs.append(iter)

        for i, idx in enumerate(remove_idxs):
            true_idx = idx - i

            multi_docs_lines[doc_num].pop(true_idx)

        if len(multi_docs_lines[doc_num]) == 0:
            pop_docs.append(doc_num)

    # Eliminate empty objects from the lists. This happens when documents
    # were totally eliminated
    if len(pop_docs) != 0:
        for i, idx in enumerate(pop_docs):
            true_idx = idx - i

            multi_docs_lines.pop(true_idx)
            multi_docs.pop(true_idx)
            langs.pop(true_idx)
            target_list.pop(true_idx)

    return (
        multi_docs_lines,
        langs,
        multi_docs,
        target_list,
    )


def fetch_DUC2004_sentences(dataset_path, path_to_clusters, cluster):
    """Retrieve the sentences in the DUC2004 documents.

    Args:
        dataset_path (str): Path to the DUC2004 dataset.

        path_to_clusters (str): Path to the document clusters.

        cluster (int): Cluster number.

    Returns:
        multi_docs (list): List containing the documents.

        multi_docs_lines (list): List of lists containing the documents' sentences.
    """
    multi_docs = []
    multi_docs_lines = []

    doc_path = Path(
        dataset_path,
        path_to_clusters,
        f"{int(cluster)}",
    )

    # Store the documents
    for i, filename in enumerate(os.listdir(doc_path)):
        word_count_tokenizer = RegexpTokenizer(r"\w+")
        with open(os.path.join(doc_path, filename), "r") as file:
            doc = file.read()

            # Update multi_docs
            multi_docs.append(doc)

            # Tokenization
            text = sent_tokenize(doc)
            text = [
                re.sub(" +", " ", re.sub("\n", " ", x))
                for x in text
                if (len(word_count_tokenizer.tokenize(x)) > 0)
            ]
            if len(text) > 0:
                multi_docs_lines.append(text)

    return multi_docs, multi_docs_lines


def fetch_DUC2004_targets(dataset_path, cluster):
    """Retrieve the target summaries from the DUC2004 dataset.

    Args:
        dataset_path (str): Path to the DUC2004 dataset.

        cluster (int): Cluster number.

    Returns:
        target_list (list): List containing all the target summaries for
        a cluster.
    """
    target_list = []

    ref_path = Path(dataset_path, "reference")

    for i in range(1, 5):
        with open(
            os.path.join(ref_path, f"Task{int(cluster)}_reference{i}.txt"),
            "r",
        ) as file:
            target = file.read()
            text = target.replace("\n", " ").strip()
            target_list.append(text)

    return target_list


def fetch_MultiNews_sentences(cluster):
    """Retrieve the sentences in the Multi-News documents.

    Args:
        cluster (int): Cluster number.

    Returns:
        multi_docs (list): List containing the documents.

        multi_docs_lines (list): List of lists containing the documents' sentences.
    """
    word_count_tokenizer = RegexpTokenizer(r"\w+")
    multi_docs = []
    multi_docs_lines = []
    # Store the documents
    cluster = cluster.split("|||||")
    # Stripping leading and trailing whitespace from each document
    cluster = [document.strip() for document in cluster]

    for doc in cluster:
        # Update multi_docs
        multi_docs.append(doc)

        # Tokenization
        text = sent_tokenize(doc)
        text = [
            re.sub(" +", " ", re.sub("\n", " ", x))
            for x in text
            if (len(word_count_tokenizer.tokenize(x)) > 0)
        ]
        if len(text) > 0:
            multi_docs_lines.append(text)

    return multi_docs, multi_docs_lines


def fetch_WCEP10_sentences(cluster):
    """Retrieve the sentences in the WCEP-10 documents.

    Args:
        cluster (int): Cluster number.

    Returns:
        multi_docs (list): List containing the documents.

        multi_docs_lines (list): List of lists containing the documents' sentences.
    """
    word_count_tokenizer = RegexpTokenizer(r"\w+")
    multi_docs = []
    multi_docs_lines = []

    # Stripping leading and trailing whitespace from each document
    cluster = [document.strip() for document in cluster]

    for doc in cluster:
        # Update multi_docs
        multi_docs.append(doc)

        # Tokenization
        text = sent_tokenize(doc)
        text = [
            re.sub(" +", " ", re.sub("\n", " ", x))
            for x in text
            if (len(word_count_tokenizer.tokenize(x)) > 0)
        ]
        if len(text) > 0:
            multi_docs_lines.append(text)

    return multi_docs, multi_docs_lines


def fetch_TAC2008_sentences(path_to_clusters, cluster):
    """Retrieve the sentences in the TAC2008 dataset.

    Args:
        path_to_clusters (str): Path to the document clusters.

        cluster (int): Cluster number.

    Returns:
        multi_docs (list): List containing the documents.

        multi_docs_lines (list): List of lists containing the documents' sentences.
    """
    multi_docs = []
    multi_docs_lines = []
    word_count_tokenizer = RegexpTokenizer(r"\w+")

    cluster_path = Path(path_to_clusters, cluster)

    cluster = read_cluster(cluster_path)

    for doc in cluster["docs"]:
        # Update multi_docs
        multi_docs.append(doc["txt"])

        # Tokenization
        text = sent_tokenize(doc["txt"])
        text = [
            re.sub(" +", " ", re.sub("\n", " ", x))
            for x in text
            if (len(word_count_tokenizer.tokenize(x)) > 0)
        ]
        if len(text) > 0:
            multi_docs_lines.append(text)

    return multi_docs, multi_docs_lines


def fetch_TAC2008_targets(path_to_clusters, cluster):
    """Retrieve the target summaries from the TAC2008 dataset

    Args:
        path_to_clusters (str): Path to the document clusters.

        cluster (int): Cluster number.

    Returns:
        target_list (list): List containing all the target summaries for
        a cluster.
    """
    target_list = []

    ref_path = Path(path_to_clusters, cluster, "models")

    for i, filename in enumerate(os.listdir(ref_path)):
        with open(os.path.join(ref_path, filename), "r") as file:
            target = file.read()
            text = target.replace("\n", " ").strip()
            target_list.append(text)

    return target_list


def eliminate_null_embeddings(multi_docs_lines, doc_features):
    """Delete cases of null embeddings from doc_features and the corresponding
    sentences from multi_docs_lines"

    Args:
        multi_docs_lines (list): List of lists containing the documents' sentences.

        doc_features (list): List containing the sentence embeddings for each document.

    Returns:
        multi_docs_lines (list): List of lists containing the documents' sentences.

        doc_features (list): List containing the sentence embeddings for each document.
    """
    # Remove cases where feature vectors are all zeros
    for doc, features in enumerate(doc_features):
        if (features.sum(1) == 0).any():
            zero_idxs = np.where(features.sum(1) == 0)[0]
            if len(zero_idxs) > 0:
                for i, zero_idx in enumerate(zero_idxs):
                    true_zero_idx = zero_idx - i
                    multi_docs_lines[doc].pop(true_zero_idx)

                    doc_features[doc] = np.concatenate(
                        [
                            doc_features[doc][:true_zero_idx],
                            doc_features[doc][true_zero_idx + 1 :],
                        ],
                        axis=0,
                    )

    return multi_docs_lines, doc_features


def read_cluster(path):
    """Read the document cluster from folder"""

    cluster = {}
    _, cluster_name = os.path.split(path)
    cluster["name"] = cluster_name
    cluster["docs"] = []
    cluster["models"] = []

    docs_path = os.path.join(path, "docs")
    for doc_name in sorted(os.listdir(docs_path)):
        doc_path = os.path.join(docs_path, doc_name)
        doc = read_document(doc_path)
        cluster["docs"].append(doc)

    models_path = os.path.join(path, "models")
    # models summaries (a.k.a. human summaries)
    for summary_file in sorted(os.listdir(models_path)):
        summary_path = os.path.join(models_path, summary_file)
        summary = read_summary(summary_path)
        cluster["models"].append(summary)

    return cluster


def read_document(path):
    """Read document txt or xml documents."""
    doc = {}
    doc["title"] = ""
    doc["date"] = "2016-01-01"
    doc["txt"] = ""

    # parse as TAC XML corpus
    if path.endswith(".xml"):
        root = etree.parse(path)
        doc["title"] = root.xpath("string(//HEADLINE)").strip()
        doc["txt"] = root.xpath("string(//TEXT)").strip()

    else:
        with open(path, encoding="utf8") as fp:
            text = fp.read().strip()
            doc["title"] = text.split("\n")[0].strip()
            doc["txt"] = "\n".join(text.split("\n")[1:]).strip()

    # date is present in the document title.
    _, filename = os.path.split(path)
    prog = re.search(r"(\d{4})(\d{2})(\d{2})", filename)
    if prog:
        doc["date"] = "-".join(prog.groups())

    return doc


def read_summary(path):
    """Read document txt or xml documents."""
    summary = ""
    with open(path, encoding="utf8") as fp:
        summary = fp.read().strip()

    return summary


def translate_summary(translator_model, summary, target_lang="en"):
    """Given a summary for a CrossSum cluster, translate its'
        sentences to english.

    Args:
        translator_model (obj): Model used for translating
        summary (list): A list of dictionaries, where each dictionary
        contains the sentence selected for the summary in its original
        language plus other information regarding this sentence.

    Returns:
        summary: A list of dictionaries, where each dictionary
        contains the english translation of the sentence selected for the summary
        plus other information regarding this sentence.
    """

    # Fetch the sentences and the languages of the original summary
    sentences = [item["sentence"] for item in summary]
    langs_summ = [item["language"] for item in summary]

    # Translate the predicted summary
    for enum, (sentence, lang_summ) in enumerate(zip(sentences, langs_summ)):
        if lang_summ != target_lang:
            result_sentence = translator_model.translate(
                sentence, source_lang=lang_summ, target_lang=target_lang
            )
        else:
            result_sentence = sentence

        summary[enum]["sentence"] = result_sentence

    return summary


def translate_multi_docs_lines(translator_model, multi_docs_lines, langs):
    """Translate the documents of the CrossSum dataset.

    Args:
        translator_model (obj): A translation model.

        multi_docs_lines (list): List of lists containing the documents' sentences.

        langs (list): List containing the ISO639-1 language code for each document.

    Returns:
       en_multi_docs_lines (list): List of lists containing the documents' sentences
       translated to english.
    """
    en_multi_docs_lines = []

    # Iterate through the data and translate it
    #################################################################
    for iterator, lang in enumerate(langs):
        tra_code = lang

        # Translate all the document sentences to English
        # if needed
        if tra_code != "en":
            try:
                sentences = translator_model.translate(
                    multi_docs_lines[iterator],
                    source_lang=tra_code,
                    target_lang="en",
                )
            # If out of memory, empty cache and re-try
            except RuntimeError as exception:
                if "CUDA out of memory" in str(exception):
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    sentences = translator_model.translate(
                        multi_docs_lines[iterator],
                        source_lang=tra_code,
                        target_lang="en",
                        batch_size=8,
                    )

            en_multi_docs_lines.append(sentences)
        else:
            # Append the document sentences in english to a list
            en_multi_docs_lines.append(
                copy.deepcopy(multi_docs_lines[iterator])
            )

    return en_multi_docs_lines


def translate_target(
    translator_model,
    target_list,
    langs,
    budget,
):
    """From a list containing the target summary of each document
    in a cluster (target_list), build one summary in english for the whole cluster.
    The summary is built by interleaving the sentences of each document
    target, in the original languages, until a maximum size, specified
    by 'budget', is reached (in our work we considered budget = 100 for the
    CrossSum data).

    Args:
        translator_model (obj): A translation model.

        target_list (list): List containing the target summary of each document.

        langs (list): List containing the ISO639-1 code for the language of each document.

        budget (int): Maximum size for the target summary, specified in number of words.


    Returns:
        new_summary_en_str (str): The resulting target summary, in english, resultant from concatenating
        the translated version of the sentences in 'new_target_list'.

        new_target_lst (list): The target sentences selected by interleaving the sentences
        of the targets of the documents, in the original languages.
    """

    word_count_tokenizer = RegexpTokenizer(r"\w+")
    target_sents_lst = []
    target_sents_lst_en = []
    target_sents_en = []

    for lang, targets in zip(langs, target_list):
        # The language saved is already in ISO639-1
        sent_code = lang
        # Instatiate the splitter object
        splitter = SentSplitter(sent_code)
        # Split the sentences in their native language
        target_sents = splitter.split(targets)
        # Append the original sentences to a list
        target_sents_lst.append(target_sents)

        # Check if the target sentences of each document should be
        # translated to english or not
        if sent_code != "en":
            try:
                target_sents_en = translator_model.translate(
                    target_sents, target_lang="en", source_lang=lang
                )
            # If out of memory, empty cache and re-try
            except RuntimeError as exception:
                if "CUDA out of memory" in str(exception):
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    target_sents_en = translator_model.translate(
                        target_sents,
                        target_lang="en",
                        source_lang=lang,
                        batch_size=8,
                    )
        else:
            target_sents_en = copy.deepcopy(target_sents)

        # Append the sentences in english to a list
        target_sents_lst_en.append(target_sents_en)

    # Save the maximum number of sentences found
    # in the original target of a document
    max_size = 0
    for inner_list in target_sents_lst:
        if len(inner_list) > max_size:
            max_size = len(inner_list)

    new_target_lst_en = []
    new_target_lst = []
    n_words = 0
    # Build the target summary for a cluster by interleaving the sentences of
    # the documents' targets, in the original languages. Everytime a sentence is
    # picked and saved in "new_target_lst", save the respective translation as well
    # in "new_target_lst_en"
    for sent_num in range(max_size):
        for target, target_en in zip(target_sents_lst, target_sents_lst_en):
            if sent_num <= len(target) - 1:
                # Append a new sentence to the summary if it does not exceed
                #  the budget
                if (
                    n_words
                    + len(word_count_tokenizer.tokenize(target[sent_num]))
                    <= budget
                ):
                    new_target_lst_en.append(target_en[sent_num])
                    new_target_lst.append(target[sent_num])
                    n_words += len(
                        word_count_tokenizer.tokenize(target[sent_num])
                    )

    # Convert the selected sentences for the cluster summary in english to a string
    new_summary_en_str = " ".join(new_target_lst_en)

    # Return the translated cluster target as a string and also, return the list of sentences
    # in the original languages, picked during the interleaving process.
    return new_summary_en_str, new_target_lst


def fetch_original_targets(
    target_list,
    langs,
    budget,
):
    """
    This function selects the sentences to be displayed in the target of
    each CrossSum cluster, in the original languages. This is done by interleaving
    the sentences of each document target, until a maximum size, specified by
    'budget', is reached (in our work we considered budget = 100).

    Args:
        target_list (list): List containing the target summary of each document.

        langs (list): List containing the ISO639-1 code for the language of each document.

        budget (int): Maximum size for the target summary, specified in number of words.

    Returns:
        new_target_lst (list): The target sentences selected by interleaving the sentences
        of the targets of the documents, in the original languages.
    """
    word_count_tokenizer = RegexpTokenizer(r"\w+")
    target_sents_lst = []

    for lang, target in zip(langs, target_list):
        # The language saved is already in iso code format
        sent_code = lang
        # Instatiate the splitter object
        splitter = SentSplitter(sent_code)
        # Split the sentences in their native language
        target_sents = splitter.split(target)
        # Append the original sentences to a list
        target_sents_lst.append(target_sents)

    # Save the maximum number of sentences found
    # in the original target of a document
    max_size = 0
    for inner_list in target_sents_lst:
        if len(inner_list) > max_size:
            max_size = len(inner_list)

    new_target_lst = []
    n_words = 0
    # Build the target summary for a cluster by interleaving the sentences of
    # the documents' targets, in the original languages.
    for sent_num in range(max_size):
        for target in target_sents_lst:
            if sent_num <= len(target) - 1:
                # Append a new sentence to the summary if it does not exceed
                #  the budget
                if (
                    n_words
                    + len(word_count_tokenizer.tokenize(target[sent_num]))
                    <= budget
                ):
                    new_target_lst.append(target[sent_num])
                    n_words += len(
                        word_count_tokenizer.tokenize(target[sent_num])
                    )

    # Return the list of sentences in the original languages, picked during the interleaving process.
    return new_target_lst
