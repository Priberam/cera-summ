import json
import os
from pathlib import Path

import iso639
from datasets import load_dataset
from icu_tokenizer import SentSplitter

from summarizer.utils import (
    fetch_DUC2004_sentences,
    fetch_DUC2004_targets,
    fetch_MultiNews_sentences,
    fetch_TAC2008_sentences,
    fetch_TAC2008_targets,
    fetch_WCEP10_sentences,
    process_CrossSum,
    translate_multi_docs_lines,
    translate_target,
)


class DUC2004Dataset:
    """Create an iterable dataset for the DUC2004 data"""

    def __init__(self, DUC2004_path):
        self.DUC2004_path = DUC2004_path
        self.path_to_clusters = Path(
            self.DUC2004_path,
            "DUC2004_Summarization_Documents",
            "duc2004_testdata",
            "tasks1and2",
            "duc2004_tasks1and2_docs",
            "docs",
        )
        self.clusters = os.listdir(self.path_to_clusters)

    def __len__(self):
        return len(self.clusters)

    def __getitem__(self, item):
        cluster = self.clusters[item]

        multi_docs, multi_docs_lines = fetch_DUC2004_sentences(
            self.DUC2004_path, self.path_to_clusters, cluster
        )
        target_list = fetch_DUC2004_targets(self.DUC2004_path, cluster)

        return {
            "multi_docs": multi_docs,
            "multi_docs_lines": multi_docs_lines,
            "target": target_list,
        }


class MultiNewsDataset:
    """Create an iterable dataset for the Multi-News train, validation,
    or test data"""

    def __init__(self, dataset, partition):
        self.clusters = dataset[partition]["document"]
        self.summaries = dataset[partition]["summary"]

    def __len__(self):
        return len(self.clusters)

    def __getitem__(self, item):
        cluster = self.clusters[item]
        target = self.summaries[item]

        multi_docs, multi_docs_lines = fetch_MultiNews_sentences(cluster)

        # Remove empty strings from multi_docs
        if "" in multi_docs:
            while "" in multi_docs:
                multi_docs.remove("")

        return {
            "multi_docs": multi_docs,
            "multi_docs_lines": multi_docs_lines,
            "target": target,
        }


class WCEP10Dataset:
    """Create an iterable dataset for the WCEP-10 train, validation, or test
    data"""

    def __init__(self, dataset, partition):
        self.clusters = dataset[partition]["document"]
        self.summaries = dataset[partition]["summary"]

    def __len__(self):
        return len(self.clusters)

    def __getitem__(self, item):
        cluster = self.clusters[item]
        target = self.summaries[item]

        multi_docs, multi_docs_lines = fetch_WCEP10_sentences(cluster)

        # Remove empty strings from multi_docs
        if "" in multi_docs:
            while "" in multi_docs:
                multi_docs.remove("")

        return {
            "multi_docs": multi_docs,
            "multi_docs_lines": multi_docs_lines,
            "target": target,
        }


class TAC2008Dataset:
    """Create an iterable dataset for the TAC2008 data"""

    def __init__(self, TAC2008_path):
        self.path_to_clusters = TAC2008_path
        self.clusters = os.listdir(self.path_to_clusters)

    def __len__(self):
        return len(self.clusters)

    def __getitem__(self, item):
        cluster = self.clusters[item]

        multi_docs, multi_docs_lines = fetch_TAC2008_sentences(
            self.path_to_clusters, cluster
        )
        target_list = fetch_TAC2008_targets(self.path_to_clusters, cluster)

        # Remove empty strings from multi_docs
        if "" in multi_docs:
            while "" in multi_docs:
                multi_docs.remove("")

        return {
            "multi_docs": multi_docs,
            "multi_docs_lines": multi_docs_lines,
            "target": target_list,
        }


class CrossSumDataset:
    """Create an iterable dataset for the CrossSum Dataset
    which is a multi-lingual sumarization dataset adapted by us from
    the original CrossSum Dataset (Bhattacharjee et al.
    https://doi.org/10.18653/v1/2021.findings-acl.413).
    """

    def __init__(self, dataset, translator_model=None, budget=None):
        self.clusters = dataset
        self.translator_model = translator_model
        self.budget = budget

    def __len__(self):
        return len(self.clusters)

    def __getitem__(self, item):
        cluster = self.clusters[item]

        # Save the cluster data into a dictionary
        cluster_data = {}
        cluster_data["multi_docs"] = [
            doc for key, doc in cluster.items() if "text" in key
        ]
        cluster_data["langs"] = [
            lang for key, lang in cluster.items() if "lang" in key
        ]
        cluster_data["target_list"] = [
            target for key, target in cluster.items() if "summary" in key
        ]

        # Initialize the unprocessed documents and target summaries.
        # Also, initalize the list which returns the documents split into
        # sentences and the list that stores the languages in the cluster.
        #################################################################
        multi_docs = cluster_data["multi_docs"]
        target_list = cluster_data["target_list"]
        multi_docs_lines = []
        langs = []
        #################################################################

        # Split all the documents from a cluster into sentences
        # and save them in multi_docs_lines
        ##############################################################
        for iterator, (doc, lang) in enumerate(
            zip(cluster_data["multi_docs"], cluster_data["langs"])
        ):
            # ISO639 code for a document's language
            sent_code = iso639.to_iso639_1(lang)
            # Instatiate the sentence splitter object
            splitter = SentSplitter(sent_code)
            # Split the document into sentences
            doc_split = splitter.split(doc)
            # Update multi_docs_lines
            multi_docs_lines.append(doc_split)
            # Update langs
            langs.append(sent_code)
        #################################################################

        # Pre-process multi_docs_lines (eliminate sentences with 'n_words'=0).
        # Process langs, multi_docs, and target_list for the
        # case that a document is completely discarded
        #################################################################
        (
            multi_docs_lines,
            langs,
            multi_docs,
            target_list,
        ) = process_CrossSum(multi_docs_lines, langs, multi_docs, target_list)
        #################################################################

        output_dict = {
            "multi_docs": multi_docs,
            "multi_docs_lines": multi_docs_lines,
            "target": target_list,
            "langs": langs,
        }

        if self.translator_model is not None:
            # Translate the documents
            en_multi_docs_lines = translate_multi_docs_lines(
                self.translator_model, multi_docs_lines, langs
            )
            # Obtain the translated target and also obtain the
            # corresponding original sentences in a list (target_list_original)
            en_target, original_targets = translate_target(
                self.translator_model, target_list, langs, self.budget
            )

            output_dict["en_target"] = en_target
            output_dict["en_multi_docs_lines"] = en_multi_docs_lines
            output_dict["original_targets"] = original_targets

        return output_dict


def load_jsonl(path):
    """To load the CrossSumm dataset which is stored
    in .jsonl format.

    Args:
        path (str): Path to the CrossSum data location.

    Returns:
        data (list): A list where each element is a cluster
        of the CrossSum data containing more than a document.
    """
    data = []

    with open(path, "r", encoding="utf-8") as reader:
        for line in reader:
            loaded_line = json.loads(line)
            if len(loaded_line) > 4:
                data.append(loaded_line)

    return data


def create_data_loader(
    dataset_path="",  # Multi-News doesn't require a dataset path
    partition="",
    translator_model=None,
    budget=None,
    dataset_name="",
):
    """Create an iterable dataloader for the dataset selected by the user.

    Args:
        dataset_path (str): Path to the dataset.

        partition (str): Dataset partition ("train", "validation", "test").

        translator_model (obj): Translator model.

        budget (int): Constrains the number of words allowed
        in the summary.

        dataset_name (str): A string with a valid name for the dataset
        ("DUC2004", "TAC2008", "MultiNews", "WCEP10", "CrossSum").

    Returns:
        ds (obj): An iterable dataloader.
    """
    if dataset_name == "CrossSum":
        try:
            dataset = load_jsonl(dataset_path)
        except:
            print(
                "Error: Please provide a valid dataset_path to create the dataloader!"
            )
            exit()

        ds = CrossSumDataset(
            dataset=dataset, translator_model=translator_model, budget=budget
        )

    elif dataset_name == "DUC2004":
        try:
            ds = DUC2004Dataset(DUC2004_path=dataset_path)
        except:
            print(
                "Error: Please provide a valid dataset_path to create the dataloader!"
            )
            exit()

    elif dataset_name == "MultiNews":
        dataset = load_dataset("multi_news")

        ds = MultiNewsDataset(dataset=dataset, partition=partition)

    elif dataset_name == "WCEP10":
        dataset = load_dataset(path="ccdv/WCEP-10", name="list")
        ds = WCEP10Dataset(dataset=dataset, partition=partition)

    elif dataset_name == "TAC2008":
        try:
            ds = TAC2008Dataset(TAC2008_path=dataset_path)
        except:
            print(
                "Error: Please provide a valid dataset_path to create the dataloader!"
            )
            exit()

    else:
        raise NotImplementedError

    return ds
