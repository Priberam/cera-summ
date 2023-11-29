import pickle
from typing import Optional

import lightning.pytorch as pl
import numpy as np
import torch
from centroid_attention.datasets_utils import (
    pad_sentences_order,
    prepare_dataloader,
    sequence_mask,
)
from summarizer.data_loaders import create_data_loader
from easynmt import EasyNMT
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader
from tqdm import tqdm


class MyDataModuleBaseClass(pl.LightningDataModule):
    def setup(self, stage: str) -> None:
        return super().setup(stage)


# DataCollator used to prepare the Multi-News and CrossSum data for the models
class DataCollator:
    def __init__(self, num_positional_embeddings):
        self.num_positional_embeddings = num_positional_embeddings

    def __call__(self, samples):
        """Processes the data within a batch. More specifically, creates an
        attention mask for the features of each cluster. This because the
        cluster features must be padded with zeros to match the maximum length
        found in a batch. Also, it pads the vector that tells the order of
        the sentences in the documents and the vector that assigns the weights
        to each sentence in the batch.
        This function will return:
        - the padded cluster features,
        - the target centroids,
        - the unsupervised centroids,
        - the attention mask,
        - the order of the sentences,
        - the number of documents,
        - the weights for each sentence in the documents,
        - the length of the documents,
        - the cluster text,
        - the target text.
        All of this for each element in the batch.

        Args:
            samples (dict): A dictionary that contains the cluster features
            ('x' key), the target centroids ('y' key), the order of the
            sentences in the clusters' documents ('order' key), the number
            of documents in the clusters ('num_docs' key), the length of
            each doc in the clusters ('docs_length' key), the clusters'
            text and target summaries text ('multi_docs_lines' and
            'target' keys, respectively).

        Returns:
           batch (dict): A dictionary of torch tensors that contains the
            padded cluster features ('x' key), the target centroids ('y' key),
            the unsupervised centroids ('clusters_centroids' key),
            the attention mask over the cluster features
            ('attention_mask' key), the padded order of the sentences in
            the cluster ('order' key), the number of documents in the cluster
            ('num_docs' key), the padded vector with the weights to assign
            to each sentence ('docs_weights' key), the length of each document
            in the cluster ('docs_length' key), the cluster text
            ('multi_docs_lines' key), and the target summary text
            ('target' key).
        """
        # Compute the clusters centroid
        clusters_centroids = [
            torch.divide(torch.sum(s["x"], dim=0), s["x"].size(0))
            for s in samples
        ]

        # Length of each element in the batch
        # (number of doc features per batch)
        lengths = torch.tensor([len(s["x"]) for s in samples])

        # Find the maximum sequence length for the cluster features
        # (Largest cluster)
        max_length = max(lengths)

        # Pad each cluster features sample with zeros to match the
        # size of the largest cluster
        x_padded_samples = [
            torch.cat(
                [
                    s["x"],
                    torch.zeros(max_length - len(s["x"]), s["x"].shape[1]),
                ]
            )
            for s in samples
        ]

        # Isolate the cluster target centroids in y_samples
        y_samples = [s["y"] for s in samples]

        # Isolate the cluster number of docs in no_docs_samples
        no_docs_samples = [s["num_docs"] for s in samples]

        # Create a list of numpy arrays that contain the weight
        # for each sentence in a document
        docs_weights = [
            np.repeat(np.divide(1, s["docs_length"]), s["docs_length"])
            for s in samples
        ]

        # Pad the previous list with zeros, for the positions corresponding to
        # padded sentences
        docs_weights_padded_samples = [
            torch.cat(
                [
                    s,
                    torch.full((max_length - len(s),), 0),
                ]
            )
            for s in docs_weights
        ]

        # Save the docs lengths
        docs_length = [s["docs_length"] for s in samples]

        # Create an attention mask for the input data (attention
        # only on not padded data)
        attention_mask = ~sequence_mask(lengths)

        # Isolate the sentences' order in z_samples
        z_samples = [s["order"] for s in samples]

        # Pad the sentences' order array as well
        z_padded_samples = pad_sentences_order(
            z_samples, self.num_positional_embeddings, max_length
        )

        # Isolate the multi_docs_lines
        multi_docs_lines_samples = [s["multi_docs_lines"] for s in samples]
        # Isolate the target text
        target_samples = [s["target"] for s in samples]

        # Return the data for a batch
        batch = {
            "x": torch.stack(x_padded_samples),  # (batch_size,max_S,embed_dim)
            "y": torch.stack(y_samples),  # (batch_size,1,embed_dim)
            "clusters_centroids": torch.stack(
                clusters_centroids
            ),  # (batch_size,embed_dim)
            "attention_mask": attention_mask,  # (batch_size, max_S)
            "order": torch.stack(z_padded_samples),  # (batch_size, max_S)
            "num_docs": torch.stack(no_docs_samples),  # (batch_size)
            "docs_weights": torch.stack(
                docs_weights_padded_samples
            ).float(),  # (batch_size, max_S)
            "docs_length": docs_length,  # (batch_size, max_S)
            "multi_docs_lines": multi_docs_lines_samples,  # (batch_size,)
            "target": target_samples,  # (batch_size,)
        }

        return batch


# Multi-News dataloader for the CeRA and CeRAI models
class MultiNewsDataset:
    """Create an iterable dataset for the Multi-News train data"""

    def __init__(self, dataset, partition):
        self.data = dataset[partition]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        x = torch.from_numpy(np.vstack(self.data[item]["doc_features"]))
        y = torch.from_numpy(self.data[item]["gold_centroid"])
        z = torch.from_numpy(self.data[item]["order"])

        no_docs = torch.from_numpy(np.array(self.data[item]["num_docs"]))
        docs_length = torch.from_numpy(
            np.array(self.data[item]["docs_length"])
        )

        # Fetch the textual data
        multi_docs_lines = self.data[item]["multi_docs_lines"]
        target = self.data[item]["target"]

        return {
            "x": x,
            "y": y,
            "order": z,
            "num_docs": no_docs,
            "docs_length": docs_length,
            "multi_docs_lines": multi_docs_lines,
            "target": target,
        }


# Multi-News PL Datamodule
class MultiNewsDataModule(MyDataModuleBaseClass):
    """Implements 3 dataloaders. One for each data split of the Multi-News
    data"""

    def __init__(
        self,
        batch_size=4,
        num_positional_embeddings=30,
        use_pickle=False,
        save_pickle=True,
        pickle_path="",
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_positional_embeddings = num_positional_embeddings
        self.use_pickle = use_pickle
        self.save_pickle = save_pickle
        self.pickle_path = pickle_path

    def setup(self, stage: Optional[str] = None):
        # Load the dataset from a pickle
        if self.use_pickle:
            print("Loading the data from a pickle...")
            file = open(
                self.pickle_path,
                "rb",
            )
            self.data = pickle.load(file)

        # Prepare the Multi-News data to be used by the CeRA/CeRAI model
        else:
            keys = ["train", "validation", "test"]
            self.data = {"train": [], "validation": [], "test": []}

            # Specify the used device
            run_device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )

            # Initialize the sentence encoder
            embeddings_model = SentenceTransformer(
                "sentence-transformers/distiluse-base-multilingual-cased-v2"
            ).to(run_device)

            for key in keys:
                if key == "train":
                    # Create the dataset using the create_dataloader script
                    dataset = create_data_loader(
                        dataset_name="MultiNews",
                        partition=key,
                    )
                if key == "validation":
                    # Create the dataset using the create_dataloader script
                    dataset = create_data_loader(
                        dataset_name="MultiNews",
                        partition=key,
                    )
                if key == "test":
                    # Create the dataset using the create_dataloader script
                    dataset = create_data_loader(
                        dataset_name="MultiNews",
                        partition=key,
                    )

                # Create the necessary data to be used by the CeRA/CeRAI model
                for iter, datapoint in enumerate(
                    tqdm(dataset, desc=f"Setting-up the {key} dataloader")
                ):
                    updated_datapoint = prepare_dataloader(
                        datapoint=datapoint,
                        partition=key,
                        embeddings_model=embeddings_model,
                        dataset_name="MultiNews",
                    )
                    # Check if the updated datapoint is valid
                    if len(updated_datapoint) != 0:
                        self.data[key].append(updated_datapoint)

            if self.save_pickle:
                print("Saving the data in a pickle...")
                # Save the CrossSum data to be used by the attention model
                with open(
                    "MultiNewsAttmodelsData.pickle",
                    "wb",
                ) as f:
                    pickle.dump(self.data, f)

        # Fetch the train split
        self.train_dataset = MultiNewsDataset(
            dataset=self.data, partition="train"
        )
        # Fetch the validation split
        self.val_dataset = MultiNewsDataset(
            dataset=self.data, partition="validation"
        )
        self.test_dataset = MultiNewsDataset(
            dataset=self.data, partition="test"
        )

    def train_dataloader(self):
        train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            collate_fn=DataCollator(self.num_positional_embeddings),
            num_workers=4,
        )
        return train_dataloader

    def val_dataloader(self):
        val_dataloader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            collate_fn=DataCollator(self.num_positional_embeddings),
            num_workers=4,
        )
        return val_dataloader

    def test_dataloader(self):
        test_dataloader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            collate_fn=DataCollator(self.num_positional_embeddings),
            num_workers=4,
        )
        return test_dataloader


# CrossSum dataloader for the CeRA and CeRAI models
class CrossSumDataset:
    """Create an iterable dataset for the CrossSum data"""

    def __init__(self, dataset, partition):
        self.data = dataset[partition]
        self.partition = partition

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        x = torch.from_numpy(np.vstack(self.data[item]["doc_features"]))
        y = torch.from_numpy(self.data[item]["gold_centroid"])
        z = torch.from_numpy(self.data[item]["order"])

        no_docs = torch.from_numpy(np.array(self.data[item]["num_docs"]))
        docs_length = torch.from_numpy(
            np.array(self.data[item]["docs_length"])
        )

        # Fetch the textual data
        if self.partition != "train":
            multi_docs_lines = self.data[item]["en_multi_docs_lines"]
            target = self.data[item]["en_target"]
        else:
            # The training data does not need the translated documents
            # or the translated targets
            multi_docs_lines = self.data[item]["multi_docs_lines"]
            target = self.data[item]["target"]

        return {
            "x": x,
            "y": y,
            "order": z,
            "num_docs": no_docs,
            "docs_length": docs_length,
            "multi_docs_lines": multi_docs_lines,
            "target": target,
        }


# CrossSum PL Datamodule
class CrossSumDataModule(MyDataModuleBaseClass):
    """Implements 3 dataloaders. One for each data split of the CrossSumAggreg
    data"""

    def __init__(
        self,
        batch_size=4,
        num_positional_embeddings=30,
        budget=100,
        use_pickle=False,
        save_pickle=True,
        pickle_path="",
        train_dataset_path="",
        validation_dataset_path="",
        test_dataset_path="",
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_positional_embeddings = num_positional_embeddings
        self.budget = budget
        self.use_pickle = use_pickle
        self.save_pickle = save_pickle
        self.pickle_path = pickle_path
        self.train_dataset_path = train_dataset_path
        self.validation_dataset_path = validation_dataset_path
        self.test_dataset_path = test_dataset_path

    def setup(self, stage: Optional[str] = None):
        # Load the dataset from a pickle
        if self.use_pickle:
            print("Loading the data from a pickle...")
            file = open(
                self.pickle_path,
                "rb",
            )
            self.data = pickle.load(file)

        # Prepare the CrossSum data to be used by the CeRA or CeRAI model
        else:
            keys = ["train", "validation", "test"]
            self.data = {"train": [], "validation": [], "test": []}

            # Specify the used device
            run_device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
            # Initialize the sentence encoder
            embeddings_model = SentenceTransformer(
                "sentence-transformers/distiluse-base-multilingual-cased-v2"
            ).to(run_device)

            # Initialize the translator model
            translator_model = EasyNMT("m2m_100_1.2B", device=run_device)

            for key in keys:
                if key == "train":
                    # Create the dataset using the create_dataloader script
                    dataset = create_data_loader(
                        dataset_path=self.train_dataset_path,
                        dataset_name="CrossSum",
                        translator_model=None,
                        budget=None,
                    )
                if key == "validation":
                    # Create the dataset using the create_dataloader script
                    dataset = create_data_loader(
                        dataset_path=self.validation_dataset_path,
                        dataset_name="CrossSum",
                        translator_model=translator_model,
                        budget=self.budget,
                    )
                if key == "test":
                    # Create the dataset using the create_dataloader script
                    dataset = create_data_loader(
                        dataset_path=self.test_dataset_path,
                        dataset_name="CrossSum",
                        translator_model=translator_model,
                        budget=self.budget,
                    )

                # Create the necessary data to be used by the CeRA/CeRAI model
                for iter, datapoint in enumerate(
                    tqdm(dataset, desc=f"Setting-up the {key} dataloader")
                ):
                    updated_datapoint = prepare_dataloader(
                        datapoint=datapoint,
                        partition=key,
                        budget=self.budget,
                        embeddings_model=embeddings_model,
                        dataset_name="CrossSum",
                    )
                    # Check if the updated datapoint is valid
                    if len(updated_datapoint) != 0:
                        self.data[key].append(updated_datapoint)

            if self.save_pickle:
                print("Saving the data in a pickle...")
                # Save the CrossSum data to be used by the attention model
                with open(
                    "CrossSumAttmodelsData.pickle",
                    "wb",
                ) as f:
                    pickle.dump(self.data, f)

        # Fetch the train split
        self.train_dataset = CrossSumDataset(
            dataset=self.data, partition="train"
        )
        # Fetch the validation split
        self.val_dataset = CrossSumDataset(
            dataset=self.data, partition="validation"
        )
        # Fetch the test split
        self.test_dataset = CrossSumDataset(
            dataset=self.data, partition="test"
        )

    def train_dataloader(self):
        train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            collate_fn=DataCollator(self.num_positional_embeddings),
            num_workers=4,
        )
        return train_dataloader

    def val_dataloader(self):
        val_dataloader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            collate_fn=DataCollator(self.num_positional_embeddings),
            num_workers=4,
        )
        return val_dataloader

    def test_dataloader(self):
        test_dataloader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            collate_fn=DataCollator(self.num_positional_embeddings),
            num_workers=4,
        )
        return test_dataloader
