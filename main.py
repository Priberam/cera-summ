import argparse
import copy
import os
import pickle

import torch
from easynmt import EasyNMT
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from centroid_attention.datasets_utils import prepare_dataloader
from centroid_attention.model import LitCentroidEstimationModel
from ROUGE.rouge_script import run_ROUGE
from summarizer.baseline import lamsiyah_sum
from summarizer.data_loaders import create_data_loader
from summarizer.summaries import sentence_selection
from summarizer.summaries_utils import create_centroid
from summarizer.utils import dict_multi_docs_lines, eliminate_null_embeddings


def main(
    args,
    embeddings_model=None,
    translator_model=None,
    centroid_model=None,
    device="cpu",
):
    # To store the summaries predicted for each cluster
    predictions = []

    # To store the target gold summaries
    targets = []

    # Check if the user gave a path to a dataset in disk
    if len(args.pickle_path) != 0:
        # If it does not exist create it and save in disk
        if not os.path.isfile(args.pickle_path):
            print("This data file does not exist! Saving the data in disk...")
            # Create the dataloader
            dataset = create_data_loader(
                dataset_path=args.dataset_path,
                partition=args.partition,
                translator_model=translator_model,
                budget=args.budget,
                dataset_name=args.dataset_name,
            )
            output_dict = {f"{args.partition}": []}
            # Update the dataloader, adding more information per cluster
            for iter, datapoint in enumerate(tqdm(dataset, desc=f"Saving...")):
                updated_datapoint = prepare_dataloader(
                    datapoint=datapoint,
                    partition=args.partition,
                    budget=args.budget,
                    embeddings_model=embeddings_model,
                    dataset_name=args.dataset_name,
                )
                # Check if the updated datapoint is valid and append it to the
                # new data
                if len(updated_datapoint) != 0:
                    output_dict[args.partition].append(updated_datapoint)
            dataset = output_dict[args.partition]
            with open(
                f"{args.pickle_path}",
                "wb",
            ) as f:
                pickle.dump(output_dict, f)
        else:
            # Load the dataset from a pickle
            print("Loading data from a pickle...")
            file = open(args.pickle_path, "rb")
            dataset = pickle.load(file)
            dataset = dataset[args.partition]

    # If the user did not give a pickle_path
    # simply create a dataloader for the dataset
    # and run the algorithm
    elif len(args.pickle_path) == 0:
        print("Creating dataloader...")
        dataset = create_data_loader(
            dataset_path=args.dataset_path,
            partition=args.partition,
            translator_model=translator_model,
            budget=args.budget,
            dataset_name=args.dataset_name,
        )

    # Iterate through the dataset
    for iter, data in enumerate(tqdm(dataset, desc="Building Summaries")):
        # Initialize the cluster centroid
        centroid = None

        if args.dataset_name != "CrossSum":
            target = data["target"]
            multi_docs_lines = data["multi_docs_lines"]
            multi_docs_lines_encode = data["multi_docs_lines"]
            langs = None

        else:
            # Fetch the translated data and the languages in the current cluster
            try:
                target = data["en_target"]
                multi_docs_lines = data["en_multi_docs_lines"]
                multi_docs_lines_encode = data["multi_docs_lines"]
            except:
                print(
                    "The translations are not present in the current data partition."
                )
                print("Exiting...")
                exit()

            langs = data["langs"]
            # Compute the target embeddings from the original sentences
            if args.centroid_type == "gold":
                if len(args.pickle_path) == 0:
                    original_targets = data["original_targets"]
                    target_features = [
                        embeddings_model.encode(original_targets)
                    ]
                else:
                    centroid = data["gold_centroid"]
            else:
                target_features = None

        # Skip iteration if the cluster is empty
        if len(multi_docs_lines) <= 0:
            print("Cluster is empty.")
            continue

        # Store the multi-reference target summaries in target_list
        # (E.g.: DUC2004, and TAC2008 datasets)
        if args.reference_type == "multi_ref":
            target_list = copy.deepcopy(target)
        # Store the single-reference target summaries in a target_list
        # (E.g.: Multi-News, WCEP-10, and CrossSum datasets)
        elif args.reference_type == "single_ref":
            target_list = copy.deepcopy([target])
        else:
            raise NotImplementedError

        # Compute the features for the target summaries
        # if not considering the CrossSum dataset
        if args.centroid_type == "gold" and args.dataset_name != "CrossSum":
            if len(args.pickle_path) == 0:
                target_features = []
                for target in target_list:
                    target_sents = sent_tokenize(target)
                    fts = embeddings_model.encode(target_sents)
                    target_features.append(fts)
            else:
                centroid = data["gold_centroid"]
        elif args.centroid_type != "gold":
            target_features = None

        # Compute the documents' embeddings
        if len(args.pickle_path) == 0:
            doc_features = []
            for text in multi_docs_lines_encode:
                # Contextual embeddings
                features = embeddings_model.encode(text)
                doc_features.append(features)
        else:
            doc_features = data["doc_features"]

        # Delete cases of null embeddings from doc_features and the corresponding
        # sentences from multi_docs_lines
        multi_docs_lines, doc_features = eliminate_null_embeddings(
            multi_docs_lines, doc_features
        )

        # Compute the cluster centroid (Unsupervised or Gold or Estimated)
        if centroid is None:
            centroid = create_centroid(
                multi_docs_lines=multi_docs_lines,
                doc_features=doc_features,
                centroid_type=args.centroid_type,
                centroid_model=centroid_model,
                target_features=target_features,
                device=device,
            )

        # Convert multi_docs_lines to a dictionary to keep track
        # of the document each sentence comes from, its position
        # in the document and its length (number of words). Also,
        # keep track of the language in the case of CrossSum data
        multi_docs_lines = dict_multi_docs_lines(
            multi_docs_lines=multi_docs_lines, langs=langs
        )

        # Generate the summaries - (Our implementation of Gholipour Ghalandari, D. (2017)
        # (https://doi.org/10.18653/v1/W17-4511) refined with Beam and Greedy searches).
        if args.summarizer_type == "ghalandari":
            summary = sentence_selection(
                multi_docs_lines=multi_docs_lines,
                doc_features=doc_features,
                centroid=centroid,
                sentences=args.sentences,
                n=args.n,
                beam_width=args.beam_width,
                counter_limit=args.counter_limit,
                budget=args.budget,
                R=args.R,
                r=args.r,
                alpha=args.alpha,
            )

        # Generate the summaries - (Our implementation of the Lamsiyah
        # et al. model) (https://www.sciencedirect.com/science/article/pii/
        # S0957417420308952)
        elif args.summarizer_type == "lamsiyah":
            summary = lamsiyah_sum(multi_docs_lines, doc_features, args.budget)

        else:
            raise NotImplementedError

        # Convert the summary to a string
        summary = " ".join([item["sentence"] for item in summary])

        # Append the generated summary and the target_list to lists
        # for ROUGE scores computation
        predictions.append(summary)
        targets.append(target_list)

    # Print the total number of clusters analyzed
    print(f"Total number of clusters: {len(predictions)}\n")

    # Compute Rouge Scores using the PERL script and print the results
    _ = run_ROUGE(
        model_summaries=targets,
        system_summaries=predictions,
        budget=args.budget,
    )


if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser()

    # Paths ######################################
    parser.add_argument(
        "--dataset_path",
        help="Path to desired dataset",
        default="",
        type=str,
    )
    parser.add_argument(
        "--centroid_model_path",
        help="Path to the pre-trained attention model checkpoint (CeRA/CeRAI models)",
        default="",
        type=str,
    )
    parser.add_argument(
        "--pickle_path",
        help="Path to a pickle with data saved in disk",
        default="",
        type=str,
    )
    ###########################################################################

    # Datasets ###############################
    parser.add_argument(
        "--partition",
        help="Choose the data partition: train, validation or test",
        default="",
        type=str,
        choices=["train", "validation", "test"],
    )
    parser.add_argument(
        "--dataset_name",
        help="Name of the dataset in use",
        default="",
        type=str,
        choices=["DUC2004", "TAC2008", "MultiNews", "WCEP10", "CrossSum"],
    )
    parser.add_argument(
        "--reference_type",
        help="Multi-reference: 'multi_ref' Single-reference: 'single_ref'.",
        default="",
        type=str,
        choices=["single_ref", "multi_ref"],
    )
    ###########################################################################

    # Summarization algorithm #############################################
    parser.add_argument(
        "--summarizer_type",
        help="Selects the summarization algorithm",
        default="gholipour",
        type=str,
        choices=["lamsiyah", "ghalandari"],
    )

    ##########################################################################

    # Algorithm Parameters #######################
    parser.add_argument("--R", help="Redundancy filter", action="store_true")
    parser.add_argument(
        "--budget",
        help="Budget for the summary",
        default=100,
        type=int,
    )
    parser.add_argument(
        "--sentences",
        help="Preselection of Sentences. 'all': No preselection, use all sentences;"
        + "'n_first': Pre-select the n first sentences from each document;"
        + "'n_best';Pre-select the n sentences from each document whose embeddings are"
        + "more similar to the cluster centroid.",
        default="n_first",
        type=str,
        choices=["all", "n_first", "n_best"],
    )
    parser.add_argument(
        "--n",
        help="Number of sentences to pre-select (used together with 'n_first' or 'n_best'"
        + "options from the '--sentences' argument).",
        default=9,
        type=int,
    )
    parser.add_argument(
        "--r",
        help="Similarity threshold for avoiding redundancy",
        default=0.6,
        type=float,
    )
    parser.add_argument(
        "--counter_limit",
        help="Number of iterations after finding a sentence that exceeds the"
        + "budget for the summary",
        default=9,
        type=int,
    )
    parser.add_argument(
        "--beam_width",
        help="Number of beams used in the beam search.",
        default=5,
        type=int,
    )
    parser.add_argument(
        "--alpha",
        help="Trade-off between cosine similarity and number of words in the"
        + " summary",
        default=0,
        type=float,
    )
    # Centroid estimator ############################################
    parser.add_argument(
        "--centroid_type",
        default="unsupervised",
        help="Select the type of cluster centroid to use: 'unsupervised'; 'gold'; 'estimated'.",
        type=str,
        choices=["unsupervised", "gold", "estimated"],
    )
    ########################################################################

    # Parse the arguments
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        print("Device:", torch.cuda.get_device_name(device))
    else:
        device = "cpu"
        print("Device: CPU only")

    # Initialize the sentence encoder
    embeddings_model = SentenceTransformer(
        "sentence-transformers/distiluse-base-multilingual-cased-v2"
    ).to(device)

    # Initialize the translator model
    if args.dataset_name == "CrossSum":
        translator_model = EasyNMT("m2m_100_1.2B", device=device)
    else:
        translator_model = None

    # Initialize the centroid estimator model
    # If using a centroid estimation model (CeRA or CeRAI)
    if args.centroid_type == "estimated":
        centroid_model = LitCentroidEstimationModel.load_from_checkpoint(
            args.centroid_model_path,
        )
    else:
        centroid_model = None

    # Call to the main function
    main(
        args,
        embeddings_model=embeddings_model,
        translator_model=translator_model,
        centroid_model=centroid_model,
        device=device,
    )
