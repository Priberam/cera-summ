import os
import random
import re
import shutil
import tempfile
from subprocess import getoutput

from tabulate import tabulate

ROUGE_PATH = "/opt/ROUGE/ROUGE-1.5.5.pl"
ROUGE_DATA = "/opt/ROUGE/data"
table = []


def rouge_output_to_dict(output):
    """
    Convert the ROUGE output into python dictionary for further processing.
    Extracted from pyrouge.
    """
    # 0 ROUGE-1 Average_R: 0.02632 (95%-conf.int. 0.02632 - 0.02632)
    pattern = re.compile(
        r"(\d+) (ROUGE-\S+) (Average_\w): (\d.\d+) "
        r"\(95%-conf.int. (\d.\d+) - (\d.\d+)\)"
    )
    results = {}
    for line in output.split("\n"):
        match = pattern.match(line)
        if match:
            (
                sys_id,
                rouge_type,
                measure,
                result,
                conf_begin,
                conf_end,
            ) = match.groups()
            measure = {
                "Average_R": "recall",
                "Average_P": "precision",
                "Average_F": "f_score",
            }[measure]
            rouge_type = rouge_type.lower().replace("-", "_")
            key = "{}_{}".format(rouge_type, measure)
            results[key] = float(result)
            results["{}_cb".format(key)] = float(conf_begin)
            results["{}_ce".format(key)] = float(conf_end)
    return results


def build_rouge_files(model_summaries, system_summaries):
    """
    models_summaries = list of a list of reference summaries
    system_summaries = list of system summaries
    """
    system_dir = tempfile.mkdtemp()
    model_dir = tempfile.mkdtemp()
    for index, (models, system) in enumerate(
        zip(model_summaries, system_summaries)
    ):
        for model_num, model in enumerate(models):
            model_char = chr(65 + model_num)
            model_name = "gold." + model_char + "." + str(index) + ".txt"
            model_path = os.path.join(model_dir, model_name)
            with open(model_path, "w", encoding="utf8") as f:
                f.write(model)
        system_name = "predicted." + str(index) + ".txt"
        system_path = os.path.join(system_dir, system_name)
        with open(system_path, "w", encoding="utf8") as f:
            f.write(system)

    return (model_dir, system_dir)


def rouge_evaluation(model_summaries, system_summaries, limit):
    """
    models_summaries = list of a list of reference summaries
    system_summaries = list of system summaries
    """
    system_dir = tempfile.mkdtemp()
    model_dir = tempfile.mkdtemp()

    # rouge config file name
    rouge_conf = str(random.randint(1, 1e5)) + ".xml"

    with open(rouge_conf, "w", encoding="utf-8") as fp_conf:
        fp_conf.write('<ROUGE-EVAL version="1.55">')
        for index, (models, system) in enumerate(
            zip(model_summaries, system_summaries)
        ):
            model_elems = []
            for model_num, model in enumerate(models):
                model_char = chr(65 + model_num)
                model_name = "gold." + model_char + "." + str(index) + ".txt"
                model_path = os.path.join(model_dir, model_name)
                model_elems.append(
                    '<M ID="{id}">{name}</M>'.format(
                        id=model_char, name=model_name
                    )
                )
                with open(model_path, "w", encoding="utf8") as f:
                    f.write(model)
            system_name = "predicted." + str(index) + ".txt"
            system_path = os.path.join(system_dir, system_name)
            peer_elem = '<P ID="{id}">{name}</P>'.format(
                id=1, name=system_name
            )
            with open(system_path, "w", encoding="utf8") as f:
                f.write(system)

            model_elems = "\n\t\t\t".join(model_elems)
            eval_string = """
    <EVAL ID="{task_id}">
        <MODEL-ROOT>{model_root}</MODEL-ROOT>
        <PEER-ROOT>{peer_root}</PEER-ROOT>
        <INPUT-FORMAT TYPE="SPL">
        </INPUT-FORMAT>
        <PEERS>
            {peer_elem}
        </PEERS>
        <MODELS>
            {model_elems}
        </MODELS>
    </EVAL>
            """.format(
                task_id=index,
                model_root=model_dir,
                model_elems=model_elems,
                peer_root=system_dir,
                peer_elem=peer_elem,
            )
            fp_conf.write(eval_string)
        fp_conf.write("</ROUGE-EVAL>")

    command = (
        ROUGE_PATH
        + " -e "
        + ROUGE_DATA
        + f" -n 4 -m -a -l {limit} -c 95 -r 1000 -f A -p 0.5 -t 0 "
        + rouge_conf
    )
    # In our experiences we considered the following values for limit:
    # 100 for TAC2008, DUC 2004, and CrossSum
    # 230 for Multi-News (average number of words on the training split
    # reference summaries)
    # 50 for the WCEP-10 dataset

    output = getoutput(command)
    shutil.rmtree(system_dir)
    shutil.rmtree(model_dir)
    os.remove(rouge_conf)

    return output, rouge_output_to_dict(output)


def run_ROUGE(model_summaries, system_summaries, budget):
    assert len(model_summaries) == len(system_summaries)

    print("Evaluating with ROUGE...")

    output, metrics = rouge_evaluation(
        model_summaries, system_summaries, budget
    )

    metrics = rouge_output_to_dict(output)

    table = []
    table.append(
        [
            "ROUGE Evaluation",
            metrics["rouge_1_precision"],
            metrics["rouge_1_recall"],
            metrics["rouge_1_f_score"],
            "",
            metrics["rouge_2_precision"],
            metrics["rouge_2_recall"],
            metrics["rouge_2_f_score"],
            "",
            metrics["rouge_4_precision"],
            metrics["rouge_4_recall"],
            metrics["rouge_4_f_score"],
            "",
            metrics["rouge_l_precision"],
            metrics["rouge_l_recall"],
            metrics["rouge_l_f_score"],
        ]
    )
    print(output)

    print(
        tabulate(
            table,
            headers=[
                "Algorithm",
                "R-1 P",
                "R-1 R",
                "R-1 F",
                "",
                "R-2 P",
                "R-2 R",
                "R-2 F",
                "",
                "R-4 P",
                "R-4 R",
                "R-4 F",
                "",
                "R-L P",
                "R-L R",
                "R-L F",
            ],
        )
    )

    return metrics
