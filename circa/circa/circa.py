"""circa dataset."""

import csv
import os

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from unidecode import unidecode


_DESCRIPTION = """
The Circa (meaning ‘approximately’) dataset aims to help machine learning systems to solve the problem of interpreting indirect answers to polar questions.

The dataset contains pairs of yes/no questions and indirect answers, together with annotations for the interpretation of the answer. The data is collected in 10 different social conversational situations (eg. food preferences of a friend).

This version generates random splits (train/val/test) for both matched/unmatched settings.
It also repeats the process for three different random seeds.
"""

# TODO(circa): BibTeX citation
_CITATION = """
@InProceedings{louis_emnlp2020,
  author =      "Annie Louis and Dan Roth and Filip Radlinski",
  title =       ""{I}'d rather just go to bed": {U}nderstanding {I}ndirect {A}nswers",
  booktitle =   "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing",
  year =        "2020",
"""

# _URL = "https://raw.githubusercontent.com/google-research-datasets/circa/main/"
# TODO: change this if/when we publish our data splits
_URL = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../data")

_SETUPS = ["matched", "unmatched"]
_SEEDS = [13, 948, 2756]
_SPLITS = ["train", "val", "test"]


class Circa(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for circa dataset."""

    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {
        "1.0.0": "Initial release.",
    }

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict(
                {
                    # These are the features of your dataset like images, labels ...
                    "id": tfds.features.Tensor(shape=(1,), dtype=tf.int64),
                    "context": tfds.features.Text(),
                    "question_x": tfds.features.Text(),
                    "canquestion_x": tfds.features.Text(),
                    "answer_y": tfds.features.Text(),
                    "judgements": tfds.features.Text(),
                    "goldstandard1": tfds.features.ClassLabel(
                        names=[
                            "Yes",
                            "Probably yes / sometimes yes",
                            "Yes, subject to some conditions",
                            "No",
                            "Probably no",
                            "In the middle, neither yes nor no",
                            "I am not sure how X will interpret Y's answer",
                            "NA",
                            "Other",
                        ]
                    ),
                    "goldstandard2": tfds.features.ClassLabel(
                        names=[
                            "Yes",
                            "Yes, subject to some conditions",
                            "No",
                            "In the middle, neither yes nor no",
                            "NA",
                            "Other",
                        ]
                    ),
                }
            ),
            # If there's a common (input, target) tuple from the
            # features, specify them here. They'll be used if
            # `as_supervised=True` in `builder.as_dataset`.
            supervised_keys=("answer_y", "goldstandard1"),  # Set to `None` to disable
            homepage="https://github.com/google-research-datasets/circa",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""

        split_generators = []
        for seed in _SEEDS:
            for setup in _SETUPS:
                for split in _SPLITS:
                    specification = f"{split}_{setup}_{seed}"
                    files = dl_manager.download_and_extract(
                        {
                            specification: [
                                os.path.join(_URL, f"circa-{split}-{setup}-{seed}.tsv")
                            ]
                        }
                    )
                    split_generators.append(
                        tfds.core.SplitGenerator(
                            name=specification,
                            gen_kwargs={"files": files[specification]},
                        )
                    )

        return split_generators

    def _generate_examples(self, files):
        """Yields all examples available in the .tsv file"""

        column_names = [
            "id",
            "context",
            "question_x",
            "canquestion_x",
            "answer_y",
            "judgements",
            "goldstandard1",
            "goldstandard2",
        ]

        for filepath in files:
            with tf.io.gfile.GFile(filepath) as f:
                tsv_reader = csv.DictReader(f, delimiter="\t", fieldnames=column_names)
                next(tsv_reader)  # skip header row

                for line in tsv_reader:
                    for k, v in line.items():
                        if "goldstandard" in k:
                            line[k] = unidecode(v)  # strange apostrophe in text
                        elif k == "judgments":
                            line[k] = list(map(unidecode, v.split("#")))

                    line_id = np.array([int(line["id"])])
                    line["id"] = line_id

                    yield line_id, line
