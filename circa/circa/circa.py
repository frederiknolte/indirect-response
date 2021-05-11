"""circa dataset."""

import csv

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from unidecode import unidecode

_DESCRIPTION = """
the circa dataset
"""

# TODO(circa): BibTeX citation
_CITATION = """
@InProceedings{louis_emnlp2020,
  author =      "Annie Louis and Dan Roth and Filip Radlinski",
  title =       ""{I}'d rather just go to bed": {U}nderstanding {I}ndirect {A}nswers",
  booktitle =   "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing",
  year =        "2020",
"""


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
            homepage="https://dataset-homepage/",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""
        path = dl_manager.download_and_extract(
            "https://raw.githubusercontent.com/google-research-datasets/circa/main/circa-data.tsv"
        )

        # all_splits = {"all_data": self._generate_examples(path)}

        per_context_splits = {
            f"context_{x}": self._generate_examples_per_context(path, x)
            for x in range(1, 11)
        }

        train_dev_test_indices = self._get_train_dev_test_split(path)
        matched_splits = {
            fold: self._generate_examples_matched(path, indices)
            for fold, indices in train_dev_test_indices.items()
        }

        # return {**all_splits, **per_context_splits, **matched_splits}
        return {**per_context_splits, **matched_splits}

    def _generate_examples(self, path):
        """Yields examples."""

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
        with open(path) as infile:
            tsv_reader = csv.DictReader(infile, delimiter="\t", fieldnames=column_names)
            next(tsv_reader)  # skip header row

            for line in tsv_reader:
                for k, v in line.items():
                    if ("goldstandard" in k) or ("judgements" in k):
                        line[k] = unidecode(v)  # strange apostrophe in text

                line_id = np.array([int(line["id"])])
                line["id"] = line_id

                yield line_id, line

    def _generate_examples_per_context(self, path, context_type: int):
        """
        split examples per scenario:
        1. X wants to know about Y's food preferences
        2. X wants to know what activities Y likes to do during weekends.
        3. X wants to know what sorts of books Y likes to read.
        4. Y has just moved into a neighbourhood and meets his/her new neighbour X.
        5. X and Y are colleagues who are leaving work on a Friday at the same time.
        6. X wants to know about Y's music preferences.
        7. Y has just travelled from a different city to meet X.
        8. X and Y are childhood neighbours who unexpectedly run into each other at a cafe.
        9. Y has just told X that he/she is thinking of buying a flat in New York.
        10. Y has just told X that he/she is considering switching his/her job.
        """

        context_keyword_mapping = {
            1: "food preference",
            2: "during weekends",
            3: "sorts of books",
            4: "new neighbour",
            5: "on a Friday",
            6: "music preferences",
            7: "different city",
            8: "childhood neighbours",
            9: "buying a flat",
            10: "switching",
        }
        context_str = context_keyword_mapping[context_type]

        for line_id, line in self._generate_examples(path):
            if context_str in line["context"]:
                yield line_id, line

    def _get_train_dev_test_split(self, path, train_frac=0.6, dev_frac=0.2):
        with open(path) as f:
            num_entries = len(f.readlines()) - 1  # 34268 entries

        entry_indices = np.arange(num_entries)
        np.random.seed(1)
        np.random.shuffle(entry_indices)

        index_train_until = int(num_entries * train_frac) + 1
        index_dev_until = int(num_entries * (train_frac + dev_frac)) + 1

        return {
            "train": set(entry_indices[:index_train_until]),
            "dev": set(entry_indices[index_train_until:index_dev_until]),
            "test": set(entry_indices[index_dev_until:]),
        }

    def _generate_examples_matched(self, path, indices):

        for line_id, line in self._generate_examples(path):
            if int(line_id) in indices:
                yield line_id, line
