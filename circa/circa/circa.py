"""circa dataset."""

import csv

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

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
        # TODO: pass this as a param or set as global?
        # path = "/home/lcur0357/indirect-response/circa/circa-data.tsv"
        path = dl_manager.download_and_extract('https://raw.githubusercontent.com/google-research-datasets/circa/main/circa-data.tsv')
        # TODO: train/test split?
        # TODO(circa): Returns the Dict[split names, Iterator[Key, Example]]
        return {
            "train": self._generate_examples(path),
        }

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
                    # line[k] = v.replace('’', "'")  # the original file had the wrong type of apostrophe
                    line[k] = v.replace(
                        "\xe2\x80\x99", "'"
                    )  # the original file had the wrong type of apostrophe

                line_id = np.array([int(line["id"])])
                line["id"] = line_id

                yield line_id, line
