"""circa dataset."""

import csv
import os

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

_URL = "https://raw.githubusercontent.com/google-research-datasets/circa/main/"


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

        files = dl_manager.download_and_extract({
            'train': [os.path.join(_URL, 'circa-data.tsv')]
        })

        return [
            tfds.core.SplitGenerator(
                name=tfds.Split.TRAIN,
                gen_kwargs={'files': files['train']},
            )
        ]

    def _generate_examples(self, files):
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

        for filepath in files:
            with tf.io.gfile.GFile(filepath) as f:
                tsv_reader = csv.DictReader(f, delimiter="\t", fieldnames=column_names)
                next(tsv_reader)  # skip header row

                for line in tsv_reader:
                    for k, v in line.items():
                        if 'goldstandard' in k:
                            line[k] = unidecode(v)
                        elif k == 'judgments':
                            line[k] = list(map(unidecode, v))

                    line_id = np.array([int(line["id"])])
                    line["id"] = line_id

                    yield line_id, line
