"""circa_matched13 dataset."""

import os
import sys

import tensorflow_datasets as tfds

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from circa_helper import CircaMixin


class CircaMatched13(CircaMixin, tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for circa_matched_13 dataset."""

    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {
        "1.0.0": "Initial release.",
    }
    _SETUP = "matched"
    _SEED = 13
