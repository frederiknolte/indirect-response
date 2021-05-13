"""circa_unmatched2756 dataset."""

import os
import sys

import tensorflow_datasets as tfds

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from circa_helper import CircaMixin


class CircaUnmatched2756(CircaMixin, tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for circa_unmatched_2756 dataset."""

    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {
        "1.0.0": "Initial release.",
    }
    _SETUP = "unmatched"
    _SEED = 2756
