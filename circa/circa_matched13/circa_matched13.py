"""circa_matched13 dataset."""

import tensorflow_datasets as tfds

import sys
sys.path.append('..')
from circa_helper import _Circa


class CircaMatched13(_Circa,tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for circa_matched_13 dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }
  _SETUP = 'matched'
  _SEED = 13
