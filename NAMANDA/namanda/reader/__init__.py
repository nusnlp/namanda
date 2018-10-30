#!/usr/bin/env python3

import os
from ..tokenizers import CoreNLPTokenizer
from .. import DATA_DIR


DEFAULTS = {
    'tokenizer': CoreNLPTokenizer,
    'model': os.path.join(DATA_DIR, 'reader/namanda.mdl'),
}


def set_default(key, value):
    global DEFAULTS
    DEFAULTS[key] = value

from .model import DocReader
from .predictor import Predictor
from . import config
from . import vector
from . import data
from . import utils
