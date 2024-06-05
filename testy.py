import unittest
from functions import *
import numpy as np
import warnings
import logging

warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)


class MyTestCase(unittest.TestCase):
    def test_training(self):
        c = 2+2

