# Define the package version
__version__ = "0.1.0-alpha"

# Import public classes and functions
from .base import SuperClass
from .prepare import Prepare
from .training import Training
from .split import split_test_train, SplitError
from .model_utils import load_model
