# Define the package version
__version__ = "0.2.1-alpha"

# Import public classes and functions
from .base import SuperClass
from .panel import SetupData
from .prepare import Prepare
from .training import Training
from .split import split_test_train, SplitError
from .model import Model
