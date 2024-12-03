# Define the package version
__version__ = "0.1.3-alpha"

# Import public classes and functions
from .base import SuperClass
from .panel_setup import SetupData
from .prepare import Prepare
from .training import Training
from .split import split_test_train, SplitError
from .model import Model
