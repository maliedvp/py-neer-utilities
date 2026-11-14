# Define the package version
__version__ = "1.2.15"


# Import public classes and functions
from .base import SuperClass
from .panel import SetupData
from .prepare import Prepare
from .training import Training
from .split import split_test_train, SplitError
from .blocking import Blocking, candidate_pairs
from .model import Model
from .custom_similarities import CustomSimilarities
from .embeddings import EmbeddingSimilarity
from .baseline_io import ModelBaseline
from .baseline_models import LogitMatchingModel, ProbitMatchingModel, GradientBoostingModel
from .similarity_features import SimilarityFeatures
from .baseline_training import BaselineTrainingPipe
from .feature_selection import FeatureSelector, FeatureSelectionResult