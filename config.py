import os
import pkg_resources
from nltk.data import path as nltk_path


def join_paths(path_1, path_2):
    # Join multiple paths together
    return os.path.join(path_1, path_2)

def package_path(relative_path):
    # Use pkg_resources to get the path to the resource
    base_path = pkg_resources.resource_filename(__name__, '')  # Get the base directory of the package
    return join_paths(base_path, relative_path)

# Set NLTK's data path to be inside the resource folder
nltk_path = []
nltk_path.append(package_path("nlp/models/nltk/data"))

# Create Global Paths to model directories
STOP_WORDS_DIR = package_path("nlp/models/nltk/stopwords.json")
EMBEDDING_DIR = package_path("nlp/models/all-MiniLM-L6-v2-onnx")
RERANKER_DIR = package_path("nlp/models/bge-reranker-onnx")
DATABASE_DIR = os.getcwd()