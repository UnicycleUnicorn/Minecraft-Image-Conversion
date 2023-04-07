import pickle
import os
from enum import Enum
import numpy as np

class Pickles(Enum):
    ImageList = "ImageList"
    AverageList = "AverageList"
    AverageTuple = "AverageTuple"
    AverageTree = "AverageTree"

PickleDump = "PickleDump"

def Save(filename: Pickles, data: any) -> None:
    """Saves data to a pickle file."""
    filepath = os.path.join(PickleDump, filename.value)
    filepath += ".pckl"
    try:
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    except IOError:
        print(f"Error: Could not write to file {filepath}")

def Load(filename: Pickles) -> any:
    """Loads data from a pickle file."""
    filepath = os.path.join(PickleDump, filename.value)
    filepath += ".pckl"
    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        return data
    except IOError:
        print(f"Error: Could not read file {filepath}")


CorrectOutputs = "CorrectOutputs"
def SaveOutput(name: str, data: np.array) -> None:
    """Saves a numpy array to a pickle file."""
    filepath = os.path.join(CorrectOutputs, name)
    filepath += ".pckl"
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)

def LoadOutput(name: str) -> np.array:
    """Loads a numpy array from a pickle file."""
    filepath = os.path.join(CorrectOutputs, name)
    filepath += ".pckl"
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    return data


CorrectedInputs = "CorrectedInputs"
def SaveInput(name: str, data: tuple) -> None:
    """Saves a tuple array to a pickle file."""
    filepath = os.path.join(CorrectedInputs, name)
    filepath += ".pckl"
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)

def LoadInput(name: str) -> tuple:
    """Loads a tuple array from a pickle file."""
    filepath = os.path.join(CorrectedInputs, name)
    filepath += ".pckl"
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    return data