from abc import ABC, abstractmethod
import numpy as np
import Pickler
import time
import random

class AbstractAlgorithm(ABC):
    @abstractmethod
    def getName() -> str:
        pass

    @abstractmethod
    def run(image: np.array) -> np.array:
        pass

    @abstractmethod
    def getAverageFormat() -> any:
        pass

class BruteForceAlgorithm(AbstractAlgorithm):
    '''
    A completely brute force algorithm. This is used as the baseline, the algorithm originally developed for the problem. One cpu process goes through each pixel individually and checks its closeness to the dictionary using a brute force approach.
    '''
    def getName() -> str:
        return "Brute Force"
    
    def getAverageFormat() -> any:
        return Pickler.Load(Pickler.Pickles.AverageList)
    
    def distance(a, b):
        ab0 = a[0] - b[0]
        ab1 = a[1] - b[1]
        ab2 = a[2] - b[2]
        return ab0 * ab0 + ab1 * ab1 + ab2 * ab2
    
    def run(image: np.array, averages: any) -> np.array:
        width, height, channels = image.shape
        new = np.zeros(shape = (width, height), dtype = np.int32)
        for x in range(width):
            for y in range(height):
                pixel = image[x, y]
                closest = 0
                closeness = BruteForceAlgorithm.distance(pixel, averages[0])
                for i in range(1, len(averages)):
                    avg = averages[i]
                    close = BruteForceAlgorithm.distance(pixel, avg)
                    if (close < closeness):
                        closeness = close
                        closest = i
                new[x, y] = closest
        return new
