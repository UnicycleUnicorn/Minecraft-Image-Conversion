from abc import ABC, abstractmethod
import numpy as np
import Pickler
import Util

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

class SNB(AbstractAlgorithm):
    '''
    Single threaded, no caching, brute force nearest neighbor
    '''
    def getName() -> str:
        return "SNB"
    
    def getAverageFormat() -> any:
        return Pickler.Load(Pickler.Pickles.AverageList)
    
    def run(image: np.array, averages: any) -> np.array:
        width = len(image)
        height = len(image[0])
        output = np.zeros(shape = (width, height), dtype = np.int16)
        for x in range(width):
            for y in range(height):
                pixel = image[x][y]
                closestIndex = 0
                closestDistance = Util.distance(pixel, averages[0])
                for i in range(1, len(averages)):
                    distance = Util.distance(pixel, averages[i])
                    if (distance < closestDistance):
                        closestDistance = distance
                        closestIndex = i
                output[x, y] = closestIndex
        return output

class SDB(AbstractAlgorithm):
    '''
    Single threaded, dictionary caching, brute force nearest neighbor
    '''
    def getName() -> str:
        return "SDB"
    
    def getAverageFormat() -> any:
        return Pickler.Load(Pickler.Pickles.AverageList)
    
    def run(image: np.array, averages: any) -> np.array:
        width = len(image)
        height = len(image[0])
        cache = {}
        output = np.zeros(shape = (width, height), dtype = np.int16)
        for x in range(width):
            for y in range(height):
                pixel = image[x][y]
                if pixel in cache.keys():
                    output[x, y] = cache[pixel]
                else:
                    closestIndex = 0
                    closestDistance = Util.distance(pixel, averages[0])
                    for i in range(1, len(averages)):
                        distance = Util.distance(pixel, averages[i])
                        if (distance < closestDistance):
                            closestDistance = distance
                            closestIndex = i
                    cache[pixel] = closestIndex
                    output[x, y] = closestIndex
        return output