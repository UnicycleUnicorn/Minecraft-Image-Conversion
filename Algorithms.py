from abc import ABC, abstractmethod
import numpy as np
import Pickler
import Util
import threading
from queue import Queue
from cachetools import LRUCache
from scipy.spatial import KDTree


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

# Pre-computed
class STF(AbstractAlgorithm):
    '''
    Single threaded, pre-computed
    '''
    def getName() -> str:
        return "STF"
    
    def getAverageFormat() -> any:
        return Pickler.Load(Pickler.Pickles.AverageTuple)
    
    def run(image: tuple[tuple[tuple[int, int, int]]], averages: tuple[tuple[int, int, int]]) -> np.array:
        width = len(image)
        height = len(image[0])
        output = np.zeros(shape = (width, height), dtype = np.int16)
        for x in range(width):
            for y in range(height):
                pixel = image[x][y]
                output[x, y] = averages[pixel[0]][pixel[1]][pixel[2]]
        return output

class PTF(AbstractAlgorithm):
    '''
    Multi threaded, pre-computed
    '''
    def getName() -> str:
        return "PTF"
    
    def getAverageFormat() -> any:
        return Pickler.Load(Pickler.Pickles.AverageTuple)

    def run(image: np.array, averages: any) -> np.array:
        width = len(image)
        height = len(image[0])
        output = np.zeros(shape = (width, height), dtype = np.int16)
        
        # Create queue and fill with all col indicies
        queue = Queue()
        for i in range(width):
            queue.put(i)

        # Create a worker thread
        def worker():
            while True:
                index = queue.get()
                if index is None:
                    break
                col = image[index]
                newCol = np.zeros(shape = (height), dtype = np.int16)
                # Process data
                for x in range(height):
                    pixel = col[x]
                    newCol[x] = averages[pixel[0]][pixel[1]][pixel[2]]
                output[index] = newCol
                queue.task_done()

        # Create new threads
        processes = 8
        threads = []
        for i in range(processes):
            thread = threading.Thread(target = worker)
            thread.start()
            threads.append(thread)

        # Await finish
        queue.join()

        # Stop threads
        for i in range(processes):
            queue.put(None)
        for t in threads:
            t.join()

        return output

# Brute force
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

class PNB(AbstractAlgorithm):
    '''
    Multi threaded, no caching, brute force nearest neighbor
    '''
    def getName() -> str:
        return "PNB"
    
    def getAverageFormat() -> any:
        return Pickler.Load(Pickler.Pickles.AverageList)

    def run(image: np.array, averages: any) -> np.array:
        width = len(image)
        height = len(image[0])
        output = np.zeros(shape = (width, height), dtype = np.int16)
        
        # Create queue and fill with all col indicies
        queue = Queue()
        for i in range(width):
            queue.put(i)

        # Create a worker thread
        def worker():
            while True:
                index = queue.get()
                if index is None:
                    break
                col = image[index]
                newCol = np.zeros(shape = (height), dtype = np.int16)
                # Process data
                for x in range(height):
                    pixel = col[x]
                    closestIndex = 0
                    closestDistance = Util.distance(pixel, averages[0])
                    for i in range(1, len(averages)):
                        distance = Util.distance(pixel, averages[i])
                        if (distance < closestDistance):
                            closestDistance = distance
                            closestIndex = i
                    newCol[x] = closestIndex
                output[index] = newCol
                queue.task_done()

        # Create new threads
        processes = 8
        threads = []
        for i in range(processes):
            thread = threading.Thread(target = worker)
            thread.start()
            threads.append(thread)

        # Await finish
        queue.join()

        # Stop threads
        for i in range(processes):
            queue.put(None)
        for t in threads:
            t.join()

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

class PDB(AbstractAlgorithm):
    '''
    Multi threaded, dictionary caching, brute force nearest neighbor
    '''
    def getName() -> str:
        return "PDB"
    
    def getAverageFormat() -> any:
        return Pickler.Load(Pickler.Pickles.AverageList)

    def run(image: np.array, averages: any) -> np.array:
        width = len(image)
        height = len(image[0])
        output = np.zeros(shape = (width, height), dtype = np.int16)
        
        cache = {}
        lock = threading.Lock()

        # Create queue and fill with all col indicies
        queue = Queue()
        for i in range(width):
            queue.put(i)

        # Create a worker thread
        def worker():
            while True:
                index = queue.get()
                if index is None:
                    break
                col = image[index]
                newCol = np.zeros(shape = (height), dtype = np.int16)
                # Process data
                for x in range(height):
                    pixel = col[x]
                    if pixel in cache.keys():
                        newCol[x] = cache[pixel]
                    else:
                        closestIndex = 0
                        closestDistance = Util.distance(pixel, averages[0])
                        for i in range(1, len(averages)):
                            distance = Util.distance(pixel, averages[i])
                            if (distance < closestDistance):
                                closestDistance = distance
                                closestIndex = i
                        newCol[x] = closestIndex
                        with lock:
                            cache[pixel] = closestIndex
                output[index] = newCol
                queue.task_done()

        # Create new threads
        processes = 8
        threads = []
        for i in range(processes):
            thread = threading.Thread(target = worker)
            thread.start()
            threads.append(thread)

        # Await finish
        queue.join()

        # Stop threads
        for i in range(processes):
            queue.put(None)
        for t in threads:
            t.join()

        return output

class SLB(AbstractAlgorithm):
    '''
    Single threaded, LRU caching, brute force nearest neighbor
    '''
    def getName() -> str:
        return "SLB"
    
    def getAverageFormat() -> any:
        return Pickler.Load(Pickler.Pickles.AverageList)
    
    def run(image: np.array, averages: any) -> np.array:
        width = len(image)
        height = len(image[0])
        cache = LRUCache(maxsize = 500)
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

class PLB(AbstractAlgorithm):
    '''
    Multi threaded, LRU caching, brute force nearest neighbor
    '''
    def getName() -> str:
        return "PLB"
    
    def getAverageFormat() -> any:
        return Pickler.Load(Pickler.Pickles.AverageList)

    def run(image: np.array, averages: any) -> np.array:
        width = len(image)
        height = len(image[0])
        output = np.zeros(shape = (width, height), dtype = np.int16)
        
        lock = threading.Lock()
        cache = LRUCache(maxsize = 500)

        # Create queue and fill with all col indicies
        queue = Queue()
        for i in range(width):
            queue.put(i)

        # Create a worker thread
        def worker():
            while True:
                index = queue.get()
                if index is None:
                    break
                col = image[index]
                newCol = np.zeros(shape = (height), dtype = np.int16)
                # Process data
                for x in range(height):
                    pixel = col[x]
                    if pixel in cache.keys():
                        try:
                            with lock:
                                newCol[x] = cache[pixel]
                        except:
                            closestIndex = 0
                            closestDistance = Util.distance(pixel, averages[0])
                            for i in range(1, len(averages)):
                                distance = Util.distance(pixel, averages[i])
                                if (distance < closestDistance):
                                    closestDistance = distance
                                    closestIndex = i
                            newCol[x] = closestIndex
                            with lock:
                                cache[pixel] = closestIndex
                    else:
                        closestIndex = 0
                        closestDistance = Util.distance(pixel, averages[0])
                        for i in range(1, len(averages)):
                            distance = Util.distance(pixel, averages[i])
                            if (distance < closestDistance):
                                closestDistance = distance
                                closestIndex = i
                        newCol[x] = closestIndex
                        with lock:
                            cache[pixel] = closestIndex
                output[index] = newCol
                queue.task_done()

        # Create new threads
        processes = 8
        threads = []
        for i in range(processes):
            thread = threading.Thread(target = worker)
            thread.start()
            threads.append(thread)

        # Await finish
        queue.join()

        # Stop threads
        for i in range(processes):
            queue.put(None)
        for t in threads:
            t.join()

        return output

# KDTree
class SNK(AbstractAlgorithm):
    '''
    Single threaded, no caching, KDTree nearest neighbor
    '''
    def getName() -> str:
        return "SNK"
    
    def getAverageFormat() -> any:
        return Pickler.Load(Pickler.Pickles.AverageTree)
    
    def run(image: np.array, averages: KDTree) -> np.array:
        width = len(image)
        height = len(image[0])
        output = np.zeros(shape = (width, height), dtype = np.int16)
        for x in range(width):
            for y in range(height):
                pixel = image[x][y]
                closestIndex = averages.query(pixel, k = 1, eps = 0)[1]
                output[x, y] = closestIndex
        return output
    
class PNK(AbstractAlgorithm):
    '''
    Multi threaded, no caching, KDTree nearest neighbor
    '''
    def getName() -> str:
        return "PNK"
    
    def getAverageFormat() -> any:
        return Pickler.Load(Pickler.Pickles.AverageTree)

    def run(image: np.array, averages: any) -> np.array:
        width = len(image)
        height = len(image[0])
        output = np.zeros(shape = (width, height), dtype = np.int16)
        
        # Create queue and fill with all col indicies
        queue = Queue()
        for i in range(width):
            queue.put(i)

        # Create a worker thread
        def worker():
            while True:
                index = queue.get()
                if index is None:
                    break
                col = image[index]
                newCol = np.zeros(shape = (height), dtype = np.int16)
                # Process data
                for x in range(height):
                    pixel = col[x]
                    closestIndex = averages.query(pixel, k = 1, eps = 0)[1]
                    newCol[x] = closestIndex
                output[index] = newCol
                queue.task_done()

        # Create new threads
        processes = 8
        threads = []
        for i in range(processes):
            thread = threading.Thread(target = worker)
            thread.start()
            threads.append(thread)

        # Await finish
        queue.join()

        # Stop threads
        for i in range(processes):
            queue.put(None)
        for t in threads:
            t.join()

        return output

class SDK(AbstractAlgorithm):
    '''
    Single threaded, dictionary caching, KDTree nearest neighbor
    '''
    def getName() -> str:
        return "SDK"
    
    def getAverageFormat() -> any:
        return Pickler.Load(Pickler.Pickles.AverageTree)
    
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
                    closestIndex = averages.query(pixel, k = 1, eps = 0)[1]
                    cache[pixel] = closestIndex
                    output[x, y] = closestIndex
        return output

class PDK(AbstractAlgorithm):
    '''
    Multi threaded, dictionary caching, KDTree nearest neighbor
    '''
    def getName() -> str:
        return "PDK"
    
    def getAverageFormat() -> any:
        return Pickler.Load(Pickler.Pickles.AverageTree)

    def run(image: np.array, averages: any) -> np.array:
        width = len(image)
        height = len(image[0])
        output = np.zeros(shape = (width, height), dtype = np.int16)
        
        cache = {}
        lock = threading.Lock()

        # Create queue and fill with all col indicies
        queue = Queue()
        for i in range(width):
            queue.put(i)

        # Create a worker thread
        def worker():
            while True:
                index = queue.get()
                if index is None:
                    break
                col = image[index]
                newCol = np.zeros(shape = (height), dtype = np.int16)
                # Process data
                for x in range(height):
                    pixel = col[x]
                    if pixel in cache.keys():
                        newCol[x] = cache[pixel]
                    else:
                        closestIndex = averages.query(pixel, k = 1, eps = 0)[1]
                        newCol[x] = closestIndex
                        with lock:
                            cache[pixel] = closestIndex
                output[index] = newCol
                queue.task_done()

        # Create new threads
        processes = 8
        threads = []
        for i in range(processes):
            thread = threading.Thread(target = worker)
            thread.start()
            threads.append(thread)

        # Await finish
        queue.join()

        # Stop threads
        for i in range(processes):
            queue.put(None)
        for t in threads:
            t.join()

        return output

class SLK(AbstractAlgorithm):
    '''
    Single threaded, LRU caching, KDTree nearest neighbor
    '''
    def getName() -> str:
        return "SLK"
    
    def getAverageFormat() -> any:
        return Pickler.Load(Pickler.Pickles.AverageTree)
    
    def run(image: np.array, averages: any) -> np.array:
        width = len(image)
        height = len(image[0])
        cache = LRUCache(maxsize = 500)
        output = np.zeros(shape = (width, height), dtype = np.int16)
        for x in range(width):
            for y in range(height):
                pixel = image[x][y]
                if pixel in cache.keys():
                    output[x, y] = cache[pixel]
                else:
                    closestIndex = averages.query(pixel, k = 1, eps = 0)[1]
                    cache[pixel] = closestIndex
                    output[x, y] = closestIndex
        return output

class PLK(AbstractAlgorithm):
    '''
    Multi threaded, LRU caching, KDTree nearest neighbor
    '''
    def getName() -> str:
        return "PLK"
    
    def getAverageFormat() -> any:
        return Pickler.Load(Pickler.Pickles.AverageTree)

    def run(image: np.array, averages: any) -> np.array:
        width = len(image)
        height = len(image[0])
        output = np.zeros(shape = (width, height), dtype = np.int16)
        
        lock = threading.Lock()
        cache = LRUCache(maxsize = 500)

        # Create queue and fill with all col indicies
        queue = Queue()
        for i in range(width):
            queue.put(i)

        # Create a worker thread
        def worker():
            while True:
                index = queue.get()
                if index is None:
                    break
                col = image[index]
                newCol = np.zeros(shape = (height), dtype = np.int16)
                # Process data
                for x in range(height):
                    pixel = col[x]
                    if pixel in cache.keys():
                        try:
                            with lock:
                                newCol[x] = cache[pixel]
                        except:
                            closestIndex = averages.query(pixel, k = 1, eps = 0)[1]
                            newCol[x] = closestIndex
                            with lock:
                                cache[pixel] = closestIndex
                    else:
                        closestIndex = averages.query(pixel, k = 1, eps = 0)[1]
                        newCol[x] = closestIndex
                        with lock:
                            cache[pixel] = closestIndex
                        
                output[index] = newCol
                queue.task_done()
        
        # Create new threads
        processes = 8
        threads = []
        for i in range(processes):
            thread = threading.Thread(target = worker)
            thread.start()
            threads.append(thread)

        # Await finish
        queue.join()

        # Stop threads
        for i in range(processes):
            queue.put(None)
        for t in threads:
            t.join()

        return output









