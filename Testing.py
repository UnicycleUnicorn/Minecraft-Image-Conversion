from PIL import Image
import time
from Algorithms import *
import numpy as np
import Pickler
import os
import Util
import pandas as pd

class Test():
    def __init__(self, name: str, image: str):
        self.name = name
        with Image.open(image).convert('RGB') as img:
            width, height = img.size
            self.n = width * height

        try:
            self.image = Pickler.LoadInput(self.name)
        except:
            print(f"Creating new test input: ", end = "")
            self.image = Util.totuple(np.array(img))
            print(self.name)
            Pickler.SaveInput(self.name, self.image)

        try:
            self.correct = Pickler.LoadOutput(self.name)
        except:
            print(f"Creating new test output: ", end = "")
            self.correct = STF.run(self.image, STF.getAverageFormat())
            print(self.name)
            Pickler.SaveOutput(self.name, self.correct)

class TestData():
    def __init__(self, tests: list[Test], algorithms: list[AbstractAlgorithm]):
        self.ExcelWorkbook = [[0 for j in range(len(algorithms) + 2)] for i in range(len(tests) + 1)]
        
        self.ExcelWorkbook[0][0] = "Name"
        self.ExcelWorkbook[0][1] = "N (pixels)"
        for i in range(len(tests)):
            t = tests[i]
            self.ExcelWorkbook[i + 1][0] = t.name
            self.ExcelWorkbook[i + 1][1] = t.n

        for i in range(len(algorithms)):
            a = algorithms[i]
            self.ExcelWorkbook[0][i + 2] = a.getName()

        self.tests = tests
        self.algorithms = algorithms
    
    def appendTest(self, test: Test, algorithm: AbstractAlgorithm, averageTime: float):
        self.ExcelWorkbook[self.tests.index(test) + 1][self.algorithms.index(algorithm) + 2] = averageTime
        self.export()

    def export(self):
        dataFrame = pd.DataFrame(self.ExcelWorkbook)
        dataFrame.to_excel('Data.xlsx', index = False, header=None)

def AssertOutputCorrect(a :np.array, b :np.array, test: Test, algorithm: AbstractAlgorithm):
    if not np.array_equal(a, b):
        s = a.shape
        avg = Pickler.Load(Pickler.Pickles.AverageList)
        for x in range(s[0]):
            for y in range(s[1]):
                av = a[x, y]
                bv = b[x, y]
                if av != bv:
                    p = test.image[x][y]
                    if (Util.distance(avg[av] ,p) != Util.distance(avg[bv], p)):
                        raise AssertionError(algorithm.getName() + " failed on: " + test.name)

class TestSuite():
    def __init__(self, count: int):
        self.tests = []
        self.algorithms = []
        self.count = count

    def addTest(self, test: Test):
        self.tests.append(test)
    
    def addAlgorithm(self, algorithm: AbstractAlgorithm):
        self.algorithms.append(algorithm)
    
    def run(self):
        data = TestData(self.tests, self.algorithms)
        for test in self.tests:
            print(f"{test.name}: {test.n}")
            for algorithm in self.algorithms:
                print(f"    {algorithm.getName()}")
                totalTime = 0
                for i in range(self.count):
                    print(f"        {i}: ", end='')
                    averageFormat = algorithm.getAverageFormat()
                    start = time.perf_counter_ns()
                    output = algorithm.run(test.image, averageFormat)
                    end = time.perf_counter_ns()
                    elapsed = end - start
                    print(f"{elapsed} ns")
                    totalTime += elapsed
                    if i == 0:
                        AssertOutputCorrect(output, test.correct, test, algorithm)
                avgTime = totalTime / self.count
                print(f"        Average: {avgTime} ns")
                data.appendTest(test, algorithm, avgTime)
        data.export()

    def saveOutputImages(self):
        textures = Pickler.Load(Pickler.Pickles.ImageList)
        for test in self.tests:
            width = len(test.correct)
            height = len(test.correct[0])
            print()
            print(f"Creating image: ", end = "")
            output = Image.new("RGB", (width * 16, height * 16))
            for x in range(width):
                for y in range(height):
                    matching = textures[test.correct[x, y]]
                    left = x * 16
                    top = y * 16
                    with Image.open(matching).convert("RGB") as matching_image:
                        output.paste(matching_image, box = (left, top, left + 16, top + 16))  
            output.rotate(-90).save(f"OutputImages/{test.name}.png")
            print(f"{test.name} with {width*16*height*16} pixels")

CONTROLLED_TESTS = False

if CONTROLLED_TESTS:
    testSuite = TestSuite(1)
    
    testSuite.addAlgorithm(SNB)
    testSuite.addAlgorithm(PNB)
    
    
    tests = ["Mario", "Creeper", "Bridge"]
    for t in tests:
        testSuite.addTest(Test(t, f"InputImages/{t}.png"))
    
    testSuite.run()

else:
    # Create a new test suite that averages on 10 runs
    testSuite = TestSuite(4)

    # Add all images as tests to the test suite
    for file in os.listdir("InputImages/"):
        if (file.endswith(".png")):
            name = file.replace(".png", "")
            path = "InputImages/" + file
            test = Test(name, path)
            testSuite.addTest(test)

    # Add algorithms to the test suite
    #TODO Run Later - make sure to copy excel file first
    testSuite.addAlgorithm(SNB)
    testSuite.addAlgorithm(PNB)
    '''
    testSuite.addAlgorithm(SDB)
    testSuite.addAlgorithm(PDB)
    testSuite.addAlgorithm(SLB)
    testSuite.addAlgorithm(PLB)

    testSuite.addAlgorithm(SNK)
    testSuite.addAlgorithm(PNK)
    testSuite.addAlgorithm(SDK)
    testSuite.addAlgorithm(PDK)
    testSuite.addAlgorithm(SLK)
    testSuite.addAlgorithm(PLK)
    
    testSuite.addAlgorithm(STF)
    testSuite.addAlgorithm(PTF)
    '''
    # Save output images
    #testSuite.saveOutputImages()

    # Run test suite
    testSuite.run()