from PIL import Image
import time
from Algorithms import *
import numpy as np
import Pickler
import os

class Test():
    def __init__(self, name: str, image: str):
        self.name = name
        with Image.open(image).convert('RGB') as img:
            width, height = img.size
            self.n = width * height
            self.image = np.array(img)
        try:
            self.correct = Pickler.LoadOutput(self.name)
        except:
            print(f"Creating new test output: ", end = "")
            self.correct = BruteForceAlgorithm.run(self.image, BruteForceAlgorithm.getAverageFormat())
            print(self.name)
            Pickler.SaveOutput(self.name, self.correct)

def AssertOutputCorrect(a :np.array, b :np.array, test: Test, algorithm: AbstractAlgorithm):
    if not np.array_equal(a, b):
        raise AssertionError(f"{test.name} failed on algorithm: {algorithm.getName()}")        

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
        for test in self.tests:
            print(f"{test.name}: {test.n}")
            for algorithm in self.algorithms:
                print(f"    {algorithm.getName()}")
                totalTime = 0
                averageFormat = algorithm.getAverageFormat()
                for i in range(self.count):
                    print(f"        {i}: ", end='')
                    start = time.process_time_ns()
                    output = algorithm.run(test.image, averageFormat)
                    end = time.process_time_ns()
                    elapsed = end - start
                    print(f"{elapsed} ns")
                    totalTime += elapsed
                    AssertOutputCorrect(output, test.correct, test, algorithm)
                avgTime = totalTime / self.count
                print(f"        Average: {avgTime} ns")
                #TODO append each test case to an excel file

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
testSuite.addAlgorithm(BruteForceAlgorithm)

# Save output images
#testSuite.saveOutputImages()

# Run test suite
#testSuite.run()
