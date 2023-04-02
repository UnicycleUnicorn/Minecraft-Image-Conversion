import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import Pickler

def plot_colors_3d(colors):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    X = [c[0] for c in colors]
    Y = [c[1] for c in colors]
    Z = [c[2] for c in colors]
    
    ax.scatter(X, Y, Z, c=colors, marker='o')
    
    ax.set_xlabel('Red')
    ax.set_ylabel('Green')
    ax.set_zlabel('Blue')
    
    plt.show()


colors = Pickler.Load(Pickler.Pickles.AverageList)
normalized_colors = [(r/255, g/255, b/255) for r, g, b in colors]

plot_colors_3d(normalized_colors)