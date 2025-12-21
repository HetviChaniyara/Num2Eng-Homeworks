"""
Numerical Mathematics for Engineering II WS 24/25
Homework 12 Programming Exercise 12.3:

# Description of the function
The function B, C, D, E = getTriangulationMatrices(N) takes the number of inner grid points per spatial dimension $N$ and returns matrices $B$, $C$, $D$, and $E$.
The matrices $B$ and $D$ have as many rows as there are nodes, i.e., $(N+2)^2$.
The $B$ matrix has two columns and contains the $x_1$ and the $x_2$ coordinate of each node.
The $D$ matrix has one column and indicates for each node if it is on the boundary (then the entry is $1$) or an inner node (then the entry is $0$).
The matrices $C$ and $E$ have as many rows as there are triangular elements in $Omega$.
The matrix $C$ has three columns and contains the three node indices (corresponding to the row index in $B$) for each triangle.
The matrix $E$ has two columns and contains for each triangle the $x_1$ and the $x_2$ coordinate of its centroid.

# Summary
The program determines some auxiliary matrices for handling the mesh in 2D FEM:
- B: contains the coordinates for each node
- C: contains the node indices for each triangle
- D: specifies for each node whether it is on the boundary (in this case the value is 1, otherwise 0)
- E: contains the coordinates of the centroid for each triangle
"""

# import needed libraries and packages:
import numpy as np
import matplotlib.pyplot as plt

def getTriangulationMatrices(N):
    # This function returns the matrices B, C, D, and E mentioned above for a standard structured
    # triangular grid of the unit square with N**2 inner grid points.

    # Total number of nodes (including boundary nodes)
    nNodes = (N + 2)**2

    # Grid size
    h = 1. / (N + 1)

    # Arrays of grid points in x and y direction
    x = np.linspace(0, 1, N + 2)
    y = x

    # Create coordinates for all nodes and reshape the resulting arrays into vectors
    X, Y = np.meshgrid(x, y)
    Xvec = np.reshape(X, (-1, 1))
    Yvec = np.reshape(Y, (-1, 1))

    # Get the matrix B by putting the x and y coordinates into its columns
    B = np.concatenate((Xvec, Yvec), axis=1)

    # Total number of triangles
    nTriangles = 2 * (N + 1)**2

    # Initialize the C matrix
    C = np.zeros((nTriangles, 3), dtype=int)

    # Create arrays of indices for the nodes which are in the bottom-left corners of the (N+1)**2
    # squares resulting from decomposing Omega along each dimension in N+1 intervals (the order of
    # the nodes is assumed to be reverse lexicographical, i.e., (x_0,y_0), (x_1,y_0), ...
    # (x_{N+1},y_0), (x_0,y_1), (x_1,y_1), ..., (x_{N+1},y_1), ..., (x_{N+1}, y_{N+1}))
    indicesOfBottomLeftNodesInSquares = np.reshape(np.add.outer((N + 2) * np.arange(N + 1), np.arange(N + 1)), (-1,))

    # Similarly, obtain the indices of the nodes in the bottom right, top left, and top right corners
    # of the squares
    indicesOfBottomRightNodesInSquares = 1 + indicesOfBottomLeftNodesInSquares
    indicesOfTopLeftNodesInSquares = N + 2 + indicesOfBottomLeftNodesInSquares
    indicesOfTopRightNodesInSquares = 1 + indicesOfTopLeftNodesInSquares

    # Get the C matrix by adding the node indices to its columns (the order of the triangles is
    # assumed such that we first go in reverse lexicographical order through the triangles formed
    # by the bottom left, bottom right, and top right nodes and then through the triangles formed
    # by the bottom left, top left, and top right nodes of each square)
    C[:, 0] = np.concatenate((indicesOfBottomLeftNodesInSquares, indicesOfBottomLeftNodesInSquares))
    C[:, 1] = np.concatenate((indicesOfBottomRightNodesInSquares, indicesOfTopLeftNodesInSquares))
    C[:, 2] = np.concatenate((indicesOfTopRightNodesInSquares, indicesOfTopRightNodesInSquares))

    # Initialize the D matrix by ones
    D = np.ones((nNodes, 1), dtype=int)

    # Get the indices of all inner nodes
    indicesOfInnerNodes = np.reshape(np.add.outer((N + 2) * np.arange(1, N + 1), np.arange(1, N + 1)), (-1, 1))

    # Set those entries of D which correspond to inner nodes to 0.
    D[indicesOfInnerNodes, 0] = np.zeros((len(indicesOfInnerNodes), 1))

    # Initialize the E matrix
    E = np.zeros((nTriangles, 2))

    # Fill the columns of the E matrix with the coordinates of the centroids (the centroid of a triangle
    # may be calculated by adding the coordinates of the vertices and dividing by 3)
    E[:, 0] = np.sum(B[C, np.zeros_like(C)], axis=1) / 3.
    E[:, 1] = np.sum(B[C, np.ones_like(C)], axis=1) / 3.

    return B, C, D, E

if __name__ == "__main__":

    # Call the function and plot the mesh for N=5 
    # TODO 
    N = 5
    B, C, D, E = getTriangulationMatrices(N)
    
    # Plotting the mesh
    plt.figure(figsize=(6,6))
    # x coordinate, y coordinate, the traingles
    plt.triplot(B[:, 0], B[:, 1], C)
    # plotting nodes as red circles
    plt.plot(B[:, 0], B[:, 1], 'ro')
    plt.title(f"Mesh for N={N}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()