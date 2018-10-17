# Convex-Hulls
The three algorithms compared are The Giftwrap algorithm, The Graham-Scan algorithm and The Monotone Chain algorihtm.  
These algorithms are compared over a wide range of data sets using pre writen tests.  

## Usage
  
### Running Algorithms  
To run each algorithm individually uncomment the line corresponding to the chosen algorithm and select the chosen  
data file then run the file. Lines to uncomment to run the algorithms can be found a the bottom of the `convexhull.py` file.  
  
### Running Acceptance Tests  
To run the acceptance tests uncomment the line `tests.output_tests()`, line 225 of convexhulls.py.  

### Running Average Time Tests  
To run average time tests uncomment the line `tests.average_tests(1, True)` line 229 of convexhulls.py and/or  
the line `tests.average_tests(1, False)` line 233 of convexhulls.py.  

### Creating Graphs  
To make graphs of the algorithms performance uncomment any of the following lines of convexhulls.py `189, 193,  
197, 201, 205, 211, 217`. Graphs are made using average times gathered by running each algorithm 200 times over all  
data files and taking the average time from each data file.
