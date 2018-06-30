"""
   Convex Hull Assignment: COSC262 (2018)
   Student Name: Ambrose Ledbrook
   Usercode: ajl190
"""
#Imports used in calculating the convex hulls with their times
from numpy import *
from outputs import *
import tests as tests
import Graphs as graphs
import time

def readDataPts(filename, N):
    """Reads the first N lines of data from the input file
          and returns a list of N tuples
          [(x0,y0), (x1, y1), ...]
    """
    #Opening file
    file = open(filename, 'r')
    listPts = []
    #Looping over the number of lines in the file and adding each point to the list
    for i in range(N):
        #Reading line of the file
        line = file.readline().strip().split(" ")
        #Appending tuple holding point that was read from file
        listPts.append((float(line[0]), float(line[1])))
    #Returning list of tuples holding all points in the file
    return listPts

def giftwrap(listPts):
    """Returns the convex hull vertices computed using the
          giftwrap algorithm as a list of m tuples
          [(u0,v0), (u1,v1), ...]
    """
    #Starting time used to test algorithm against others
    start = time.time()
    #Checking for base case where the list is of size 3 or less
    if len(listPts) <= 3:
        return listPts
    #Geting minimum y value point of the list
    min_y, k_index = minYValue(listPts)
    #Setting up inital variables used in algorithm
    last_angle = 0
    index = 0
    chull = []
    listPts.append(min_y)
    n = len(listPts) - 1
    #Looping over points until the end point of the hull is found
    while k_index != n:
        #Swapping points at k and index
        listPts[index], listPts[k_index] = listPts[k_index], listPts[index]
        #Adding point to hull
        chull.append(listPts[index])
        minAngle = 361
        #Looping over points
        for j in range(index+1, n+1):
            #Getting the angle between horozontal line and point at j
            angle = theta(listPts[index], listPts[j])
            #Checking for minimum angle
            if angle < minAngle and angle > last_angle and listPts[j] != listPts[index]:
                minAngle = angle
                k_index = j
        #Setting variable for the last abgle
        last_angle = minAngle
        #Incrementing index
        index += 1
    #Gettign end time used for analysis
    end = time.time()
    #Returning convex hull and time taken
    return chull, end - start

def minYValue(listPts):
    """Returns the point with the minimum y value
        from a passed list of points
    """
    k_index = 0
    #Setting default value for minimum y point
    min_y = [float('inf'), float('inf')]
    #Looping over all points
    for index, point in enumerate(listPts):
        #Checking if points y value is less than the y value of the current minimum
        if point[1] < min_y[1]:
            min_y = point
            k_index = index
        #Handling when there are two points of equal minimum y value
        elif point[1] == min_y[1]:
            if point[0] > min_y[0]:
                min_y = point
                k_index = index
    #Returning minimum y point and index
    return min_y, k_index

def theta(A, B):
    """Calculates an approximation of the
       angle between the line AB and a horozontal
       line through A
    """
    #Handling case where the points are equal
    if A == B:
        return 0
    #Calculating dx and dy
    dx = B[0] - A[0]
    dy = B[1] - A[1]
    #Checking degenerate case
    if abs(dx) < 1.e-6 and abs(dy) < 1.e-6: t = 0
    #Calculating t
    else: t = dy/((abs(dx)) + abs(dy))
    #Finding angle dependent of the value of t
    if dx < 0: t = 2 - t
    elif dy < 0: t = 4 + t
    #Returning the angle
    if t == 0: return 360
    else: return t*90

def grahamscan(listPts):
    """Returns the convex hull vertices computed using the
         Graham-scan algorithm as a list of m tuples
         [(u0,v0), (u1,v1), ...]
    """
    #Starting timer used for analysis
    start = time.time()
    #Checking for base case where the list is of size 3 or less
    if len(listPts) <= 3:
        return listPts
    #Getting the minimum y point
    min_y, k = minYValue(listPts)
    #Calculating the angles of all the points compared to the minimum y point
    pts_with_angles = []
    for point in listPts:
        pts_with_angles.append((point, theta(min_y, point)))
    #Sorting the list of points swith there angles
    sorted_pts = sorted(pts_with_angles, key=lambda i: i[1])
    #Addign the first three points to the stack
    stack = [sorted_pts[0][0], sorted_pts[1][0], sorted_pts[2][0]]
    #Looping over all points
    for index in range(3, len(sorted_pts)):
        #Checking that the top 2 values of the stack and the next point form a counter clock wise turn
        while not is_counter(stack[-2], stack[-1], sorted_pts[index][0]):
            stack.pop()
        #Adding the next point to the stack
        stack.append(sorted_pts[index][0])
    #Getting the end time used for analysis
    end = time.time()
    #Returning the convex hull and the time taken
    return stack,end - start

def is_counter(A, B, Y):
    """Returns boolean flag holding if the turn
        of the three points passed is counter clockwise
    """
    #Using the line function to check if the line forms a counter clock wise turn
    return (((B[0] - A[0])*(Y[1] - A[1]) -
            ((B[1] - A[1])*(Y[0] - A[0])))) > 0

def monotone_chain(listPts):
    """Returns the convex hull vertices computed using the
         Monotone chain algorithm as a list of m tuples
         [(u0,v0), (u1,v1), ...]
    """
    #Starting timer
    start = time.time()
    #Checking for base case where the list is of size 3 or less
    if len(listPts) <= 3:
        return listPts
    #Sorting the points
    sorted_pts = sorted(listPts)
    #Computing the bottom half of the hull
    bottom_hull = []
    for point in sorted_pts:
        while len(bottom_hull) >= 2 and cross_product(bottom_hull[-2], bottom_hull[-1], point) <= 0:
            bottom_hull.pop()
        bottom_hull.append(point)
    #Computing the top half of the hull
    upper_hull = []
    for point in reversed(sorted_pts):
        while len(upper_hull) >= 2 and cross_product(upper_hull[-2], upper_hull[-1], point) <= 0:
            upper_hull.pop()
        upper_hull.append(point)
    #Ending the timer
    end = time.time()
    #Returning the hull and time taken
    return bottom_hull[:-1] + upper_hull[:-1], end - start

def cross_product(X, A, B):
    """
    returns the cross product of passed points
    """
    return (A[0] - X[0]) * (B[1] - X[1]) - (A[1] - X[1]) * (B[0] - X[0])

def main():
    #------------------------------------------------------------------
    """
        File names may need to be changed dependent on their location
    """
    #Uncomment to produce the hull of a selected file with the time taken
    listPts = readDataPts("Set_A/A_3000.dat", 3000)
    gft_hull, gift_time = giftwrap(listPts[:])
    print(gft_hull, "Time taken: ", gift_time)
    grs_hull, grs_time = grahamscan(listPts[:])
    print(grs_hull, "Time taken: ", grs_time)
    mono_hull, mono_time = monotone_chain(listPts[:])
    print(mono_hull, "Time taken: ", mono_time)
    #------------------------------------------------------------------

if __name__  ==  "__main__":
    main()