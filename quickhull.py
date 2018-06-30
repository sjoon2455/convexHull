def quickhull(listPts):
    """Returns the convex hull vertices computed using the
         Quickhull algorithm as a list of m tuples
         [(u0,v0), (u1,v1), ...]
    """
    #Finding the minimum and mximum x points
    min_x = (float('inf'), 0)
    max_x = (0, 0)
    for point in listPts:
        if point[0] < min_x[0]: min_x = point
        if point[0] > max_x[0]: max_x = point
    #Clculating the convex hull using the quickhull_helper function
    chull = quickhull_helper(listPts, min_x, max_x)
    chull += quickhull_helper(listPts, max_x, min_x)
    #Returning the convex hull
    return chull

def quickhull_helper(listPts, min_x, max_x):
    """
        Helper function for the quickhull algorithm,
        which performs the bulk of the algorithm and returns
        semi complete convex hulls
    """
    #Getting all the points to the left of the line formed by the minimum and maximum x points
    left_points = [point for point in listPts if is_counter(min_x, max_x, point)]
    #Getting the farest left point
    far_point = farest_point(min_x, max_x, left_points)
    #Checking if there is no points to the left
    if far_point == (-1, -1): return [max_x]
    #Recursively getting the points of the convex hull
    hullPts = quickhull_helper(left_points, min_x, far_point)
    hullPts = hullPts + quickhull_helper(left_points, far_point, max_x)
    #Returning the hull points from the passed set of points
    return hullPts

def farest_point(A, B, listPts):
    """
        Returns the point from a passed list which is
        the farest from the passed line AB
    """
    #Setting up the minimum and maximum x points
    max_point = (-1, -1)
    max_distance = 0
    #Looping over all points
    for point in listPts:
        #Checking the current point isnt one of the bounding points
        if point not in [A, B]:
            #Calculating the distance from the bounding line to the point
            temp = abs((B[1] - A[1]) * point[0] - (B[0] - A[0]) * point[1] + B[0] * A[1] - B[s1] * A[0])
            distance = (temp) / (((B[1] - A[1])**2 + (B[0] - A[0]) ** 2) ** 0.5)
            #Checking if the current point is further away than the current furthest point
            if distance > max_distance:
                max_distance = distance
                max_point = point
    #Returning the furthest point
    return max_point(A[1] - X[1]) * (B[0] - X[0])



def is_counter(A, B, Y):
    """Returns boolean flag holding if the turn
        of the three points passed is counter clockwise
    """
    #Using the line function to check if the line forms a counter clock wise turn
    return (((B[0] - A[0])*(Y[1] - A[1]) -
            ((B[1] - A[1])*(Y[0] - A[0])))) > 0