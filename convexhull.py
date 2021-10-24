"""
   Convex Hull Assignment: COSC262 (2018)
   Student Name: Ambrose Ledbrook
   Usercode: ajl190
"""
# Imports used in calculating the convex hulls
# from numpy import *
from functools import reduce
import numpy as np
import tests as tests
import Graphs as graphs
import time
import math
from random import randint
from collections import namedtuple
from typing import Iterable, List, Set, Union
from binarytree import binarytree
from symmetric import symmetric


def readDataPts(filename, N):
    """Reads the first N lines of data from the input file
          and returns a list of N tuples
          [(x0,y0), (x1, y1), ...]
    """
    # Opeing file
    file = open(filename, 'r')
    listPts = []
    # Looping over the number of lines in the file and adding each point to the list
    for i in range(N):
        # Reading line of the file
        line = file.readline().strip().split(" ")
        # Appending tuple holding point that was read from file
        listPts.append((float(line[0]), float(line[1])))
    # Returning list of tuples holding all points in the file
    return listPts


def giftwrap(listPts):
    """Returns the convex hull vertices computed using the
          giftwrap algorithm as a list of m tuples
          [(u0,v0), (u1,v1), ...]
    """
    start = time.time()
    # Checking for base case where the list is of size 3 or less
    if len(listPts) <= 3:
        return listPts
    # Geting minimum y value point of the list
    min_y, k_index = minYValue(listPts)
    # Setting up inital variables used in algorithm
    last_angle = 0
    index = 0
    chull = []
    listPts.append(min_y)
    n = len(listPts) - 1
    # Looping over points until the end point of the hull is found
    while k_index != n:
        # Swapping points at k and index
        listPts[index], listPts[k_index] = listPts[k_index], listPts[index]
        # Adding point to hull
        chull.append(listPts[index])
        minAngle = 361
        # Looping over points
        for j in range(index+1, n+1):
            # Getting the angle between horozontal line and point at j
            angle = theta(listPts[index], listPts[j])
            # Checking for minimum angle
            if angle < minAngle and angle > last_angle and listPts[j] != listPts[index]:
                minAngle = angle
                k_index = j
        # Setting variable for the last abgle
        last_angle = minAngle
        # Incrementing index
        index += 1
    # Returning convex hull
    end = time.time()
    return chull, end - start


def minYValue(listPts):
    """Returns the point with the minimum y value
        from a passed list of points
    """
    k_index = 0
    # Setting default value for minimum y point
    min_y = [float('inf'), float('inf')]
    # Looping over all points
    for index, point in enumerate(listPts):
        # Checking if points y value is less than the y value of the current minimum
        if point[1] < min_y[1]:
            min_y = point
            k_index = index
        # Handling when there are two points of equal minimum y value
        elif point[1] == min_y[1]:
            if point[0] > min_y[0]:
                min_y = point
                k_index = index
    # Returning minimum y point and index
    return min_y, k_index


def theta(A, B):
    """Calculates an approximation of the
       angle between the line AB and a horozontal
       line through A
    """
    # Handling case where the points are equal
    if A == B:
        return 0
    # Calculating dx and dy
    dx = B[0] - A[0]
    dy = B[1] - A[1]
    # Checking degenerate case
    if abs(dx) < 1.e-6 and abs(dy) < 1.e-6:
        t = 0
    # Calculating t
    else:
        t = dy/((abs(dx)) + abs(dy))
    # Finding angle dependent of the value of t
    if dx < 0:
        t = 2 - t
    elif dy < 0:
        t = 4 + t
    # Returning the angle
    if t == 0:
        return 360
    else:
        return t*90


def grahamscan(listPts):
    """Returns the convex hull vertices computed using the
         Graham-scan algorithm as a list of m tuples
         [(u0,v0), (u1,v1), ...]
    """
    start = time.time()
    # Checking for base case where the list is of size 3 or less
    if len(listPts) <= 3:
        return listPts
    # Getting the minimum y point
    min_y, k = minYValue(listPts)
    # Calculating the angles of all the points compared to the minimum y point
    pts_with_angles = []
    for point in listPts:
        pts_with_angles.append((point, theta(min_y, point)))
    # Sorting the list of points swith there angles
    sorted_pts = sorted(pts_with_angles, key=lambda i: i[1])
    # Addign the first three points to the stack
    stack = [sorted_pts[0][0], sorted_pts[1][0], sorted_pts[2][0]]
    # Looping over all points
    for index in range(3, len(sorted_pts)):
        # Checking that the top 2 values of the stack and the next point form a counter clock wise turn
        while not is_counter(stack[-2], stack[-1], sorted_pts[index][0]):
            stack.pop()
        # Adding the next point to the stack
        stack.append(sorted_pts[index][0])
    # Returning the convex hull
    end = time.time()
    return stack, end - start


def is_counter(A, B, Y):
    """Returns boolean flag holding if the turn
        of the three points passed is counter clockwise
    """
    # Using the line function to check if the line forms a counter clock wise turn
    return (((B[0] - A[0])*(Y[1] - A[1]) -
             ((B[1] - A[1])*(Y[0] - A[0])))) > 0


def monotone_chain(listPts):
    """Returns the convex hull vertices computed using the
         Monotone chain algorithm as a list of m tuples
         [(u0,v0), (u1,v1), ...]
    """
    start = time.time()
    # Checking for base case where the list is of size 3 or less
    if len(listPts) <= 3:
        return listPts
    # Sorting the points
    sorted_pts = sorted(listPts)

    # Computing the bottom half of the hull
    bottom_hull = []
    for point in sorted_pts:
        while len(bottom_hull) >= 2 and cross_product(bottom_hull[-2], bottom_hull[-1], point) <= 0:
            bottom_hull.pop()
        bottom_hull.append(point)
    # Computing the top half of the hull
    upper_hull = []
    for point in reversed(sorted_pts):
        while len(upper_hull) >= 2 and cross_product(upper_hull[-2], upper_hull[-1], point) <= 0:
            upper_hull.pop()
        upper_hull.append(point)
    # Returning the hull
    end = time.time()
    return bottom_hull[:-1] + upper_hull[:-1], end - start

# incremental part

# Returns true if point p is left of the line ab


def isLeftOf(p, a, b):
    return (np.sign((b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])) >= 0)

# Returns true if point p is right of the line ab


def isRightOf(p, a, b):
    return (np.sign((b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])) <= 0)

# Returns true if the point p is upper tangent of point p.
# q1 is the previous point of p and q2 is the next point, when moving CCW in a polygon


def isUpperTangent(p, q, q1, q2):
    return isLeftOf(p, q, q2) and isRightOf(p, q1, q)


def isLowerTangent(p, q, q1, q2):
    return isRightOf(p, q, q2) and isLeftOf(p, q1, q)


def incremental(listPts):
    start = time.time()
    allPoints = sorted(listPts)

    # Start with a trivial hull(a triangle of the first points)
    hullPoints = allPoints[:3]

    # Store edges in CCW (counter-clock wise) order
    hullEdge = {}
    # print(33,type(hullPoints[0]), type(hullPoints[1]), type(hullPoints[2]))
    if (isRightOf(hullPoints[0], hullPoints[1], hullPoints[2])):
        hullEdge = {
            hullPoints[0]: {'prev': hullPoints[1], 'next': hullPoints[2]},
            hullPoints[1]: {'prev': hullPoints[2], 'next': hullPoints[0]},
            hullPoints[2]: {'prev': hullPoints[0], 'next': hullPoints[1]}
        }
    else:
        hullEdge = {
            hullPoints[0]: {'prev': hullPoints[2], 'next': hullPoints[1]},
            hullPoints[2]: {'prev': hullPoints[1], 'next': hullPoints[0]},
            hullPoints[1]: {'prev': hullPoints[0], 'next': hullPoints[2]}
        }

    n = len(allPoints)

    # One by one add the remaining vertices to the convex hull
    # and remove vertices that are inside it
    for i in range(3, n):
        pi = allPoints[i]
        # print('Adding point pi=%s'%(pi))

        # Let j be the rightmost index of the convex hull
        j = len(hullPoints) - 1

        # Look for upper tangent point
        u = j
        upperTangent = hullPoints[u]
        while(not isUpperTangent(pi, upperTangent, hullEdge[upperTangent]['prev'], hullEdge[upperTangent]['next'])):
            # print('- its not %s'%(upperTangent))
            u -= 1
            upperTangent = hullPoints[u]
        # print('  Upper tangent point: %s' %(upperTangent))

        # Look for lower tangent point by iterating over the vertices that are
        # previous of upperTangent, one by one until it is found
        lowerTangent = hullEdge[upperTangent]['prev']
        while(not isLowerTangent(pi, lowerTangent, hullEdge[lowerTangent]['prev'], hullEdge[lowerTangent]['next'])):
            # print('     Removing %s' %(lowerTangent))
            temp = lowerTangent
            lowerTangent = hullEdge[lowerTangent]['prev']
            hullEdge.pop(temp, None)
            hullPoints.remove(temp)
        # print('  Lower tangent point: %s' %(lowerTangent))

        # Update convex hull by adding the new point
        hullPoints.append(pi)

        # Update edges
        hullEdge[pi] = {'prev': lowerTangent, 'next': upperTangent}
        hullEdge[lowerTangent]['next'] = pi
        hullEdge[upperTangent]['prev'] = pi
    # print('Generating plot...')
    # Convert points and edges into a np.arrays in order to plot them
    pointsArray = list()
    for point in allPoints:
        pointsArray.append([point[0], point[1]])
    # pointsArray = np.array(pointsArray)

    hullEdge[pi] = {'prev': lowerTangent, 'next': upperTangent}
    hullArray = list()
    point = hullPoints[0]
    for i in range(len(hullPoints)):
        # hullArray.append([point[0], point[1]])
        hullArray.append((point[0], point[1]))
        point = hullEdge[point]['next']

    hullArray.append(hullArray[0])
    # hullArray = np.array(hullArray)
    end = time.time()

    return hullArray, end - start

# Chan ------------


TURN_LEFT, TURN_RIGHT, TURN_NONE = (1, -1, 0)


def cmp(a, b):
    return (a > b) - (a < b)


def turn(p, q, r):
    """Returns -1, 0, 1 if p,q,r forms a right, straight, or left turn."""
    return cmp((q[0] - p[0])*(r[1] - p[1]) - (r[0] - p[0])*(q[1] - p[1]), 0)


def _keep_left(hull, r):
    while len(hull) > 1 and turn(hull[-2], hull[-1], r) != TURN_LEFT:
        hull.pop()
    return (not len(hull) or hull[-1] != r) and hull.append(r) or hull


def _graham_scan(points):
    """Returns points on convex hull of an array of points in CCW order."""
    points.sort()
    lh = reduce(_keep_left, points, [])
    uh = reduce(_keep_left, reversed(points), [])
    return lh.extend(uh[i] for i in range(1, len(uh) - 1)) or lh


def _rtangent(hull, p):
    """Return the index of the point in hull that the right tangent line from p
    to hull touches.
    """
    l, r = 0, len(hull)
    l_prev = turn(p, hull[0], hull[-1])
    l_next = turn(p, hull[0], hull[(l + 1) % r])
    while l < r:
        c = (l + r) // 2
        c_prev = turn(p, hull[c], hull[(c - 1) % len(hull)])
        c_next = turn(p, hull[c], hull[(c + 1) % len(hull)])
        c_side = turn(p, hull[l], hull[c])
        if c_prev != TURN_RIGHT and c_next != TURN_RIGHT:
            return c
        elif c_side == TURN_LEFT and (l_next == TURN_RIGHT or
                                      l_prev == l_next) or \
                c_side == TURN_RIGHT and c_prev == TURN_RIGHT:
            r = c               # Tangent touches left chain
        else:
            l = c + 1           # Tangent touches right chain
            l_prev = -c_next    # Switch sides
            l_next = turn(p, hull[l], hull[(l + 1) % len(hull)])
    return l


def _min_hull_pt_pair(hulls):
    """Returns the hull, point index pair that is minimal."""
    h, p = 0, 0
    for i in range(len(hulls)):
        j = min(range(len(hulls[i])), key=lambda j: hulls[i][j])
        if hulls[i][j] < hulls[h][p]:
            h, p = i, j
    return (h, p)


def _next_hull_pt_pair(hulls, pair):
    """
    Returns the (hull, point) index pair of the next point in the convex
    hull.
    """
    p = hulls[pair[0]][pair[1]]
    next = (pair[0], (pair[1] + 1) % len(hulls[pair[0]]))
    for h in (i for i in range(len(hulls)) if i != pair[0]):
        s = _rtangent(hulls[h], p)
        q, r = hulls[next[0]][next[1]], hulls[h][s]
        t = turn(p, q, r)
        if t == TURN_RIGHT or t == TURN_NONE and _dist(p, r) > _dist(p, q):
            next = (h, s)
    return next


def Chan(pts):
    start = time.time()
    """Returns the points on the convex hull of pts in CCW order."""
    for m in (1 << (1 << t) for t in range(len(pts))):
        hulls = [_graham_scan(pts[i:i + m]) for i in range(0, len(pts), m)]
        hull = [_min_hull_pt_pair(hulls)]
        for throw_away in range(m):
            p = _next_hull_pt_pair(hulls, hull[-1])
            if p == hull[0]:
                end = time.time()
                return [hulls[h][i] for h, i in hull], end-start
            hull.append(p)
    end = time.time()
    return hull, end - start

# -----------------Quick Hull


def quickHull(S):
    start = time.time()
    hull = []
    S = sorted(S, key=lambda x: x[0])
    hull.append(S[0])
    hull.append(S[-1])
    S1 = []
    S2 = []

    for x in S:
        if (sideOfLinePointIsOn([S[0], S[-1]], x) > 0.00):
            S2.append(x)

        if(sideOfLinePointIsOn([S[0], S[-1]], x) < 0.00):
            S1.append(x)

    findHull(S1, S[0], S[-1], hull)
    findHull(S2, S[-1], S[0], hull)
    return hull, time.time()-start


def findHull(Sk, P, Q, hull):
    if(len(Sk) == 0):
        return
    furthestPoint = Sk[0]
    maxDist = 0
    for x in Sk:

        dist = findDistPointLine(x, [P, Q])
        if(dist > maxDist):
            maxDist = dist
            furthestPoint = x

    Sk.remove(furthestPoint)

    hull.insert(1, furthestPoint)
    S1 = []
    S2 = []
    for p in Sk:
        if(isInsideTriangle(P, furthestPoint, Q, p)):
            Sk.remove(p)
        if(sideOfLinePointIsOn([P, furthestPoint], p) > 0.00):
            S2.append(x)
        if(sideOfLinePointIsOn([furthestPoint, Q], x) < 0.00):
            S1.append(x)
    findHull(S1, P, furthestPoint, hull)
    findHull(S2, furthestPoint, Q, hull)


def slopeCalc(P, Q):
    return (Q[1] - P[1])/(Q[0]-P[0])


# y =ax+b, y=cx+d
# x = (d-b)/(a-c)
def findNearestPoint(point, line):
    line = sorted(line, key=lambda x: x[0])
    # reg line
    slope = slopeCalc(line[0], line[1])

    yInt = line[0][1] - (slope * line[0][0])

    # neg reciprocal
    if slope != 0:
        slopeRecip = (-1)/slope
        yIntRecip = point[1] - (slopeRecip * line[0][0])

        nearestPoint = [0, 0]
        nearestPoint[0] = (yIntRecip - yInt)/(slope-slopeRecip)
        nearestPoint[1] = slope * nearestPoint[0] + yInt
    else:
        nearestPoint = [0, 0]
    # if nearestPoint x val is less than min x val on segment PQ
    if(nearestPoint[0] < line[0][0]):
        return line[0]

    # if nearestPoint x val is greater than max x val on segment PQ
    if(nearestPoint[0] > line[-1][0]):
        return line[-1]

    return nearestPoint


def findPointDistance(P1, P2):

    return math.hypot(P2[0] - P1[0], P2[1] - P1[1])


def findDistPointLine(point, line):
    return findPointDistance(point, findNearestPoint(point, line))


def sideOfLinePointIsOn(line, x):
    vectAB = ((line[1][0] - line[0][0]), line[1][1] - line[0][1])
    vectAX = ((x[0] - line[0][0]), x[1] - line[0][1])
    zCoord = (vectAB[0] * vectAX[1]) - (vectAB[1] * vectAX[0])
    return zCoord


def isInsideTriangle(A, B, C, p):
    if(
        (sideOfLinePointIsOn((A, B), p) > 0) and
        (sideOfLinePointIsOn((B, C), p) > 0) and
            (sideOfLinePointIsOn((C, A), p) > 0)):
        return True


def cross_product(X, A, B):
    """
    returns the cross product of passed points
    """
    return (A[0] - X[0]) * (B[1] - X[1]) - (A[1] - X[1]) * (B[0] - X[0])


# -----------------Kirkpatrick-Seidel
Point = namedtuple('Point', 'x y')


def flipped(points):
    return [Point(-point.x, -point.y) for point in points]


def quickselect(ls, index, lo=0, hi=None, depth=0):
    if hi is None:
        hi = len(ls)-1
    if lo == hi:
        return ls[lo]
    pivot = randint(lo, hi)
    ls = list(ls)
    ls[lo], ls[pivot] = ls[pivot], ls[lo]
    cur = lo
    for run in range(lo+1, hi+1):
        if ls[run] < ls[lo]:
            cur += 1
            ls[cur], ls[run] = ls[run], ls[cur]
    ls[cur], ls[lo] = ls[lo], ls[cur]
    if index < cur:
        return quickselect(ls, index, lo, cur-1, depth+1)
    elif index > cur:
        return quickselect(ls, index, cur+1, hi, depth+1)
    else:
        return ls[cur]


def bridge(points, vertical_line):
    candidates = set()
    if len(points) == 2:
        return tuple(sorted(points))
    pairs = []
    modify_s = set(points)
    while len(modify_s) >= 2:
        pairs += [tuple(sorted([modify_s.pop(), modify_s.pop()]))]
    if len(modify_s) == 1:
        candidates.add(modify_s.pop())
    slopes = []
    for pi, pj in pairs[:]:
        if pi.x == pj.x:
            pairs.remove((pi, pj))
            candidates.add(pi if pi.y > pj.y else pj)
        else:
            slopes += [(pi.y-pj.y)/(pi.x-pj.x)]
    median_index = len(slopes)//2 - (1 if len(slopes) % 2 == 0 else 0)
    median_slope = quickselect(slopes, median_index)
    small = {pairs[i]
             for i, slope in enumerate(slopes) if slope < median_slope}
    equal = {pairs[i]
             for i, slope in enumerate(slopes) if slope == median_slope}
    large = {pairs[i]
             for i, slope in enumerate(slopes) if slope > median_slope}
    max_slope = max(point.y-median_slope*point.x for point in points)
    max_set = [point for point in points if point.y -
               median_slope*point.x == max_slope]
    left = min(max_set)
    right = max(max_set)
    if left.x <= vertical_line and right.x > vertical_line:
        return (left, right)
    if right.x <= vertical_line:
        candidates |= {point for _, point in large | equal}
        candidates |= {point for pair in small for point in pair}
    if left.x > vertical_line:
        candidates |= {point for point, _ in small | equal}
        candidates |= {point for pair in large for point in pair}
    return bridge(candidates, vertical_line)


def connect(lower, upper, points):
    if lower == upper:
        return [lower]
    max_left = quickselect(points, len(points)//2-1)
    min_right = quickselect(points, len(points)//2)
    left, right = bridge(points, (max_left.x + min_right.x)/2)
    points_left = {left} | {point for point in points if point.x < left.x}
    points_right = {right} | {point for point in points if point.x > right.x}
    return connect(lower, left, points_left) + connect(right, upper, points_right)


def upper_hull(points):
    lower = min(points)
    lower = max({point for point in points if point.x == lower.x})
    upper = max(points)
    points = {lower, upper} | {p for p in points if lower.x < p.x < upper.x}
    return connect(lower, upper, points)


def kirk_seidel(points):
    start = time.time()
    points = {Point(x, y) for (x, y) in points}
    upper = upper_hull(points)
    lower = flipped(upper_hull(flipped(points)))
    if upper[-1] == lower[0]:
        del upper[-1]
    if upper[0] == lower[-1]:
        del lower[-1]
    res = [(p.x, p.y) for p in upper+lower]
    end = time.time()
    return res, end-start

#---------------- Divide and Conquer


class Point:
    """
    Defines a 2-d point for use by all convex-hull algorithms.
    Parameters
    ----------
    x: an int or a float, the x-coordinate of the 2-d point
    y: an int or a float, the y-coordinate of the 2-d point
    Examples
    --------
    >>> Point(1, 2)
    (1.0, 2.0)
    >>> Point("1", "2")
    (1.0, 2.0)
    >>> Point(1, 2) > Point(0, 1)
    True
    >>> Point(1, 1) == Point(1, 1)
    True
    >>> Point(-0.5, 1) == Point(0.5, 1)
    False
    >>> Point("pi", "e")
    Traceback (most recent call last):
        ...
    ValueError: could not convert string to float: 'pi'
    """

    def __init__(self, x, y):
        self.x, self.y = float(x), float(y)

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __ne__(self, other):
        return not self == other

    def __gt__(self, other):
        if self.x > other.x:
            return True
        elif self.x == other.x:
            return self.y > other.y
        return False

    def __lt__(self, other):
        return not self > other

    def __ge__(self, other):
        if self.x > other.x:
            return True
        elif self.x == other.x:
            return self.y >= other.y
        return False

    def __le__(self, other):
        if self.x < other.x:
            return True
        elif self.x == other.x:
            return self.y <= other.y
        return False

    def __repr__(self):
        return f"({self.x}, {self.y})"

    def __hash__(self):
        return hash(self.x)


def _construct_points(
    list_of_tuples: Union[List[Point],
                          List[List[float]], Iterable[List[float]]]
) -> List[Point]:
    """
    constructs a list of points from an array-like object of numbers
    Arguments
    ---------
    list_of_tuples: array-like object of type numbers. Acceptable types so far
    are lists, tuples and sets.
    Returns
    --------
    points: a list where each item is of type Point. This contains only objects
    which can be converted into a Point.
    Examples
    -------
    >>> _construct_points([[1, 1], [2, -1], [0.3, 4]])
    [(1.0, 1.0), (2.0, -1.0), (0.3, 4.0)]
    >>> _construct_points([1, 2])
    Ignoring deformed point 1. All points must have at least 2 coordinates.
    Ignoring deformed point 2. All points must have at least 2 coordinates.
    []
    >>> _construct_points([])
    []
    >>> _construct_points(None)
    []
    """

    points: List[Point] = []
    if list_of_tuples:
        for p in list_of_tuples:
            if isinstance(p, Point):
                points.append(p)
            else:
                try:
                    points.append(Point(p[0], p[1]))
                except (IndexError, TypeError):
                    print(
                        f"Ignoring deformed point {p}. All points"
                        " must have at least 2 coordinates."
                    )
    return points


def _validate_input(points: Union[List[Point], List[List[float]]]) -> List[Point]:
    """
    validates an input instance before a convex-hull algorithms uses it
    Parameters
    ---------
    points: array-like, the 2d points to validate before using with
    a convex-hull algorithm. The elements of points must be either lists, tuples or
    Points.
    Returns
    -------
    points: array_like, an iterable of all well-defined Points constructed passed in.
    Exception
    ---------
    ValueError: if points is empty or None, or if a wrong data structure like a scalar
                 is passed
    TypeError: if an iterable but non-indexable object (eg. dictionary) is passed.
                The exception to this a set which we'll convert to a list before using
    Examples
    -------
    >>> _validate_input([[1, 2]])
    [(1.0, 2.0)]
    >>> _validate_input([(1, 2)])
    [(1.0, 2.0)]
    >>> _validate_input([Point(2, 1), Point(-1, 2)])
    [(2.0, 1.0), (-1.0, 2.0)]
    >>> _validate_input([])
    Traceback (most recent call last):
        ...
    ValueError: Expecting a list of points but got []
    >>> _validate_input(1)
    Traceback (most recent call last):
        ...
    ValueError: Expecting an iterable object but got an non-iterable type 1
    """

    if not hasattr(points, "__iter__"):
        raise ValueError(
            f"Expecting an iterable object but got an non-iterable type {points}"
        )

    if not points:
        raise ValueError(f"Expecting a list of points but got {points}")

    return _construct_points(points)


def _det(a: Point, b: Point, c: Point) -> float:
    """
    Computes the sign perpendicular distance of a 2d point c from a line segment
    ab. The sign indicates the direction of c relative to ab.
    A Positive value means c is above ab (to the left), while a negative value
    means c is below ab (to the right). 0 means all three points are on a straight line.
    As a side note, 0.5 * abs|det| is the area of triangle abc
    Parameters
    ----------
    a: point, the point on the left end of line segment ab
    b: point, the point on the right end of line segment ab
    c: point, the point for which the direction and location is desired.
    Returns
    --------
    det: float, abs(det) is the distance of c from ab. The sign
    indicates which side of line segment ab c is. det is computed as
    (a_xb_y + c_xa_y + b_xc_y) - (a_yb_x + c_ya_x + b_yc_x)
    Examples
    ----------
    >>> _det(Point(1, 1), Point(1, 2), Point(1, 5))
    0.0
    >>> _det(Point(0, 0), Point(10, 0), Point(0, 10))
    100.0
    >>> _det(Point(0, 0), Point(10, 0), Point(0, -10))
    -100.0
    """

    det = (a.x * b.y + b.x * c.y + c.x * a.y) - \
        (a.y * b.x + b.y * c.x + c.y * a.x)
    return det


def convex_hull_recursive(points: List[Point]) -> List[Point]:
    """
    Constructs the convex hull of a set of 2D points using a divide-and-conquer strategy
    The algorithm exploits the geometric properties of the problem by repeatedly
    partitioning the set of points into smaller hulls, and finding the convex hull of
    these smaller hulls.  The union of the convex hull from smaller hulls is the
    solution to the convex hull of the larger problem.
    Parameter
    ---------
    points: array-like of object of Points, lists or tuples.
    The set of  2d points for which the convex-hull is needed
    Runtime: O(n log n)
    Returns
    -------
    convex_set: list, the convex-hull of points sorted in non-decreasing order.
    Examples
    ---------
    >>> convex_hull_recursive([[0, 0], [1, 0], [10, 1]])
    [(0.0, 0.0), (1.0, 0.0), (10.0, 1.0)]
    >>> convex_hull_recursive([[0, 0], [1, 0], [10, 0]])
    [(0.0, 0.0), (10.0, 0.0)]
    >>> convex_hull_recursive([[-1, 1],[-1, -1], [0, 0], [0.5, 0.5], [1, -1], [1, 1],
    ...                        [-0.75, 1]])
    [(-1.0, -1.0), (-1.0, 1.0), (1.0, -1.0), (1.0, 1.0)]
    >>> convex_hull_recursive([(0, 3), (2, 2), (1, 1), (2, 1), (3, 0), (0, 0), (3, 3),
    ...                        (2, -1), (2, -4), (1, -3)])
    [(0.0, 0.0), (0.0, 3.0), (1.0, -3.0), (2.0, -4.0), (3.0, 0.0), (3.0, 3.0)]
    """
    start = time.time()

    points = sorted(_validate_input(points))
    n = len(points)

    # divide all the points into an upper hull and a lower hull
    # the left most point and the right most point are definitely
    # members of the convex hull by definition.
    # use these two anchors to divide all the points into two hulls,
    # an upper hull and a lower hull.

    # all points to the left (above) the line joining the extreme points belong to the
    # upper hull
    # all points to the right (below) the line joining the extreme points below to the
    # lower hull
    # ignore all points on the line joining the extreme points since they cannot be
    # part of the convex hull

    left_most_point = points[0]
    right_most_point = points[n - 1]

    convex_set = {left_most_point, right_most_point}
    upper_hull = []
    lower_hull = []

    for i in range(1, n - 1):
        det = _det(left_most_point, right_most_point, points[i])

        if det > 0:
            upper_hull.append(points[i])
        elif det < 0:
            lower_hull.append(points[i])

    _construct_hull(upper_hull, left_most_point, right_most_point, convex_set)
    _construct_hull(lower_hull, right_most_point, left_most_point, convex_set)
    end = time.time()

    return sorted(convex_set), end-start


def _construct_hull(
    points: List[Point], left: Point, right: Point, convex_set: Set[Point]
) -> None:
    """
    Parameters
    ---------
    points: list or None, the hull of points from which to choose the next convex-hull
        point
    left: Point, the point to the left  of line segment joining left and right
    right: The point to the right of the line segment joining left and right
    convex_set: set, the current convex-hull. The state of convex-set gets updated by
        this function
    Note
    ----
    For the line segment 'ab', 'a' is on the left and 'b' on the right.
    but the reverse is true for the line segment 'ba'.
    Returns
    -------
    Nothing, only updates the state of convex-set
    """
    if points:
        extreme_point = None
        extreme_point_distance = float("-inf")
        candidate_points = []

        for p in points:
            det = _det(left, right, p)

            if det > 0:
                candidate_points.append(p)

                if det > extreme_point_distance:
                    extreme_point_distance = det
                    extreme_point = p

        if extreme_point:
            _construct_hull(candidate_points, left, extreme_point, convex_set)
            convex_set.add(extreme_point)
            _construct_hull(candidate_points, extreme_point, right, convex_set)

#---------------- Melkman


def convex_hull_melkman(points: List[Point]) -> List[Point]:
    """
    Constructs the convex hull of a set of 2D points using the melkman algorithm.
    The algorithm works by iteratively inserting points of a simple polygonal chain
    (meaning that no line segments between two consecutive points cross each other).
    Sorting the points yields such a polygonal chain.
    For a detailed description, see http://cgm.cs.mcgill.ca/~athens/cs601/Melkman.html
    Runtime: O(n log n) - O(n) if points are already sorted in the input
    Parameters
    ---------
    points: array-like of object of Points, lists or tuples.
    The set of 2d points for which the convex-hull is needed
    Returns
    ------
    convex_set: list, the convex-hull of points sorted in non-decreasing order.
    See Also
    --------
    Examples
    ---------
    >>> convex_hull_melkman([[0, 0], [1, 0], [10, 1]])
    [(0.0, 0.0), (1.0, 0.0), (10.0, 1.0)]
    >>> convex_hull_melkman([[0, 0], [1, 0], [10, 0]])
    [(0.0, 0.0), (10.0, 0.0)]
    >>> convex_hull_melkman([[-1, 1],[-1, -1], [0, 0], [0.5, 0.5], [1, -1], [1, 1],
    ...                 [-0.75, 1]])
    [(-1.0, -1.0), (-1.0, 1.0), (1.0, -1.0), (1.0, 1.0)]
    >>> convex_hull_melkman([(0, 3), (2, 2), (1, 1), (2, 1), (3, 0), (0, 0), (3, 3),
    ...                 (2, -1), (2, -4), (1, -3)])
    [(0.0, 0.0), (0.0, 3.0), (1.0, -3.0), (2.0, -4.0), (3.0, 0.0), (3.0, 3.0)]
    """
    start = time.time()

    points = sorted(_validate_input(points))
    n = len(points)

    convex_hull = points[:2]
    for i in range(2, n):
        det = _det(convex_hull[1], convex_hull[0], points[i])
        if det > 0:
            convex_hull.insert(0, points[i])
            break
        elif det < 0:
            convex_hull.append(points[i])
            break
        else:
            convex_hull[1] = points[i]
    i += 1

    for i in range(i, n):
        if (
            _det(convex_hull[0], convex_hull[-1], points[i]) > 0
            and _det(convex_hull[-1], convex_hull[0], points[1]) < 0
        ):
            # The point lies within the convex hull
            continue

        convex_hull.insert(0, points[i])
        convex_hull.append(points[i])
        while _det(convex_hull[0], convex_hull[1], convex_hull[2]) >= 0:
            del convex_hull[1]
        while _det(convex_hull[-1], convex_hull[-2], convex_hull[-3]) <= 0:
            del convex_hull[-2]
    end = time.time()
    # `convex_hull` is contains the convex hull in circular order
    return sorted(convex_hull[1:] if len(convex_hull) > 3 else convex_hull), end-start


'''
def main():
    # Data used to compute the, test and analyse the convex hulls
    sizes = [3000, 6000, 9000, 12000, 15000, 18000, 21000, 24000, 27000, 30000]
    hull_sizes_A = [3, 3, 3, 4, 4, 4, 4, 5, 5, 5]
    hull_sizes_B = [30, 60, 90, 120, 150, 180, 210, 240, 270, 300]
    filenames_a = ["Set_A/A_3000.dat", "Set_A/A_6000.dat", "Set_A/A_9000.dat", "Set_A/A_12000.dat", "Set_A/A_15000.dat",
                   "Set_A/A_18000.dat", "Set_A/A_21000.dat", "Set_A/A_24000.dat", "Set_A/A_27000.dat", "Set_A/A_30000.dat"]
    filenames_b = ["Set_B/B_3000.dat", "Set_B/B_6000.dat", "Set_B/B_9000.dat", "Set_B/B_12000.dat", "Set_B/B_15000.dat",
                   "Set_B/B_18000.dat", "Set_B/B_21000.dat", "Set_B/B_24000.dat", "Set_B/B_27000.dat", "Set_B/B_30000.dat"]
'''


def main_time():
    # ------------------------------------------------------------------
    """
        File names may need to be changed dependent on their location
    """
    # Uncomment to produce the hull of a selected file with the time taken
    listPts = readDataPts("Set_A/A_3000.dat", 3000)

    gft_hull, gift_time = giftwrap(listPts[:])
    print(gft_hull, "Time taken: ", " - Gift Wrapping")

    grs_hull, grs_time = grahamscan(listPts[:])
    print(grs_hull, "Time taken: ", grs_time, " - Graham Scan")

    mono_hull, mono_time = monotone_chain(listPts[:])
    print(mono_hull, "Time taken: ", mono_time, " - Monotone Chain")

    incrmt_hull, incrmt_time = incremental(listPts[:])
    print(incrmt_hull, "Time taken: ", incrmt_time, " - Incremental")

    chan_hull, chan_time = Chan(listPts[:])
    print(chan_hull, "Time taken: ", chan_time, " - Chan")
    # listPts = [[732.0, 590.0], [415.0, 360.0], [
    #     276.0, 276.0], [229.0, 544.0], [299.0, 95.0]]
    quick_hull, quick_time = quickHull(listPts[:])
    print(quick_hull, "Time taken: ", quick_time, " - QuickHull")

    ks_hull, ks_time = kirk_seidel(listPts[:])
    print(ks_hull, "Time taken: ", ks_time, " - Kirk-Seidel")

    dnq_hull, dnq_time = convex_hull_recursive(listPts[:])
    print(dnq_hull, "Time taken: ", dnq_time, " - DivideNConquer")

    melk_hull, melk_time = convex_hull_melkman(listPts[:])
    print(melk_hull, "Time taken: ", melk_time, " - Melkman")

    bt_hull, bt_time = binarytree(listPts[:])
    print(bt_hull, "Time taken: ", bt_time, " - BinaryTree")

    sym_hull, sym_time = symmetric(listPts[:])
    print(sym_hull, "Time taken: ", sym_time, " - Symmetric")
    # ------------------------------------------------------------------


if __name__ == "__main__":
    sizes = [3000, 6000, 9000, 12000, 15000, 18000, 21000, 24000, 27000, 30000]
    # sizes = [3000, 6000]
    # main_time()
    # tests.average_tests(1, True)
    # tests.average_tests(1, False)
    # graphs.graph_set_A(sizes)
    graphs.graph_set_B(sizes)
