import time


def ccw(p1, p2, p3):
    a = (p1[0] - p2[0], p1[1] - p2[1])
    b = (p3[0] - p2[0], p3[1] - p2[1])
    return a[0]*b[1] < a[1]*b[0]


class Node:
    def __init__(self, x, y, n):
        self.x = x
        self.y = y
        self.n = n
        self.ymin = y
        self.ymax = y
        self.yminn = n
        self.ymaxn = n
        self.left = None
        self.right = None

    def insert(self, x, y, n):
        if self.x == x:
            if self.ymin > y:
                self.ymin = y
                self.yminn = n
            if self.ymax < y:
                self.ymax = y
                self.ymaxn = n
        elif self.x > x:
            if self.left:
                self.left.insert(x, y, n)
            else:
                self.left = Node(x, y, n)
        else:
            if self.right:
                self.right.insert(x, y, n)
            else:
                self.right = Node(x, y, n)


upperhull = []
lowerhull = []


def binarytree(points):
    start = time.time()
    root = Node(*points[0], 0)
    for i in range(1, len(points)):
        root.insert(*points[i], i)
    getupperhull(points, root)
    getlowerhull(points, root)
    res = list(map(lambda x: points[x], lowerhull)) + \
        list(map(lambda x: points[x], upperhull))
    end = time.time()
    return res, end-start


def getupperhull(points, x):
    global upperhull
    if x.left:
        getupperhull(points, x.left)
    while len(upperhull) > 1 and ccw(points[upperhull[-2]], points[upperhull[-1]], points[x.ymaxn]):
        upperhull.pop()
    upperhull.append(x.ymaxn)
    if x.right:
        getupperhull(points, x.right)


def getlowerhull(points, x):
    global lowerhull
    global upperhull
    if x.left:
        getlowerhull(points, x.left)
    while len(lowerhull) > 1 and not ccw(points[lowerhull[-2]], points[lowerhull[-1]], points[x.yminn]):
        lowerhull.pop()
    lowerhull.append(x.yminn)
    if x.right:
        getlowerhull(points, x.right)
