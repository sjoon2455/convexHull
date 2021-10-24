
import time


def ccw(p1, p2, p3):
    a = (p1[0] - p2[0], p1[1] - p2[1])
    b = (p3[0] - p2[0], p3[1] - p2[1])
    return a[0]*b[1] < a[1]*b[0]


def cw(p1, p2, p3):
    a = (p1[0] - p2[0], p1[1] - p2[1])
    b = (p3[0] - p2[0], p3[1] - p2[1])
    return a[0]*b[1] > a[1]*b[0]


def symmetric(points):
    start = time.time()

    def line_formula(p1, p2):
        # Ax - By + C = 0
        A = p2[1] - p1[1]
        B = p2[0] - p1[0]
        C = p2[0] * p1[1] - p1[0] * p2[1]
        # y = (A/B)x + C/B
        return (A/B, C/B)
    Xmax, Xmin, Ymax, Ymin = 0, 0, 0, 0
    for i in range(1, len(points)):
        if points[Xmax][0] < points[i][0]:
            Xmax = i
        if points[Xmin][0] > points[i][0]:
            Xmin = i
        if points[Ymax][1] < points[i][1]:
            Ymax = i
        if points[Ymin][1] > points[i][1]:
            Ymin = i
    R1_exist = Xmax != Ymax and points[Xmax][0] != points[Ymax][0] and points[Xmax][1] != points[Ymax][1]
    R2_exist = Xmin != Ymin and points[Xmin][0] != points[Ymax][0] and points[Xmin][1] != points[Ymax][1]
    R3_exist = Xmin != Ymin and points[Xmin][0] != points[Ymin][0] and points[Xmin][1] != points[Ymin][1]
    R4_exist = Xmax != Ymin and points[Xmax][0] != points[Ymin][0] and points[Xmax][1] != points[Ymin][1]
    R1_points, R2_points, R3_points, R4_points = [], [], [], []
    if R1_exist:
        R1_a, R1_b = line_formula(points[Xmax], points[Ymax])
        R1_points.extend([Xmax, Ymax])
    if R2_exist:
        R2_a, R2_b = line_formula(points[Xmin], points[Ymax])
        R2_points.extend([Xmin, Ymax])
    if R3_exist:
        R3_a, R3_b = line_formula(points[Xmin], points[Ymin])
        R3_points.extend([Xmin, Ymin])
    if R4_exist:
        R4_a, R4_b = line_formula(points[Xmax], points[Ymin])
        R4_points.extend([Xmax, Ymin])

    for i in range(len(points)):
        if R1_exist and points[i][1] > R1_a * points[i][0] + R1_b:
            R1_points.append(i)
        elif R2_exist and points[i][1] > R2_a * points[i][0] + R2_b:
            R2_points.append(i)
        elif R3_exist and points[i][1] < R3_a * points[i][0] + R3_b:
            R3_points.append(i)
        elif R4_exist and points[i][1] < R4_a * points[i][0] + R4_b:
            R4_points.append(i)
    R1_points.sort(key=lambda x: points[x][0]+R1_a*points[x][1])
    R2_points.sort(key=lambda x: points[x][0]+R2_a*points[x][1])
    R3_points.sort(key=lambda x: points[x][0]+R3_a*points[x][1])
    R4_points.sort(key=lambda x: points[x][0]+R4_a*points[x][1])
    R1_hull, R2_hull, R3_hull, R4_hull = [], [], [], []
    for p in R1_points:
        while len(R1_hull) > 1 and not cw(points[R1_hull[-2]], points[R1_hull[-1]], points[p]):
            R1_hull.pop()
        R1_hull.append(p)
    for p in R2_points:
        while len(R2_hull) > 1 and not cw(points[R2_hull[-2]], points[R2_hull[-1]], points[p]):
            R2_hull.pop()
        R2_hull.append(p)
    for p in R3_points:
        while len(R3_hull) > 1 and not ccw(points[R3_hull[-2]], points[R3_hull[-1]], points[p]):
            R3_hull.pop()
        R3_hull.append(p)
    for p in R4_points:
        while len(R4_hull) > 1 and not ccw(points[R4_hull[-2]], points[R4_hull[-1]], points[p]):
            R4_hull.pop()
        R4_hull.append(p)

    res = list(map(lambda x: points[x], R1_hull)) + list(map(lambda x: points[x], R2_hull)) + list(
        map(lambda x: points[x], R3_hull)) + list(map(lambda x: points[x], R4_hull))
    end = time.time()
    return res, end-start
    # # draw image
    # from PIL import Image, ImageDraw, ImageOps
    # im = Image.new(mode='RGB', size=(500, 500), color='white')
    # draw = ImageDraw.Draw(im)
    # for point in points:
    #     draw.ellipse(((point[0]-1, point[1]-1),
    #                   (point[0]+1, point[1]+1)), fill=(0, 0, 0))
    # if R1_exist:
    #     draw.line(list(map(lambda x: points[x], R1_hull)), fill=(
    #         255, 0, 0), width=1)
    # else:
    #     draw.line([points[Xmax], points[Ymax]], fill=(255, 0, 0), width=1)
    # if R2_exist:
    #     draw.line(list(map(lambda x: points[x], R2_hull)), fill=(
    #         0, 255, 0), width=1)
    # else:
    #     draw.line([points[Xmin], points[Ymax]], fill=(0, 255, 0), width=1)
    # if R3_exist:
    #     draw.line(list(map(lambda x: points[x], R3_hull)), fill=(
    #         0, 0, 255), width=1)
    # else:
    #     draw.line([points[Xmin], points[Ymin]], fill=(0, 0, 255), width=1)
    # if R4_exist:
    #     draw.line(list(map(lambda x: points[x], R4_hull)), fill=(
    #         255, 0, 255), width=1)
    # else:
    #     draw.line([points[Xmax], points[Ymin]], fill=(255, 0, 255), width=1)
    # # As coorinate system of PIL regards left upper point as origin, flip upside down for visibility
    # ImageOps.flip(im).save('result_symmetric.bmp')


if __name__ == '__main__':
    with open('point.txt', 'r') as f:
        points = []
        for line in f.readlines():
            points.append(tuple(map(float, line.strip().split())))
        symmetric(points)
