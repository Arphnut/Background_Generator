# -*- coding: utf-8 -*-
"""
Date: 10:49 on 31-05-2019
Name: wallpaper.py
Description: Creation of a wallpaper formed of hexagons with shading colors.
             It takes around 1 min to create a (1920, 180, 3) wallpaper.
"""

import math
import tqdm
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal

WIDTH = 1920
HEIGHT = 1080

image = np.zeros((HEIGHT, WIDTH, 3), dtype=np.int64)


def draw_line(img, point1, point2, color=np.array([255, 255, 255])):
    """
    Draw a line of color 'color' between 'point1' and 'point2' on the 'img'.

    Parameters:
    -----------
    img (array): The image where we should draw the line.
    point1 (tuple 2): A point (not necessarily on the image).
    point2 (tuple 2): A point (not necessarily on the image).
    color (array 3): The color of the line to draw
    """
    u, v, w = img.shape
    x1, y1 = point1
    x2, y2 = point2

    # Check if the color is of the good shape
    assert len(color) == w

    nb_step = 2 * max(np.abs([x2 - x1, y2 - y1]))
    stepx = np.linspace(x1, x2, nb_step, dtype=np.int64)
    stepy = np.linspace(y1, y2, nb_step, dtype=np.int64)

    # If all the coordinates are in the image :
    if np.all(0 <= stepx) and np.all(stepx < u) and np.all(
            0 <= stepy) and np.all(stepy < v):
        img[stepx, stepy] = color  # we can trace the line
    else:
        # Else, we only print the possible points one by one.
        for x, y in zip(stepx, stepy):
            if (0 <= x < u) and (0 <= y < v):
                img[x, y] = color


# Bézier Curve drawer.


def binomial(k, n):
    """ The binomial coefficient of k amongst n. """
    return math.factorial(n) / (math.factorial(k) * math.factorial(n - k))


def bernstein(t, i, n):
    """The 'i'-th coefficient of the Bernstein polynom of degree 'n', evaluated for 't'"""
    return binomial(i, n) * (t**i) * ((1 - t)**(n - i))


def Bézier(t, points):
    """The evaluation of the Bézier polynom at time t, with regard to the set 'points' of point."""
    n = len(points) - 1
    x, y = 0, 0
    for i, pos in enumerate(points):
        bern = bernstein(t, i, n)
        x += pos[0] * bern
        y += pos[1] * bern
    return round(x), round(y)


def drawBézier(img, points, nb):
    """
    Draw a Bézier curve on an image.

    Parameters:
    -----------
    img (array) : The image the Bézizer curve will be drawn to.
    points (list): The list of the director points for the Bézier curve.
    nb (int): The step of the Bézier curve.
    """
    béziercurb = []
    for t in np.linspace(0, 1, nb):
        béziercurb.append(Bézier(t, points))

    for i in range(len(béziercurb) - 1):
        draw_line(img, béziercurb[i], béziercurb[i + 1])


def curve_hexagone(heximage,
                   center=(0, 0),
                   radius=10,
                   nb=100,
                   set_angle=list()):
    """
    Draw an hexagon on an image.

    Parameters:
    -----------
    heximage (array): The image we want to draw the hexagon to.
    center (tuple 2) : The beginning point to draw the hexagon (the point at the top of it).
    radius (float): Half of the size of a side of the hexagon.
    """
    u, v, _ = heximage.shape
    deb = [
        np.array([
            center[0], center[1], center[0], center[1] + radius,
            center[0] + radius * np.sin(np.pi / 3),
            center[1] + radius + radius * np.cos(np.pi / 3)
        ])
    ]  # The first two edges of the hexagon

    # We add to 'last_points' all the coordinate of the points needed for the Bézier curve.
    for i in range(5):
        last_points = deb[-1].copy()
        last_points[1] += radius * np.cos(np.pi / 3 * i) + radius * np.cos(
            np.pi / 3 * (i + 1))
        last_points[0] += radius * np.sin(np.pi / 3 * i) + radius * np.sin(
            np.pi / 3 * (i + 1))
        last_points[3] += 2 * radius * np.cos(np.pi / 3 * (i + 1))
        last_points[2] += 2 * radius * np.sin(np.pi / 3 * (i + 1))
        last_points[5] += radius * np.cos(
            np.pi / 3 * (i + 1)) + radius * np.cos(np.pi / 3 * (i + 2))
        last_points[4] += radius * np.sin(
            np.pi / 3 * (i + 1)) + radius * np.sin(np.pi / 3 * (i + 2))
        deb.append(last_points)
        if (0 <= last_points[2] < u) and (0 <= last_points[3] < v):
            set_angle.append((int(last_points[2]), int(last_points[3])))
    for el in deb:  # We draw the Bézier curve of the point forming an hexagon.
        bez = array_to_point(el)
        drawBézier(heximage, bez, nb)


def array_to_point(arr):
    """
    Transform an array with the coordianted of three points, into a list of points,
    that can be used for drawing a Bézier curve.
    """
    el = np.array(np.around(arr), dtype=np.int64)
    return [(el[0], el[1]), (el[2], el[3]), (el[2], el[3]), (el[4], el[5])]


def hexa_pattern(size=(1080, 1920, 3), radius=50, nb=100):
    """
    Draw a pattern of hexagone, on an image of size 'size', with a radius of 'radius'.

    Parameters:
    -----------
    size (tuple 3): The size of the image. Should be of shape 3.
    radius (int): The size of the hexagon will depend of 'radius'.
    The side of the hexagone will be twice sa long as 'radius'.
    nb (int): The precision of the drawing.
    """
    set_angle = list()
    image = np.zeros(size)
    nb_y = size[0] // (radius * 3)
    nb_x = size[1] // (radius * 4)
    print(nb_x, nb_y)
    center = (0, 0)
    for i in tqdm.tqdm(range(-1, nb_y)):
        for j in range(-1, nb_x):
            center_x = center[0] + i * 4 * np.sin(np.pi / 3) * radius
            center_y = center[1] + j * 6 * radius
            curve_hexagone(image, (center_x, center_y), radius, nb, set_angle)
            curve_hexagone(image,
                           (center_x + 2 * np.sin(np.pi / 3) * radius,
                            center_y + 2 * (1 + np.cos(np.pi / 3)) * radius),
                           radius, nb, set_angle)
    return image, set_angle


def double_points(image_thin):
    """
    Double every points of an image
    """
    imshape = np.array(image_thin.shape)

    imshape = imshape[[2, 0, 1]]
    image = signal.convolve2d(image_thin[:, :, 0],
                              np.ones((3, 3)),
                              mode='same')
    image = 255 * np.ones(tuple(imshape)) * (image > 0)
    return image.transpose([1, 2, 0])


def transpose(point):
    """
    Transpose a point.
    """
    return (point[1], point[0])


def fill_contour(image, center=(0, 0)):
    """
    Fill with the white color all the points that are contained in the same contour as the
    center point.

    We iterate over 'list_point'.
    'list_point' contains all the point that are in the contour and that has not been seen yet.
    If a point is in this list, we check all of this neightour, and if they weren't check before,
    and are in the contour (because they are in black), then we add them to the list.
    We keep memory of the visited points thanks to 'final_set'.
    We do this until there is no more point in the list.
    """
    u, v, _ = image.shape
    list_point = [center
                  ]  # A list containing the points that are in the contour.
    final_set = set([center])  # The set of points to color in white
    neighbour = np.array([[1, 0], [-1, 0], [0, 1], [0, -1], [1, 1], [1, -1],
                          [-1, 1], [-1, -1]])
    while len(list_point) > 0:
        p = list_point[0]
        # This point is in the contour
        for nei in neighbour:
            nei_p = (p[0] + nei[0], p[1] + nei[1])
            if (0 <= nei_p[0] < u) and (0 <= nei_p[1] < v):
                # If the neighbour is in the image :
                if np.all(image[nei_p] == 0):
                    # If this point is black :
                    if nei_p not in final_set:
                        # If we haven't visited this point yet :
                        list_point.append(
                            nei_p)  # We add it to the list of points to check.
                        final_set.add(
                            nei_p)  # We add it to the point in the contour.
        del list_point[0]
    for p in final_set:
        image[p] = np.array([255, 255, 255])


def fill_hexa(image, angles):
    """
    We fill the contour of all hexa.

    Parameters:
    -----------
    image (array): The image we wan't to draw on.
    angles (list): The list of all angles of the different hexgons.
                   They are used as the center of the zone to find the contour.
    """
    angles = set(angles)
    for el in tqdm.tqdm(angles):
        fill_contour(image, el)


def color_gradient(image,
                   colors=np.array([[0, 0, 0], [255, 255, 255], [0, 255, 0],
                                    [255, 0, 0]])):
    """
    Draw two different gradients from the top left corner to the bottom right corner.
    The first gradient is for initially white pixels.
    The second gradient is for initially black pixels.

    Parameters:
    -----------
    image (array): An image in black and white (without shade of grey)

    Returns:
    --------
    final_image (array): A colored image based on the black and white image.
        All previous black pixels are replaced by a shading from 'colors[0]' to 'colors[1]'.
        All previous white pixels are replaced by a shading from 'colors[2]' to 'colors[3]'.
    """
    u, v, _ = image.shape
    final_image = image.copy()
    for i in tqdm.tqdm(range(u)):
        for j in range(v):
            if np.all(image[i, j] == 0):
                final_image[i, j] = (i + j) / (u + v) * colors[0] + (
                    1 - (i + j) / (u + v)) * colors[1]
            else:
                final_image[i, j] = (i + j) / (u + v) * colors[2] + (
                    1 - (i + j) / (u + v)) * colors[3]
    return final_image


def background(size=(1080, 1920, 3),
               radius=50,
               colors=np.array([[0, 0, 0], [255, 255, 255], [0, 255, 0],
                                [255, 0, 0]])):
    """
    Create a new background image
    """
    image, angles = hexa_pattern(size=size, radius=radius)
    image2 = double_points(image)
    fill_hexa(image2, angles)
    image3 = color_gradient(image2, colors=colors)
    return image3


def show_image(image):
    """ Show the image """
    plt.imshow(image / 255)
    plt.show()


if __name__ == '__main__':
    image = background(size=(1080, 1920, 3),
                       radius=50,
                       colors=np.array([[0, 0, 100], [100, 0, 100],
                                        [0, 0, 100], [100, 0, 0]]))

    file.open("/mnt/etienne/detente/python/backgound_generator/")
