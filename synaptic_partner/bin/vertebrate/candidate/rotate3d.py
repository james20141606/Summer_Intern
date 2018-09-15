import numpy as np


"""
Basic rotations of a 3d matrix.
----------
Example:

cube = array([[[0, 1],
               [2, 3]],

              [[4, 5],
               [6, 7]]])

axis 0: perpendicular to the face [[0,1],[2,3]] (front-rear)
axis 1: perpendicular to the face [[1,5],[3,7]] (lateral right-left)
axis 2: perpendicular to the face [[0,1],[5,4]] (top-bottom)
----------
Note: the command m[:, ::-1, :].swapaxes(0, 1)[::-1, :, :].swapaxes(0, 2) rotates the cube m
around the diagonal axis 0-7.
"""


def basic_rot_ax(m, ax=0):
    """
    :param m: 3d matrix
    :return: rotate the cube around axis ax, perpendicular to the face [[0,1],[2,3]]
    """

    ax %= 3

    if ax == 0:
        return np.rot90(m[:, ::-1, :].swapaxes(0, 1)[::-1, :, :].swapaxes(0, 2), 3)
    if ax == 1:
        return np.rot90(m, 1)
    if ax == 2:
        return m.swapaxes(0, 2)[::-1, :, :]


def axial_rotations(m, rot=1, ax=2):
    """
    :param m: 3d matrix
    :param rot: number of rotations
    :param ax: axis of rotation
    :return: m rotate rot times around axis ax, according to convention.
    """

    if len(m.shape) is not 3:
        assert IOError

    rot %= 4

    if rot == 0:
        return m

    for _ in range(rot):
        m = basic_rot_ax(m, ax=ax)

    return m
