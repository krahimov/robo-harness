"""DROID flange poses versus the Menagerie Panda fingertip control site.

DROID config/panda/franka_panda.yaml specifies panda_link8 (flange).
Menagerie places the hand at link7 + .107m, with a -45-degree Z rotation;
Studio's fingertip site is another .1034m along the hand's local Z.
Both observations and commanded poses must use this same rigid transform.
"""
import math
import numpy as np
from studio.adapters.franka import rotation_matrix, euler_xyz

TIP_OFFSET = np.array([0., 0., .1034])
c, s = math.cos(math.pi/4), math.sin(math.pi/4)
FLANGE_FROM_HAND = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.]])


def tip_to_flange(position, euler):
    rotation = rotation_matrix(euler)
    return (np.asarray(position)-rotation@TIP_OFFSET,
            euler_xyz(rotation@FLANGE_FROM_HAND))


def flange_to_tip(position, euler):
    rotation = rotation_matrix(euler)@FLANGE_FROM_HAND.T
    return np.asarray(position)+rotation@TIP_OFFSET, euler_xyz(rotation)
