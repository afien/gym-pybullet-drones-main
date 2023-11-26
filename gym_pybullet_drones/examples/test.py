import sys
sys.path.insert(0, 'C:/Users/USER/gym-pybullet-drones-main')
from utils.enums import DroneModel, Physics, ImageType

import pybullet

from transforms3d.quaternions import rotate_vector, qconjugate