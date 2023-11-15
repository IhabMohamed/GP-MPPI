# from distutils.core import setup
# from catkin_pkg.python_setup import generate_distutils_setup

# setup(**generate_distutils_setup(
#     packages=['jackal_bringup', 'jackal_bringup.multicast'],
#     package_dir={'': 'src'}
# ))

from setuptools import setup

setup(
    name='jackal_bringup',
    version='0.6.1',
    packages=['jackal_bringup', 'jackal_bringup.multicast'],
    package_dir={'': 'src'}
)

