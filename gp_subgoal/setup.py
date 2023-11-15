# from distutils.core import setup
# from catkin_pkg.python_setup import generate_distutils_setup

# d = generate_distutils_setup(
#     packages=['gp_subgoal'],
#     package_dir={'script': '.'}
# )

# setup(**d)

from setuptools import setup

setup(
    name='gp_subgoal',
    version='0.0.0',
    packages=['gp_subgoal'],
    package_dir={'script': '.'}
)
