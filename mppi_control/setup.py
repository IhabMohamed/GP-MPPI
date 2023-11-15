from setuptools import setup
setup(
    name='mppi_control',
    version='0.0.0',
    packages=['mppi_control'],
    install_requires=[
        'rospy',
        'roscpp',
        'std_msgs',
        'geometry_msgs',
        # Other dependencies required by your package
    ],
    package_dir={'': 'scripts'}
)