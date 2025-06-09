from setuptools import find_packages, setup

package_name = 'blib2_ros'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(include=['blib2_ros', 'blib2_ros.*']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ubuntu',
    maintainer_email='ubuntu@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'blib2_ros_node = blib2_ros.blib2_ros_node:main',
            'test_blib2 = blib2_ros.test.test_blib2:main',
        ],
    },
)
