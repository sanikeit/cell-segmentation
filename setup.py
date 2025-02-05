from setuptools import setup, find_packages

setup(
    name='cell-segmentation',
    version='0.1.0',
    author='Anikeit Sethi, and Tushar Lamba',
    author_email='your.email@example.com',
    description='A project for cell segmentation and counting using UNet.',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'torch',
        'torchvision',
        'pytorch-lightning',
        'tensorboard',
        'timm',
        'seaborn',
        'numpy',
        'scikit-image',
        'opencv-python'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)