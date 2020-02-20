from setuptools import setup

setup(name='conet',
      packages=["conet",
                "conet.datasets",
                "conet.models",
                "conet.loss_functions",
                "conet.script",
                "conet.dataset_utils"
                ],
      version='0.6',
      description='conet.',
      url='https://github.com/MIC-DKFZ/nnUNet',
      author='hangli',
      author_email='hangli@stu.xmu.edu.cn',
      license='Apache License Version 2.0, January 2004',
      install_requires=[
            "torch",
            "tqdm",
            # "scikit-image",
            # "medpy",
            "scipy",
            "tensorboardX",
            # "batchgenerators>=0.19.4",
            "numpy",
            "sklearn",
            "SimpleITK",
            "pandas",
      ],
      keywords=['']
      )
