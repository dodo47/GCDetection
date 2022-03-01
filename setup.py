import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="globsML",
    version="1.0.0",
    author="Dominik Dold, Katja Fahrion",
    author_email="dodo.science@web.de",
    description="ML tools for GC detection.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8.8',
    install_requires=[
          'numpy>=1.21.2',
          'scipy>=1.7.1',
          'matplotlib',
          'torch>=1.10.0',
          'torchvision>=0.11.1',
          'jupyter',
          'scikit-learn>=0.24.2',
          'seaborn',
          'pandas>=1.3.3',
          'tqdm',
          'astropy>=4.3.1',
          'lime',
          'lime_stability',
          'catboost>=1.0.3',
          'pytorch_tabnet',
      ]
)
