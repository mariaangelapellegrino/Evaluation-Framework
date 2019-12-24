
from distutils.core import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
  name = 'evaluation_framework',         # How you named your package folder (MyLib)
  packages = ['evaluation_framework'],   # Chose the same as "name"
  version = '01.7',      # Start with a small number and increase it with every change you make
  license='apache-2.0',        # Chose a license from here: https://help.github.com/articles/licensing-a-repository
  description = 'Evaluation Framework for testing and comparing graph embedding techniques',   # Give a short description about your library
  long_description=long_description,
  long_description_content_type="text/markdown",
  author = 'Maria Angela Pellegrino',                   # Type in your name
  author_email = 'mariaangelapellegrino94@gmail.com',      # Type in your E-Mail
  url = 'https://github.com/mariaangelapellegrino/Evaluation-Framework',   # Provide either the link to your github or to your website
  download_url = 'https://github.com/mariaangelapellegrino/Evaluation-Framework/archive/v_01.7.tar.gz',    # I explain this later on
  keywords = ['evaluation', 'graph-embedding', 'python', 'library', 'benchmark-framework', 'machine learning tasks', 'semantic tasks'],   # Keywords that define your package best
  install_requires=[            # I get to this in a second
          'numpy==1.14.0',
          'pandas==0.22.0',
          'scikit-learn==0.19.2',
          'scipy==1.1.0',
          'h5py==2.8.0',
          'unicodecsv'
      ],
  classifiers=[
    'Development Status :: 3 - Alpha',      # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
    'Intended Audience :: Developers',      # Define that your audience are developers
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: Apache Software License',   # Again, pick a license
    'Programming Language :: Python :: 2.7',      #Specify which pyhton versions that you want to support
  ],
)
