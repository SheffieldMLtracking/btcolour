from distutils.core import setup
setup(
  name = 'btcolour',
  packages = ['btcolour'],
  version = '0.0.1',
  description = 'Classifies colour tags for bee differentiation',
  author = 'Euan Emery',
  author_email = 'euanemery@gmail.com',
  url = '',
  download_url = '',
  keywords = ['fitting','colour classification','colour','tag','PSF','focus'],
  classifiers = [],
  install_requires=['numpy','opencv-python','scipy']
)