from distutils.core import setup
setup(
  name = 'btcolour',
  packages = ['btcolour'],
  version = '1.0',
  description = 'Identifies the colour of retroreflectors in a series of flash illuminated photos',
  author = 'Mike Smith',
  author_email = 'm.t.smith@sheffield.ac.uk',
  url = 'https://github.com/SheffieldMLtracking/btcolour.git',
  download_url = 'https://github.com/SheffieldMLtracking/btcolour.git',
  keywords = ['image processing','retroreflectors'],
  classifiers = [],
  install_requires=['numpy'],
  scripts=['bin/btcolour'],
)
