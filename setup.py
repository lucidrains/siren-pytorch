from setuptools import setup, find_packages

setup(
  name = 'siren-pytorch',
  packages = find_packages(),
  version = '0.1.7',
  license='MIT',
  description = 'Implicit Neural Representations with Periodic Activation Functions',
  long_description_content_type = 'text/markdown',
  author = 'Phil Wang',
  author_email = 'lucidrains@gmail.com',
  url = 'https://github.com/lucidrains/siren-pytorch',
  keywords = ['artificial intelligence', 'deep learning'],
  install_requires=[
      'einops',
      'torch'
  ],
  classifiers=[
      'Development Status :: 4 - Beta',
      'Intended Audience :: Developers',
      'Topic :: Scientific/Engineering :: Artificial Intelligence',
      'License :: OSI Approved :: MIT License',
      'Programming Language :: Python :: 3.6',
  ],
)