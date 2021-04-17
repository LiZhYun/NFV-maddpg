from setuptools import setup, find_packages

setup(name='nfvmaddpg',
      version='0.0.1',
      description='NFV Multi-Agent Deep Deterministic Policy Gradient',
      author='ZhiYuan Li',
      author_email='zhiyuanli@std.uestc.edu.cn',
      packages=find_packages(),
      include_package_data=True,
      zip_safe=False,
      install_requires=['gym', 'numpy-stl']
)
