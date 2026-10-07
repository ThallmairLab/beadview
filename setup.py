from setuptools import setup, find_packages

with open('README.md', 'r') as f:
    long_description = f.read()

setup(
    name='CGSmiles_utils',
    version='1.7.15',
    packages=find_packages(),
    include_package_data=True,
    package_data={
        # 'M3_parametrization': ['citations.bib'],
    },
    python_requires='>=3.11',
    description='Parametrization of M3 photoswitches for CG simulations',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Thilo Duve',
    author_email='thiloduve@web.de',
    url='https://github.com/ThiloDu/M3-parametrization',
    license='MIT',
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.11',
        'Operating System :: OS Independent',
    ],

    # CHANGE
    # install_requires=["bson >= 0.5.10"],
    # extras_require={
    #     "dev": ["pytest>=7.0", "twine>=4.0.2"],
    # },
    # python_requires=">=3.10",
)