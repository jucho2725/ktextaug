from setuptools import setup, find_packages

setup(
    name="ktextaug",
    version="0.1.9c3",
    description="data augmentation tool for Korean",
    author="jinuk.cho, eddie.jeon, jonghyeok.park, junghoon.lee, minsu.jeong. all from ING Lab, SKKU.",
    author_email="cju2725@gmail.com",
    url="https://github.com/jucho2725/ktextaug",
    download_url="https://github.com/jucho2725/ktextaug/archive/master.zip",
    install_requires=["beautifulsoup4>=4.6.0", "googletrans==3.1.0a0", "konlpy>=0.5.2", "pykomoran>=0.1.5", "transformers>=2.6.0"],
    packages=find_packages(exclude=["DataAug_CNN, DataAug_CNN.*, test, etc, etc.*"]),
    keywords=["text augmentation", "korean"],
    python_requires=">=3.6",
    package_data={'ktextaug': ['stopwords-ko.txt', 'vocab_noised.txt']},
    extras_require={
        'dev': ['pytest', 'flake8'],
    },
    include_package_data=True,
    zip_safe=False,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.2",
        "Programming Language :: Python :: 3.3",
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
)
