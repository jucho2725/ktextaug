from setuptools import setup, find_packages

setup(
    name="ktextaug",
    version="0.1.1",
    description="data augmentation tool for Korean",
    author="jinuk.cho, eddie.jeon, jonghyeok.park, junghoon.lee, minsu.jeong. all from ING Lab, SKKU.",
    author_email="cju2725@gmail.com",
    url="https://github.com/jucho2725/ktextaug",
    download_url="https://github.com/jucho2725/ktextaug/archive/master.zip",
    install_requires=["beautifulsoup4>=4.6.0", "googletrans>=2.4.0",
                      "pandas>=1.0.4", "konlpy>=0.5.2"],
    packages=find_packages(exclude=[]),
    keywords=["text augmentation", "korean"],
    python_requires=">=3.3",
    package_data={},
    zip_safe=False,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.2",
        "Programming Language :: Python :: 3.3",
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
    ],
)
