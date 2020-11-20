from setuptools import setup, find_packages

setup(
    name='ktextaug',
    version='0.1',
    description='data augmentation tool for Korean',
    author='jinuk.cho',
    author_email='cju2725@gmail.com',
    url='https://github.com/jucho2725/ktextaug',
    download_url='https://github.com/jucho2725/ktextaug/archive/master.zip',
    # 해당 패키지를 사용하기 위해 필요한 패키지를 적어줍니다. ex. install_requires= ['numpy', 'django']
    # 여기에 적어준 패키지는 현재 패키지를 install할때 함께 install됩니다.
    install_requires=['bs4', 'googletrans', 'pandas'],
    packages=find_packages(exclude=[]),
    keywords=['text augmentation', 'korean'],
    python_requires='>=3.3',
    # 파이썬 파일이 아닌 다른 파일을 포함시키고 싶다면 package_data에 포함시켜야 합니다.
    package_data={},
    # 위의 package_data에 대한 설정을 하였다면 zip_safe설정도 해주어야 합니다.
    zip_safe=False,
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.2',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
)
