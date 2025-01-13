from setuptools import setup, find_packages

setup(
    name='fraud_detection_system',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'pandas',
        'scikit-learn',
        'seaborn',
        'matplotlib',
        'joblib',
        'tensorflow',
        'numpy'
    ],
    entry_points={
        'console_scripts': [
            'train_models=scripts.train_models:main'
        ]
    },
    author='Dr. Bappaditya Ghosh',
    description='A fraud detection system using anomaly detection models.',
    license='MIT'
)
