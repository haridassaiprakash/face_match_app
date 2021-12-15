from setuptools import setup

setup(
    name="src",
    version="0.0.1",
    author="sai prakash",
    description="A small package for To whom does your face match",
    author_email="haridassai7893@gmail.com",
    packages=["src"],
    python_requires=">3.7",
    install_requires=[
        'mtcnn==0.1.0',
        'tensorflow==2.3.1.0',
        'keras==2.4.3',
        'keras-vggface==0.6',
        'keras_applications==1.0.8',
        'pyYAML',
        'tqdm',
        'scikit-learn',
        'streamlit',
        'bing-image-downloader'
 
    ],
)