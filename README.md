# CV (Computer Vision)

This repository contains python scripts which performs various CV operation, with and without deep learning.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Ubuntu system dependencies

To run the scripts, you'll need some extra software :
```
sudo apt-get install build-essential cmake git unzip pkg-config
sudo apt-get install libjpeg-dev libtiff5-dev libjasper-dev libpng12-dev
sudo apt-get install libavcodec-dev libavformat-dev libswscale-dev libv4l-dev
sudo apt-get install libxvidcore-dev libx264-dev
sudo apt-get install libgtk-3-dev
sudo apt-get install libhdf5-serial-dev graphviz
sudo apt-get install libopenblas-dev libatlas-base-dev gfortran
sudo apt-get install python-tk python3-tk python-imaging-tk
```

And the python headers packages :

```
sudo apt-get install python2.7-dev python3-dev
```

### Create a python virtual environment (optional)

It is strongly advised to use python virtual environment :
```
pip install virtualenv virtualenvwrapper
```

Then, we need to update the ~/.bashrc file:

```
echo -e "\n# virtualenv and virtualenvwrapper" >> ~/.bashrc
echo "export WORKON_HOME=$HOME/.virtualenvs" >> ~/.bashrc
echo "export VIRTUALENVWRAPPER_PYTHON=/usr/bin/python3" >> ~/.bashrc
echo "source /usr/local/bin/virtualenvwrapper.sh" >> ~/.bashrc
```

And reload it with `source ~/.bashrc` (only once for this shell session, for next sessions it will be loaded automaticaly)

After the global virtualenv parameters are correctly configured, you can create the environment (here with python3 bindings) with :
```
mkvirtualenv cv -p python3
```
You can name your virtualenv (here cv) as you want. To work on it, simply do `workon cv`.
### Install OpenCV
You will need numpy to run OpenCV : `pip install numpy`

To install OpenCV, simply do `pip install opencv-contrib-python`

OR

Follow one of this guides to install it with custom parameters : [OpenCV Tutorials](https://www.pyimagesearch.com/opencv-tutorials-resources-guides/)

### Python packages

```
pip install scipy matplotlib pillow
pip install imutils h5py requests progressbar2
pip install scikit-learn scikit-image dlib
pip install mahotas
pip install tensorflow
pip install keras
```
### PiCamera module

```
pip install picamera[array]
```

### pyimagesearch package

The pyimagesearch package used in many examples and projects is located at the project root. To be able to use the scripts, don't forget to either :
1. copy/paste the folder in your projects.
2. add the path to your PYTHONPATH environment variable :
   - modify your ~/.bashrc file
   - add `export PYTHONPATH="${PYTHONPATH}:path/to/package"` (exemple : `export PYTHONPATH="${PYTHONPATH}:${HOME}/CV`
3. sym-link your repository

Other methods are also acceptable. Remember, the package is mandatory for running scripts.

### Testing the installation

To test your installation, try to do `import keras`, `import cv2` and `import pyimagesearch`in a Python shell. If you don't have any error, your installation is working properly.

Note : the success of pyimagesearch import depends on the choice your made at previous step.

## Authors

See the list of [contributors](https://github.com/sergentd/CV/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

Thanks to [pyimagesearch](https://www.pyimagesearch.com) for the excellent tutorials and books from where a lot of exemples present in this project are inspired from.
