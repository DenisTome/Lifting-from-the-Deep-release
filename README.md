# Lifting from the Deep
Denis Tome', Chris Russell, Lourdes Agapito

[Lifting from the Deep: Convolutional 3D Pose Estimation from a Single Image](https://arxiv.org/abs/1701.00295), CVPR 2017

This project is licensed under the terms of the GNU GPLv3 license. By using the software, you are agreeing to the terms of the license agreement ([link](https://github.com/DenisTome/cpm_tensorflow/blob/public/LICENSE)).

![Teaser?](https://github.com/DenisTome/cpm_tensorflow/blob/public/images/teaser-github.png)
## Dependencies

The code is compatible with python2.7
- [Tensorflow 1.0](https://www.tensorflow.org/)
- [OpenCv](http://opencv.org/)

## Models

The architecture extends the one proposed in [Convolutional Pose Machines (CPM)](https://github.com/shihenw/convolutional-pose-machines-release).

For this demo, CPM's caffe-models trained on the MPI datasets ([link](https://github.com/shihenw/convolutional-pose-machines-release/tree/master/model)) are used for **2D pose estimation**, whereas for **3D pose estimation** our probabilistic 3D pose model is trained on the [Human3.6M dataset](http://vision.imar.ro/human3.6m/description.php).

## Testing
- First, run `setup.sh` to retreive the trained models and to install the external utilities.
- Run `demo.py` to evaluate the test image.

## Additional material
- Project [webpage](http://visual.cs.ucl.ac.uk/pubs/liftingFromTheDeep/)
- Some [videos](https://youtu.be/tKfkGttx0qs).

## Citation

	@article{tome2017lifting,
	    title={Lifting from the Deep: Convolutional 3D Pose Estimation from a Single Image},
	    author={Tome, Denis and Russell, Chris and Agapito, Lourdes},
	    journal={arXiv preprint arXiv:1701.00295},
	    year={2017}
	}
