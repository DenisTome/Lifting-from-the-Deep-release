[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/lifting-from-the-deep-convolutional-3d-pose/3d-human-pose-estimation-on-human36m)](https://paperswithcode.com/sota/3d-human-pose-estimation-on-human36m?p=lifting-from-the-deep-convolutional-3d-pose)

# Lifting from the Deep
Denis Tome', Chris Russell, Lourdes Agapito

[Lifting from the Deep: Convolutional 3D Pose Estimation from a Single Image](http://openaccess.thecvf.com/content_cvpr_2017/papers/Tome_Lifting_From_the_CVPR_2017_paper.pdf), CVPR 2017

This project is licensed under the terms of the GNU GPLv3 license. By using the software, you are agreeing to the terms of the license agreement ([link](https://github.com/DenisTome/Lifting-from-the-Deep-release/blob/master/LICENSE)).

![Teaser?](https://github.com/DenisTome/Lifting-from-the-Deep-release/blob/master/data/images/teaser-github.png)
## Abstract

We propose a unified formulation for the problem of 3D human pose estimation from a single raw RGB image
that reasons jointly about 2D joint estimation and 3D pose reconstruction to improve both tasks. We take an integrated
approach that fuses probabilistic knowledge of 3D human pose with a multi-stage CNN architecture and uses
the knowledge of plausible 3D landmark locations to refine the search for better 2D locations. The entire process is
trained end-to-end, is extremely efficient and obtains stateof-the-art results on Human3.6M outperforming previous
approaches both on 2D and 3D errors.

## Dependencies

The code is compatible with python2.7
- [Tensorflow 1.0](https://www.tensorflow.org/)
- [OpenCV](http://opencv.org/)

## Models

For this demo, CPM's caffe-models trained on the MPI datasets ([link](https://github.com/shihenw/convolutional-pose-machines-release/tree/master/model)) are used for **2D pose estimation**, whereas for **3D pose estimation** our probabilistic 3D pose model is trained on the [Human3.6M dataset](http://vision.imar.ro/human3.6m/description.php).

## Testing
- First, run `setup.sh` to retreive the trained models and to install the external utilities.
- Run `demo.py` to evaluate the test image.

## Additional material
- Project [webpage](https://denistome.com/papers/LiftingFromTheDeep.html)
- Some [videos](https://youtu.be/tKfkGttx0qs).

## Citation

	@InProceedings{Tome_2017_CVPR,
	author = {Tome, Denis and Russell, Chris and Agapito, Lourdes},
	title = {Lifting From the Deep: Convolutional 3D Pose Estimation From a Single Image},
	booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
	month = {July},
	year = {2017}
	}

## Notes

The models provided for the demo are NOT the ones that have been used to generate results for the paper. We are still in the process of converting all the code.

## Acknowledgements

This work has been supported by the SecondHands project, funded from the EU Horizon 2020 Research and Innovation programme under grant agreement No 643950.

## References

- [Convolutional Pose Machines (CPM)](https://github.com/shihenw/convolutional-pose-machines-release).
