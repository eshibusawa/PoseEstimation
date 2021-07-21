# PoseEstimation
## Overview
* C++ template library for:
    * EPnP algorithm [1,2] for projective-n-point
    * P3P algorithm [3] for projective-3-point
    * Five point algorithm [4] for Essential matrix esitimation
    * Seven point algorithm for Fundamental matrix esitimation (almost equivalent to [5])
    * Three point algorithm [6] for 3D similitude transformation estimation

* These algorithms support:
    * RANSAC scheme for robust estimation
    * double / float precision as input
    * unit test by using CPPUnit (except for EPnP)

## Usage
### Dependencies
* [CMake](https://cmake.org/)
* [Eigen 3](https://eigen.tuxfamily.org/)
* [CPPUnit](https://sourceforge.net/projects/cppunit/) (optional)
### Build
```
$ cd PoseEstimation
$ mkdir build
$ cd build
$ cmake -DCMAKE_VERBOSE_MAKEFILE=true -DCMAKE_BUILD_TYPE=Release ../
$ make -j8
$ make test
```

## References
[1] V. Lepetit, F. Moreno-Noguer and P. Fua. "EPnP: An Accurate O(n) Solution to the PnP Problem". *International Journal of Computer Vision*, vol. 81, pp. 155-166, 2009.

[2] F. Moreno-Noguer, V. Lepetit and P. Fua. "Accurate Non-Iterative O(n) Solution to the PnP Problem". In *Proc. of IEEE International Conference on Computer Vision, Rio de Janeiro, Brazil*, October 2007, pp. 1-8.

[3] L. Kneip, D. Scaramuzza and R. Siegwart. "A novel parametrization of the perspective-three-point problem for a direct computation of absolute camera position and orientation". In *Proc. of IEEE International Conference on Computer Vision and Pattern Recognition, Colorado Springs, CO, USA*, June 2011, pp. 2969-2976.

[4] D. Nistér. "An Efficient Solution to the Five-Point Relative Pose Problem". *IEEE Transactions on Pattern Analysis and Machine Intelligence*, vol. 26, no. 6, pp. 756-777, 2004.

[5] O. Faugeras. "Stratification of three-dimensional vision: projective, affine, and metric representations", *Journal of the Optical Society of America A*, vol. 12, issue 3, pp. 465-484, 1995.

[6] B. K. P. Horn. "Closed-form solution of absolute orientation using unit quaternions", *Journal of the Optical Society of America A*, vol. 4, issue 4, pp. 629-642, 1987.
