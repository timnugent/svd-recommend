Incremental Singular Value Decomposition-like Matrix Factorization for Collaborative Filtering using Stochastic Gradient Descent
-------------------------------------------------------------------------------------------------------------------------------- 

Based on Simon Funk's Netflix implementation [1-4] which processes one latent feature at a time. Uses L2 regularization to control the magnitude of the user-feature (P) and item-feature (Q) matrices, based on [3]. General SVD background can be found at [5]. Optional learning rate decay, e.g. [8].

Compile
-------

Includes the Eigen C++ headers [6]. Modify the include flag in the Makefile to use a different version.

Requires Boost libraries (runtime and development headers). On Ubuntu, 'apt-get install libboost-all-dev', on OS X, 'brew install boost'

Compile with 'make'

Test with 'make test'


Sample Output
-------------

bin/svd_collabfilt -o data/predictions.tsv -i data/training.tsv
Input file:      data/training.tsv
Output file:     data/predictions.tsv
Predictions:     25
Latent features: 240
Threads:         4
Read file data/training.tsv (282931 entries)
Users with data: 33217
Items with data: 8319
Writing user-feature matrix data/P_demo.mat ...
Writing item-feature matrix data/Q_demo.mat ...
Writing predictions file data/predictions.tsv ...

The predictions file contains the top 25 recommendations. First column is the user, then the item, then the rating.

Links
-----

[1] http://sifter.org/~simon/journal/20061211.html
[2] http://www.timelydevelopment.com/demos/NetflixPrize.aspx
[3] http://www.netflixprize.com/community/viewtopic.php?id=1423
[4] http://www.netflixprize.com/community/viewtopic.php?id=481
[5] http://www.quuxlabs.com/blog/2010/09/matrix-factorization-a-simple-tutorial-and-implementation-in-python/
[6] http://alias-i.com/lingpipe/docs/api/com/aliasi/matrix/SvdMatrix.html
[7] http://eigen.tuxfamily.org/
[8] http://bengio.abracadoudou.com/lectures/old/tex_ann.pdf

timnugent@gmail.com

