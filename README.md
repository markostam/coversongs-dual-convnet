# Detecting Cover Songs with Convnets in Tensorflow
<a href='https://www.recurse.com' title='Made with love at the Recurse Center'><img src='https://cloud.githubusercontent.com/assets/2883345/11325206/336ea5f4-9150-11e5-9e90-d86ad31993d8.png' height='20px'/></a>

Deep learning model in python and [tensorflow](https://www.tensorflow.org) trained to automatically identify cover songs. Simple 4-layer siamese convnet with an added affine layer on the output. Precision of 65% shows that the dataset can be framed this way. Data augmentation, soft attention and triplet architecture planned for future work.

Dataset is the [secondhand songs set](http://labrosa.ee.columbia.edu/millionsong/secondhand), which is a subset of the [million song dataset](http://labrosa.ee.columbia.edu/millionsong/).

