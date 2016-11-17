# Towards Automatic Cover Song Detection Using Parallel Convolutional Neural Networks
<a href='https://www.recurse.com' title='Made with love at the Recurse Center'><img src='https://cloud.githubusercontent.com/assets/2883345/11325206/336ea5f4-9150-11e5-9e90-d86ad31993d8.png' height='20px'/></a>

Deep learning model in python and [tensorflow](https://www.tensorflow.org) trained to automatically identify cover songs. 

Dataset is the [secondhand songs set](http://labrosa.ee.columbia.edu/millionsong/secondhand), which is a subset of the [million song dataset](http://labrosa.ee.columbia.edu/millionsong/).

# Description
A cover song, by definition, is a new performance or recording of a previously recorded, commercially released song. It may be by the original artist themselves or a different artist altogether and can vary from the original in unpredictable ways including key, arrangement, instrumentation, timbre and more. We train a convolutional neural network to automatically detect cover song pairs. The network architecture consists of parallel four-layer convolutional neural networks tied together by a fully-connected softmax layer to binary classifier. The input to each 4-layer convolutional neural network is a constant-q transform (CQT) spectral representation of the song in question and the output is binary - either cover song pair, or not. Our network is trained on audio pulled from the Second Hand Song dataset, which contains a training set of approximately 25,000 unique cover song pairs.

![thing](https://github.com/markostam/coversongs-dual-convnet/blob/master/Poster/WNYISPW_2016_CoverSongs.png)

