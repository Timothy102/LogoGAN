import h5py
import tensorflow as tf
import matplotlib.pyplot as plt

path ='/media/tim/Elements/Tim/logos/'
filepath = path+'LLD-icon-sharp.hdf5'
# simple way to load the complete dataset (for a more sophisticated generator example, see LLD-logo script)
# open hdf5 file
hdf5_file = h5py.File(filepath, 'r')
# load data into memory as numpy array
images, labels = (hdf5_file['data'][:], hdf5_file['labels/resnet/rc_64'][:])

# alternatively, h5py objects can be used like numpy arrays without loading the whole dataset into memory:
images, labels = (hdf5_file['data'], hdf5_file['labels/resnet/rc_64'])
x = tf.reshape(images[0],(32,32,3))

images = images[:labels.len()]
data = tf.data.Dataset.from_tensor_slices((images,labels))
def resize(image,label):
    return tf.reshape(image,(32,32,3)),label

data.map(resize).shuffle(1000)

print(data.cardinality().numpy())