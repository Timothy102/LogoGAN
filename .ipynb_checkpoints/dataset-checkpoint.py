import h5py
from tensorflow.data import Dataset

path = '/media/tim/Elements/Tim/logos/LLD-icon_PNG/LLD_favicons_clean_png'

# simple way to load the complete dataset (for a more sophisticated generator example, see LLD-logo script)
# open hdf5 file
hdf5_file = h5py.File('LLD-icon.hdf5', 'r')
# load data into memory as numpy array
images, labels = (hdf5_file['data'][:], hdf5_file['labels/resnet/rc_64'][:])

# alternatively, h5py objects can be used like numpy arrays without loading the whole dataset into memory:
images, labels = (hdf5_file['data'], hdf5_file['labels/resnet/rc_64'])
# here, images[0] will be again returned as a numpy array and can eg. be viewed with matplotlib using
dataset = Dataset.from_tensor_slices((images,labels))
plt.imshow(images[0])
        