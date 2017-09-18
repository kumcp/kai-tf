from self_cifar10 import KCifarClassification

from readerwriter import KReaderWriter
import os.path
from self_7food import K7FoodClassification
import tensorflow as tf



path = os.path.join(os.getcwd(),"food7")
export_dir= '/media/kum/DATA/export_food/'

# krw = KReaderWriter()
# krw.convert_real_image_to_tfrecord(path,export_dir,True)



myclsf = K7FoodClassification()
train_dir = export_dir+"train/"
eval_dir = export_dir+"test/"
myclsf.train(train_dir)
# myclsf.eval(eval_dir)



