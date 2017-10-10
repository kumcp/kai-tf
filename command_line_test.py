from readerwriter_detection import KReaderWriter
import os.path
from self_7food import K7FoodClassification
import tensorflow as tf



path = os.path.join(os.getcwd(),"data","signtest")
export_dir= '/media/kum/DATA/export_sign/'
label_map_path = '/home/kum/project/ML/models/object_detection/data/sign_label_map.pbtxt'

krw = KReaderWriter()
krw.set_label_map_path(label_map_path)
krw.convert_real_image_to_tfrecord(path,export_dir)



# myclsf = K7FoodClassification()
# train_dir = export_dir+"train/"
# eval_dir = export_dir+"test/"
# myclsf.train(train_dir)
# myclsf.eval(train_dir)



