import tensorflow as tf

class KDataManipulating(object):
    def __init__(self):
        pass    

    @staticmethod
    def modify(modify_doc):
        pass

    @staticmethod
    def cast(input, to):
        return tf.cast(input,to)

    @staticmethod
    def crop(input,type,shape=[],crop_size=None):
        if type == 'random':
            return tf.random_crop(input,shape)
        else:
            # return tf.image.crop_and_resize(input,boxes=boxes,box_ind=box_ind,crop_size=crop_size)
            resized_image = tf.image.resize_images(input,size=[500,500])

            #Crop caculation:

            random_offset_x = tf.random_uniform([],0,1)*tf.constant(50.) 
            offset_x = tf.cast(random_offset_x,tf.int32)

            random_offset_y = tf.random_uniform([],0,1)*tf.constant(50.)
            offset_y = tf.cast(random_offset_y, tf.int32)
            height = tf.cast(tf.constant(500.) - random_offset_x - tf.random_uniform([],0,1)*tf.constant(50.), tf.int32)
            width = tf.cast(tf.constant(500.) - random_offset_y - tf.random_uniform([],0,1)*tf.constant(50.) , tf.int32)

            random_crop_image = tf.image.crop_to_bounding_box(resized_image,offset_x,offset_y,height,width)
            
            result = tf.image.resize_images(random_crop_image,size=crop_size)
            
            
            return result

    @staticmethod
    def flip(input_data,type='lr',random=True):
        """
        Flip images left-right or up-down.

        Args:
            type:
                'lr': flip left-right
                'ud': flip up-down
                'lrud': flip both left-right and up-down
            random:
                True: random flip or not
                False: always flip

        Return:
            Flipped data
        """
        if random:
            if type == 'lr':
                return tf.image.random_flip_left_right(input_data)
            elif type == 'ud':
                return tf.image.random_flip_up_down(input_data)
            else:
                return tf.image.random_flip_up_down(tf.image.random_flip_left_right(input_data))
        else:
            if type == 'lr':
                return tf.image.flip_left_right(input_data)
            elif type == 'ud':
                return tf.image.flip_up_down(input_data)
            else:
                return tf.image.flip_up_down(tf.image.flip_left_right(input_data))


    @staticmethod
    def standardize(input):
        return tf.image.per_image_standardization(input)


    @staticmethod
    def normalize(input,original_minval, original_maxval, target_minval,
                    target_maxval):
        with tf.name_scope('NormalizeImage', values=[input]):
            original_minval = float(original_minval)
            original_maxval = float(original_maxval)
            target_minval = float(target_minval)
            target_maxval = float(target_maxval)
            image = tf.to_float(image)
            image = tf.subtract(image, original_minval)
            image = tf.multiply(image, (target_maxval - target_minval) /
                                (original_maxval - original_minval))
            image = tf.add(image, target_minval)
            return image
    
