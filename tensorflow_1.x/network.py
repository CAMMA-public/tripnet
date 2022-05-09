import tensorflow as tf
import numpy as np
import os
import sys


SCOPE = 'tripnet'


class Modules():
    def __init__(self):
        super(Modules, self).__init__()

    def conv2d(self, inputs, kernel_size, filters, name, strides=1, batch_norm=False, activation=True, rate=None, is_training=True):
        x_shape = inputs.get_shape().as_list()
        x = inputs
        with tf.variable_scope(name) as scope:
            w   = tf.get_variable(name='weights', shape=[kernel_size, kernel_size, int(x_shape[3]), filters])
            if rate == None:
                x = tf.nn.conv2d(input=x, filter=w,  padding='SAME', strides=[1, strides, strides, 1], name='conv')
            else:
                x = tf.nn.atrousconv2d(value=x,  filters=w, padding='SAME', rate=rate, name='conv')
            if batch_norm:
                with tf.variable_scope('BatchNorm'):
                    x = self.batch_norm(x, is_training=is_training)
            else:
                b = tf.get_variable(name='biases', shape=[filters])
                x = x + b
            if activation:
                x = self.elu(x)
            outputs = x
        print('\tBuilding unit: {}: {} --> {}'.format( scope.name, inputs.get_shape(),  outputs.get_shape()))
        return outputs

    def batch_norm(self, x, global_step=None, is_training=True, name='bn'):
        moving_average_decay = 0.9
        with tf.variable_scope(name):
            decay = moving_average_decay
            batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2])
            with tf.device('/CPU:0'):
                mu = tf.get_variable('mu', batch_mean.get_shape(), tf.float32, initializer=tf.zeros_initializer(), trainable=False)
                sigma = tf.get_variable('sigma', batch_var.get_shape(), tf.float32, initializer=tf.ones_initializer(), trainable=False)
                beta = tf.get_variable('beta', batch_mean.get_shape(), tf.float32, initializer=tf.zeros_initializer())
                gamma = tf.get_variable('gamma', batch_var.get_shape(), tf.float32, initializer=tf.ones_initializer())
            # These ops will only be preformed when training.
            update = 1.0 - decay
            update_mu = mu.assign_sub(update*(mu - batch_mean))
            update_sigma = sigma.assign_sub(update*(sigma - batch_var))
            tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_mu)
            tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_sigma)    
            mean, var = tf.cond(is_training, lambda: (batch_mean, batch_var), lambda: (mu, sigma))
            bn = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-5)
        return bn

    def batch_norm_1d(self, x, global_step=None, is_training=True, name='bn'):
        moving_average_decay = 0.9
        with tf.variable_scope(name):
            decay = moving_average_decay
            batch_mean, batch_var = tf.nn.moments(x, [0, 1])
            with tf.device('/CPU:0'):
                mu = tf.get_variable('mu', batch_mean.get_shape(), tf.float32, initializer=tf.zeros_initializer(), trainable=False)
                sigma = tf.get_variable('sigma', batch_var.get_shape(), tf.float32, initializer=tf.ones_initializer(), trainable=False)
                beta = tf.get_variable('beta', batch_mean.get_shape(), tf.float32, initializer=tf.zeros_initializer())
                gamma = tf.get_variable('gamma', batch_var.get_shape(), tf.float32, initializer=tf.ones_initializer())
            # These ops will only be preformed when training.
            update = 1.0 - decay
            update_mu = mu.assign_sub(update*(mu - batch_mean))
            update_sigma = sigma.assign_sub(update*(sigma - batch_var))
            tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_mu)
            tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_sigma)    
            mean, var = tf.cond(is_training, lambda: (batch_mean, batch_var), lambda: (mu, sigma))
            bn = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-5)
        return bn    

    def elu(self, x, name='elu'):
        return tf.nn.elu(x, name=name)    

    def relu(self, x, name='relu'):
        return tf.nn.relu(x, name=name)        
    
    def sigmoid(self, x, name="sigmoid"):
        return tf.sigmoid(x, name=name)                

    def softmax(self, x, axis=-1, name="softmax"):
        return tf.nn.softmax(x, axis=axis, name=name)    

    def tanh(self, x, name='tanh'):
        return tf.nn.tanh(x, name=name)    
       
    def avg_pool(self, x, k=2, s=1, padding='SAME', name='avg_pool'):
        with tf.name_scope(name):
            return tf.nn.avg_pool(x, ksize=[1, k, k, 1], strides=[1, s, s, 1], padding=padding, name=name)

    def max_pool(self, x, k=4, s=1, padding='SAME', name='max_pool'):
        with tf.name_scope(name):
            return tf.nn.max_pool(value=x, ksize=[1,k,k,1], strides=[1,s,s,1], padding=padding, name=name)

    def wildcat_pool(self, x, k=4, s=1, padding='SAME'):
        with tf.name_scope(name):
            return tf.reduce_max(x, axis=[1,2]) + 0.6*tf.reduce_min(x, axis=[1,2], name='wildcat_pool')
             
    def global_avg_pool(self, x, k=4, s=1, padding='SAME', name='global_avg_pool'):
        with tf.name_scope(name):
            x = self.avg_pool(x, k=k, s=s, padding=padding, name=name)
            return tf.reduce_mean(x, axis=[1, 2])     
    
    def global_max_pool(self, x, k=4, s=1, padding='SAME', name='global_max_pool'):   
        with tf.name_scope(name): 
            x = self.max_pool(x, k=k, s=s, padding=padding, name=name)
            return tf.reduce_max(x, axis=[1,2])        

    def global_wildcat_pool(self, x, k=4, s=1, padding='SAME', name='global_wildcat_pool'):
        with tf.name_scope(name):
            return tf.reduce_max(x, axis=[1,2]) + 0.6*tf.reduce_min(x, axis=[1,2], name=name)    

    def flatten(self, inputs):
        shape   = inputs.get_shape().as_list()
        dim     = np.prod(shape[1:])         #dim     = tf.reduce_prod(tf.shape(inputs)[1:])
        return tf.reshape(inputs, shape=[-1, dim])

    def fc(self, inputs, units, name='dense'):
        with tf.variable_scope(name) as scope:
            x  = self.flatten(inputs)
            n  = x.get_shape().as_list()[-1]
            w  = tf.get_variable('weight', [n, units], initializer=tf.truncated_normal_initializer(stddev=0.02))
            b  = tf.get_variable('bias', [units], initializer=tf.constant_initializer(0))
            outputs  = tf.matmul(x, w, name='fc') + b
            print('\tBuilding unit: {}: {} --> {}'.format( scope.name, inputs.get_shape(),  outputs.get_shape()))
        return outputs
        
    def dense(self, inputs, units, name="fc"):
        with tf.variable_scope(name) as scope:
            in_dims   = inputs.get_shape().as_list()[1:]
            in_units  = np.prod(in_dims)
            out_units = (units * in_units)/in_dims[-1]
            out_dims  = in_dims.copy()
            out_dims  = [-1]+out_dims
            out_dims[-1] = units
            weight    = tf.get_variable('weight', [in_units, out_units], initializer=tf.truncated_normal_initializer(stddev=0.02))
            bias      = tf.get_variable('bias', [out_units], initializer=tf.constant_initializer(0))
            inputs    = tf.reshape(inputs, shape=[-1, np.prod(inputs.get_shape().as_list()[1:]) ])
            outputs   = tf.matmul(inputs, weight, name='fc') + bias
            outputs   = tf.reshape(outputs, shape=out_dims)                
            print('\tBuilding unit: {}: {} --> {}'.format( scope.name, inputs.get_shape(),  outputs.get_shape()))
        return outputs


class Tripnet(object):
    def __init__(self, model='tripnet', basename="resnet18", num_tool=6, num_verb=10, num_target=15, num_triplet=100, hr_output=False, dict_map_url="./"):
        super(Tripnet, self).__init__()
        self.encoder = Encoder(basename, num_tool, num_verb, num_target, num_triplet, hr_output=hr_output)
        self.decoder = Decoder(num_tool, num_verb, num_target, num_triplet, dict_map_url)

    def __call__(self, inputs, is_training):
        with tf.variable_scope(SCOPE):
            enc_i, enc_v, enc_t = self.encoder(inputs=inputs, is_training=is_training)
            dec_ivt = self.decoder(enc_i, enc_v, enc_t, is_training=is_training)
        return enc_i, enc_v, enc_t, dec_ivt


class Encoder(Modules):
    def __init__(self, basename, num_tool, num_verb, num_target, num_triplet, hr_output):
        super(Encoder, self).__init__()
        self.num_tool      = num_tool
        self.num_verb      = num_verb
        self.num_target    = num_target
        self.num_triplet   = num_triplet
        self.hr_output     = hr_output
        self.wsl           = self.tool_detector
        self.cagam         = self.cag
        if basename        == 'resnet18':
            self.basemodel =  self.resnet18
        elif basename      == 'resnet50':
            self.basemodel = self.resnet50

    def __call__(self, inputs, is_training):
        with tf.variable_scope('encoder'):  
            high_x, low_x = self.basemodel(inputs, is_training, hr_output=self.hr_output)
            enc_i         = self.wsl(high_x, is_training)
            enc_v, enc_t  = self.cagam(high_x, enc_i[0], is_training)
        return enc_i, enc_v, enc_t

    def resnet18(self, inputs, is_training=tf.constant(True), hr_output=False):  
        import resnet         
        with tf.variable_scope('backbone') as scope:
            resnet_network       = resnet.ResNet(images=inputs, version=18, is_train=is_training, hr_output=hr_output)
            high_features, end_points = resnet_network._build_model()
            low_features = end_points['block_2']
        print('\tBuilding unit: {}: {} --> {}'.format( scope.name, inputs.get_shape(),  high_features.get_shape()))
        return high_features, low_features
    
    def resnet50(self, inputs, is_training=tf.constant(True), hr_output=False): 
        from tensorflow.contrib.slim.nets import resnet_v2
        slim = tf.contrib.slim
        weight_decay       = 1e-5
        batch_norm_decay   = 0.9997
        batch_norm_epsilon = 1e-4
        with tf.variable_scope('backbone') as scope:
            with slim.arg_scope(
                 resnet_v2.resnet_arg_scope(
                         weight_decay=weight_decay,
                         batch_norm_decay=batch_norm_decay,
                         batch_norm_epsilon=batch_norm_epsilon )
                 ):
                 resnet = getattr(resnet_v2, 'resnet_v2_50')
                 _, end_points = resnet(inputs=inputs,
                                   is_training=is_training,
                                   num_classes=None,
                                   global_pool=False,
                                   output_stride=16,
                                   reuse=False)
                 high_features = end_points['{}/backbone/resnet_v2_50/block4'.format(SCOPE)]
                 low_features  = end_points['{}/backbone/resnet_v2_50/block1'.format(SCOPE)]
        print('\tBuilding unit: {}: {} --> {}'.format( scope.name, inputs.get_shape(),  high_features.get_shape()))
        return high_features, low_features

    def tool_detector(self, features, is_training):
        with tf.variable_scope('wsl'):
            features    = self.conv2d(features, kernel_size=3, filters=64, name='conv1', batch_norm=True, activation=True, is_training=is_training)
            tool_maps   = self.conv2d(features, kernel_size=1, filters=self.num_tool, name='cam', batch_norm=False, activation=False, is_training=is_training)
            tool_logits = self.global_max_pool(tool_maps, name='gmp')
        return tool_maps, tool_logits

    def cag(self, base_features, cam_features, is_training):  
        with tf.variable_scope('cag'):  
            x_shape = tf.shape(cam_features) 
            with tf.variable_scope('verb_subnet'):
                verb_pred   = self.conv2d(base_features, kernel_size=3, filters=64, name='conv1', batch_norm=True, activation=True, is_training=is_training)    
                verb_pred   = tf.image.resize_bilinear(verb_pred, size=[x_shape[1], x_shape[2]])     
                verb_pred   = tf.concat([verb_pred, cam_features], axis=-1)
                verb_maps   = self.conv2d(verb_pred, kernel_size=1, filters=64, name='conv2', batch_norm=True, activation=True, is_training=is_training)           
                verb_logits = self._fc(verb_maps, units=self.num_verb, name='verb_dense')        
            with tf.variable_scope('target_subnet'):      
                target_pred = self.conv2d(base_features, kernel_size=3, filters=64, name='conv1', batch_norm=True, activation=True, is_training=is_training)  
                target_pred = tf.image.resize_bilinear(target_pred, size=[x_shape[1], x_shape[2]])   
                target_pred = tf.concat([target_pred, cam_features], axis=-1)
                target_maps = self.conv2d(target_pred, kernel_size=1, filters=64, name='conv2', batch_norm=True, activation=True, is_training=is_training)           
                target_logits = self._fc(target_maps, units=self.num_target, name='target_dense')
        return verb_maps, verb_logits, target_maps, target_logits


class Decoder(Modules):
    def __init__(self, num_tool, num_verb, num_target, num_class, dict_map_url):
        super(Decoder, self).__init__()
        self.num_class  = num_class
        self.num_tool   = num_tool
        self.num_verb   = num_verb
        self.num_target = num_target
        self.dict_map   = dict_map_url
        self.3dis       = self.triplet_3dis

    def __call__(self, enc_i, enc_v, enc_t, is_training):
        with tf.variable_scope('decoder'):   
            feat_i   = self.elu(self.batch_norm(enc_i[0], is_training=is_training, name='bn_I'), name='elu_I')
            feat_v   = self.elu(self.batch_norm(enc_v[0], is_training=is_training, name='bn_V'), name='elu_V')
            feat_t   = self.elu(self.batch_norm(enc_t[0], is_training=is_training, name='bn_T'), name='elu_T')
            logits   = self.3dis(feat_i, feat_v, feat_t, is_training)
        return logits

    def get_valid_triplet_indices(self, num_verb, num_target, url):
        # 3D Interaction space constraints mask
        indexes = []
        with open(url) as f:              
            for line in f:
                values = line.split(',')
                if '#' in values[0]:
                    continue
                indexes.append( list(map(int, values[1:4])) )
            indexes = np.array(indexes)
        index_pos = []  
        for index in indexes:
            index_pos.append(index[0]*(num_target*num_verb) + index[1]*(num_target) + index[2])            
        return np.array(index_pos)

    def get_valid_triplet_from_3D_space(self, ivts, url, num_tool, num_verb, num_target):
        # Map 3D ispace to a vector of valid triplets
        dict_map_url   = os.path.join(os.getcwd(), 'dict/maps.txt') if str(url).lower()=='none' else url
        triplet_idx    = self.get_valid_triplet_indices(num_verb, num_target, dict_map_url) 
        ivt_flatten    = tf.reshape(ivts, [tf.shape(ivts)[0], num_tool*num_verb*num_target]) 
        valid_triplets =  tf.gather(params=ivt_flatten, indices=triplet_idx, axis=-1, name="extract_valid_triplet")
        return valid_triplets

    def triplet_3dis(self, tool_logits, verb_logits, target_logits, is_training):
        # 3D Interaction space definitions 
        with tf.variable_scope('3dis'):
            # weights for project the 3 logits onto a 3D interation space
            alpha     = tf.get_variable('alpha', [tool_logits.get_shape().as_list()[-1],   self.num_tool  ], initializer=tf.truncated_normal_initializer(stddev=0.02)) 
            beta      = tf.get_variable('beta' , [verb_logits.get_shape().as_list()[-1],   self.num_verb  ], initializer=tf.truncated_normal_initializer(stddev=0.02)) 
            gamma     = tf.get_variable('gamma', [target_logits.get_shape().as_list()[-1], self.num_target], initializer=tf.truncated_normal_initializer(stddev=0.02)) 
            tool      = tf.matmul(tool_logits, alpha, name='alpha_tool')
            verb      = tf.matmul(verb_logits, beta, name='beta_verb')
            target    = tf.matmul(target_logits, gamma, name='gamma_target')        
            # projection onto 3D ispace
            ivt_maps  = tf.einsum('bi,bv,bt->bivt', tool, verb, target ) 
            ivt_mask  = self.get_valid_triplet_from_3D_space(ivts=ivt_maps,  dict_map_url=os.path.join(self.dict_map_url, 'maps.txt'))  
            ivt_mask  = self.fc(ivt_mask, units=self.num_class, name='triplet_logits') 
        return ivt_mask 