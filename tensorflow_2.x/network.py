#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 18 19:09:36 2021

@author: nwoye
"""

import tensorflow as tf

SCOPE           = 'tripnet'
INPUT_SHAPE     = (256,448,3)
OUTPUT_SHAPE    = (256,448,3)
NETSCOPE        = {
            'mobilenet':{
                    'high_level_feature':'out_relu', 
                    'low_level_feature':'block_1_project_BN', 
                    'low_output_shape_ratio':(4,4), 
                    'high_output_shape_ratio':(8,8)
                    },
            'mobilenetv2':{
                    'high_level_feature':'out_relu', 
                    'low_level_feature':'block_1_project_BN', 
                    'low_output_shape_ratio':(4,4), 
                    'high_output_shape_ratio':(8,8)
                    },
            'xception':{
                    'high_level_feature':'block14_sepconv2_act', 
                    'low_level_feature':'block1_conv2_act', 
                    'low_output_shape_ratio':(4,4), 
                    'high_output_shape_ratio':(8,8)
                    },
            'resnet50':{
                    'high_level_feature':'conv5_block3_out', 
                    'low_level_feature':'pool1_pool', 
                    'low_output_shape_ratio':(4,4), 
                    'high_output_shape_ratio':(8,8)
                    },
            'resnet50v2':{
                    'high_level_feature':'post_relu', 
                    'low_level_feature':'pool1_pool', 
                    'low_output_shape_ratio':(4,4), 
                    'high_output_shape_ratio':(8,8)
                    },
            'resnet18v2':{
                    'high_level_feature':'post_relu', 
                    'low_level_feature':'pool1_pool', 
                    'low_output_shape_ratio':(4,4), 
                    'high_output_shape_ratio':(8,8)
                    },
            'densenet169':{
                    'high_level_feature':'bn', 
                    'low_level_feature':'pool1', 
                    'low_output_shape_ratio':(4,4), 
                    'high_output_shape_ratio':(8,8)

                    }
        }    


class Tripnet(tf.keras.Model):
    """
    Rendezvous: surgical action triplet recognition by Nwoye, C.I. et.al. 2020
    @args:
        image_shape: a tuple (height, width) e.g: (224,224)
        basename: Feature extraction network: e.g: "resnet50", "VGG19"
        num_tool: default = 6, 
        num_verb: default = 10, 
        num_target: default = 15, 
        num_triplet: default = 100, 
        dict_map_url: path to the map file for the triplet decomposition
    @call:
        inputs: Batch of input images of shape [batch, height, width, channel]
        training: True or False. Python boolean indicating whether the layer should behave in training mode (adding dropout) or in inference mode (doing nothing)
    @output: 
        enc_i: tuple (cam, logits) for instrument
        enc_v: logits for verb
        enc_t: logits for target
        dec_ivt: logits for triplet
    """
    def __init__(self, image_shape=(256,448,3), basename="resnet50", pretrained='imagenet', num_tool=6, num_verb=10, num_target=15, num_triplet=100, dict_map_url="./"):
        super(Tripnet, self).__init__()
        inputs          = tf.keras.Input(shape=image_shape)
        self.encoder    = Encoder(basename, pretrained, image_shape, num_tool, num_verb, num_target, num_triplet)
        self.decoder    = Decoder(num_tool, num_verb, num_target, num_triplet, dict_map_url)
        enc_i, enc_v, enc_t = self.encoder(inputs)
        dec_ivt         = self.decoder(enc_i, enc_v, enc_t)
        self.tripnet    = tf.keras.models.Model(inputs=inputs, outputs=(enc_i, enc_v, enc_t, dec_ivt), name='tripnet')

    def call(self, inputs, training):
        enc_i, enc_v, enc_t, dec_ivt = self.tripnet(inputs, training=training)
        return enc_i, enc_v, enc_t, dec_ivt
    

# Model Encoder
class Encoder(tf.keras.layers.Layer):
    def __init__(self, basename, pretrained, image_shape, num_tool=6, num_verb, num_target, num_triplet):
        super(Encoder, self).__init__()
        self.basemodel  = Basemodel(basename, image_shape, pretrained)
        self.wsl        = WSL(num_tool)
        self.cagam      = CAG(num_verb, num_target)

    def call(self, inputs, training):
        low_x, high_x   = self.basemodel(inputs, training)
        enc_i           = self.wsl(high_x, training)
        enc_v, enc_t    = self.cagam(high_x, enc_i[0], training)
        return enc_i, enc_v, enc_t



# Backbone
class Basemodel(tf.keras.layers.Layer):
    def __init__(self, basename, image_shape, pretrained='imagenet'):
        super(Basemodel, self).__init__()
        if basename ==  'mobilenet':
            base_model = tf.keras.applications.MobileNetV2(
                                    input_shape=image_shape,
                                    include_top=False,
                                    weights='imagenet')
        elif basename ==  'mobilenetv2':
            base_model = tf.keras.applications.mobilenet_v2.MobileNetV2(
                                    input_shape=image_shape,
                                    include_top=False,
                                    weights='imagenet')
        elif basename ==  'xception':
            base_model = tf.keras.applications.Xception(
                                    weights='imagenet',  # Load weights pre-trained on ImageNet.
                                    input_shape=image_shape,
                                    include_top=False)  # Do not include the ImageNet classifier at the top.
        elif basename ==  'resnet50':
            base_model = tf.keras.applications.resnet50.ResNet50(
                                    weights=pretrained,  # Load weights pre-trained on ImageNet.
                                    input_shape=image_shape,
                                    include_top=False)  # Do not include the ImageNet classifier at the top.
        elif basename ==  'resnet50v2':
            base_model = tf.keras.applications.resnet_v2.ResNet50V2(
                                    weights='imagenet',  # Load weights pre-trained on ImageNet.
                                    input_shape=image_shape,
                                    include_top=False)  # Do not include the ImageNet classifier at the top.
        elif basename ==  'resnet18v2':
            import resnet_v2
            base_model = resnet_v2.ResNet18V2(
                                    weights=None,  # Load weights pre-trained on ImageNet.
                                    input_shape=image_shape,
                                    stride=1,
                                    include_top=False)  # Do not include the ImageNet classifier at the top.
        elif basename =='densenet169':
            base_model = tf.keras.applications.densenet.DenseNet169(
                                    include_top=False, 
                                    weights='imagenet',
                                    input_shape=image_shape )
        else: base_model = tf.keras.applications.resnet18.ResNet18( # not impl.
                                    weights='imagenet', # Load weights pre-trained on ImageNet.
                                    input_shape=image_shape,
                                    include_top=False)  # Do not include the ImageNet classifier at the top.
        self.base_model = tf.keras.models.Model(inputs=base_model.input, 
                                                outputs=(base_model.get_layer(NETSCOPE[basename]['low_level_feature']).output, base_model.output),
                                                name='backbone')
        # self.base_model.trainable = trainable        

    def call(self, inputs, training):
        return self.base_model(inputs, training=training)
            
            
# WSL of Tools
class WSL(tf.keras.layers.Layer):
    def __init__(self, num_class, depth=64):
        super(WSL, self).__init__()
        self.num_class = num_class
        self.conv1 = tf.keras.layers.Conv2D(depth, 3, activation=None, padding='same')
        self.conv2 = tf.keras.layers.Conv2D(num_class, 1, activation=None, name='cam')
        self.gmp   = tf.keras.layers.GlobalMaxPooling2D()
        self.bn    = tf.keras.layers.BatchNormalization()
        self.elu   = tf.keras.activations.elu

    def call(self, inputs, training):
        feature = self.conv1(inputs, training=training)
        feature = self.elu(self.bn(feature, training=training))
        cam     = self.conv2(feature, training=training)
        logits  = self.gmp(cam)
        return cam, logits


# Class Activation Guide
class CAG(tf.keras.layers.Layer):
    def __init__(self, num_verb, num_target):
        super(CAG, self).__init__()
        self.depth  = 64
        self.conv1  = tf.keras.layers.Conv2D(self.depth, 3, activation="elu", padding='same')
        self.cmap1  = tf.keras.layers.Conv2D(self.depth, 1, activation="elu")
        self.logit1 = tf.keras.layers.Dense(num_verb, activation=None)
        self.conv2  = tf.keras.layers.Conv2D(self.depth, 3, activation="elu", padding='same')
        self.cmap2  = tf.keras.layers.Conv2D(self.depth, 1, activation="elu")
        self.logit2 = tf.keras.layers.Dense(num_target, activation=None)

    def call(self, inputs, cam, training):
        # verb
        verb_x = self.conv1(inputs, training=training)
        verb_x = tf.concat((verb_x, cam), axis=-1)
        verb_x = self.conv2(verb_x, training=training)
        logit_v = self.logit1(verb_x, training=training)
        # target
        target_x = self.conv1(inputs, training=training)
        target_x = tf.concat((target_x, cam), axis=-1)
        target_x = self.conv2(target_x, training=training)
        logit_t = self.logit1(target_x, training=training)
        return (verb_x,logit_v), (target_x,logit_t)


# 3D interaction space
class Decoder(tf.keras.layers.Layer):
    def __init__(self, num_tool, num_verb, num_target, num_triplet, dict_map_url):
        super(Decoder, self).__init__()
        self.num_tool       = num_tool
        self.num_verb       = num_verb
        self.num_target     = num_target
        self.valid_position = self.constraint(num_verb, num_target, url=os.path.join(dict_map_url, 'maps.txt'))
        self.alpha          = self.add_weight("decoder/3dis/triplet/alpha", shape=(self.num_tool, self.num_tool))
        self.beta           = self.add_weight("decoder/3dis/triplet/beta", shape=(self.num_verb, self.num_verb))
        self.gamma          = self.add_weight("decoder/3dis/triplet/gamma", shape=(self.num_target, self.num_target))
        self.fc             = tf.keras.layers.Dense(num_triplet, name='mlp')

    def constraint(self, num_verb, num_target, url):
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

    def mask(self, ivts):
        # Map 3D ispace to a vector of valid triplets
        ivt_flatten    = tf.reshape(ivts, [tf.shape(ivts)[0], self.num_tool*self.num_verb*self.num_target])
        valid_triplets = tf.gather(params=ivt_flatten, indices=self.valid_position, axis=-1, name="extract_valid_triplet")
        return valid_triplets

    def call(self, tool_logits, verb_logits, target_logits, is_training):
        tool      = tf.matmul(tool_logits, self.alpha, name='alpha_tool')
        verb      = tf.matmul(verb_logits, self.beta, name='beta_verb')
        target    = tf.matmul(target_logits, self.gamma, name='gamma_target')   
        ivt_maps  = tf.einsum('bi,bv,bt->bivt', tool, verb, target ) 
        ivt_mask  = self.mask(ivts=ivt_maps)  
        ivt_mask  = self.fc(self.fc)
        return ivt_mask  
