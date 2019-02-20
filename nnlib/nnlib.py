import os
import sys
import contextlib
import numpy as np

from utils import std_utils
from .device import device

class nnlib(object):
    device = device #forwards nnlib.devicelib to device in order to use nnlib as standalone lib
    DeviceConfig = device.Config
    active_DeviceConfig = DeviceConfig() #default is one best GPU

    dlib = None
    
    keras = None
    keras_contrib = None
    
    tf = None
    tf_sess = None
    
    PML = None
    PMLK = None
    PMLTile= None
    
    code_import_keras = None
    code_import_keras_contrib = None
    code_import_all = None
    
    code_import_dlib = None

    
    ResNet = None
    UNet = None
    UNetTemporalPredictor = None
    NLayerDiscriminator = None

    code_import_keras_string = \
"""
keras = nnlib.keras
K = keras.backend

Input = keras.layers.Input

Dense = keras.layers.Dense
Conv2D = keras.layers.Conv2D
Conv2DTranspose = keras.layers.Conv2DTranspose
SeparableConv2D = keras.layers.SeparableConv2D
MaxPooling2D = keras.layers.MaxPooling2D
UpSampling2D = keras.layers.UpSampling2D
BatchNormalization = keras.layers.BatchNormalization

LeakyReLU = keras.layers.LeakyReLU
ReLU = keras.layers.ReLU
PReLU = keras.layers.PReLU
tanh = keras.layers.Activation('tanh')
sigmoid = keras.layers.Activation('sigmoid')
Dropout = keras.layers.Dropout
Softmax = keras.layers.Softmax

Lambda = keras.layers.Lambda
Add = keras.layers.Add
Concatenate = keras.layers.Concatenate

Flatten = keras.layers.Flatten
Reshape = keras.layers.Reshape

ZeroPadding2D = keras.layers.ZeroPadding2D       

RandomNormal = keras.initializers.RandomNormal
Model = keras.models.Model

Adam = keras.optimizers.Adam

modelify = nnlib.modelify
gaussian_blur = nnlib.gaussian_blur
style_loss = nnlib.style_loss
dssim = nnlib.dssim

#ReflectionPadding2D = nnlib.ReflectionPadding2D
PixelShuffler = nnlib.PixelShuffler
SubpixelUpscaler = nnlib.SubpixelUpscaler
#AddUniformNoise = nnlib.AddUniformNoise
"""
    code_import_keras_contrib_string = \
"""
keras_contrib = nnlib.keras_contrib
GroupNormalization = keras_contrib.layers.GroupNormalization
InstanceNormalization = keras_contrib.layers.InstanceNormalization
Padam = keras_contrib.optimizers.Padam
"""
    code_import_dlib_string = \
"""
dlib = nnlib.dlib
"""    

    code_import_all_string = \
"""
DSSIMMSEMaskLoss = nnlib.DSSIMMSEMaskLoss
ResNet = nnlib.ResNet
UNet = nnlib.UNet
UNetTemporalPredictor = nnlib.UNetTemporalPredictor
NLayerDiscriminator = nnlib.NLayerDiscriminator
"""
    
            
    @staticmethod
    def _import_tf(device_config):
        if nnlib.tf is not None:
            return nnlib.code_import_tf

        if 'TF_SUPPRESS_STD' in os.environ.keys() and os.environ['TF_SUPPRESS_STD'] == '1':
            suppressor = std_utils.suppress_stdout_stderr().__enter__()
        else:
            suppressor = None
            
        if 'CUDA_VISIBLE_DEVICES' in os.environ.keys():
            os.environ.pop('CUDA_VISIBLE_DEVICES')
        
        os.environ['TF_MIN_GPU_MULTIPROCESSOR_COUNT'] = '2'
        
        import tensorflow as tf
        nnlib.tf = tf
        
        if device_config.cpu_only:
            config = tf.ConfigProto(device_count={'GPU': 0})
        else:   
            config = tf.ConfigProto()
            
            if device_config.backend != "tensorflow-generic":
                #tensorflow-generic is system with NVIDIA card, but w/o NVSMI
                #so dont hide devices and let tensorflow to choose best card
                visible_device_list = ''
                for idx in device_config.gpu_idxs:
                    visible_device_list += str(idx) + ','
                config.gpu_options.visible_device_list=visible_device_list[:-1]
            
        config.gpu_options.force_gpu_compatible = True            
        config.gpu_options.allow_growth = device_config.allow_growth

        nnlib.tf_sess = tf.Session(config=config)
            
        if suppressor is not None:  
            suppressor.__exit__()
        
    @staticmethod
    def import_keras(device_config = None):
        if nnlib.keras is not None:
            return nnlib.code_import_keras

        if device_config is None:
            device_config = nnlib.active_DeviceConfig
            
        nnlib.active_DeviceConfig = device_config

        if "tensorflow" in device_config.backend:
            nnlib._import_tf(device_config)
            device_config = nnlib.active_DeviceConfig
        elif device_config.backend == "plaidML":
            os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
            os.environ["PLAIDML_DEVICE_IDS"] = ",".join ( [ nnlib.device.getDeviceID(idx) for idx in device_config.gpu_idxs] )

        if 'TF_SUPPRESS_STD' in os.environ.keys() and os.environ['TF_SUPPRESS_STD'] == '1':
            suppressor = std_utils.suppress_stdout_stderr().__enter__()
 
        import keras as keras_
        nnlib.keras = keras_
        
        if device_config.backend == "plaidML":
            import plaidml
            import plaidml.tile
            nnlib.PML = plaidml
            nnlib.PMLK = plaidml.keras.backend
            nnlib.PMLTile = plaidml.tile
            
        if device_config.use_fp16:
            nnlib.keras.backend.set_floatx('float16')
        
        if "tensorflow" in device_config.backend:
            nnlib.keras.backend.set_session(nnlib.tf_sess)
            
        nnlib.keras.backend.set_image_data_format('channels_last')
        
        if 'TF_SUPPRESS_STD' in os.environ.keys() and os.environ['TF_SUPPRESS_STD'] == '1':        
            suppressor.__exit__()

        nnlib.__initialize_keras_functions()  
        nnlib.code_import_keras = compile (nnlib.code_import_keras_string,'','exec')
        return nnlib.code_import_keras
        
    @staticmethod
    def __initialize_keras_functions():
        keras = nnlib.keras
        K = keras.backend
        
        def modelify(model_functor):
            def func(tensor):
                return keras.models.Model (tensor, model_functor(tensor))
            return func
        
        nnlib.modelify = modelify
        
        def gaussian_blur(radius=2.0):
            def gaussian(x, mu, sigma):
                return np.exp(-(float(x) - float(mu)) ** 2 / (2 * sigma ** 2))

            def make_kernel(sigma):
                kernel_size = max(3, int(2 * 2 * sigma + 1))
                mean = np.floor(0.5 * kernel_size)
                kernel_1d = np.array([gaussian(x, mean, sigma) for x in range(kernel_size)])
                np_kernel = np.outer(kernel_1d, kernel_1d).astype(dtype=K.floatx())
                kernel = np_kernel / np.sum(np_kernel)
                return kernel
          
            gauss_kernel = make_kernel(radius)
            gauss_kernel = gauss_kernel[:, :,np.newaxis, np.newaxis]

            def func(input):
                inputs = [ input[:,:,:,i:i+1]  for i in range( K.int_shape( input )[-1] ) ]

                outputs = []
                for i in range(len(inputs)):
                    outputs += [ K.conv2d( inputs[i] , K.constant(gauss_kernel) , strides=(1,1), padding="same") ]

                return K.concatenate (outputs, axis=-1)
            return func
        nnlib.gaussian_blur = gaussian_blur
        
        def style_loss(gaussian_blur_radius=0.0, loss_weight=1.0, wnd_size=0, step_size=1):
            if gaussian_blur_radius > 0.0:
                gblur = gaussian_blur(gaussian_blur_radius)
            
            def sd(content, style, loss_weight):
                content_nc = K.int_shape(content)[-1]
                style_nc = K.int_shape(style)[-1]
                if content_nc != style_nc:
                    raise Exception("style_loss() content_nc != style_nc")
                    
                axes = [1,2]
                c_mean, c_var = K.mean(content, axis=axes, keepdims=True), K.var(content, axis=axes, keepdims=True)
                s_mean, s_var = K.mean(style, axis=axes, keepdims=True), K.var(style, axis=axes, keepdims=True)
                c_std, s_std = K.sqrt(c_var + 1e-5), K.sqrt(s_var + 1e-5)

                mean_loss = K.sum(K.square(c_mean-s_mean))
                std_loss = K.sum(K.square(c_std-s_std))
                
                return (mean_loss + std_loss) * ( loss_weight / float(content_nc) )
                
            def func(target, style):
                if wnd_size == 0:
                    if gaussian_blur_radius > 0.0:
                        return sd( gblur(target), gblur(style), loss_weight=loss_weight)
                    else:
                        return sd( target, style, loss_weight=loss_weight )
                else:
                    #currently unused
                    if nnlib.tf is not None:
                        sh = K.int_shape(target)[1]
                        k = (sh-wnd_size) // step_size + 1                        
                        if gaussian_blur_radius > 0.0:
                            target, style = gblur(target), gblur(style)                        
                        target = nnlib.tf.image.extract_image_patches(target, [1,k,k,1], [1,1,1,1], [1,step_size,step_size,1], 'VALID')
                        style  = nnlib.tf.image.extract_image_patches(style,  [1,k,k,1], [1,1,1,1], [1,step_size,step_size,1], 'VALID')
                        return sd( target, style, loss_weight )
                    if nnlib.PML is not None:
                        print ("Sorry, plaidML backend does not support style_loss")
                        return 0
            return func
        nnlib.style_loss = style_loss  
        
        def dssim(k1=0.01, k2=0.03, max_value=1.0):
            # port of tf.image.ssim to pure keras in order to work on plaidML backend.

            def func(y_true, y_pred):
                ch = K.shape(y_pred)[-1]

                def _fspecial_gauss(size, sigma):
                    #Function to mimic the 'fspecial' gaussian MATLAB function.
                    coords = np.arange(0, size, dtype=K.floatx())
                    coords -= (size - 1 ) / 2.0
                    g = coords**2
                    g *= ( -0.5 / (sigma**2) )
                    g = np.reshape (g, (1,-1)) + np.reshape(g, (-1,1) )
                    g = K.constant ( np.reshape (g, (1,-1)) )
                    g = K.softmax(g)
                    g = K.reshape (g, (size, size, 1, 1)) 
                    g = K.tile (g, (1,1,ch,1))
                    return g
                              
                kernel = _fspecial_gauss(11,1.5)

                def reducer(x):
                    return K.depthwise_conv2d(x, kernel, strides=(1, 1), padding='valid')

                c1 = (k1 * max_value) ** 2
                c2 = (k2 * max_value) ** 2
                
                mean0 = reducer(y_true)
                mean1 = reducer(y_pred)
                num0 = mean0 * mean1 * 2.0
                den0 = K.square(mean0) + K.square(mean1)
                luminance = (num0 + c1) / (den0 + c1)
                
                num1 = reducer(y_true * y_pred) * 2.0
                den1 = reducer(K.square(y_true) + K.square(y_pred))
                c2 *= 1.0 #compensation factor
                cs = (num1 - num0 + c2) / (den1 - den0 + c2)

                ssim_val = K.mean(luminance * cs, axis=(-3, -2) )
                return K.mean( (1.0 - ssim_val ) / 2.0 )

            return func
        
        nnlib.dssim = dssim
        
        class PixelShuffler(keras.layers.Layer):
            def __init__(self, size=(2, 2), data_format=None, **kwargs):
                super(PixelShuffler, self).__init__(**kwargs)
                self.data_format = K.normalize_data_format(data_format)
                self.size = keras.utils.conv_utils.normalize_tuple(size, 2, 'size')

            def call(self, inputs):

                input_shape = K.int_shape(inputs)
                if len(input_shape) != 4:
                    raise ValueError('Inputs should have rank ' +
                                     str(4) +
                                     '; Received input shape:', str(input_shape))

                if self.data_format == 'channels_first':
                    batch_size, c, h, w = input_shape
                    if batch_size is None:
                        batch_size = -1
                    rh, rw = self.size
                    oh, ow = h * rh, w * rw
                    oc = c // (rh * rw)

                    out = K.reshape(inputs, (batch_size, rh, rw, oc, h, w))
                    out = K.permute_dimensions(out, (0, 3, 4, 1, 5, 2))
                    out = K.reshape(out, (batch_size, oc, oh, ow))
                    return out

                elif self.data_format == 'channels_last':
                    batch_size, h, w, c = input_shape
                    if batch_size is None:
                        batch_size = -1
                    rh, rw = self.size
                    oh, ow = h * rh, w * rw
                    oc = c // (rh * rw)

                    out = K.reshape(inputs, (batch_size, h, w, rh, rw, oc))
                    out = K.permute_dimensions(out, (0, 1, 3, 2, 4, 5))
                    out = K.reshape(out, (batch_size, oh, ow, oc))
                    return out

            def compute_output_shape(self, input_shape):

                if len(input_shape) != 4:
                    raise ValueError('Inputs should have rank ' +
                                     str(4) +
                                     '; Received input shape:', str(input_shape))

                if self.data_format == 'channels_first':
                    height = input_shape[2] * self.size[0] if input_shape[2] is not None else None
                    width = input_shape[3] * self.size[1] if input_shape[3] is not None else None
                    channels = input_shape[1] // self.size[0] // self.size[1]

                    if channels * self.size[0] * self.size[1] != input_shape[1]:
                        raise ValueError('channels of input and size are incompatible')

                    return (input_shape[0],
                            channels,
                            height,
                            width)

                elif self.data_format == 'channels_last':
                    height = input_shape[1] * self.size[0] if input_shape[1] is not None else None
                    width = input_shape[2] * self.size[1] if input_shape[2] is not None else None
                    channels = input_shape[3] // self.size[0] // self.size[1]

                    if channels * self.size[0] * self.size[1] != input_shape[3]:
                        raise ValueError('channels of input and size are incompatible')

                    return (input_shape[0],
                            height,
                            width,
                            channels)

            def get_config(self):
                config = {'size': self.size,
                          'data_format': self.data_format}
                base_config = super(PixelShuffler, self).get_config()

                return dict(list(base_config.items()) + list(config.items()))        
        
        nnlib.PixelShuffler = PixelShuffler
        nnlib.SubpixelUpscaler = PixelShuffler    
        '''
        
        class ReflectionPadding2D(keras.layers.Layer):
            def __init__(self, padding=(1, 1), **kwargs):
                self.padding = tuple(padding)
                self.input_spec = [keras.layers.InputSpec(ndim=4)]
                super(ReflectionPadding2D, self).__init__(**kwargs)

            def compute_output_shape(self, s):
                """ If you are using "channels_last" configuration"""
                return (s[0], s[1] + 2 * self.padding[0], s[2] + 2 * self.padding[1], s[3])

            def call(self, x, mask=None):
                w_pad,h_pad = self.padding
                return tf.pad(x, [[0,0], [h_pad,h_pad], [w_pad,w_pad], [0,0] ], 'REFLECT')
        nnlib.ReflectionPadding2D = ReflectionPadding2D

       
        class AddUniformNoise(keras.layers.Layer):
            def __init__(self, power=1.0, minval=-1.0, maxval=1.0, **kwargs):
                super(AddUniformNoise, self).__init__(**kwargs)
                self.power = power
                self.supports_masking = True
                self.minval = minval
                self.maxval = maxval

            def call(self, inputs, training=None):
                def noised():
                    return inputs + self.power*K.random_uniform(shape=K.shape(inputs), minval=self.minval, maxval=self.maxval)
                return K.in_train_phase(noised, inputs, training=training)

            def get_config(self):
                config = {'power': self.power, 'minval': self.minval, 'maxval': self.maxval}
                base_config = super(AddUniformNoise, self).get_config()
                return dict(list(base_config.items()) + list(config.items()))
        nnlib.AddUniformNoise = AddUniformNoise       
        '''        
    @staticmethod
    def import_keras_contrib(device_config = None):
        if nnlib.keras_contrib is not None:
            return nnlib.code_import_keras_contrib
        
        import keras_contrib as keras_contrib_    
        nnlib.keras_contrib = keras_contrib_
        nnlib.__initialize_keras_contrib_functions()    
        nnlib.code_import_keras_contrib = compile (nnlib.code_import_keras_contrib_string,'','exec')
    
    @staticmethod
    def __initialize_keras_contrib_functions():
        pass
        
    @staticmethod
    def import_dlib( device_config = None):
        if nnlib.dlib is not None:
            return nnlib.code_import_dlib

        import dlib as dlib_
        nnlib.dlib = dlib_
        if not device_config.cpu_only and "tensorflow" in device_config.backend and len(device_config.gpu_idxs) > 0:
            nnlib.dlib.cuda.set_device(device_config.gpu_idxs[0])           
        
        nnlib.code_import_dlib = compile (nnlib.code_import_dlib_string,'','exec')
    
    @staticmethod
    def import_all(device_config = None):
        if nnlib.code_import_all is None:  
            nnlib.import_keras(device_config)
            nnlib.import_keras_contrib(device_config)                                                
            nnlib.code_import_all = compile (nnlib.code_import_keras_string + '\n'
                                            + nnlib.code_import_keras_contrib_string 
                                            + nnlib.code_import_all_string,'','exec')        
            nnlib.__initialize_all_functions()
        
        return nnlib.code_import_all
    
    @staticmethod
    def __initialize_all_functions():
        exec (nnlib.import_keras(), locals(), globals())
        exec (nnlib.import_keras_contrib(), locals(), globals())
        
        class DSSIMMSEMaskLoss(object):
            def __init__(self, mask, is_mse=False):
                self.mask = mask
                self.is_mse = is_mse                
            def __call__(self,y_true, y_pred):
                total_loss = None
                mask = self.mask
                if self.is_mse:                
                    blur_mask = gaussian_blur(max(1, K.int_shape(mask)[1] // 64))(mask)
                    return K.mean ( 50*K.square( y_true*blur_mask - y_pred*blur_mask ) )
                else:
                    return 10*dssim() (y_true*mask, y_pred*mask)                    
        nnlib.DSSIMMSEMaskLoss = DSSIMMSEMaskLoss
        
        '''
        def ResNet(output_nc, use_batch_norm, ngf=64, n_blocks=6, use_dropout=False):
            exec (nnlib.import_all(), locals(), globals())

            if not use_batch_norm:
                use_bias = True
                def XNormalization(x):
                    return InstanceNormalization (axis=3, gamma_initializer=RandomNormal(1., 0.02))(x)#GroupNormalization (axis=3, groups=K.int_shape (x)[3] // 4, gamma_initializer=RandomNormal(1., 0.02))(x)
            else:
                use_bias = False
                def XNormalization(x):
                    return BatchNormalization (axis=3, gamma_initializer=RandomNormal(1., 0.02))(x)
                    
            def Conv2D (filters, kernel_size, strides=(1, 1), padding='valid', data_format=None, dilation_rate=(1, 1), activation=None, use_bias=use_bias, kernel_initializer=RandomNormal(0, 0.02), bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None):
                return keras.layers.Conv2D( filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, data_format=data_format, dilation_rate=dilation_rate, activation=activation, use_bias=use_bias, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer, activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint, bias_constraint=bias_constraint )
        
            def Conv2DTranspose(filters, kernel_size, strides=(1, 1), padding='valid', output_padding=None, data_format=None, dilation_rate=(1, 1), activation=None, use_bias=use_bias, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None):
                return keras.layers.Conv2DTranspose(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, output_padding=output_padding, data_format=data_format, dilation_rate=dilation_rate, activation=activation, use_bias=use_bias, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer, activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint, bias_constraint=bias_constraint)
            
            def func(input):

               
                def ResnetBlock(dim):    
                    def func(input):
                        x = input
                        
                        x = ReflectionPadding2D((1,1))(x)
                        x = Conv2D(dim, 3, 1, padding='valid')(x)
                        x = XNormalization(x)
                        x = ReLU()(x)
                        
                        if use_dropout:
                            x = Dropout(0.5)(x)
                            
                        x = ReflectionPadding2D((1,1))(x)
                        x = Conv2D(dim, 3, 1, padding='valid')(x)
                        x = XNormalization(x)
                        x = ReLU()(x)       
                        return Add()([x,input])            
                    return func

                x = input
                
                x = ReflectionPadding2D((3,3))(x)
                x = Conv2D(ngf, 7, 1, 'valid')(x)
                
                x = ReLU()(XNormalization(Conv2D(ngf*2, 4, 2, 'same')(x)))
                x = ReLU()(XNormalization(Conv2D(ngf*4, 4, 2, 'same')(x)))
                
                for i in range(n_blocks):
                    x = ResnetBlock(ngf*4)(x)
                    
                x = ReLU()(XNormalization(PixelShuffler()(Conv2D(ngf*2 *4, 3, 1, 'same')(x))))
                x = ReLU()(XNormalization(PixelShuffler()(Conv2D(ngf   *4, 3, 1, 'same')(x))))
                
                x = ReflectionPadding2D((3,3))(x)
                x = Conv2D(output_nc, 7, 1, 'valid')(x)
                x = tanh(x)
                
                return x
                
            return func  
            
        nnlib.ResNet = ResNet
             
        # Defines the Unet generator.
        # |num_downs|: number of downsamplings in UNet. For example,
        # if |num_downs| == 7, image of size 128x128 will become of size 1x1
        # at the bottleneck
        def UNet(output_nc, use_batch_norm, num_downs, ngf=64, use_dropout=False):
            exec (nnlib.import_all(), locals(), globals())

            if not use_batch_norm:
                use_bias = True
                def XNormalization(x):
                    return InstanceNormalization (axis=3, gamma_initializer=RandomNormal(1., 0.02))(x)#GroupNormalization (axis=3, groups=K.int_shape (x)[3] // 4, gamma_initializer=RandomNormal(1., 0.02))(x)
            else:
                use_bias = False
                def XNormalization(x):
                    return BatchNormalization (axis=3, gamma_initializer=RandomNormal(1., 0.02))(x)
                    
            def Conv2D (filters, kernel_size, strides=(1, 1), padding='valid', data_format=None, dilation_rate=(1, 1), activation=None, use_bias=use_bias, kernel_initializer=RandomNormal(0, 0.02), bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None):
                return keras.layers.Conv2D( filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, data_format=data_format, dilation_rate=dilation_rate, activation=activation, use_bias=use_bias, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer, activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint, bias_constraint=bias_constraint )

            def Conv2DTranspose(filters, kernel_size, strides=(1, 1), padding='valid', output_padding=None, data_format=None, dilation_rate=(1, 1), activation=None, use_bias=use_bias, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None):
                return keras.layers.Conv2DTranspose(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, output_padding=output_padding, data_format=data_format, dilation_rate=dilation_rate, activation=activation, use_bias=use_bias, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer, activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint, bias_constraint=bias_constraint)
                
            def UNetSkipConnection(outer_nc, inner_nc, sub_model=None, outermost=False, innermost=False, use_dropout=False):       
                def func(inp):
                    x = inp
                    
                    x = Conv2D(inner_nc, 4, 2, 'valid')(ReflectionPadding2D( (1,1) )(x))
                    x = XNormalization(x)
                    x = ReLU()(x)
                        
                    if not innermost:
                        x = sub_model(x)
                        
                    if not outermost:
                        x = Conv2DTranspose(outer_nc, 3, 2, 'same')(x)
                        x = XNormalization(x)
                        x = ReLU()(x)
                        
                        if not innermost:
                            if use_dropout:
                                x = Dropout(0.5)(x)
                            
                        x = Concatenate(axis=3)([inp, x])
                    else:
                        x = Conv2DTranspose(outer_nc, 3, 2, 'same')(x)
                        x = tanh(x)   
                        

                    return x           
                    
                return func
                    
            def func(input):                    

                unet_block = UNetSkipConnection(ngf * 8, ngf * 8, sub_model=None, innermost=True)

                for i in range(num_downs - 5):
                    unet_block = UNetSkipConnection(ngf * 8, ngf * 8, sub_model=unet_block, use_dropout=use_dropout)
                
                unet_block = UNetSkipConnection(ngf * 4  , ngf * 8, sub_model=unet_block)
                unet_block = UNetSkipConnection(ngf * 2  , ngf * 4, sub_model=unet_block)
                unet_block = UNetSkipConnection(ngf      , ngf * 2, sub_model=unet_block)
                unet_block = UNetSkipConnection(output_nc, ngf    , sub_model=unet_block, outermost=True)
                
                return unet_block(input)
            return func
        nnlib.UNet = UNet
        
        #predicts based on two past_image_tensors
        def UNetTemporalPredictor(output_nc, use_batch_norm, num_downs, ngf=64, use_dropout=False):
            exec (nnlib.import_all(), locals(), globals())
            def func(inputs):
                past_2_image_tensor, past_1_image_tensor = inputs
                    
                x = Concatenate(axis=3)([ past_2_image_tensor, past_1_image_tensor ])
                x = UNet(3, use_batch_norm, num_downs=num_downs, ngf=ngf, use_dropout=use_dropout) (x)

                return x
                    
            return func                
        nnlib.UNetTemporalPredictor = UNetTemporalPredictor
        
        def NLayerDiscriminator(use_batch_norm, ndf=64, n_layers=3):
            exec (nnlib.import_all(), locals(), globals())
            
            if not use_batch_norm:
                use_bias = True
                def XNormalization(x):
                    return InstanceNormalization (axis=3, gamma_initializer=RandomNormal(1., 0.02))(x)#GroupNormalization (axis=3, groups=K.int_shape (x)[3] // 4, gamma_initializer=RandomNormal(1., 0.02))(x)
            else:
                use_bias = False
                def XNormalization(x):
                    return BatchNormalization (axis=3, gamma_initializer=RandomNormal(1., 0.02))(x)

            def Conv2D (filters, kernel_size, strides=(1, 1), padding='valid', data_format=None, dilation_rate=(1, 1), activation=None, use_bias=use_bias, kernel_initializer=RandomNormal(0, 0.02), bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None):
                return keras.layers.Conv2D( filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, data_format=data_format, dilation_rate=dilation_rate, activation=activation, use_bias=use_bias, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer, activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint, bias_constraint=bias_constraint )

            def func(input):
                x = input
                
                x = ZeroPadding2D((1,1))(x)
                x = Conv2D( ndf, 4, 2, 'valid')(x)
                x = LeakyReLU(0.2)(x)
                    
                for i in range(1, n_layers):        
                    x = ZeroPadding2D((1,1))(x)
                    x = Conv2D( ndf * min(2 ** i, 8), 4, 2, 'valid')(x)
                    x = XNormalization(x)
                    x = LeakyReLU(0.2)(x)
                    
                x = ZeroPadding2D((1,1))(x)
                x = Conv2D( ndf * min(2 ** n_layers, 8), 4, 1, 'valid')(x)
                x = XNormalization(x)    
                x = LeakyReLU(0.2)(x)
                
                x = ZeroPadding2D((1,1))(x)
                return Conv2D( 1, 4, 1, 'valid')(x)
            return func
        nnlib.NLayerDiscriminator = NLayerDiscriminator
        '''
    @staticmethod
    def finalize_all():
        if nnlib.keras_contrib is not None:
            nnlib.keras_contrib = None
        
        if nnlib.keras is not None:
            nnlib.keras.backend.clear_session()
            nnlib.keras = None

        if nnlib.tf is not None:
            nnlib.tf_sess = None
            nnlib.tf = None


