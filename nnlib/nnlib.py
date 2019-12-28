import os
import sys
import contextlib
import numpy as np

from .CAInitializer import CAGenerateWeights
import multiprocessing
from joblib import Subprocessor

from utils import std_utils
from .device import device
from interact import interact as io

class nnlib(object):
    device = device #forwards nnlib.devicelib to device in order to use nnlib as standalone lib
    DeviceConfig = device.Config
    active_DeviceConfig = DeviceConfig() #default is one best GPU

    backend = ""

    dlib = None

    torch = None
    torch_device = None

    keras = None
    keras_contrib = None

    tf = None
    tf_sess = None
    tf_sess_config = None
    
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
KL = keras.layers

Input = KL.Input

Dense = KL.Dense
Conv2D = KL.Conv2D
Conv2DTranspose = KL.Conv2DTranspose
EqualConv2D = nnlib.EqualConv2D
SeparableConv2D = KL.SeparableConv2D
DepthwiseConv2D = KL.DepthwiseConv2D
MaxPooling2D = KL.MaxPooling2D
AveragePooling2D = KL.AveragePooling2D
GlobalAveragePooling2D = KL.GlobalAveragePooling2D
UpSampling2D = KL.UpSampling2D
BatchNormalization = KL.BatchNormalization
PixelNormalization = nnlib.PixelNormalization

Activation = KL.Activation
LeakyReLU = KL.LeakyReLU
ELU = KL.ELU
GeLU = nnlib.GeLU
ReLU = KL.ReLU
PReLU = KL.PReLU
tanh = KL.Activation('tanh')
sigmoid = KL.Activation('sigmoid')
Dropout = KL.Dropout
Softmax = KL.Softmax

Lambda = KL.Lambda
Add = KL.Add
Multiply = KL.Multiply
Concatenate = KL.Concatenate


Flatten = KL.Flatten
Reshape = KL.Reshape

ZeroPadding2D = KL.ZeroPadding2D

RandomNormal = keras.initializers.RandomNormal
Model = keras.models.Model

Adam = nnlib.Adam
RMSprop = nnlib.RMSprop
LookaheadOptimizer = nnlib.LookaheadOptimizer
SGD = nnlib.keras.optimizers.SGD

modelify = nnlib.modelify
gaussian_blur = nnlib.gaussian_blur
style_loss = nnlib.style_loss
dssim = nnlib.dssim

DenseMaxout = nnlib.DenseMaxout
PixelShuffler = nnlib.PixelShuffler
SubpixelUpscaler = nnlib.SubpixelUpscaler
SubpixelDownscaler = nnlib.SubpixelDownscaler
Scale = nnlib.Scale
BilinearInterpolation = nnlib.BilinearInterpolation
BlurPool = nnlib.BlurPool
FUNITAdain = nnlib.FUNITAdain
SelfAttention = nnlib.SelfAttention

CAInitializerMP = nnlib.CAInitializerMP

#ReflectionPadding2D = nnlib.ReflectionPadding2D
#AddUniformNoise = nnlib.AddUniformNoise
"""
    code_import_keras_contrib_string = \
"""
keras_contrib = nnlib.keras_contrib
GroupNormalization = keras_contrib.layers.GroupNormalization
InstanceNormalization = keras_contrib.layers.InstanceNormalization
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
    def import_torch(device_config=None):
        if nnlib.torch is not None:
            return

        if device_config is None:
            device_config = nnlib.active_DeviceConfig
        else:
            nnlib.active_DeviceConfig = device_config

        if 'CUDA_VISIBLE_DEVICES' in os.environ.keys():
            os.environ.pop('CUDA_VISIBLE_DEVICES')

        io.log_info ("Using PyTorch backend.")
        import torch
        nnlib.torch = torch

        if device_config.cpu_only or device_config.backend == 'plaidML':
            nnlib.torch_device = torch.device(type='cpu')
        else:
            nnlib.torch_device = torch.device(type='cuda', index=device_config.gpu_idxs[0] )
            torch.cuda.set_device(nnlib.torch_device)

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

        os.environ['CUDA_​CACHE_​MAXSIZE'] = '536870912' #512Mb (32mb default)

        os.environ['TF_MIN_GPU_MULTIPROCESSOR_COUNT'] = '2'
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #tf log errors only

        import warnings
        warnings.simplefilter(action='ignore', category=FutureWarning)

        import tensorflow as tf
        nnlib.tf = tf

        if device_config.cpu_only:
            config = tf.ConfigProto(device_count={'GPU': 0})
        else:
            config = tf.ConfigProto()

            visible_device_list = ''
            for idx in device_config.gpu_idxs:
                visible_device_list += str(idx) + ','
            config.gpu_options.visible_device_list=visible_device_list[:-1]

        config.gpu_options.force_gpu_compatible = True
        config.gpu_options.allow_growth = device_config.allow_growth
        nnlib.tf_sess_config = config
        
        nnlib.tf_sess = tf.Session(config=config)

        if suppressor is not None:
            suppressor.__exit__()

    @staticmethod
    def import_keras(device_config):
        if nnlib.keras is not None:
            return nnlib.code_import_keras

        nnlib.backend = device_config.backend
        if "tensorflow" in nnlib.backend:
            nnlib._import_tf(device_config)
        elif nnlib.backend == "plaidML":
            os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
            os.environ["PLAIDML_DEVICE_IDS"] = ",".join ( [ nnlib.device.getDeviceID(idx) for idx in device_config.gpu_idxs] )

        #if "tensorflow" in nnlib.backend:
        #    nnlib.keras = nnlib.tf.keras
        #else:
        import keras as keras_
        nnlib.keras = keras_

        if 'KERAS_BACKEND' in os.environ:
            os.environ.pop('KERAS_BACKEND')

        if nnlib.backend == "plaidML":
            import plaidml
            import plaidml.tile
            nnlib.PML = plaidml
            nnlib.PMLK = plaidml.keras.backend
            nnlib.PMLTile = plaidml.tile

        if device_config.use_fp16:
            nnlib.keras.backend.set_floatx('float16')

        if "tensorflow" in nnlib.backend:
            nnlib.keras.backend.set_session(nnlib.tf_sess)

        nnlib.keras.backend.set_image_data_format('channels_last')

        nnlib.code_import_keras = compile (nnlib.code_import_keras_string,'','exec')
        nnlib.__initialize_keras_functions()

        return nnlib.code_import_keras

    @staticmethod
    def __initialize_keras_functions():
        keras = nnlib.keras
        K = keras.backend
        KL = keras.layers
        backend = nnlib.backend

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

        def dssim(kernel_size=11, k1=0.01, k2=0.03, max_value=1.0):
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

                kernel = _fspecial_gauss(kernel_size,1.5)

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
                return(1.0 - ssim_val ) / 2.0

            return func

        nnlib.dssim = dssim

        if 'tensorflow' in backend:
            class PixelShuffler(keras.layers.Layer):
                def __init__(self, size=(2, 2),  data_format='channels_last', **kwargs):
                    super(PixelShuffler, self).__init__(**kwargs)
                    self.data_format = data_format
                    self.size = size

                def call(self, inputs):
                    input_shape = K.shape(inputs)
                    if K.int_shape(input_shape)[0] != 4:
                        raise ValueError('Inputs should have rank 4; Received input shape:', str(K.int_shape(inputs)))

                    if self.data_format == 'channels_first':
                        return K.tf.depth_to_space(inputs, self.size[0], 'NCHW')

                    elif self.data_format == 'channels_last':
                        return K.tf.depth_to_space(inputs, self.size[0], 'NHWC')

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
        else:
            class PixelShuffler(KL.Layer):
                def __init__(self, size=(2, 2), data_format='channels_last', **kwargs):
                    super(PixelShuffler, self).__init__(**kwargs)
                    self.data_format = data_format
                    self.size = size

                def call(self, inputs):

                    input_shape = K.shape(inputs)
                    if K.int_shape(input_shape)[0] != 4:
                        raise ValueError('Inputs should have rank 4; Received input shape:', str(K.int_shape(inputs)))

                    if self.data_format == 'channels_first':
                        batch_size, c, h, w = input_shape[0], K.int_shape(inputs)[1], input_shape[2], input_shape[3]
                        rh, rw = self.size
                        oh, ow = h * rh, w * rw
                        oc = c // (rh * rw)

                        out = K.reshape(inputs, (batch_size, rh, rw, oc, h, w))
                        out = K.permute_dimensions(out, (0, 3, 4, 1, 5, 2))
                        out = K.reshape(out, (batch_size, oc, oh, ow))
                        return out

                    elif self.data_format == 'channels_last':
                        batch_size, h, w, c = input_shape[0], input_shape[1], input_shape[2], K.int_shape(inputs)[-1]
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

        if 'tensorflow' in backend:
            class SubpixelDownscaler(KL.Layer):
                def __init__(self, size=(2, 2), data_format='channels_last', **kwargs):
                    super(SubpixelDownscaler, self).__init__(**kwargs)
                    self.data_format = data_format
                    self.size = size

                def call(self, inputs):

                    input_shape = K.shape(inputs)
                    if K.int_shape(input_shape)[0] != 4:
                        raise ValueError('Inputs should have rank 4; Received input shape:', str(K.int_shape(inputs)))

                    return K.tf.space_to_depth(inputs, self.size[0], 'NHWC')

                def compute_output_shape(self, input_shape):
                    if len(input_shape) != 4:
                        raise ValueError('Inputs should have rank ' +
                                        str(4) +
                                        '; Received input shape:', str(input_shape))

                    height = input_shape[1] // self.size[0] if input_shape[1] is not None else None
                    width = input_shape[2] // self.size[1] if input_shape[2] is not None else None
                    channels = input_shape[3] * self.size[0] * self.size[1]

                    return (input_shape[0], height, width, channels)

                def get_config(self):
                    config = {'size': self.size,
                            'data_format': self.data_format}
                    base_config = super(SubpixelDownscaler, self).get_config()

                    return dict(list(base_config.items()) + list(config.items()))
        else:
            class SubpixelDownscaler(KL.Layer):
                def __init__(self, size=(2, 2), data_format='channels_last', **kwargs):
                    super(SubpixelDownscaler, self).__init__(**kwargs)
                    self.data_format = data_format
                    self.size = size

                def call(self, inputs):

                    input_shape = K.shape(inputs)
                    if K.int_shape(input_shape)[0] != 4:
                        raise ValueError('Inputs should have rank 4; Received input shape:', str(K.int_shape(inputs)))

                    batch_size, h, w, c = input_shape[0], input_shape[1], input_shape[2], K.int_shape(inputs)[-1]
                    rh, rw = self.size
                    oh, ow = h // rh, w // rw
                    oc = c * (rh * rw)

                    out = K.reshape(inputs, (batch_size, oh, rh, ow, rw, c))
                    out = K.permute_dimensions(out, (0, 1, 3, 2, 4, 5))
                    out = K.reshape(out, (batch_size, oh, ow, oc))
                    return out

                def compute_output_shape(self, input_shape):
                    if len(input_shape) != 4:
                        raise ValueError('Inputs should have rank ' +
                                        str(4) +
                                        '; Received input shape:', str(input_shape))

                    height = input_shape[1] // self.size[0] if input_shape[1] is not None else None
                    width = input_shape[2] // self.size[1] if input_shape[2] is not None else None
                    channels = input_shape[3] * self.size[0] * self.size[1]

                    return (input_shape[0], height, width, channels)

                def get_config(self):
                    config = {'size': self.size,
                            'data_format': self.data_format}
                    base_config = super(SubpixelDownscaler, self).get_config()

                    return dict(list(base_config.items()) + list(config.items()))

        nnlib.SubpixelDownscaler = SubpixelDownscaler

        class BlurPool(KL.Layer):
            """
            https://arxiv.org/abs/1904.11486 https://github.com/adobe/antialiased-cnns
            """
            def __init__(self, filt_size=3, stride=2, **kwargs):
                self.strides = (stride,stride)
                self.filt_size = filt_size
                self.padding = ( (int(1.*(filt_size-1)/2), int(np.ceil(1.*(filt_size-1)/2)) ), (int(1.*(filt_size-1)/2), int(np.ceil(1.*(filt_size-1)/2)) ) )
                if(self.filt_size==1):
                    self.a = np.array([1.,])
                elif(self.filt_size==2):
                    self.a = np.array([1., 1.])
                elif(self.filt_size==3):
                    self.a = np.array([1., 2., 1.])
                elif(self.filt_size==4):
                    self.a = np.array([1., 3., 3., 1.])
                elif(self.filt_size==5):
                    self.a = np.array([1., 4., 6., 4., 1.])
                elif(self.filt_size==6):
                    self.a = np.array([1., 5., 10., 10., 5., 1.])
                elif(self.filt_size==7):
                    self.a = np.array([1., 6., 15., 20., 15., 6., 1.])

                super(BlurPool, self).__init__(**kwargs)

            def compute_output_shape(self, input_shape):
                height = input_shape[1] // self.strides[0]
                width = input_shape[2] // self.strides[1]
                channels = input_shape[3]
                return (input_shape[0], height, width, channels)

            def call(self, x):
                k = self.a
                k = k[:,None]*k[None,:]
                k = k / np.sum(k)
                k = np.tile (k[:,:,None,None], (1,1,K.int_shape(x)[-1],1) )
                k = K.constant (k, dtype=K.floatx() )

                x = K.spatial_2d_padding(x, padding=self.padding)
                x = K.depthwise_conv2d(x, k, strides=self.strides, padding='valid')
                return x

        nnlib.BlurPool = BlurPool

        class FUNITAdain(KL.Layer):
            """
            differents from NVLabs/FUNIT:
            I moved two dense blocks inside this layer,
                so we don't need to slice outter MLP block and assign weights every call, just pass MLP inside.
                also size of dense blocks is calculated automatically
            """
            def __init__(self, axis=-1, epsilon=1e-5, momentum=0.99, kernel_initializer='glorot_uniform', **kwargs):
                self.axis = axis
                self.epsilon = epsilon
                self.momentum = momentum
                self.kernel_initializer = kernel_initializer
                super(FUNITAdain, self).__init__(**kwargs)

            def build(self, input_shape):
                self.input_spec = None
                x, mlp = input_shape
                units = x[self.axis]

                self.kernel1 = self.add_weight(shape=(units, units), initializer=self.kernel_initializer, name='kernel1')
                self.bias1 = self.add_weight(shape=(units,), initializer='zeros', name='bias1')
                self.kernel2 = self.add_weight(shape=(units, units), initializer=self.kernel_initializer, name='kernel2')
                self.bias2 = self.add_weight(shape=(units,), initializer='zeros', name='bias2')

                self.built = True

            def call(self, inputs, training=None):
                x, mlp = inputs

                gamma = K.dot(mlp, self.kernel1)
                gamma = K.bias_add(gamma, self.bias1, data_format='channels_last')

                beta = K.dot(mlp, self.kernel2)
                beta = K.bias_add(beta, self.bias2, data_format='channels_last')

                input_shape = K.int_shape(x)

                reduction_axes = list(range(len(input_shape)))
                del reduction_axes[self.axis]
                del reduction_axes[0]

                broadcast_shape = [1] * len(input_shape)
                broadcast_shape[self.axis] = input_shape[self.axis]
                mean = K.mean(x, reduction_axes, keepdims=True)
                stddev = K.std(x, reduction_axes, keepdims=True) + self.epsilon
                normed = (x - mean) / stddev
                normed *= K.reshape(gamma,[-1]+broadcast_shape[1:] )
                normed += K.reshape(beta, [-1]+broadcast_shape[1:] )
                return normed

            def get_config(self):
                config = {'axis': self.axis, 'epsilon': self.epsilon }

                base_config = super(FUNITAdain, self).get_config()
                return dict(list(base_config.items()) + list(config.items()))

            def compute_output_shape(self, input_shape):
                return input_shape
        nnlib.FUNITAdain = FUNITAdain

        class Scale(KL.Layer):
            """
            GAN Custom Scal Layer
            Code borrows from https://github.com/flyyufelix/cnn_finetune
            """
            def __init__(self, weights=None, axis=-1, gamma_init='zero', **kwargs):
                self.axis = axis
                self.gamma_init = keras.initializers.get(gamma_init)
                self.initial_weights = weights
                super(Scale, self).__init__(**kwargs)

            def build(self, input_shape):
                self.input_spec = [keras.engine.InputSpec(shape=input_shape)]

                # Compatibility with TensorFlow >= 1.0.0
                self.gamma = K.variable(self.gamma_init((1,)), name='{}_gamma'.format(self.name))
                self.trainable_weights = [self.gamma]

                if self.initial_weights is not None:
                    self.set_weights(self.initial_weights)
                    del self.initial_weights

            def call(self, x, mask=None):
                return self.gamma * x

            def get_config(self):
                config = {"axis": self.axis}
                base_config = super(Scale, self).get_config()
                return dict(list(base_config.items()) + list(config.items()))
        nnlib.Scale = Scale
        
 
        """
        unable to work in plaidML, due to unimplemented ops
        
        class BilinearInterpolation(KL.Layer):
            def __init__(self, size=(2,2), **kwargs):
                self.size = size
                super(BilinearInterpolation, self).__init__(**kwargs)

            def compute_output_shape(self, input_shape):
                return (input_shape[0], input_shape[1]*self.size[1], input_shape[2]*self.size[0], input_shape[3])


            def call(self, X):
                _,h,w,_ = K.int_shape(X)

                X = K.concatenate( [ X, X[:,:,-2:-1,:] ],axis=2 )
                X = K.concatenate( [ X, X[:,:,-2:-1,:] ],axis=2 )
                X = K.concatenate( [ X, X[:,-2:-1,:,:] ],axis=1 )
                X = K.concatenate( [ X, X[:,-2:-1,:,:] ],axis=1 )

                X_sh = K.shape(X)
                batch_size, height, width, num_channels = X_sh[0], X_sh[1], X_sh[2], X_sh[3]

                output_h, output_w = (h*self.size[1]+4, w*self.size[0]+4)
                
                x_linspace = np.linspace(-1. , 1. - 2/output_w, output_w)#
                y_linspace = np.linspace(-1. , 1. - 2/output_h, output_h)#
            
                x_coordinates, y_coordinates = np.meshgrid(x_linspace, y_linspace)
                x_coordinates = K.flatten(K.constant(x_coordinates, dtype=K.floatx() ))
                y_coordinates = K.flatten(K.constant(y_coordinates, dtype=K.floatx() ))

                grid = K.concatenate([x_coordinates, y_coordinates, K.ones_like(x_coordinates)], 0)
                grid = K.flatten(grid)


                grids = K.tile(grid, ( batch_size, ) )
                grids = K.reshape(grids, (batch_size, 3, output_h * output_w ))


                x = K.cast(K.flatten(grids[:, 0:1, :]), dtype='float32')
                y = K.cast(K.flatten(grids[:, 1:2, :]), dtype='float32')
                x = .5 * (x + 1.0) * K.cast(width, dtype='float32')
                y = .5 * (y + 1.0) * K.cast(height, dtype='float32')
                x0 = K.cast(x, 'int32')
                x1 = x0 + 1
                y0 = K.cast(y, 'int32')
                y1 = y0 + 1
                max_x = int(K.int_shape(X)[2] -1)
                max_y = int(K.int_shape(X)[1] -1)

                x0 = K.clip(x0, 0, max_x)
                x1 = K.clip(x1, 0, max_x)
                y0 = K.clip(y0, 0, max_y)
                y1 = K.clip(y1, 0, max_y)


                pixels_batch = K.constant ( np.arange(0, batch_size) * (height * width), dtype=K.floatx() ) 
                
                pixels_batch = K.expand_dims(pixels_batch, axis=-1)

                base = K.tile(pixels_batch, (1, output_h * output_w ) )
                base = K.flatten(base)

                # base_y0 = base + (y0 * width)
                base_y0 = y0 * width
                base_y0 = base + base_y0
                # base_y1 = base + (y1 * width)
                base_y1 = y1 * width
                base_y1 = base_y1 + base

                indices_a = base_y0 + x0
                indices_b = base_y1 + x0
                indices_c = base_y0 + x1
                indices_d = base_y1 + x1

                flat_image = K.reshape(X, (-1, num_channels) )
                flat_image = K.cast(flat_image, dtype='float32')
                pixel_values_a = K.gather(flat_image, indices_a)
                pixel_values_b = K.gather(flat_image, indices_b)
                pixel_values_c = K.gather(flat_image, indices_c)
                pixel_values_d = K.gather(flat_image, indices_d)

                x0 = K.cast(x0, 'float32')
                x1 = K.cast(x1, 'float32')
                y0 = K.cast(y0, 'float32')
                y1 = K.cast(y1, 'float32')

                area_a = K.expand_dims(((x1 - x) * (y1 - y)), 1)
                area_b = K.expand_dims(((x1 - x) * (y - y0)), 1)
                area_c = K.expand_dims(((x - x0) * (y1 - y)), 1)
                area_d = K.expand_dims(((x - x0) * (y - y0)), 1)

                values_a = area_a * pixel_values_a
                values_b = area_b * pixel_values_b
                values_c = area_c * pixel_values_c
                values_d = area_d * pixel_values_d
                interpolated_image = values_a + values_b + values_c + values_d
        
                new_shape = (batch_size, output_h, output_w, num_channels)
                interpolated_image = K.reshape(interpolated_image, new_shape)

                interpolated_image = interpolated_image[:,:-4,:-4,:]
                return interpolated_image

            def get_config(self):
                config = {"size": self.size}
                base_config = super(BilinearInterpolation, self).get_config()
                return dict(list(base_config.items()) + list(config.items()))
        """      
        class BilinearInterpolation(KL.Layer):
            def __init__(self, size=(2,2), **kwargs):
                self.size = size
                super(BilinearInterpolation, self).__init__(**kwargs)

            def compute_output_shape(self, input_shape):
                return (input_shape[0], input_shape[1]*self.size[1], input_shape[2]*self.size[0], input_shape[3])
                
            def call(self, X):
                _,h,w,_ = K.int_shape(X)

                return K.cast( K.tf.image.resize_images(X, (h*self.size[1],w*self.size[0]) ), K.floatx() )

            def get_config(self):
                config = {"size": self.size}
                base_config = super(BilinearInterpolation, self).get_config()
                return dict(list(base_config.items()) + list(config.items()))
     
        nnlib.BilinearInterpolation = BilinearInterpolation

        
        
        

        class SelfAttention(KL.Layer):
            def __init__(self, nc, squeeze_factor=8, **kwargs):
                assert nc//squeeze_factor > 0, f"Input channels must be >= {squeeze_factor}, recieved nc={nc}"

                self.nc = nc
                self.squeeze_factor = squeeze_factor
                super(SelfAttention, self).__init__(**kwargs)

            def compute_output_shape(self, input_shape):
                return (input_shape[0], input_shape[1], input_shape[2], self.nc)

            def call(self, inp):
                x = inp
                shape_x = x.get_shape().as_list()

                f = Conv2D(self.nc//self.squeeze_factor, 1, kernel_regularizer=keras.regularizers.l2(1e-4))(x)
                g = Conv2D(self.nc//self.squeeze_factor, 1, kernel_regularizer=keras.regularizers.l2(1e-4))(x)
                h = Conv2D(self.nc, 1, kernel_regularizer=keras.regularizers.l2(1e-4))(x)

                shape_f = f.get_shape().as_list()
                shape_g = g.get_shape().as_list()
                shape_h = h.get_shape().as_list()
                flat_f = Reshape( (-1, shape_f[-1]) )(f)
                flat_g = Reshape( (-1, shape_g[-1]) )(g)
                flat_h = Reshape( (-1, shape_h[-1]) )(h)

                s = Lambda(lambda x: K.batch_dot(x[0], keras.layers.Permute((2,1))(x[1]) ))([flat_g, flat_f])
                beta = keras.layers.Softmax(axis=-1)(s)
                o = Lambda(lambda x: K.batch_dot(x[0], x[1]))([beta, flat_h])

                o = Reshape(shape_x[1:])(o)
                o = Scale()(o)

                out = Add()([o, inp])
                return out
        nnlib.SelfAttention = SelfAttention

        class RMSprop(keras.optimizers.Optimizer):
            """RMSProp optimizer.
            It is recommended to leave the parameters of this optimizer
            at their default values
            (except the learning rate, which can be freely tuned).
            # Arguments
                learning_rate: float >= 0. Learning rate.
                rho: float >= 0.
            # References
                - [rmsprop: Divide the gradient by a running average of its recent magnitude
                ](http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf)

                tf_cpu_mode: only for tensorflow backend
                              0 - default, no changes.
                              1 - allows to train x2 bigger network on same VRAM consuming RAM
                              2 - allows to train x3 bigger network on same VRAM consuming RAM*2 and CPU power.
            """

            def __init__(self, learning_rate=0.001, rho=0.9, lr_dropout=0, tf_cpu_mode=0, **kwargs):
                self.initial_decay = kwargs.pop('decay', 0.0)
                self.epsilon = kwargs.pop('epsilon', K.epsilon())
                self.lr_dropout = lr_dropout
                self.tf_cpu_mode = tf_cpu_mode

                learning_rate = kwargs.pop('lr', learning_rate)
                super(RMSprop, self).__init__(**kwargs)
                with K.name_scope(self.__class__.__name__):
                    self.learning_rate = K.variable(learning_rate, name='learning_rate')
                    self.rho = K.variable(rho, name='rho')
                    self.decay = K.variable(self.initial_decay, name='decay')
                    self.iterations = K.variable(0, dtype='int64', name='iterations')

            def get_updates(self, loss, params):
                grads = self.get_gradients(loss, params)


                e = K.tf.device("/cpu:0") if self.tf_cpu_mode > 0 else None
                if e: e.__enter__()
                accumulators = [K.zeros(K.int_shape(p),
                                dtype=K.dtype(p),
                                name='accumulator_' + str(i))
                                for (i, p) in enumerate(params)]
                if self.lr_dropout != 0:
                    lr_rnds = [ K.random_binomial(K.int_shape(p), p=self.lr_dropout, dtype=K.dtype(p)) for p in params ]
                if e: e.__exit__(None, None, None)

                self.weights = [self.iterations] + accumulators
                self.updates = [K.update_add(self.iterations, 1)]

                lr = self.learning_rate
                if self.initial_decay > 0:
                    lr = lr * (1. / (1. + self.decay * K.cast(self.iterations,
                                                            K.dtype(self.decay))))

                for i, (p, g, a) in enumerate(zip(params, grads, accumulators)):
                    # update accumulator
                    e = K.tf.device("/cpu:0") if self.tf_cpu_mode == 2 else None
                    if e: e.__enter__()
                    new_a = self.rho * a + (1. - self.rho) * K.square(g)
                    p_diff = - lr * g / (K.sqrt(new_a) + self.epsilon)
                    if self.lr_dropout != 0:
                        p_diff *= lr_rnds[i]
                    new_p = p + p_diff
                    if e: e.__exit__(None, None, None)

                    self.updates.append(K.update(a, new_a))

                    # Apply constraints.
                    if getattr(p, 'constraint', None) is not None:
                        new_p = p.constraint(new_p)

                    self.updates.append(K.update(p, new_p))
                return self.updates

            def set_weights(self, weights):
                params = self.weights
                # Override set_weights for backward compatibility of Keras 2.2.4 optimizer
                # since it does not include iteration at head of the weight list. Set
                # iteration to 0.
                if len(params) == len(weights) + 1:
                    weights = [np.array(0)] + weights
                super(RMSprop, self).set_weights(weights)

            def get_config(self):
                config = {'learning_rate': float(K.get_value(self.learning_rate)),
                        'rho': float(K.get_value(self.rho)),
                        'decay': float(K.get_value(self.decay)),
                        'epsilon': self.epsilon,
                        'lr_dropout' : self.lr_dropout }
                base_config = super(RMSprop, self).get_config()
                return dict(list(base_config.items()) + list(config.items()))
        nnlib.RMSprop = RMSprop

        class Adam(keras.optimizers.Optimizer):
            """Adam optimizer.

            Default parameters follow those provided in the original paper.

            # Arguments
                lr: float >= 0. Learning rate.
                beta_1: float, 0 < beta < 1. Generally close to 1.
                beta_2: float, 0 < beta < 1. Generally close to 1.
                epsilon: float >= 0. Fuzz factor. If `None`, defaults to `K.epsilon()`.
                decay: float >= 0. Learning rate decay over each update.
                amsgrad: boolean. Whether to apply the AMSGrad variant of this
                    algorithm from the paper "On the Convergence of Adam and
                    Beyond".
                lr_dropout: float [0.0 .. 1.0] Learning rate dropout https://arxiv.org/pdf/1912.00144
                tf_cpu_mode: only for tensorflow backend
                              0 - default, no changes.
                              1 - allows to train x2 bigger network on same VRAM consuming RAM
                              2 - allows to train x3 bigger network on same VRAM consuming RAM*2 and CPU power.

            # References
                - [Adam - A Method for Stochastic Optimization]
                  (https://arxiv.org/abs/1412.6980v8)
                - [On the Convergence of Adam and Beyond]
                  (https://openreview.net/forum?id=ryQu7f-RZ)
            """

            def __init__(self, lr=0.001, beta_1=0.9, beta_2=0.999,
                         epsilon=None, decay=0., amsgrad=False, lr_dropout=0, tf_cpu_mode=0, **kwargs):
                super(Adam, self).__init__(**kwargs)
                with K.name_scope(self.__class__.__name__):
                    self.iterations = K.variable(0, dtype='int64', name='iterations')
                    self.lr = K.variable(lr, name='lr')
                    self.beta_1 = K.variable(beta_1, name='beta_1')
                    self.beta_2 = K.variable(beta_2, name='beta_2')
                    self.decay = K.variable(decay, name='decay')
                if epsilon is None:
                    epsilon = K.epsilon()
                self.epsilon = epsilon
                self.initial_decay = decay
                self.amsgrad = amsgrad
                self.lr_dropout = lr_dropout
                self.tf_cpu_mode = tf_cpu_mode

            def get_updates(self, loss, params):
                grads = self.get_gradients(loss, params)
                self.updates = [K.update_add(self.iterations, 1)]

                lr = self.lr
                if self.initial_decay > 0:
                    lr = lr * (1. / (1. + self.decay * K.cast(self.iterations,
                                                              K.dtype(self.decay))))

                t = K.cast(self.iterations, K.floatx()) + 1
                lr_t = lr * (K.sqrt(1. - K.pow(self.beta_2, t)) /
                                   (1. - K.pow(self.beta_1, t)))

                e = K.tf.device("/cpu:0") if self.tf_cpu_mode > 0 else None
                if e: e.__enter__()
                ms = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
                vs = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
                if self.amsgrad:
                    vhats = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
                else:
                    vhats = [K.zeros(1) for _ in params]


                if self.lr_dropout != 0:
                    lr_rnds = [ K.random_binomial(K.int_shape(p), p=self.lr_dropout, dtype=K.dtype(p)) for p in params ]

                if e: e.__exit__(None, None, None)

                self.weights = [self.iterations] + ms + vs + vhats

                for i, (p, g, m, v, vhat) in enumerate( zip(params, grads, ms, vs, vhats) ):
                    e = K.tf.device("/cpu:0") if self.tf_cpu_mode == 2 else None
                    if e: e.__enter__()
                    m_t = (self.beta_1 * m) + (1. - self.beta_1) * g
                    v_t = (self.beta_2 * v) + (1. - self.beta_2) * K.square(g)

                    if self.amsgrad:
                        vhat_t = K.maximum(vhat, v_t)
                        self.updates.append(K.update(vhat, vhat_t))
                    if e: e.__exit__(None, None, None)

                    if self.amsgrad:
                        p_diff = - lr_t * m_t / (K.sqrt(vhat_t) + self.epsilon)
                    else:
                        p_diff = - lr_t * m_t / (K.sqrt(v_t) + self.epsilon)

                    if self.lr_dropout != 0:
                        p_diff *= lr_rnds[i]

                    self.updates.append(K.update(m, m_t))
                    self.updates.append(K.update(v, v_t))
                    new_p = p + p_diff

                    # Apply constraints.
                    if getattr(p, 'constraint', None) is not None:
                        new_p = p.constraint(new_p)

                    self.updates.append(K.update(p, new_p))
                return self.updates

            def get_config(self):
                config = {'lr': float(K.get_value(self.lr)),
                          'beta_1': float(K.get_value(self.beta_1)),
                          'beta_2': float(K.get_value(self.beta_2)),
                          'decay': float(K.get_value(self.decay)),
                          'epsilon': self.epsilon,
                          'amsgrad': self.amsgrad,
                          'lr_dropout' : self.lr_dropout}
                base_config = super(Adam, self).get_config()
                return dict(list(base_config.items()) + list(config.items()))
        nnlib.Adam = Adam

        class LookaheadOptimizer(keras.optimizers.Optimizer):
            def __init__(self, optimizer, sync_period=5, slow_step=0.5, tf_cpu_mode=0, **kwargs):
                super(LookaheadOptimizer, self).__init__(**kwargs)
                self.optimizer = optimizer
                self.tf_cpu_mode = tf_cpu_mode

                with K.name_scope(self.__class__.__name__):
                    self.sync_period = K.variable(sync_period, dtype='int64', name='sync_period')
                    self.slow_step = K.variable(slow_step, name='slow_step')

            @property
            def lr(self):
                return self.optimizer.lr

            @lr.setter
            def lr(self, lr):
                self.optimizer.lr = lr

            @property
            def learning_rate(self):
                return self.optimizer.learning_rate

            @learning_rate.setter
            def learning_rate(self, learning_rate):
                self.optimizer.learning_rate = learning_rate

            @property
            def iterations(self):
                return self.optimizer.iterations

            def get_updates(self, loss, params):
                sync_cond = K.equal((self.iterations + 1) // self.sync_period * self.sync_period, (self.iterations + 1))

                e = K.tf.device("/cpu:0") if self.tf_cpu_mode > 0 else None
                if e: e.__enter__()
                slow_params = [K.variable(K.get_value(p), name='sp_{}'.format(i)) for i, p in enumerate(params)]
                if e: e.__exit__(None, None, None)


                self.updates = self.optimizer.get_updates(loss, params)
                slow_updates = []
                for p, sp in zip(params, slow_params):

                    e = K.tf.device("/cpu:0") if self.tf_cpu_mode == 2 else None
                    if e: e.__enter__()
                    sp_t = sp + self.slow_step * (p - sp)
                    if e: e.__exit__(None, None, None)

                    slow_updates.append(K.update(sp, K.switch(
                        sync_cond,
                        sp_t,
                        sp,
                    )))
                    slow_updates.append(K.update_add(p, K.switch(
                        sync_cond,
                        sp_t - p,
                        K.zeros_like(p),
                    )))

                self.updates += slow_updates
                self.weights = self.optimizer.weights + slow_params
                return self.updates

            def get_config(self):
                config = {
                    'optimizer': keras.optimizers.serialize(self.optimizer),
                    'sync_period': int(K.get_value(self.sync_period)),
                    'slow_step': float(K.get_value(self.slow_step)),
                }
                base_config = super(LookaheadOptimizer, self).get_config()
                return dict(list(base_config.items()) + list(config.items()))

            @classmethod
            def from_config(cls, config):
                optimizer = keras.optimizers.deserialize(config.pop('optimizer'))
                return cls(optimizer, **config)
        nnlib.LookaheadOptimizer = LookaheadOptimizer

        class DenseMaxout(keras.layers.Layer):
            """A dense maxout layer.
            A `MaxoutDense` layer takes the element-wise maximum of
            `nb_feature` `Dense(input_dim, output_dim)` linear layers.
            This allows the layer to learn a convex,
            piecewise linear activation function over the inputs.
            Note that this is a *linear* layer;
            if you wish to apply activation function
            (you shouldn't need to --they are universal function approximators),
            an `Activation` layer must be added after.
            # Arguments
                output_dim: int > 0.
                nb_feature: number of Dense layers to use internally.
                init: name of initialization function for the weights of the layer
                    (see [initializations](../initializations.md)),
                    or alternatively, Theano function to use for weights
                    initialization. This parameter is only relevant
                    if you don't pass a `weights` argument.
                weights: list of Numpy arrays to set as initial weights.
                    The list should have 2 elements, of shape `(input_dim, output_dim)`
                    and (output_dim,) for weights and biases respectively.
                W_regularizer: instance of [WeightRegularizer](../regularizers.md)
                    (eg. L1 or L2 regularization), applied to the main weights matrix.
                b_regularizer: instance of [WeightRegularizer](../regularizers.md),
                    applied to the bias.
                activity_regularizer: instance of [ActivityRegularizer](../regularizers.md),
                    applied to the network output.
                W_constraint: instance of the [constraints](../constraints.md) module
                    (eg. maxnorm, nonneg), applied to the main weights matrix.
                b_constraint: instance of the [constraints](../constraints.md) module,
                    applied to the bias.
                bias: whether to include a bias
                    (i.e. make the layer affine rather than linear).
                input_dim: dimensionality of the input (integer). This argument
                    (or alternatively, the keyword argument `input_shape`)
                    is required when using this layer as the first layer in a model.
            # Input shape
                2D tensor with shape: `(nb_samples, input_dim)`.
            # Output shape
                2D tensor with shape: `(nb_samples, output_dim)`.
            # References
                - [Maxout Networks](http://arxiv.org/abs/1302.4389)
            """

            def __init__(self, output_dim,
                        nb_feature=4,
                        kernel_initializer='glorot_uniform',
                        weights=None,
                        W_regularizer=None,
                        b_regularizer=None,
                        activity_regularizer=None,
                        W_constraint=None,
                        b_constraint=None,
                        bias=True,
                        input_dim=None,
                        **kwargs):
                self.output_dim = output_dim
                self.nb_feature = nb_feature
                self.kernel_initializer = keras.initializers.get(kernel_initializer)

                self.W_regularizer = keras.regularizers.get(W_regularizer)
                self.b_regularizer = keras.regularizers.get(b_regularizer)
                self.activity_regularizer = keras.regularizers.get(activity_regularizer)

                self.W_constraint = keras.constraints.get(W_constraint)
                self.b_constraint = keras.constraints.get(b_constraint)

                self.bias = bias
                self.initial_weights = weights
                self.input_spec = keras.layers.InputSpec(ndim=2)

                self.input_dim = input_dim
                if self.input_dim:
                    kwargs['input_shape'] = (self.input_dim,)
                super(DenseMaxout, self).__init__(**kwargs)

            def build(self, input_shape):
                input_dim = input_shape[1]
                self.input_spec = keras.layers.InputSpec(dtype=K.floatx(),
                                            shape=(None, input_dim))

                self.W = self.add_weight(shape=(self.nb_feature, input_dim, self.output_dim),
                                        initializer=self.kernel_initializer,
                                        name='W',
                                        regularizer=self.W_regularizer,
                                        constraint=self.W_constraint)
                if self.bias:
                    self.b = self.add_weight(shape=(self.nb_feature, self.output_dim,),
                                            initializer='zero',
                                            name='b',
                                            regularizer=self.b_regularizer,
                                            constraint=self.b_constraint)
                else:
                    self.b = None

                if self.initial_weights is not None:
                    self.set_weights(self.initial_weights)
                    del self.initial_weights
                self.built = True

            def compute_output_shape(self, input_shape):
                assert input_shape and len(input_shape) == 2
                return (input_shape[0], self.output_dim)

            def call(self, x):
                # no activation, this layer is only linear.
                output = K.dot(x, self.W)
                if self.bias:
                    output += self.b
                output = K.max(output, axis=1)
                return output

            def get_config(self):
                config = {'output_dim': self.output_dim,
                        'kernel_initializer': initializers.serialize(self.kernel_initializer),
                        'nb_feature': self.nb_feature,
                        'W_regularizer': regularizers.serialize(self.W_regularizer),
                        'b_regularizer': regularizers.serialize(self.b_regularizer),
                        'activity_regularizer': regularizers.serialize(self.activity_regularizer),
                        'W_constraint': constraints.serialize(self.W_constraint),
                        'b_constraint': constraints.serialize(self.b_constraint),
                        'bias': self.bias,
                        'input_dim': self.input_dim}
                base_config = super(DenseMaxout, self).get_config()
                return dict(list(base_config.items()) + list(config.items()))
        nnlib.DenseMaxout = DenseMaxout
        
        class GeLU(KL.Layer):
            """Gaussian Error Linear Unit.
            A smoother version of ReLU generally used
            in the BERT or BERT architecture based models.
            Original paper: https://arxiv.org/abs/1606.08415
            Input shape:
                Arbitrary. Use the keyword argument `input_shape`
                (tuple of integers, does not include the samples axis)
                when using this layer as the first layer in a model.
            Output shape:
                Same shape as the input.
            """

            def __init__(self, approximate=True, **kwargs):
                super(GeLU, self).__init__(**kwargs)
                self.approximate = approximate
                self.supports_masking = True

            def call(self, inputs):
                cdf = 0.5 * (1.0 + K.tanh((np.sqrt(2 / np.pi) * (inputs + 0.044715 * K.pow(inputs, 3)))))
                return inputs * cdf

            def get_config(self):
                config = {'approximate': self.approximate}
                base_config = super(GeLU, self).get_config()
                return dict(list(base_config.items()) + list(config.items()))

            def compute_output_shape(self, input_shape):
                return input_shape
        nnlib.GeLU = GeLU

        def CAInitializerMP( conv_weights_list ):
            #Convolution Aware Initialization https://arxiv.org/abs/1702.06295
            data = [ (i, K.int_shape(conv_weights)) for i, conv_weights in enumerate(conv_weights_list) ]
            data = sorted(data, key=lambda data: np.prod(data[1]) )
            result = CAInitializerMPSubprocessor (data, K.floatx(), K.image_data_format() ).run()
            for idx, weights in result:
                K.set_value ( conv_weights_list[idx], weights )
        nnlib.CAInitializerMP = CAInitializerMP


        if backend == "plaidML":
            class TileOP_ReflectionPadding2D(nnlib.PMLTile.Operation):
                def __init__(self, input, w_pad, h_pad):
                    if K.image_data_format() == 'channels_last':
                        if input.shape.ndims == 4:
                            H, W = input.shape.dims[1:3]
                            if (type(H) == int and h_pad >= H) or \
                                (type(W) == int and w_pad >= W):
                                raise ValueError("Paddings must be less than dimensions.")

                            c = """ function (I[B, H, W, C] ) -> (O) {{
                                    WE = W + {w_pad}*2;
                                    HE = H + {h_pad}*2;
                                """.format(h_pad=h_pad, w_pad=w_pad)
                            if w_pad > 0:
                                c += """
                                    LEFT_PAD [b, h, w , c : B, H, WE, C ] = =(I[b, h, {w_pad}-w,            c]), w < {w_pad} ;
                                    HCENTER  [b, h, w , c : B, H, WE, C ] = =(I[b, h, w-{w_pad},            c]), w < W+{w_pad}-1 ;
                                    RIGHT_PAD[b, h, w , c : B, H, WE, C ] = =(I[b, h, 2*W - (w-{w_pad}) -2, c]);
                                    LCR = LEFT_PAD+HCENTER+RIGHT_PAD;
                                """.format(h_pad=h_pad, w_pad=w_pad)
                            else:
                                c += "LCR = I;"

                            if h_pad > 0:
                                c += """
                                    TOP_PAD   [b, h, w , c : B, HE, WE, C ] = =(LCR[b, {h_pad}-h,            w, c]), h < {h_pad};
                                    VCENTER   [b, h, w , c : B, HE, WE, C ] = =(LCR[b, h-{h_pad},            w, c]), h < H+{h_pad}-1 ;
                                    BOTTOM_PAD[b, h, w , c : B, HE, WE, C ] = =(LCR[b, 2*H - (h-{h_pad}) -2, w, c]);
                                    TVB = TOP_PAD+VCENTER+BOTTOM_PAD;
                                """.format(h_pad=h_pad, w_pad=w_pad)
                            else:
                                c += "TVB = LCR;"

                            c += "O = TVB; }"

                            inp_dims = input.shape.dims
                            out_dims = (inp_dims[0], inp_dims[1]+h_pad*2, inp_dims[2]+w_pad*2, inp_dims[3])
                        else:
                            raise NotImplemented
                    else:
                        raise NotImplemented

                    super(TileOP_ReflectionPadding2D, self).__init__(c, [('I', input) ],
                            [('O', nnlib.PMLTile.Shape(input.shape.dtype, out_dims ) )])

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
                if "tensorflow" in backend:
                    return K.tf.pad(x, [[0,0], [h_pad,h_pad], [w_pad,w_pad], [0,0] ], 'REFLECT')
                elif backend == "plaidML":
                    return TileOP_ReflectionPadding2D.function(x, self.padding[0], self.padding[1])
                else:
                    if K.image_data_format() == 'channels_last':
                        if x.shape.ndims == 4:
                            w = K.concatenate ([ x[:,:,w_pad:0:-1,:],
                                                x,
                                                x[:,:,-2:-w_pad-2:-1,:] ], axis=2 )
                            h = K.concatenate ([ w[:,h_pad:0:-1,:,:],
                                                w,
                                                w[:,-2:-h_pad-2:-1,:,:] ], axis=1 )
                            return h
                        else:
                            raise NotImplemented
                    else:
                        raise NotImplemented

        nnlib.ReflectionPadding2D = ReflectionPadding2D

        class Conv2D():
            def __init__ (self, *args, **kwargs):
                self.reflect_pad = False
                padding = kwargs.get('padding','')
                if padding == 'zero':
                    kwargs['padding'] = 'same'
                if padding == 'reflect':
                    kernel_size = kwargs['kernel_size']
                    if (kernel_size % 2) == 1:
                        self.pad = (kernel_size // 2,)*2
                        kwargs['padding'] = 'valid'
                        self.reflect_pad = True
                self.func = keras.layers.Conv2D (*args, **kwargs)

            def __call__(self,x):
                if self.reflect_pad:
                    x = ReflectionPadding2D( self.pad ) (x)
                return self.func(x)
        nnlib.Conv2D = Conv2D

        class Conv2DTranspose():
            def __init__ (self, *args, **kwargs):
                self.reflect_pad = False
                padding = kwargs.get('padding','')
                if padding == 'zero':
                    kwargs['padding'] = 'same'
                if padding == 'reflect':
                    kernel_size = kwargs['kernel_size']
                    if (kernel_size % 2) == 1:
                        self.pad = (kernel_size // 2,)*2
                        kwargs['padding'] = 'valid'
                        self.reflect_pad = True
                self.func = keras.layers.Conv2DTranspose (*args, **kwargs)

            def __call__(self,x):
                if self.reflect_pad:
                    x = ReflectionPadding2D( self.pad ) (x)
                return self.func(x)
        nnlib.Conv2DTranspose = Conv2DTranspose

        class EqualConv2D(KL.Conv2D):
            def __init__(self, filters,
                        kernel_size,
                        strides=(1, 1),
                        padding='valid',
                        data_format=None,
                        dilation_rate=(1, 1),
                        activation=None,
                        use_bias=True,
                        gain=np.sqrt(2),
                        **kwargs):
                super().__init__(
                    filters=filters,
                    kernel_size=kernel_size,
                    strides=strides,
                    padding=padding,
                    data_format=data_format,
                    dilation_rate=dilation_rate,
                    activation=activation,
                    use_bias=use_bias,
                    kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=1.0),
                    bias_initializer='zeros',
                    kernel_regularizer=None,
                    bias_regularizer=None,
                    activity_regularizer=None,
                    kernel_constraint=None,
                    bias_constraint=None,
                    **kwargs)
                self.gain = gain

            def build(self, input_shape):
                super().build(input_shape)

                self.wscale = self.gain / np.sqrt( np.prod( K.int_shape(self.kernel)[:-1]) )
                self.wscale_t = K.constant (self.wscale, dtype=K.floatx() )

            def call(self, inputs):
                k = self.kernel * self.wscale_t

                outputs = K.conv2d(
                        inputs,
                        k,
                        strides=self.strides,
                        padding=self.padding,
                        data_format=self.data_format,
                        dilation_rate=self.dilation_rate)

                if self.use_bias:
                    outputs = K.bias_add(
                        outputs,
                        self.bias,
                        data_format=self.data_format)

                if self.activation is not None:
                    return self.activation(outputs)
                return outputs
        nnlib.EqualConv2D = EqualConv2D

        class PixelNormalization(KL.Layer):
            # initialize the layer
            def __init__(self, **kwargs):
                super(PixelNormalization, self).__init__(**kwargs)

            # perform the operation
            def call(self, inputs):
                # calculate square pixel values
                values = inputs**2.0
                # calculate the mean pixel values
                mean_values = K.mean(values, axis=-1, keepdims=True)
                # ensure the mean is not zero
                mean_values += 1.0e-8
                # calculate the sqrt of the mean squared value (L2 norm)
                l2 = K.sqrt(mean_values)
                # normalize values by the l2 norm
                normalized = inputs / l2
                return normalized

            # define the output shape of the layer
            def compute_output_shape(self, input_shape):
                return input_shape
        nnlib.PixelNormalization = PixelNormalization

    @staticmethod
    def import_keras_contrib(device_config):
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
            if device_config is None:
                device_config = nnlib.active_DeviceConfig
            else:
                nnlib.active_DeviceConfig = device_config

            nnlib.import_keras(device_config)
            nnlib.import_keras_contrib(device_config)
            nnlib.code_import_all = compile (nnlib.code_import_keras_string + '\n'
                                            + nnlib.code_import_keras_contrib_string
                                            + nnlib.code_import_all_string,'','exec')
            nnlib.__initialize_all_functions()

        return nnlib.code_import_all

    @staticmethod
    def __initialize_all_functions():
        exec (nnlib.import_keras(nnlib.active_DeviceConfig), locals(), globals())
        exec (nnlib.import_keras_contrib(nnlib.active_DeviceConfig), locals(), globals())

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


class CAInitializerMPSubprocessor(Subprocessor):
    class Cli(Subprocessor.Cli):

        #override
        def on_initialize(self, client_dict):
            self.floatx = client_dict['floatx']
            self.data_format = client_dict['data_format']

        #override
        def process_data(self, data):
            idx, shape = data
            weights = CAGenerateWeights (shape, self.floatx, self.data_format)
            return idx, weights

        #override
        def get_data_name (self, data):
            #return string identificator of your data
            return "undefined"

    #override
    def __init__(self, idx_shapes_list, floatx, data_format ):
        self.idx_shapes_list = idx_shapes_list
        self.floatx = floatx
        self.data_format = data_format

        self.result = []
        super().__init__('CAInitializerMP', CAInitializerMPSubprocessor.Cli)

    #override
    def on_clients_initialized(self):
        io.progress_bar ("Initializing CA weights", len (self.idx_shapes_list))

    #override
    def on_clients_finalized(self):
        io.progress_bar_close()

    #override
    def process_info_generator(self):
        for i in range(multiprocessing.cpu_count()):
            yield 'CPU%d' % (i), {}, {'device_idx': i,
                                      'device_name': 'CPU%d' % (i),
                                      'floatx' : self.floatx,
                                      'data_format' : self.data_format
                                      }

    #override
    def get_data(self, host_dict):
        if len (self.idx_shapes_list) > 0:
            return self.idx_shapes_list.pop(0)

        return None

    #override
    def on_data_return (self, host_dict, data):
        self.idx_shapes_list.insert(0, data)

    #override
    def on_result (self, host_dict, data, result):
        self.result.append ( result )
        io.progress_bar_inc(1)

    #override
    def get_result(self):
        return self.result
