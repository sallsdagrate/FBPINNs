from fbpinns.networks import FCN

FCN_8_Config = dict(layer_sizes=[1,8,1])
FCN_8 = ("FCN_8", FCN, FCN_8_Config)

FCN_32_Config = dict(layer_sizes=[1,32,1])
FCN_32 = ("FCN_32", FCN, FCN_32_Config)

FCN_128_Config = dict(layer_sizes=[1,128,1])
FCN_128 = ("FCN_128", FCN, FCN_128_Config)