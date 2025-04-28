from fbpinns.networks import (
    FCN, 
    AdaptiveFCN, 
    ChebyshevKAN, 
    ChebyshevAdaptiveKAN, 
    LegendreKAN,
    LegendreAdaptiveKAN
    )


# FCN
def FCN_Generator(mid_dim):
    return lambda indim, outdim: (f"FCN_{mid_dim}", FCN, dict(layer_sizes=[indim,mid_dim,outdim]))
FCN_8 = FCN_Generator(8)
FCN_32 = FCN_Generator(32)
FCN_128 = FCN_Generator(128)

# AdaptiveFCN
def AdaptiveFCN_Generator(mid_dim):
    return lambda indim, outdim: (f"FCN_A_{mid_dim}", AdaptiveFCN, dict(layer_sizes=[indim,mid_dim,outdim]))
AdaptiveFCN_8 = AdaptiveFCN_Generator(8)
AdaptiveFCN_32 = AdaptiveFCN_Generator(32)
AdaptiveFCN_128 = AdaptiveFCN_Generator(128)

# CKAN
def ChebyshevKAN_Generator(degree):
    return lambda indim, outdim: (f"CKAN_{degree}", ChebyshevKAN, dict(
        input_dim=indim,
        output_dim=outdim,
        degree=degree))
CKAN_8 = ChebyshevKAN_Generator(8)
CKAN_16 = ChebyshevKAN_Generator(16)
CKAN_32 = ChebyshevKAN_Generator(32)
CKAN_64 = ChebyshevKAN_Generator(64)
CKAN_128 = ChebyshevKAN_Generator(128)

# CKAN Adaptive
def ChebyshevAdaptiveKAN_Generator(degree):
    return lambda indim, outdim: (f"CKAN_A_{degree}", ChebyshevAdaptiveKAN, dict(
        input_dim=indim,
        output_dim=outdim,
        degree=degree))
CKAN_8_Adaptive = ChebyshevAdaptiveKAN_Generator(8)
CKAN_16_Adaptive = ChebyshevAdaptiveKAN_Generator(16)
CKAN_32_Adaptive = ChebyshevAdaptiveKAN_Generator(32)
CKAN_64_Adaptive = ChebyshevAdaptiveKAN_Generator(64)
CKAN_128_Adaptive = ChebyshevAdaptiveKAN_Generator(128)

# LegendreKAN
def LegendreKAN_Generator(degree):
    return lambda indim, outdim: (f"LKAN_{degree}", LegendreKAN, dict(
        input_dim=indim,
        output_dim=outdim,
        degree=degree))
LegendreKAN_8 = LegendreKAN_Generator(8)
LegendreKAN_16 = LegendreKAN_Generator(16)
LegendreKAN_32 = LegendreKAN_Generator(32)
LegendreKAN_64 = LegendreKAN_Generator(64)
LegendreKAN_128 = LegendreKAN_Generator(128)

# LegendreKAN Adaptive
def LegendreAdaptiveKAN_Generator(degree):
    return lambda indim, outdim: (f"LKAN_A_{degree}", LegendreAdaptiveKAN, dict(
        input_dim=indim,
        output_dim=outdim,
        degree=degree))
LegendreAdaptiveKAN_8 = LegendreAdaptiveKAN_Generator(8)
LegendreAdaptiveKAN_16 = LegendreAdaptiveKAN_Generator(16)
LegendreAdaptiveKAN_32 = LegendreAdaptiveKAN_Generator(32)
LegendreAdaptiveKAN_64 = LegendreAdaptiveKAN_Generator(64)
LegendreAdaptiveKAN_128 = LegendreAdaptiveKAN_Generator(128)

def get_all_models():
    return [
        FCN_8,
        FCN_32,
        FCN_128,
        AdaptiveFCN_8,
        AdaptiveFCN_32,
        AdaptiveFCN_128,
        CKAN_8,
        CKAN_16,
        CKAN_64,
        CKAN_8_Adaptive,
        CKAN_16_Adaptive,
        CKAN_64_Adaptive,
        LegendreKAN_8,
        LegendreKAN_16,
        LegendreKAN_64,
        LegendreAdaptiveKAN_8,
        LegendreAdaptiveKAN_16,
        LegendreAdaptiveKAN_64,
    ]