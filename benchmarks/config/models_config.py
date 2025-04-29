from fbpinns.networks import (
    FCN, AdaptiveFCN, 
    ChebyshevKAN, ChebyshevAdaptiveKAN, 
    LegendreKAN, LegendreAdaptiveKAN
    )


# FCN
def FCN_Generator(mid_dim):
    return lambda indim, outdim: (f"FCN_{mid_dim}", FCN, dict(layer_sizes=[indim,mid_dim,outdim]))


# AdaptiveFCN
def AdaptiveFCN_Generator(mid_dim):
    return lambda indim, outdim: (f"FCN_A_{mid_dim}", AdaptiveFCN, dict(layer_sizes=[indim,mid_dim,outdim]))


# CKAN
def ChebyshevKAN_Generator(degree):
    return lambda indim, outdim: (f"CKAN_{degree}", ChebyshevKAN, dict(
        input_dim=indim,
        output_dim=outdim,
        degree=degree))

# CKAN Adaptive
def ChebyshevAdaptiveKAN_Generator(degree):
    return lambda indim, outdim: (f"CKAN_A_{degree}", ChebyshevAdaptiveKAN, dict(
        input_dim=indim,
        output_dim=outdim,
        degree=degree))

# LegendreKAN
def LegendreKAN_Generator(degree):
    return lambda indim, outdim: (f"LKAN_{degree}", LegendreKAN, dict(
        input_dim=indim,
        output_dim=outdim,
        degree=degree))

# LegendreKAN Adaptive
def LegendreAdaptiveKAN_Generator(degree):
    return lambda indim, outdim: (f"LKAN_A_{degree}", LegendreAdaptiveKAN, dict(
        input_dim=indim,
        output_dim=outdim,
        degree=degree))

def get_all_models():
    return [
        FCN_Generator(8),
        FCN_Generator(32),
        FCN_Generator(128),
        AdaptiveFCN_Generator(8),
        AdaptiveFCN_Generator(32),
        AdaptiveFCN_Generator(128),
        ChebyshevKAN_Generator(8),
        ChebyshevKAN_Generator(16),
        ChebyshevKAN_Generator(32),
        ChebyshevKAN_Generator(64),
        ChebyshevKAN_Generator(128),
        ChebyshevAdaptiveKAN_Generator(8),
        ChebyshevAdaptiveKAN_Generator(16),
        ChebyshevAdaptiveKAN_Generator(32),
        ChebyshevAdaptiveKAN_Generator(64),
        ChebyshevAdaptiveKAN_Generator(128),
        LegendreKAN_Generator(8),
        LegendreKAN_Generator(16),
        LegendreKAN_Generator(32),
        LegendreKAN_Generator(64),
        LegendreKAN_Generator(128),
        LegendreAdaptiveKAN_Generator(8),
        LegendreAdaptiveKAN_Generator(16),
        LegendreAdaptiveKAN_Generator(32),
        LegendreAdaptiveKAN_Generator(64),
        LegendreAdaptiveKAN_Generator(128),
    ]