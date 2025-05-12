from fbpinns.networks import (
    FCN, AdaptiveFCN, 
    ChebyshevKAN, ChebyshevAdaptiveKAN, StackedChebyshevKAN,
    LegendreKAN, LegendreAdaptiveKAN
    )

from typing import Tuple
# FCN
def FCN_Generator(mid_dim):
    if type(mid_dim) == int:
        mid_dim = [mid_dim]
    return lambda indim, outdim: (f"FCN_{mid_dim}", FCN, dict(layer_sizes=[indim,*mid_dim,outdim]))


# AdaptiveFCN
def AdaptiveFCN_Generator(mid_dim):
    return lambda indim, outdim: (f"FCN_A_{mid_dim}", AdaptiveFCN, dict(layer_sizes=[indim,mid_dim,outdim]))


# CKAN
def ChebyshevKAN_Generator(degree, kind=1):
    return lambda indim, outdim: (f"CKAN_{degree}", ChebyshevKAN, dict(
        input_dim=indim,
        output_dim=outdim,
        degree=degree,
        kind=kind))

def StackedCKAN_Generator(degree, hidden_dim, kind=1):
    return lambda indim, outdim: (f"StackedCKAN_d{degree}_h{hidden_dim}", StackedChebyshevKAN, dict(
        input_dim=indim,
        hidden_dim=hidden_dim,
        output_dim=outdim,
        degree=degree,
        kind=kind))


# CKAN Adaptive
def ChebyshevAdaptiveKAN_Generator(degree, kind=1):
    return lambda indim, outdim: (f"CKAN_A_{degree}", ChebyshevAdaptiveKAN, dict(
        input_dim=indim,
        output_dim=outdim,
        degree=degree,
        kind=kind))

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

def get_all_models_1():
    return [
        FCN_Generator(8),
        FCN_Generator(16),
        FCN_Generator(32),
        ChebyshevKAN_Generator(8),
        ChebyshevKAN_Generator(16),
        ChebyshevKAN_Generator(32),
        ChebyshevKAN_Generator(8, kind=2),
        ChebyshevKAN_Generator(16, kind=2),
        ChebyshevKAN_Generator(32, kind=2),
        LegendreKAN_Generator(8),
        LegendreKAN_Generator(16),
        LegendreKAN_Generator(32),
    ]

def get_all_models_2():
    return [
        FCN_Generator((8, 8)),
        FCN_Generator((16, 16)),
        FCN_Generator((32, 32)),
        # StackedCKAN_Generator(4, hidden_dim=1),
        # StackedCKAN_Generator(4, hidden_dim=2),
        # StackedCKAN_Generator(4, hidden_dim=4),
        # StackedCKAN_Generator(8, hidden_dim=1),
        # StackedCKAN_Generator(8, hidden_dim=2),
        # StackedCKAN_Generator(8, hidden_dim=4),
        # StackedCKAN_Generator(16, hidden_dim=1),
        # StackedCKAN_Generator(16, hidden_dim=2),
        # StackedCKAN_Generator(16, hidden_dim=4),
    ]

if __name__ == "__main__":
    m = FCN_Generator(16)(2, 1)
    print(m)