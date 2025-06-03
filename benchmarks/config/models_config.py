from fbpinns.networks import (
    FCN, AdaptiveFCN, 
    ChebyshevKAN, ChebyshevAdaptiveKAN, 
    StackedChebyshevKAN, StackedLegendreKAN,
    LegendreKAN, LegendreAdaptiveKAN,
    OptimizedChebyshevKAN, OptimizedStackedChebyshevKAN,
    MixedBasisKAN,
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
def ChebyshevKAN_Generator(degree, kind=2):
    return lambda indim, outdim: (f"CKAN_d{degree}_k{kind}", ChebyshevKAN, dict(
        input_dim=indim,
        output_dim=outdim,
        degree=degree,
        kind=kind))

def StackedCKAN_Generator(degrees, hidden_dims, kind=2):
    return lambda indim, outdim: (f"StackedCKAN_deg{degrees}_h{hidden_dims}", StackedChebyshevKAN, dict(
        dims=[indim] + hidden_dims + [outdim],
        degrees=degrees,
        kinds=kind))

def OptimizedCKAN_Generator(degree, kind=2):
    return lambda indim, outdim: (f"OptimizedCKAN_d{degree}_k{kind}", OptimizedChebyshevKAN, dict(
        input_dim=indim,
        output_dim=outdim,
        degree=degree,
        kind=kind))

def OptimizedStackedCKAN_Generator(degrees, hidden_dims, kind=2):
    return lambda indim, outdim: (f"OptimizedStackedCKAN_deg{degrees}_h{hidden_dims}", OptimizedStackedChebyshevKAN, dict(
        dims=[indim] + hidden_dims + [outdim],
        degrees=degrees,
        kinds=kind))

# CKAN Adaptive
def ChebyshevAdaptiveKAN_Generator(degree, kind=1):
    return lambda indim, outdim: (f"CKAN_A_d{degree}_k{kind}", ChebyshevAdaptiveKAN, dict(
        input_dim=indim,
        output_dim=outdim,
        degree=degree,
        kind=kind))

# LegendreKAN
def LegendreKAN_Generator(degree):
    return lambda indim, outdim: (f"LKAN_d{degree}", LegendreKAN, dict(
        input_dim=indim,
        output_dim=outdim,
        degree=degree))

def StackedLKAN_Generator(degrees, hidden_dims):
    return lambda indim, outdim: (f"StackedLKAN_deg{degrees}_h{hidden_dims}", StackedLegendreKAN, dict(
        dims=[indim] + hidden_dims + [outdim],
        degrees=degrees))

# LegendreKAN Adaptive
def LegendreAdaptiveKAN_Generator(degree):
    return lambda indim, outdim: (f"LKAN_A_d{degree}", LegendreAdaptiveKAN, dict(
        input_dim=indim,
        output_dim=outdim,
        degree=degree))

def ChebLegStackedKAN_Generator(degrees, hidden_dims, kind=2):
    return lambda indim, outdim: (f"ChebLegStackedKAN_deg{degrees}_h{hidden_dims}", MixedBasisKAN, dict(
        dims=[indim] + hidden_dims + [outdim],
        degrees=degrees,
        kinds=kind))

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
        FCN_Generator([4]),
        FCN_Generator([4, 4]),
        FCN_Generator([4, 4, 4]),
        FCN_Generator([4, 4, 4, 4]),
        FCN_Generator([8]),
        FCN_Generator([8, 8]),
        FCN_Generator([8, 8, 8]),
        FCN_Generator([8, 8, 8, 8]),
        FCN_Generator([16]),
        FCN_Generator([16, 16]),
        FCN_Generator([16, 16, 16]),
        # FCN_Generator([16, 16, 16, 16]),
        FCN_Generator([32]),
        FCN_Generator([32, 32]),
        # FCN_Generator([32, 32, 32]),
        # FCN_Generator([32, 32, 32, 32]),
    ]


def get_models_stacked_ckan_varying_stack():
    return [
        StackedCKAN_Generator(degrees=[4, 4], hidden_dims=[4]),
        StackedCKAN_Generator(degrees=[4, 4, 4], hidden_dims=[4, 4]),
        StackedCKAN_Generator(degrees=[4, 4, 4, 4], hidden_dims=[4, 4, 4]),
        StackedCKAN_Generator(degrees=[4, 4, 4, 4, 4], hidden_dims=[4, 4, 4, 4]),
    ]

def get_models_stacked_ckan_varying_hidden_dims():
    return [
        StackedCKAN_Generator(degrees=[4, 4], hidden_dims=[2]),
        StackedCKAN_Generator(degrees=[4, 4], hidden_dims=[4]),
        StackedCKAN_Generator(degrees=[4, 4], hidden_dims=[6]),
        StackedCKAN_Generator(degrees=[4, 4], hidden_dims=[8]),
        StackedCKAN_Generator(degrees=[4, 4], hidden_dims=[10]),
    ]

def get_models_stacked_ckan_varying_degrees_1():
    return [
        StackedCKAN_Generator(degrees=[2, 4], hidden_dims=[4]),
        StackedCKAN_Generator(degrees=[4, 4], hidden_dims=[4]),
        StackedCKAN_Generator(degrees=[8, 4], hidden_dims=[4]),
        StackedCKAN_Generator(degrees=[10, 4], hidden_dims=[4]),
        StackedCKAN_Generator(degrees=[20, 4], hidden_dims=[4]),
    ]

def get_models_stacked_ckan_varying_degrees_2():
    return [
        StackedCKAN_Generator(degrees=[4, 2], hidden_dims=[4]),
        StackedCKAN_Generator(degrees=[4, 4], hidden_dims=[4]),
        StackedCKAN_Generator(degrees=[4, 8], hidden_dims=[4]),
        StackedCKAN_Generator(degrees=[4, 10], hidden_dims=[4]),
        StackedCKAN_Generator(degrees=[4, 20], hidden_dims=[4]),
    ]

def get_models_stacked_lkan_varying_hidden_dims():
    return [
        StackedLKAN_Generator(degrees=[4, 4], hidden_dims=[2]),
        StackedLKAN_Generator(degrees=[4, 4], hidden_dims=[4]),
        StackedLKAN_Generator(degrees=[4, 4], hidden_dims=[6]),
        StackedLKAN_Generator(degrees=[4, 4], hidden_dims=[8]),
        StackedLKAN_Generator(degrees=[4, 4], hidden_dims=[10]),
    ]

def get_models_optimized_ckan():
    return [
        # OptimizedCKAN_Generator(8),
        # OptimizedCKAN_Generator(16),
        # OptimizedCKAN_Generator(32),
    ]

def get_models_optimized_stacked_ckan():
    return [
        # OptimizedStackedCKAN_Generator(degrees=[4, 4], hidden_dims=[2]),
        # OptimizedStackedCKAN_Generator(degrees=[4, 4], hidden_dims=[4]),
        # OptimizedStackedCKAN_Generator(degrees=[4, 4], hidden_dims=[6]),
        # OptimizedStackedCKAN_Generator(degrees=[4, 4], hidden_dims=[8]),
        # OptimizedStackedCKAN_Generator(degrees=[4, 4], hidden_dims=[10]),
        # OptimizedStackedCKAN_Generator(degrees=[4, 4, 4], hidden_dims=[4, 4]),
        OptimizedStackedCKAN_Generator(degrees=[4, 4, 4, 4], hidden_dims=[4, 4, 4]),
        OptimizedStackedCKAN_Generator(degrees=[4, 4, 4, 4, 4], hidden_dims=[4, 4, 4, 4]),
    ]


def get_models_taylor():
    return [
        # OptimizedStackedCKAN_Generator(degrees=[4, 4], hidden_dims=[1]),
        # OptimizedStackedCKAN_Generator(degrees=[4, 4], hidden_dims=[2]),
        # OptimizedStackedCKAN_Generator(degrees=[4, 4], hidden_dims=[3]),
        # OptimizedStackedCKAN_Generator(degrees=[4, 4], hidden_dims=[4]),
        # OptimizedStackedCKAN_Generator(degrees=[4, 4], hidden_dims=[5]),
        # OptimizedStackedCKAN_Generator(degrees=[4, 4], hidden_dims=[6]),
        # OptimizedStackedCKAN_Generator(degrees=[5, 5], hidden_dims=[4]),
        # OptimizedStackedCKAN_Generator(degrees=[6, 6], hidden_dims=[4]),
        # OptimizedStackedCKAN_Generator(degrees=[7, 7], hidden_dims=[4]),
        OptimizedStackedCKAN_Generator(degrees=[7, 7], hidden_dims=[6]),
        # StackedCKAN_Generator(degrees=[5, 5], hidden_dims=[4]),
        # StackedCKAN_Generator(degrees=[6, 6], hidden_dims=[4]),
        # StackedCKAN_Generator(degrees=[7, 7], hidden_dims=[4]),
        # FCN_Generator([4, 4]),
        # FCN_Generator([8, 8]),
        # FCN_Generator([16, 16]),
        # FCN_Generator([32, 32]),
    ][::-1]

def get_models_mixed_basis_kan():
    return [
        ChebLegStackedKAN_Generator(degrees=[4, 4], hidden_dims=[2]),
        ChebLegStackedKAN_Generator(degrees=[4, 4], hidden_dims=[4]),
        ChebLegStackedKAN_Generator(degrees=[4, 4], hidden_dims=[6]),
        ChebLegStackedKAN_Generator(degrees=[4, 4], hidden_dims=[8]),
        ChebLegStackedKAN_Generator(degrees=[4, 4], hidden_dims=[10]),
        ChebLegStackedKAN_Generator(degrees=[5, 5], hidden_dims=[4]),
        ChebLegStackedKAN_Generator(degrees=[6, 6], hidden_dims=[4]),
        ChebLegStackedKAN_Generator(degrees=[7, 7], hidden_dims=[4]),
    ]

def get_models_scheduling():
    return [
        # OptimizedStackedCKAN_Generator(degrees=[4, 4], hidden_dims=[16]),
        # OptimizedStackedCKAN_Generator(degrees=[4, 4], hidden_dims=[14]),
        # OptimizedStackedCKAN_Generator(degrees=[4, 4], hidden_dims=[12]),
        # OptimizedStackedCKAN_Generator(degrees=[8, 8], hidden_dims=[10]),
        # OptimizedStackedCKAN_Generator(degrees=[6, 6], hidden_dims=[10]),
        # OptimizedStackedCKAN_Generator(degrees=[4, 4], hidden_dims=[10]),
        # OptimizedStackedCKAN_Generator(degrees=[4, 4], hidden_dims=[8]),
        OptimizedStackedCKAN_Generator(degrees=[4, 4], hidden_dims=[6]),
        OptimizedStackedCKAN_Generator(degrees=[4, 4], hidden_dims=[4]),
        # OptimizedStackedCKAN_Generator(degrees=[4, 4, 4], hidden_dims=[10, 10]),
        # OptimizedStackedCKAN_Generator(degrees=[4, 4, 4, 4], hidden_dims=[10, 10, 10]),
        FCN_Generator([8, 8]),
        FCN_Generator([16, 16]),
        # FCN_Generator([32, 32]),
        # FCN_Generator([32, 32, 32]),
        # FCN_Generator([32, 32, 32, 32]),
        # FCN_Generator([64, 64]),
    ]


def get_models_gauss():
    return [
        FCN_Generator([4, 4]),
        FCN_Generator([8, 8]),
        FCN_Generator([16, 16]),
        FCN_Generator([32, 32]),
        # OptimizedStackedCKAN_Generator(degrees=[4, 4], hidden_dims=[1]),
        # OptimizedStackedCKAN_Generator(degrees=[4, 4], hidden_dims=[2]),
        # OptimizedStackedCKAN_Generator(degrees=[4, 4], hidden_dims=[3]),
        # OptimizedStackedCKAN_Generator(degrees=[4, 4], hidden_dims=[4]),
        # OptimizedStackedCKAN_Generator(degrees=[4, 4], hidden_dims=[5]),
        # OptimizedStackedCKAN_Generator(degrees=[4, 4], hidden_dims=[6]),
        OptimizedStackedCKAN_Generator(degrees=[5, 5], hidden_dims=[4]),
        OptimizedStackedCKAN_Generator(degrees=[6, 6], hidden_dims=[4]),
        OptimizedStackedCKAN_Generator(degrees=[7, 7], hidden_dims=[4]),
    ]


if __name__ == "__main__":
    m = FCN_Generator(16)(2, 1)
    print(m)