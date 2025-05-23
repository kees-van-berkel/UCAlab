from .UCA1D import *
from .UCA2D import *

matplotlib.rcParams['animation.embed_limit'] = 2**28
sp.init_printing(use_latex=True)

__all__ = ['UCA1D', 'W_plane1D', 'W_packet1D', 'W_box1D', 'W_harmonic1D',   
           'UCA2D', 'W_plane2D', 'W_packet2D']