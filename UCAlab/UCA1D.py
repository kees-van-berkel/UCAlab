"""
Copyright © 2025  Kees van Berkel
                                                                                
Permission is hereby granted, free of charge, to any person obtaining a copy of 
this software and associated documentation files (the “Software”), to deal in 
the Software without restriction, including without limitation the rights to 
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies 
of the Software, and to permit persons to whom the Software is furnished to do 
so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all 
copies or substantial portions of the Software.
                                         
THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS 
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR 
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER 
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN 
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import  os
import  sys
import  types
from    pathlib import Path
import  time   as cpu_time
import  pickle
import  scipy
import  sympy  as sp
import  numpy  as np
import  math
import  matplotlib
from    matplotlib.ticker    import MaxNLocator
import  matplotlib.pyplot    as plt
import  matplotlib.cm        as cm
import  matplotlib.patches   as patches
import  matplotlib.animation as animation
from    IPython.display import display, clear_output, Image, display_pretty,HTML

def W_plane1D(x, x0, k):
    return (np.exp(1j*k*(x-x0)))

def W_packet1D(x, x0, sigma, k):
    return (np.power(1.0/(sigma**2 *np.pi), 1/4) * 
            np.exp(-np.power((x-x0)/sigma, 2.0)/ 2) *
            np.exp(1j*k*(x-x0)))

def W_box1D(x, xc, L, n):
    k = n*np.pi/L
    return np.sin(k*(x-xc+L/2)) if xc-L/2 < x <= xc+L/2 else 0
                                            
def W_harmonic1D(x, N, rho, n):
    xs = (x-N//2)/rho                   # scale from CA-x to xi coordinates
    f0 = 1/np.sqrt(1.0* 2**n * math.factorial(n))
    f1 = scipy.special.hermite(n, monic=True)(xs) # look up hermite function
    f2 = np.exp(-0.5*xs**2)
    return f0*f1*f2                     # normalization tbd by .init() 
  
class UCA1D:
    _SUBPLOT_TITLE  = False
    _UCA_PICKLE     = './pickle'           # folder for pickled animations
    _UCA_png        = './png'              # folder for images
    _UCA_mp4        = './mp4'              # folder for animations
    _Psi2_offset    = 1e-8                 # offset of abs(Psi)**2, for color
    _color_Psi_real = 'tab:orange'         # colors for first particle
    _color_Psi_imag = 'tab:green'
    _color_Psi2     = 'tab:blue'
    _color_Psi_real_1 = 'tab:pink'         # colors for second particle
    _color_Psi_imag_1 = 'tab:olive'
    _color_Psi2_1   = 'tab:cyan'
    _color_Psi_trace= 'gold'
    _color_theta    = 'dodgerblue'
    _color_X        = 'black'              
    _color_V        = 'gray'                
    _color_A        = 'lightgray'
    _fig_width      = 10                   # rather high; for nice resolution
    _fig_height     = _fig_width/3                   
    _fontsize_subtitle = 12
    _fontsize_title = 14
             
    def __init__(self, N, M=1, name='uca'):
        """ create a 1D UCA of N (even) cells for M particles """
        if N%2!=0:
            raise ValueError('cell count N must be even')
        if not (0<M<=2):
            raise ValueError('particle count M must be at most 2')
        self.N              = N
        self.M              = M
        self.uca_name       = name
        self.theta          = None          # enables check if Hamiltonian set
        self.update_U0U1    = False
        self.time           = 0             # before reset() to support V(x, t=0)
        self.T_update       = None          # time interval [cycles] to update U
        self.N_report       = 10            # #cycles between run-report updates
        self.Psi2max        = None
        self.log_sum_abs_Psi_imag = False    
        self.log_trace_max  = False
        self.trace_max      = None
        self.probe_boxes    = []
        self.probe_boxes_user= []
        self.pickle_results = False         # set to True for UCA2D
        self.UCA_settings   = None
        if os.path.isfile('../UCA_settings.pkl'):
            pkl_file = open('../UCA_settings.pkl', 'rb')
            self.UCA_settings = pickle.load(pkl_file)
            pkl_file.close()
                
    def reset(self):
        """ reset the UCA to its initial state """ 
        self.time   = 0
        self.Psi    = self.Psi_init.copy()
        self.sum_abs_Psi_imag = []
        self._set_U()
        if self.log_trace_max:
            self.trace_max = [self._get_Psi2_xy()]
        self.Psi_probes= [np.zeros((box[1]-box[0]), dtype=float)
                          for box in self.probe_boxes]

    def _Psi_abs(self):
        return abs(self.Psi)**2

    def _Psi_view(self, Psi, real=False, imag=False):
        """ add _Psi2_offset (1e-8) for better visual results, esp for 2D plots"""
        return (Psi.real if real else
                Psi.imag if imag else 
                abs(Psi)**2 +UCA1D._Psi2_offset)

    def _Psi_init_normalize(self):
        S   = np.sum(abs(self.Psi_init)**2)
        self.Psi_init/= np.sqrt(S)
                                                                                
    def _exp_H22(a,b,c,d):
        """ return the exponent of hermitean 2x2 matrix -1j*[[a,b],[c,d]] 
        """
        def f_rho(a,b,c,d):
            r   = -a**2 +2*a*d -4*b*c -d**2         # r always negative
            return np.sqrt(-r)              
        rho     = f_rho(a,b,c,d)
        if rho<1e-12: 
            return np.identity(2, dtype=complex)
        cos_rh  = np.cos(rho/2)
        sin_rh  = np.sin(rho/2)
        sin_rhi = (1j)*sin_rh/rho
        exp_ab  = np.exp(-(1j)*(a+d)/2)
        U       = np.empty(shape=[2, 2], dtype=complex)
        U[0,0]  = (cos_rh - (a-d)*sin_rhi)*exp_ab
        U[0,1]  = (       - (2*b)*sin_rhi)*exp_ab
        U[1,0]  = (       - (2*c)*sin_rhi)*exp_ab
        U[1,1]  = (cos_rh + (a-d)*sin_rhi)*exp_ab
        return U
   
    def _val_x(self, Q, x, time=0):
        return (0    if Q is None                 else
                Q(x) if Q.__code__.co_argcount==1 else 
                Q(x, time))

    def _get_U22(self, x, theta, X, V, A, I, Psi): 
        """ construct local 2x2 block Hamiltonian (tau/hbar H) for cell x;
            theta = (tau/hbar) delta
            X(x)  = reflective cell
            V(x)  = (tau/hbar) q * phi(x);  phi is the scalar field
            A(x)  = (tau/hbar) q *  A'(x);  A'  is the vector field
            and return exp(-1j*(tau/hbar)*H) 
        """
        theta_x = theta(x) if type(theta) is types.LambdaType else theta
        X_x = self._val_x(X, x, time=self.time)
        if X_x:                                     # return "eXclusion U" 
            return -np.exp(-1j*theta_x)*np.identity(2, dtype=complex)       
        V_x = self._val_x(V, x, time=self.time)
        A_x = self._val_x(A, x, time=self.time)
        I_x = (0    if I is None                 else
               I(Psi))
        I_x0, I_x1 =  I_x if hasattr(I_x, "__len__") else (I_x, 0)
        dim = 2 if hasattr(self, 'Ny') else 1
        C   =  theta_x + (0 if theta_x==0 else (V_x + A_x**2/theta_x)/2/dim + I_x0)
        D   = -theta_x - 2j*A_x + I_x1
        U   = UCA1D._exp_H22(C, D, np.conjugate(D), C)
        return U
                                                                            
    def _get_U(self, N, theta, X, V, A):
        """ return U0, U1, 1-dimensional; also used by 2D _set_U() """
        shape   = (N//2,2,2) if self.M==1 else (self.M, N//2,2,2)
        shape   = (self.M, N//2,2,2)        
        U0      = np.empty(shape=shape, dtype=complex)
        U1      = np.empty(shape=shape, dtype=complex)
        if self.M==1:                       # single particle, no interaction
            for x in range(N//2):
                U0[0,x]   = self._get_U22(2*x,   theta, X, V, A, None, None)
                U1[0,x]   = self._get_U22(2*x+1, theta, X, V, A, None, None)
        else:
            for x in range(N//2):           # two particle, interaction I
                U0[0,x]   = self._get_U22(2*x,   theta, X, V, A, I, self.Psi[1][2*x])
                U0[1,x]   = self._get_U22(2*x,   theta, X, V, A, I, self.Psi[0][2*x])
                U1[0,x]   = self._get_U22(2*x+1, theta, X, V, A, I, self.Psi[1][2*x+1])
                U1[1,x]   = self._get_U22(2*x+1, theta, X, V, A, I, self.Psi[0][2*x+1])         
        return U0, U1                       

    def _set_X_line(self, time):
        """ set V_line = for time-dependent V """
        if not self.X is None:
            self.X_line = np.array([self._val_x(self.X, x, time=time) 
                                    for x in range(self.N)])

    def _set_V_line(self, time):
        """ set V_line = for time-dependent V """
        if not self.V is None:
            self.V_line = np.array([self._val_x(self.V, x, time=time) 
                                    for x in range(self.N)])    
            if time==0:
                self.V_max  = max(self.V_line)

    def _set_U(self):
        """ set U0, U1, 1-dimensional """
        self.U0, self.U1  = self._get_U(self.N, self.theta, 
                                       self.X, self.V, self.A)
        self._set_X_line(self.time)
        self._set_V_line(self.time)

    def set_hamiltonian(self, theta, X=None, V=None, A=None):
        """ configures the CA with the specified Hamiltonian
            theta: fixed rotation angle of a free particle
                 : alternatively:        x* --> float
            X    : eXclusion (boundary): x* --> bool
            V    : scalar potential:     x* --> float
            A    : vector potential:     x* --> float*
        """
        self.theta  = theta
        self.X  = X
        self.V  = V
        self.A  = A                       
        dim     = 2 if hasattr(self, 'Ny') else 1  
        args    = ('theta', 'X', 'V', 'A')
        for i, q in enumerate((theta, X, V, A)):
            argcount= 0  
            if type(q) is types.LambdaType:
                argcount= q.__code__.co_argcount
            if dim==1 and argcount >2:
                s   = ' must have at most two args'
                raise ValueError(args[i] +s)
            if dim==2 and not argcount in (0,2):
                s   = ' must have 0/2 args and must be time independent'
                raise ValueError(args[i] +s)
                                                                                
    def set_probe_boxes(self, boxes):
        """ boxes is a list of intervals, 2-tuples (xl, xh). 
            During UCA evolution abs(Psi)**2 is accumulated per cell per box,
        """
        self.probe_boxes_user = boxes
        self.probe_boxes      = []        
        count   = 0
        if self.theta is None:
            raise ValueError('Hamiltonian must be set before probe boxes')       
        for b in boxes:
            if not b in ('left', 'right'):
                self.probe_boxes.append(b) 
            else:
                box = [0, self.N]
                Xis =  not self.X is None
                if   b=='left'  : 
                    if Xis and self._val_x(self.X, box[0]): box[0] +=1
                    box[1] = box[0]+1
                elif b=='right' : 
                    if Xis and self._val_x(self.X, box[1]): box[1] -=1
                    box[0] = box[1]-1
                count  += 1
                self.probe_boxes.append(box) 
        if count>2:
            raise ValueError('more than 2 probes not supported')
    
    def init(self, f, particle=0, name='init', title=''):
        """ initialize Psi using function f(x);
            name can be used to identify the initial state;
            The initial state is normalized by .reset.
        """
        self.name   = name                  # wave form name
        self.title  = title                 # title of experiment
        if particle==0:
            self.Psi_init   = np.zeros((self.M, self.N), dtype=complex)
        for x in range(self.N):
            self.Psi_init[particle][x] = f(x)
        self._Psi_init_normalize()

    def _check_orthogonality(self):
        if self.M==1:
            return 0
        else:
            return np.sum(np.multiply(self.Psi[0], np.conjugate(self.Psi[1])))
    
    def set_title(self, title=''):
        self.title = title

    def _ax_plot_shared(self, ax, real, imag, time=0):
        """ shared (common) parts of _ax_plot() and _ax_plot_animate() """
        y0max   = 1.1*np.max(abs(self.Psi_init))**2     
        if not self.Psi2max is None: y0max = self.Psi2max
        ax.set_ylim(-y0max, y0max)
        ax.tick_params(axis='y', colors=UCA1D._color_Psi2)
        X   = np.arange(0, self.N)
        ax1 = None
        ax2 = None
        if real or imag:
            ax1     = ax.twinx()
            y1max   = 1.1*np.max(abs(self.Psi_init))
            color   = UCA1D._color_Psi_real if real else UCA1D._color_Psi_imag
            ax1.set_ylim(-y1max, y1max)
            ax1.tick_params(axis='y', colors=color)
            ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
        do_X= not self.X is None
        do_V= not self.V is None
        if do_X or do_V:
            ax2     = ax.twinx()
            y2max   = self.V_max if do_V else 1
            ax2.set_ylim([0, y2max])
            ax2.set_axis_off()
            ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
        return ax1, ax2, X

    def _ax_plot(self, ax, Psi, real, imag, time, is_RHS=False, title='time'): 
        """ plot 1D abs(Psi)**2 on ax;
            If (Boolean) real: add Psi.real;
            If (Boolean) is_RHS: add 2nd y-axis for Psi.real """
        if   title=='time':
            ax.set_title('time='+str(time), fontsize=UCA1D._fontsize_subtitle)   
        elif title=='full':
            ax.set_title(self.title, fontsize=UCA1D._fontsize_title)
        elif UCA1D._SUBPLOT_TITLE:
            if   imag:
                subtitle = 'Re(\u03a8(x)), Im(\u03a8(x)), P(x) in orange, green, blue @'
            elif real:
                subtitle = 'Re(\u03a8(x)), P(x) in orange, blue @'
            else:
                subtitle = 'P(x) in blue @'
            subtitle  += 'time='+str(time)
            ax.set_title(subtitle, fontsize=UCA1D._fontsize_subtitle)
        X   = np.arange(0, self.N)                
        ax.plot(X, self._Psi_view(Psi[0], real=False), color=UCA1D._color_Psi2)
        if self.M>1: 
            ax.plot(X, self._Psi_view(Psi[1], real=False), color=UCA1D._color_Psi2_1)
        ax1 = None
        ax1, ax2, X = self._ax_plot_shared(ax, real, imag, time=time)
        if not self.V is None:
            self._set_V_line(time)
            ax2.plot(X, self.V_line, color=UCA1D._color_V, linestyle='dashed')
        if not self.X is None:
            self._set_X_line(time)
            ax2.plot(X, self.X_line, color=UCA1D._color_X)
        if real or imag:
            if real:
                ax1.plot(X, self._Psi_view(Psi[0], real=True), 
                         color=UCA1D._color_Psi_real)
                if self.M>1: 
                    ax1.plot(X, self._Psi_view(Psi[1], real=True), 
                             color=UCA1D._color_Psi_real_1)  
            if imag:
                ax1.plot(X, self._Psi_view(Psi[0], imag=True), 
                         color=UCA1D._color_Psi_imag)
                if self.M>1: 
                    ax1.plot(X, self._Psi_view(Psi[1], imag=True), 
                             color=UCA1D._color_Psi_imag_1)
            if not is_RHS:
                ax1.yaxis.set_tick_params(labelright=False)
            # https://stackoverflow.com
            # /questions/38687887/how-to-define-zorder-when-using-2-y-axis
            ax.set_zorder(ax1.get_zorder()+1)
            ax.patch.set_visible(False)

    def _ax_plot_animate(self, real, imag):
        """ return initial plot for animate() """
        fig     = plt.figure()
        y0max   = 1.1*np.max(abs(self.Psi))**2     
        ax0     = plt.axes(xlim=(0, self.N)) 
        ax1, ax2, X  = self._ax_plot_shared(ax0, real, imag, time=self.time)
        plt.xlabel('x', fontsize=UCA1D._fontsize_title)
        return fig, ax0, ax1, ax2, X
        
    def plot(self, real=False, imag=False, Psi_slice=None, Psi_shape=None, 
             title='time', save=False):
        """ plot the current state 
            - real     : plot Psi.real
            - imag     : plot Psi.imag
            - if .Psi_save>1: plot intermediate states as well
            - Psi_slice: slice applied to the list of saved intermediate states
            - Psi_shape: tuple defining shape of a multi-state plot
            - save     : save plot as png
        """
        def get_figsize(V,H):
            fig_W   = UCA1D._fig_width
            if V==1:
                fig_H = UCA1D._fig_height
            else:
                HW    = self.Ny/self.Nx if hasattr(self, 'Ny') else 1.1
                fig_H = V/H * (HW+0.3) * fig_W
            return fig_W, fig_H

        V,H = (1,1) if Psi_shape is None or type(Psi_shape)!=tuple else Psi_shape
        fig, AX = plt.subplots(V, H, sharex='col', sharey='row', 
                               figsize=get_figsize(V,H))                                  
        Psi_saves, Psi_times = ((self.Psi_saves, self.Psi_times)
                                 if hasattr(self, 'Psi_saves') else
                               ([self.Psi], [self.time]))
        if not Psi_slice is None:
            Psi_saves, Psi_times = Psi_saves[Psi_slice], Psi_times[Psi_slice] 
        Psi_A       = np.array(Psi_saves[:V*H])
        xlabel      = 'cell index'
        for i, psi in enumerate(Psi_saves[:V*H]):
            ax = AX if V+H==2 else AX[i] if H==1 or V==1 else AX[i//H][i%H]
            self._ax_plot(ax, psi, real, imag, Psi_times[i], title=title, 
                          is_RHS=(i%H==H-1))                 
            if i//H == V-1:
                ax.set_xlabel(xlabel, fontsize=12)
            if (i//H == V-1) and (V>3 or H>3):  # recude xticks count
                xticks = ax.get_xticks()
                start  = int(xticks[0]<0)
                lx, lc = len(xticks)-start, xticks[-1]>=100
                if lx>6 and lc:
                    ax.set_xticks(xticks[start::2])
        fig.suptitle(self.title, fontsize=UCA1D._fontsize_title)
        fig.tight_layout()
        if save:
            os.makedirs(UCA1D._UCA_png, exist_ok=True)
            ext = '_real' if real else ''
            file_name = './png/' +self.name +ext +'.png'
            plt.savefig(file_name, bbox_inches = "tight")
        plt.show()
    
    def plot_time(self, T_real=False, xmin=None, xmax=None,
                  add_time0=False, real=False, imag=False, save=False):
        """ plot the time evolution of abs(Psi)**2;
            - T_real    : plot Psi.real in stead of abs(Psi)**2
            - xmin, xmax: cell range
            - add_time0 : add a panel with the initial state
            - real      : plot Psi.real in the initial state
            - imag      : plot Psi.imag in the initial state
            - save      : save plot as png
        """
        if xmin is None: xmin=0
        if xmax is None: xmax=self.N
        x       = np.arange(xmin, xmax)
        curves  = self.Psi_saves
        xlabel  = 'cell index'
        if add_time0:
            fig_W   = UCA1D._fig_width
            fig_H   = UCA1D._fig_height
            fig, ax = plt.subplots(1, 2, figsize=(fig_W, fig_H))
            ax1     = ax[1]
            psi     = curves[0] 
            self._ax_plot(ax[0], psi, real, imag, 0, is_RHS=True, title='')
            ax[0].set_xlabel(xlabel, fontsize=12)
        else:
            fig, ax = plt.subplots(1, 1, figsize=(7, 5))
            ax1     = ax
        ax1.set_xlabel(xlabel, fontsize=12)
        T_delta = self.time/len(curves)
        Psi_max = np.max(self._Psi_view(curves[0], real=T_real, imag=imag))
        F_scale = self.time * (0.14/Psi_max)
        for i, psi in enumerate(curves):
            psi2    = self._Psi_view(psi[0], real=T_real, imag=False)[xmin:xmax] 
            color   = 'black'   
            ax1.plot(x, F_scale*psi2+i*T_delta, color=color, linewidth=0.5)
            if self.M>1:
                psi2    = self._Psi_view(psi[1], real=T_real, imag=False)[xmin:xmax]
                color   = UCA1D._color_Psi2_1
                ax1.plot(x, F_scale*psi2+i*T_delta, color=color, linewidth=0.5)    
        subtitle   = ('Re(\u03a8(x))' if T_real else 'P(x)') +' over time'
        ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
        if UCA1D._SUBPLOT_TITLE:
            ax1.set_title(subtitle,  fontsize=UCA1D._fontsize_subtitle)
            fig.suptitle(self.title, fontsize=UCA1D._fontsize_title)
        fig.tight_layout()  
        if save:
            os.makedirs(UCA1D._UCA_png, exist_ok=True)
            file_name = './png/' +self.name +'_time.png'
            plt.savefig(file_name, bbox_inches = "tight")
        plt.show()
    
    def _update_U0(self, Psi, PsiN, U0):
        """ PsiN = U0*Psi """
        N       = Psi.shape[0]
        PsiN    = np.einsum('ijk,ik->ij', U0, Psi.reshape(N//2,2)).reshape((N))
        return PsiN
        
    def _update_U1(self, Psi, PsiN, U1):
        """ PsiN = roll(U1*roll(Psi,1), -1); faster rolls """
        N       = Psi.shape[0]
        a       = Psi[0]                   # roll PsiN by -1  
        Psi[:-1]= Psi[1:]
        Psi[-1] = a      
        PsiN    = np.einsum('ijk,ik->ij', U1, Psi.reshape(N//2,2)).reshape((N))
        a       = PsiN[-1]                 # roll Psi by  1 
        PsiN[1:]= PsiN[:-1]
        PsiN[0] = a      
        return PsiN
                
    def _update(self, Psi, PsiN, U0, U1):
        """ takes care of U0U1-U1U0 alternation (also for 2D) 
            Arguments are local to enable reuse by UCA2D.
            Argument PsiN is a dummy for reuse.
        """
        if self.time%2==0 or self.update_U0U1:
            PsiN= self._update_U0(Psi, PsiN, U0)
            Psi = self._update_U1(PsiN, Psi, U1)
        else:
            PsiN= self._update_U1(Psi, PsiN, U1)
            Psi = self._update_U0(PsiN, Psi, U0)    
        if self.log_sum_abs_Psi_imag:
            Psi_sum = np.sum(np.abs(Psi.imag))
            self.sum_abs_Psi_imag.append(Psi_sum)
            Psi_sum_x = np.sum(Psi.imag)
        return Psi
                                            
    def _run_init(self, T=0, Psi_save=1):
        """ preparation for run/animate """
        self.reset()
        self.tcpu0  = cpu_time.process_time()
        if hasattr(Psi_save, "__len__"):
            self.Psi_times  = Psi_save      # user specified Psi times
        elif Psi_save==1:
            self.Psi_times  = [T]
        else:
            period          = int(T/(Psi_save-1))
            self.Psi_times  = [k*period for k in range(Psi_save)]
        self.Psi_saves  = [self.Psi.copy()]
        #self.jupyter_book = Path('../_toc.yml').is_file()
        if self.N_report>0:
            self._run_report()
            
    def _run_pickle_open(self):
        """ prepare for pickle """
        os.makedirs(UCA1D._UCA_PICKLE, exist_ok=True)
        pkl_name= UCA1D._UCA_PICKLE +'/' +self.name
        self.Psi_pkl_name   = pkl_name +'_Psi.pkl'
        self.html_pkl_name  = pkl_name +'_html.pkl'
        if not self.UCA_settings is None: 
            self.UCA_settings['timer'] = True   # not running for jupyter-book

    def _run_pickle_load_psi(self):
        """ load evolution state from pickle file """
        pkl_file = open(self.Psi_pkl_name, 'rb')
        (self.Psi_saves, self.Psi_times, self.Psi_probes, self.trace_max) \
            = pickle.load(pkl_file)
        self.Psi  = self.Psi_saves[-1]
        self.time = self.Psi_times[-1]
        pkl_file.close()
                                                                            
    def _run_pickle_close(self):
        """ dump evolution state in pickle file """
        pkl_file = open(self.Psi_pkl_name, 'wb')
        if self.time!=self.Psi_times[-1]:
            self.Psi_times.append(self.time)
            self.Psi_saves.append(self.Psi.copy())
        pickle.dump((self.Psi_saves, self.Psi_times, self.Psi_probes, 
                     self.trace_max), pkl_file)
        pkl_file.close()

    def _run_report(self):
        """ print simulation progress report during evolution """
        Psi_abs = self._Psi_abs()
        vtcpu   = cpu_time.process_time()-self.tcpu0     # cpu time
        g       = '    '
        tcyc    = 'time: '  +'{:> 5}' .format(self.time) +g
        tcpu    = 'tcpu: '  +'{0:.2f}'.format(vtcpu) +g
        prob    = '|P-1|: ' +'{:.1E}'.format(abs(np.sum(Psi_abs)-1))
        if self.M>1:
            S       = self._check_orthogonality()
            ortho   = g +'|Sab|: ' + '{:.1E}'.format(np.abs(S))
        else:
            ortho   = ''
        self.timer= tcyc +tcpu +prob +ortho +'      '    # reused for animation
        if self.UCA_settings is None or self.UCA_settings['timer']:
            sys.stdout.write("\r"+self.timer)

    def _get_Psi2_xy(self):
        """ return cell position for max Psi**2 """
        Psi2 = self._Psi_view(self.Psi)
        return np.unravel_index(Psi2.argmax(), Psi2.shape)

    def _run_step_probes(self):
        """ accumulate state in probe boxes """
        for boxi, box in enumerate(self.probe_boxes):
            Psi_box = self.Psi[0][box[0]:box[1]]
            self.Psi_probes[boxi] += np.abs(Psi_box)**2
 
    def _run_step_inc(self):
        """ update reporting and trackers """
        self.time +=1
        if self.N_report>0 and self.time%(self.N_report)==0:
            self._run_report()
        if self.time in self.Psi_times:
            self.Psi_saves.append(self.Psi.copy())
        if self.log_trace_max:
            self.trace_max.append(self._get_Psi2_xy())
        if self.probe_boxes_user!=[]:
            self._run_step_probes()
    
    def _run_step(self, T, cpf=1):
        """ advances the UCA state by one step;
            used both by run() and by animate() """  
        PsiN   = np.empty(self.Psi.shape, dtype=complex)
        c = 0
        while self.time<T and c<cpf:
            c += 1
            self.Psi[0] = self._update(self.Psi[0], PsiN, self.U0[0], self.U1[0])
            if self.M>1:                    # update 2nd particle
                self.Psi[1] = self._update(self.Psi[1], PsiN, self.U0[1], self.U1[1])    
                if self.F:                  # apply fermion cross
                    Psi_a = 1/np.sqrt(2)*(self.Psi[0] + self.Psi[1])
                    Psi_b = 1/np.sqrt(2)*(self.Psi[0] - self.Psi[1])
                    if self.time%2==0:      # alternate fermion cross each cycle
                        self.Psi[0], self.Psi[1] = Psi_a, Psi_b
                    else:
                        self.Psi[0], self.Psi[1] = Psi_a, Psi_b
            if not self.T_update is None and self.time%self.T_update==0:
                self._set_U()    
            self._run_step_inc()

    def run(self, T, Psi_save=1, pickle_results=True):
        """ evolve the UCA for T time steps;
            - Psi_save: save Psi_save states at regular intervals  
            - if pickle_results: ONLY FOR 2D UCA:
            -                    reuse earlier results or save for reuse 
        """   
        self.reset()
        if len(self.Psi.shape)>2:                   # 2D simulation    
            self.pickle_results = pickle_results
        if self.pickle_results:
            self._run_pickle_open()
            if os.path.isfile(self.Psi_pkl_name):   # no run, restore Psi_save
                self._run_pickle_load_psi()
                return
        self._run_init(T=T, Psi_save=Psi_save)
        while self.time<T:
            self._run_step(T)
        if self.pickle_results:  
            self._run_pickle_close()
                                                                                
    def animate(self, T,  cpf=1, real=True, imag=False, Psi2max=None, 
                Psi_save=1, save=False):        
        """ create an animation of the evolution of the UCA for T time steps:
            - cpf     : number of cycles (steps) per video frame
            - real    : include Psi.real
            - imag    : include Psi.imag
            - Psi2max : overrules max ylim in animaton
            - Psi_save: save Psi_save states at regular intervals  
            - save    : save animation as mp4
        """        
        def set_line_colors():
            self.line_colors = []
            if not self.X is None: 
                self.line_colors.append(UCA1D._color_X)
            if not self.V is None: 
                self.line_colors.append(UCA1D._color_V)
            if True: 
                self.line_colors.append(UCA1D._color_Psi2)
                if self.M>1: self.line_colors.append(UCA1D._color_Psi2_1)
            if real: 
                self.line_colors.append(UCA1D._color_Psi_real)
                if self.M>1: self.line_colors.append(UCA1D._color_Psi_real_1)
            if imag: 
                self.line_colors.append(UCA1D._color_Psi_imag)       
                if self.M>1: self.line_colors.append(UCA1D._color_Psi_imag_1)
        def init():
            self.first_frame = True 
            time_txt.set_text('time = 0')
            for l in lines[:-1]:
                l.set_data([], [])
            return lines                   
        
        def animate_step(i):
            if self.first_frame:
                self.first_frame = False
            else:
                self._run_step(T, cpf=cpf)
            time_txt.set_text(self.timer)
            for l, color in enumerate(self.line_colors):
                if   color==UCA1D._color_X:            
                    lines[l].set_data([X], [self.X_line])
                elif color==UCA1D._color_V:            
                    lines[l].set_data([X], [self.V_line])
                elif color==UCA1D._color_Psi2:       
                    lines[l].set_data( X, self._Psi_view(self.Psi[0]))           
                elif color==UCA1D._color_Psi_real:   
                    lines[l].set_data([X], [self.Psi[0].real])           
                elif color==UCA1D._color_Psi_imag:   
                    lines[l].set_data([X], [self.Psi[0].imag])            
                elif color==UCA1D._color_Psi2_1:     
                    lines[l].set_data( X, self._Psi_view(self.Psi[1]))          
                elif color==UCA1D._color_Psi_real_1: 
                    lines[l].set_data([X], [self.Psi[1].real])           
                elif color==UCA1D._color_Psi_imag_1: 
                    lines[l].set_data([X], [self.Psi[1].imag])            
            return lines
                    
        self.Psi2max= Psi2max
        self._run_init(T=T, Psi_save=Psi_save)
        fig, ax0, ax1, ax2, X = self._ax_plot_animate(real, imag)
        ax0.set_title(self.title, fontsize=UCA1D._fontsize_title)
        fig.set_size_inches(8, 4.5)         # aim at 16x9 frames
        tx, ty  = (0.02, 0.02)
        time_txt= ax0.text(tx, ty, '', horizontalalignment='left',
                    verticalalignment='bottom', transform=ax0.transAxes)
        set_line_colors()
        line_ax = []
        if not self.X is None: line_ax += [ax2]
        if not self.V is None: line_ax += [ax2]
        line_ax += [ax0]
        if self.M>1: line_ax += [ax0]
        if real: 
            line_ax += [ax1]
            if self.M>1: line_ax += [ax1]
        if imag: 
            line_ax += [ax1]
            if self.M>1: line_ax += [ax1]
        lines   = []
        for i, color in enumerate(self.line_colors):
            linestyle   = 'dashed' if color==UCA1D._color_V else 'solid'
            line,   = line_ax[i].plot([], [], color=color, linestyle=linestyle)
            lines.append(line)
        if not ax1 is None and not ax1 in line_ax:
            ax1.set_axis_off()
        if not ax2 is None and not ax2 in line_ax:
            ax2.set_axis_off()
                
        self.ani= animation.FuncAnimation(fig, animate_step, init_func=init,
                                          frames=T//cpf+2, interval=100, blit=True)
        plt.close()
        if save:                            # avoid redundant plot
            writer      = animation.writers['ffmpeg'](fps=10, bitrate=1e6)
            writer.scale= (1920,1080)          
            os.makedirs(UCA1D._UCA_mp4, exist_ok=True)
            file_name   = './mp4/'+self.name+'.mp4'
            self.ani.save(file_name, writer=writer, dpi=240)
        return HTML(self.ani.to_jshtml(default_mode='once'))
