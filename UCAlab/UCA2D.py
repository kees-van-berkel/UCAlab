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

from .UCA1D import *
from mpl_toolkits.axes_grid1 import make_axes_locatable

def W_plane2D(x, y, x0, y0, kx, ky):
    return (np.exp(1j*kx*(x-x0) + 1j*ky*(y-y0)))
    
def W_packet2D(x, y, x0, y0, sigma, sigma_y=None, kx=0, ky=0):
    if sigma_y is None:
        sigma_y = sigma
    return (np.exp(-1/(2*sigma**2)*(x-x0)**2 -1/(2*sigma_y**2)*(y-y0)**2) *
            np.exp(1j*kx*(x-x0) +1j*ky*(y-y0)))
                                                                            
class UCA2D(UCA1D):
                                            
    def __init__(self, Nx, Ny, name='uca'):
        """ create a 2D UCA of Nx by Ny cells """
        if Nx%2!=0 or Ny%2!=0:
            raise ValueError('cell countx Nx and Ny must be even')
        self.Nx     = Nx
        self.Ny     = Ny
        super(UCA2D, self).__init__(Nx, name=name)

    def reset(self):
        """ reset the UCA to its initial state """ 
        super(UCA2D, self).reset()
        self.Psi_probes= [np.zeros((box[3]-box[2],box[1]-box[0]), dtype=float)
                          for box in self.probe_boxes]

    def init(self, f, particle=0, name='init', title=''):
        """ initialize Psi by applying lambda function f(x) or f(x,y);
            name can be used to identify the initial state;
            The initial state is normalized.
        """
        self.name       = name
        self.title      = title             # title of experiment
        self.Psi_init   = np.zeros((self.M, self.Ny, self.Nx), dtype=complex)
        for x in range(self.Nx):
            for y in range(self.Ny):
                self.Psi_init[particle][y,x] = f(x,y)
        self._Psi_init_normalize()
       
    def set_XVA(self):
        """ set plot overlays """
        self.Xmesh, self.Ymesh = np.meshgrid(range(self.Nx), range(self.Ny)) 
        if type(self.theta) is types.LambdaType:
            theta_vect  = np.vectorize(self.theta)
            self.thetaZ = theta_vect(self.Xmesh, self.Ymesh) 
        if not self.X is None:
            X_vect  = np.vectorize(self.X)
            self.XZ = X_vect(self.Xmesh, self.Ymesh) 
        if not self.V is None:
            V_vect  = np.vectorize(self.V, otypes=[float])
            self.VZ = V_vect(self.Xmesh, self.Ymesh)
        if not self.A is None:
            self.Ax, self.Ay  = self.A(self.Xmesh, self.Ymesh)
    
    def _set_U(self):
        """ set U0, U1, 2-dimensional; overrides UCA1D._set_U() """
        self.X_U0   = np.empty((self.Ny), dtype=object)
        self.X_U1   = np.empty((self.Ny), dtype=object)
        self.Y_U0   = np.empty((self.Nx), dtype=object)
        self.Y_U1   = np.empty((self.Nx), dtype=object)
        T, X, V, A  = self.theta, self.X, self.V, self.A 
        for y in range(self.Ny):
            T1  = T if isinstance(T, float) else lambda x: T(x,y)
            X1  = X if X is None            else lambda x: X(x,y)
            V1  = V if V is None            else lambda x: V(x,y)
            A1  = A if A is None            else lambda x: A(x,y)[0]
            self.X_U0[y], self.X_U1[y] = self._get_U(self.Nx, T1, X1, V1, A1)
        for x in range(self.Nx):         
            T1  = T if isinstance(T, float) else lambda y: T(x,y)
            X1  = X if X is None            else lambda y: X(x,y)
            V1  = V if V is None            else lambda y: V(x,y)
            A1  = A if A is None            else lambda y: A(x,y)[1]
            self.Y_U0[x], self.Y_U1[x] = self._get_U(self.Ny, T1, X1, V1, A1) 
        self.set_XVA()

    def set_probe_boxes(self, boxes):
        """ boxes is a list of box, 4-tuples (xl, xh, yl, yh). 
            During UCA evolution abs(Psi)**2 is accumulated per cell per box,
        """
        self.probe_boxes_user = boxes
        self.probe_boxes      = []        
        count   = 0
        if self.theta is None:
            raise ValueError('Hamiltonian must be set before probe boxes')       
        for b in boxes:
            if not b in ('left', 'right', 'top', 'bottom'):
                self.probe_boxes.append(b) 
            else:
                box = [0, self.Nx, 0, self.Ny]
                Xis =  not self.X is None
                if   b=='left'  : 
                    if Xis and self.X(box[0],1): box[0] +=1
                    box[1] = box[0]+1
                elif b=='right' : 
                    if Xis and self.X(box[1],1): box[1] -=1
                    box[0] = box[1]-1
                elif b=='bottom': 
                    if Xis and self.X(1,box[2]): box[2] +=1
                    box[3] = box[2]+1
                elif b=='top'   : 
                    if Xis and self.X(1,box[3]): box[3] -=1
                    box[2] = box[3]-1
                count  += 1
                self.probe_boxes.append(box) 
        if count>2:
            raise ValueError('more than 2 probes not supported')
                            
    def _set_cmap(self):
        self.vmax   = None
        self.cmap_real  = cm.bwr.copy()     # blue-white-red
        self.cmap   = cm.Paired.copy()
        self.cmap.set_over('white')     # white for high; looks better
        self.vmax   = np.max(self._Psi_abs())/2
        self.tcolor = 'gray'
            
    def _ax_plot(self, ax, Psi, real, imag, time, is_RHS, title='time'):
        """ plot a single axis with Psi;
            used by UCA1D .plot() for (multi-panel) 2D plots,
            used by UCA2D .animate() for 2D animations
        """
        self._set_cmap()                    
        ax.set_aspect('equal')              # square pixels
        ax.invert_yaxis()                   # origin in lower-left corner
        if title=='time':
            subtitle = 'time='+str(time)
            ax.set_title(subtitle, fontsize=UCA1D._fontsize_subtitle)   
        Psi2    = self._Psi_view(Psi[0], real=real)
        Psi_min = np.min(Psi2)
        cmap = self.cmap_real if real else self.cmap
        vmax = None if real else self.vmax
        ax.imshow(Psi2, origin='lower', interpolation='gaussian',
                  cmap=cmap, vmax=vmax)
        if type(self.theta) is types.LambdaType:
            ax.contour(self.Xmesh, self.Ymesh, self.thetaZ, 
                       colors=UCA1D._color_theta, linewidths=0.5)
        if not self.V is None:              # add self.V contours
            ax.contour(self.Xmesh, self.Ymesh, self.VZ, 
                       colors=UCA1D._color_V, linewidths=0.5)
        if not self.A is None:              # add self.A streamplot            
            density = 180/max(self.Nx, self.Ny)
            ax.streamplot(self.Xmesh, self.Ymesh, self.Ax, self.Ay,
                          broken_streamlines=False, density=density,
                          linewidth=1.0, color=UCA1D._color_A) 
        if not self.X is None:              # add self.X "overlay"
            ax.contour(self.Xmesh, self.Ymesh, self.XZ, 
                       colors=UCA1D._color_X, linewidths=1)            
        if self.log_trace_max and self.time>0:
            N_frame = len(self.trace_max)*time//self.time
            if N_frame>0:
                p,Y,X = list(zip(*self.trace_max[:N_frame]))
                ax.plot(X, Y, color=UCA1D._color_Psi_trace) 
        ax.use_sticky_edges = True
                
    def _run_step_probes(self):
        """ accumulate state in probe boxes """
        for boxi, box in enumerate(self.probe_boxes):
            Psi_box = self.Psi[0][box[2]:box[3], box[0]:box[1]]
            self.Psi_probes[boxi] += np.abs(Psi_box)**2
 
    def _run_step(self, T, cpf=1):
        """ advances the UCA state by one step; 
            used both by run() and by annimate(); overrides UCA1D._run_step() """
        c = 0
        while self.time<T and c<cpf:
            c += 1
            PsiN = np.empty(self.Psi[0,:].shape, dtype=complex)
            for y in range(self.Ny):         # apply H-rule to rows
                Psi     = self.Psi[0][y,:]
                U0, U1  = self.X_U0[y], self.X_U1[y]
                Psi     = self._update(Psi, PsiN, U0[0], U1[0])
                self.Psi[0][y,:] = Psi
            PsiN = np.empty(self.Psi[:,0].shape, dtype=complex)
            for x in range(self.Nx):         # apply V-rule to columns
                Psi     = self.Psi[0][:,x]
                U0, U1  = self.Y_U0[x], self.Y_U1[x]
                Psi     = self._update(Psi, PsiN, U0[0], U1[0])
                self.Psi[0][:,x] = Psi   
            self._run_step_inc()
        
    def _get_animate_probes(self, ax, ax_probes, probe_max):
        """ prepare probe screens """
        if 'left'   in ax_probes: 
            ax.yaxis.set_tick_params(labelleft=False)
        if 'bottom' in ax_probes: 
            ax.xaxis.set_tick_params(labelbottom=False)    
        if len(ax_probes)>0:
            # Set aspect of the main Axes.
            ax.set_aspect(1.)
            # create new Axes on the left and on the right of the current Axes
            divider     = make_axes_locatable(ax)
        ax_probe0, ax_probe1 = None, None
        for i, axp in enumerate(ax_probes): # max two
            LR      = ax_probes[i] in ('left', 'right')
            sharex, sharey = (None, ax) if LR else (ax, None)
            # below width/height and pad are in inches
            ax_probe= divider.append_axes(ax_probes[i], 1, pad=0.1, 
                                          sharex=sharex, sharey=sharey)
            if LR:                          # x-axes
                ax_probe.set_xlim([0, probe_max])
                ax_probe.set_xticks([])
                if ax_probes[i]=='right':
                    ax_probe.yaxis.set_tick_params(labelleft=False)
            else:                           # y-axes
                ax_probe.set_ylim([0, probe_max])
                ax_probe.set_yticks([])
                if ax_probes[i]=='top':
                    ax_probe.xaxis.set_tick_params(labelbottom=False)
            if i==0: ax_probe0 = ax_probe   
            else   : ax_probe1 = ax_probe
        return ax_probe0, ax_probe1         # possibly None
   
    def animate(self, T, cpf=1, real=False, probe_max=0,
                Psi_save=1, pickle_results=True, save=False):        
        """ create an animation of the evolution of the UCA for T time steps
            - cpf     : number of cycles (steps) per video frame
            - real             : show Psi.real else: Psi.abs squared
            - if probe_max>0   : animate probe boxes, using probe_max as x/ylim
            - integer cpf      : the number of cycles (steps) per video frame
            - Psi_save         : save Psi_save states at regular intervals  
            - save             : save annimation as mp4
            - if pickle_results: reuse or update pickled html
        """                
        def get_Psi_probe_vals(side):
            pbi = self.probe_boxes_user.index(side)
            box = self.Psi_probes[pbi]      # select from probe_boxes
            return box[0,:] if box.shape[0]==1 else box[:,0]
                                        
        def animate_step(i):
            """ relies on non-local side0, Y_probe0, pb0, etc """
            im.set_data(self._Psi_view(self.Psi[0], real=real))
            time_txt.set_text(self.timer)
            if self.first_frame:
                self.first_frame = False
            else:
                self._run_step(T, cpf=cpf)
            if ax_trace is None and ax_probe0 is None and ax_probe1 is None:
                return im, time_txt,
            elif not ax_trace is None:
                l   = len(self.trace_max)
                M   = 3
                MM  = 2*M+1
                if l>=MM:                # smoothen path over MM values
                    s   = tuple(sum(x)/MM for x in zip(*self.trace_max[l-MM:l])) 
                    self.trace_max[l-M-1] = s
                p,Y,X = list(zip(*self.trace_max))
                pt.set_data(X,Y)  
                return im, time_txt, pt,              
            else:  
                vals= get_Psi_probe_vals(side0)
                if side0 in ('left', 'bottom'):
                    vals= probe_max-vals            # mirror
                X,Y = ((vals, Y_probe0) if ax_probes[0] in ('left', 'right') else
                       (Y_probe0, vals))              
                pb0.set_data(X,Y)  
                if ax_probe1 is None:
                    return im, time_txt, pb0, 
                else: 
                    vals= get_Psi_probe_vals(side1)
                    if side1 in ('left', 'bottom'):
                        vals= probe_max-vals        # mirror
                    X,Y = ((vals, Y_probe1) if ax_probes[1] in ('left', 'right') else
                           (Y_probe1, vals))
                    pb1.set_data(X,Y)
                    return im, time_txt, pb0, pb1,
                    
        self.pickle_results = pickle_results
        if self.pickle_results:
            self._run_pickle_open()
        if (not save and self.pickle_results and os.path.isfile(self.html_pkl_name)):
            self.reset()
            pkl_file = open(self.html_pkl_name, 'rb')
            html_obj = pickle.load(pkl_file)
            if os.path.isfile(self.Psi_pkl_name):   
                self._run_pickle_load_psi()
        else:
            self._run_init(T=T, Psi_save=Psi_save)
            fig     = plt.figure(figsize=(8, 4.5))      # aim at 16x9 frames
            ax      = fig.add_subplot(111)
            imag    = False   
            self._ax_plot(ax, self.Psi, real, imag, time=-1, is_RHS=True)
            tx, ty  = (0.02, 0.02)
            time_txt= ax.text(tx, ty, '', horizontalalignment='left',
                            color=self.tcolor, fontsize=12, 
                            verticalalignment='bottom', transform=ax.transAxes)
            time_txt.set_text('time = 0')
            im = plt.imshow(self._Psi_view(self.Psi[0], real=real), animated=True,
                            origin='lower', interpolation='gaussian', 
                            cmap=self.cmap, vmax=self.vmax)
            ax_title = ax                   # default axis to attach title
            ax_trace, ax_probe0, ax_probe1 = None, None, None
            if self.log_trace_max:
                ax_trace= ax
                p, Y, X = list(zip(*self.trace_max))
                pt,     = ax_trace.plot(Y, X, color=UCA1D._color_Psi_trace) 
            elif probe_max>0:               # animate probe_boxes 
                ax_probes = [b for b in self.probe_boxes_user 
                             if b in ('left', 'right', 'top', 'bottom')]
                ax_probe0, ax_probe1 = \
                    self._get_animate_probes(ax, ax_probes, probe_max) 
                if not ax_probe0 is None:
                    side0   = ax_probes[0]
                    if side0=='top': ax_title = ax_probe0
                    vals    = get_Psi_probe_vals(side0)
                    Y_probe0= np.arange(0, len(vals))
                    pb0,    = ax_probe0.plot(vals, Y_probe0, color=UCA1D._color_Psi2)
                if not ax_probe1 is None:
                    side1   = ax_probes[1]
                    if side1=='top': ax_title = ax_probe1
                    vals    = get_Psi_probe_vals(side1)
                    Y_probe1= np.arange(0, len(vals))
                    pb1,    = ax_probe1.plot(Y_probe1, vals, color=UCA1D._color_Psi2) 
            ax_title.set_title(self.title, fontsize=UCA1D._fontsize_title)
            if not self.UCA_settings is None: 
                self.UCA_settings['timer'] = True
            self.first_frame= True
            ani= animation.FuncAnimation(fig, animate_step, frames=T//cpf+2,
                                         interval=100, blit=True)
            plt.close()
            if save:           
                html_obj= None
                writer  = animation.writers['ffmpeg'](fps=10, bitrate=1e6)
                os.makedirs(UCA1D._UCA_mp4, exist_ok=True)
                ani.save(UCA1D._UCA_mp4+self.name+'.mp4', writer=writer, dpi=240)
            else: 
                html_obj = HTML(ani.to_jshtml(default_mode='once'))
                if self.pickle_results:
                    pkl_file = open(self.html_pkl_name, 'wb')
                    pickle.dump(html_obj, pkl_file)
        if self.pickle_results and not os.path.isfile(self.Psi_pkl_name):
            self._run_pickle_close()
        return html_obj
 