#!/usr/bin/env python

import numpy as np
from seb.plot import Plot3D, Plot, Container, Animation

def grad_descent(x, y, dfnx, dfny, alpha=0.2, length=50):
    trace = [(x, y)]
    for _ in range(length):
        x = x - alpha * dfnx(x)
        y = y - alpha * dfny(y)
        trace.append((x, y))
    return np.array(trace), (x, y)

if __name__ == '__main__':
    point_considered = -36
    x_init = -1.9
    y_init = -1
    x = np.linspace(-7, 7, 50)

    # 3D example
    fn = lambda x, y: -np.sin(x / 2.0) + y**2
    dfnx = lambda x: -0.5 * np.cos(x/2.0)
    dfny = lambda y: 2*y
    fig3d = Plot3D()
    fig3d.surface(x, np.cos(x + 0.5), fn)
    # fig3d.projection(x, np.cos(x + 0.5), fn)
    fig3d.set_camera(45, 66)
    fig3d.set_axis('x axis', 'y axis', 'z axis')

    trace, (x_final, y_final) = grad_descent(x_init, y_init, dfnx, dfny)
    fig3d.scatter(x=[trace[point_considered, 0], ], 
                  y=[trace[point_considered, 1], ], 
                  z=fn, 
                  s=350.0, label='Trust Region')
    fig3d.plot(x=trace[:, 0], y=trace[:, 1], z=fn, label='Trajectory')
    fig3d.save('trpo3d.png')

    # 1D Example
    fig1d = Plot()
    trace = trace[:-15]
    point_considered = point_considered + 15
    z = 10 * np.array([fn(a[0], a[1]) for a in trace])
    iterations = np.arange(len(trace))
    fig1d.circle(x=iterations[point_considered], y=z[point_considered], radius=1.0)
    fig1d.plot(x=iterations, y=z, label='True Loss')
    fig1d.scatter(x=[iterations[point_considered], ], y=[z[point_considered], ], label='Current params', s=10.0)
    fig1d.annotate('Trust Region', (18, 17), (15, 5), rad=0.3)
    fig1d.set_axis('Parameters', 'Cost')
    # Hypothetical curves
    x_trunc = iterations[point_considered:]
    z_trunc = z[point_considered:]
    z2 = [z_trunc[0] + np.sin((a - z_trunc[0])) for a in z_trunc]
    fig1d.plot(x=x_trunc, y=z2)
    z2 = [z_trunc[0] + np.sin((a - z_trunc[0])) for a in z_trunc]
    fig1d.plot(x=x_trunc, y=z2)
    z3 = [z_trunc[0] + 2*(a - z_trunc[0]) for a in z_trunc]
    fig1d.plot(x=x_trunc, y=z3)
    fig1d.save('conv.png')

    cont = Container(1, 2)
    cont.set_plot(0, 0, fig3d)
    cont.set_plot(0, 1, fig1d)
    cont.save('full.png')

    # anim = Animation()
    # fig3d.canvas.axis('off')
    # anim.rotate_3d(fig3d)
    # anim.save('trpo3d.gif')
    
