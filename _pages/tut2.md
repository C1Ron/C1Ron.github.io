---
layout: post
title:  ""
categories: jekyll update
mathjax: true
permalink: /tut2/
---
## Linear System Simulation using Python
This post will cover how to simulate a linear system using `Python`.

### Mathematical model
The system will a basic DC motor with an attached load, commonly represented as
![dcmotor]({{site.baseurl}}/images/dcmotor.jpg)
and in block diagram form with a simple P position controller
![dcmotor2]({{site.baseurl}}/images/dcmotor2.jpg)
We have Kirchoffs voltage law for the electrical system: 

$$ v(t)=v_R(t)+v_L(t)+e(t)$$

where $$v$$ is the supply voltage, $$v_R$$ and $$v_L$$ are voltages across the armature resistor and inductor, and $$e$$ is the back-emf.

We have Newton's 2nd law for the mechanical system:

$$ J_m\alpha(t)=T_{em}(t)-T_{f}(t)-T_L(t) $$

where $$J_m$$ and $$\alpha$$ are the moment of inertia and acceleration of the shaft. Furthermore $$T_{em}$$, $$T_f$$, $$T_L$$ are the electromagnetic, friction and load torques, respectively.

Our two systems get coupled electromechanically by: 
<ul>
<li>torque is propertional to armature current:  \(\hspace{0.5cm}T_{em}(t)=K_i\cdot i(t)\)</li>
<li>back-emf is proportional to motor speed:   \(\hspace{0.7cm}e(t)=K_e \cdot\omega(t)\)</li>
</ul>
where $$K_e$$ and $$K_i$$ are motor constants and $$i$$ and $$\omega$$ are motor current and speed. <br/>
Also the voltage drops across the circuit elements can be written $$v_R=R_a\cdot i$$ and $$v_L=L_a\cdot di/dt$$.

Finally, the friction is assumed to be a viscous friction: $$T_f=b\cdot\omega$$, always working against the speed.

The coupled system can be written:

$$\begin{align}
v(t)&=R_ai(t)+L_a\frac{di}{dt}+K_e\frac{d\theta}{dt}\\
J_m\frac{d^2\theta}{dt^2} &= K_ii(t)-b\frac{d\theta}{dt}-T_L
\end{align}$$

For position control using a linear state-space representation $$\dot{\mathbf{x}}=\mathbf{A}\mathbf{x}+\mathbf{B}\mathbf{u}$$ we use state variables $$\mathbf{x}=\begin{bmatrix}\theta\\ \dot{\theta}\\ i\end{bmatrix}$$, 
and input vector $$\mathbf{u}=\begin{bmatrix}0\\T_L\\v\end{bmatrix}$$ where $$v$$ is the SISO system voltage input and $$T_L$$ is the load torque.

The matrices turns out to be

$$\mathbf{A}=\begin{bmatrix}0&1&0\\0&-b/J_m&K_i/J_m\\0&-K_e/L_a&-R_a/L_a\end{bmatrix}
\hspace{1cm} \mathbf{B}=\begin{bmatrix}0&0&0\\0&-1/J_m&0\\0&0&1/L_a\end{bmatrix}$$

To test the model we simulate with a simple P controller $$v=K_P\cdot(\theta_r-\theta)$$ and neglect saturation of the system input.
The motor parameters are randomly chosen without care.

### Simulation using Python
We simply show the script 
{% highlight python %}
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint 
plt.close('all')
plt.rcParams["axes.labelsize"] = 18
plt.rc('xtick', labelsize=18)
plt.rc('ytick', labelsize=18)
# Equation of motion
def eom(x,t,Jm,Ra,La,Ke,Ki,b,Kp,T):
    A = np.array([[0, 1, 0],[0, -b/Jm, Ki/Jm], [0, -Ke/La, -Ra/La]])
    B = np.array([[0, 0, 0], [0, -1/Jm, 0], [0, 0, 1/La]])
    if t < 0.5:
        theta_ref = 10*t
    else:
        theta_ref = 5
    u = np.array([0,T,Kp*(theta_ref - x[0])])
    return A.dot(x) + B.dot(u)

# Parameter definitions
Jm = 1e-1
b = 10
Ra = 30
La = 2
Ke = 10
Ki = 10

Kp = 1000
T = 50

# Time integration
x0 = np.array([0,0,0])
t = np.linspace(0,1,1000)
sol = odeint(eom, x0, t, args=(Jm,Ra,La,Ke,Ki,b,Kp,T))

# Visualizing results
theta_ref = [10*x if x < 0.5 else 5 for x in t]
fig = plt.figure(figsize=[12,7])
ax1 = fig.add_subplot(211)
ax1.plot(t, sol[:,0], t, theta_ref, '--')
ax1.autoscale(enable=True, axis='both', tight=True)
ax1.set_xlabel('time')
ax1.set_ylabel('rad')
ax1.legend(['theta','theta_ref'], fontsize=18, loc='center right')
ax1.grid()

ax2 = fig.add_subplot(212)
ax2.plot(t, sol[:,1])
ax2.autoscale(enable=True, axis='both', tight=True)
ax2.set_xlabel('time')
ax2.set_ylabel('rad/s')
ax2.legend(['theta_dot'], fontsize=18)
ax2.grid()
{% endhighlight %}
and the resulting plot
![dcmotor3]({{site.baseurl}}/images/dcmotor3.jpg)