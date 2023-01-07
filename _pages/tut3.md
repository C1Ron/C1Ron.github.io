---
layout: post
title:  ""
categories: jekyll update
mathjax: true
permalink: /tut3/
---
## Cart-Pole Modeling and LQR using MATLAB
This post will share how to model a cart-pole and simulate the dynamics using `MATLAB`.
The goal is to first do this in `MATLAB` and later to do the same steps using `Python` and also to do the animation using `Blender`.

### Mathematical Model
The system consists of a frictionless cart and a pendulum fixed to the centroid of the cart body.   
The cart is actuated by a force $$F$$. The system and a free-body-diagram of the two bodies are shown:
![cart-pole2]({{site.baseurl}}/images/cart-pole2.jpg)
We denote the position of $$m_1$$ by $$(x,y)$$ and use $$\theta$$ as the pendulum angle relative to static equilibrium.
First we express the two vectors and take their time-derivatives for later use

$$\mathbf{r}_1=\begin{bmatrix}x\\y\end{bmatrix}, \hspace{1cm} \mathbf{r}_2=\mathbf{r}_1+\ell\begin{bmatrix}\sin(\theta)\\-\cos(\theta)\end{bmatrix} $$

We assume $$\dot{y}=\ddot{y}=0$$, i.e. that the vertical component of $$m_1$$ is constant:

$$\begin{align}\dot{\mathbf{r}}_1 &= \begin{bmatrix}\dot{x}\\0\end{bmatrix}, \hspace{1cm} \dot{\mathbf{r}}_2=\dot{\mathbf{r}}_1+\ell\dot{\theta}\begin{bmatrix}\cos(\theta)\\\sin(\theta)\end{bmatrix}\\
\ddot{\mathbf{r}}_1 &= \begin{bmatrix}\ddot{x}\\0\end{bmatrix}, \hspace{1cm} \ddot{\mathbf{r}}_2=\ddot{\mathbf{r}}_1 + \ell\ddot{\theta}\begin{bmatrix}\cos(\theta)\\\sin(\theta)\end{bmatrix} + \ell\dot{\theta}^2\begin{bmatrix}-\sin(\theta)\\\cos(\theta)\end{bmatrix}\end{align}$$

Following the FBD we find the translational equations of motion for body 1:

$$m_1\ddot{\mathbf{r}}_1 = \mathbf{T} + \begin{bmatrix}F\\N-m_1g\end{bmatrix}$$

Note that $$\mathbf{T}=\begin{bmatrix}T_x&T_y\end{bmatrix}^T$$ are the reaction forces in the pendulum joint. Consider the other FBD to find the translational equations of motion for body 2:

$$m_2\ddot{\mathbf{r}}_2=-\mathbf{T}+\begin{bmatrix}0\\-m_2g\end{bmatrix}$$

Finally, we consider the rotational equation of motion for body 2, about the pivot-point $$O$$. Note that the point $$O$$ is accelerating with acceleration $$\ddot{x}$$. Also, note that positive moment is in the CCW direction:

$$J_O\ddot{\theta} = m_2\ell^2\ddot{\theta}= -m_2g\ell\sin(\theta) - m_2\ddot{x}\ell\cos(\theta)$$

The vector equations can be written

$$m_1\begin{bmatrix}\ddot{x}\\0\end{bmatrix} = \begin{bmatrix}T_x\\T_y\end{bmatrix}+\begin{bmatrix}F\\N-m_1g\end{bmatrix}$$

and

$$m_2\begin{bmatrix}\ddot{x} + \ell\ddot{\theta}\cos(\theta)-\ell\dot{\theta}^2\sin(\theta)\\\ell\ddot{\theta}\sin(\theta)+\ell\dot{\theta}^2\cos(\theta)\end{bmatrix} = -\begin{bmatrix}T_x\\T_y\end{bmatrix}+\begin{bmatrix}0\\-m_2g\end{bmatrix}$$

The point now is to get two differential equations that relates the two state variables, $$x$$ and $$\theta$$ and the input force $$F$$.
In this pursuit, we don't need the vertical component of neither vector equation. Furthermore, we use the first equation to express the unknown reaction force $$T_x$$ as

$$m_1\ddot{x}=F+T_x \hspace{0.5cm} \Longrightarrow \hspace{0.5cm}T_x=m_1\ddot{x}-F$$

Then we end up with two governing equations of motion:

$$\begin{align} m_1\ddot{x}+m_2\ddot{x}+m_2\ell\ddot{\theta}\cos(\theta)&=F+m_2\ell\dot{\theta}^2\sin(\theta)\\
m_2\ell^2\ddot{\theta} + m_2\ell\ddot{x}\cos(\theta) &= -m_2g\ell\sin(\theta)\end{align}$$

In matrix form:

$$\begin{bmatrix}m_1+m_2&m_2\ell\cos(\theta)\\m_2\ell\cos(\theta)&m_2\ell^2\end{bmatrix}\begin{bmatrix}\ddot{x}\\\ddot{\theta}\end{bmatrix}=\begin{bmatrix}F\\0\end{bmatrix}+\begin{bmatrix}m_2\ell\dot{\theta}^2\sin(\theta)\\-m_2g\ell\sin(\theta)\end{bmatrix}$$

Finally, we want a system of the form $$\dot{\mathbf{x}}=\mathbf{f}(\mathbf{x},\mathbf{u})$$ where $$\mathbf{x}$$ is a state vector, $$\mathbf{f}$$ is some nonlinear vector function and $$\mathbf{u}$$ is the input for our system. In this regard, we let $$\mathbf{x}=\begin{bmatrix}x&\theta & \dot{x}&\dot{\theta}\end{bmatrix}^T$$ and $$\mathbf{u}=\begin{bmatrix}F&0\end{bmatrix}^T$$. The system can then be written as

$$\dot{\mathbf{x}}=\begin{bmatrix}\dot{x}\\\dot{\theta}\\\mathbf{M}^{-1}(\mathbf{u}+\mathbf{v})\end{bmatrix}$$

### Linearization and State-Feedback
To employ LQR in feedback control, we need a linear system $$\dot{\mathbf{x}}=\mathbf{A}\mathbf{x}+\mathbf{B}\mathbf{u}$$.
We linearize about the unstable fixed point $$\mathbf{x}_e=\begin{bmatrix}x&\pi&0&0\end{bmatrix}^T$$ and $$\mathbf{u}_e=\mathbf{0}$$. 
The matrices $$\mathbf{A}$$ and $$\mathbf{B}$$ can be determined analytically by

$$\mathbf{A}=\frac{\partial\mathbf{f}}{\partial\mathbf{x}}\Biggr\rvert_{\mathbf{x}=\mathbf{x}_e, \mathbf{u}=\mathbf{u}_e}, 
\hspace{1cm}\mathbf{B}=\frac{\partial \mathbf{f}}{\partial\mathbf{u}}\Biggr\rvert_{\mathbf{x}=\mathbf{x}_e,\mathbf{u}=\mathbf{u}_e}$$

However, this requires us to solve the matrix equation of motion algebraically for $$\ddot{x}$$ and $$\ddot{\theta}$$, typically using Cramer's rule.
To avoid that we use the symbolic toolbox of MATLAB. The following section shows MATLAB code to simulate the nonlinear system.
Also, we linearize the system using the `jacobian()` function of the symbolic toolbox in MATLAB, 
hence finding the matrices $$\mathbf{A}$$ and $$\mathbf{B}$$ which are required for state-feedback control.
### Equations of motion and Animation
The following code-snippet shows how to specify the nonlinear equations of motion in a form that MATLAB's `ode` suite will accept the function.
There are two functions in this snippet: one for the cart-pole model and one for animation. 
**Note that you should place these in the bottom of the script**.
{% highlight MATLAB %}
%% Equations of motion in MATLAB ODE's notation
function xxdot = model(t, xx, u, params)
    % States
    x = xx(1);
    theta = xx(2);
    xdot = xx(3);
    thetadot = xx(4);
    % Parameters
    m1 = params.m1;
    m2 = params.m2;
    L = params.L;
    g = params.g;
    % Model
    M = [m1 + m2, m2*L*cos(theta); m2*L*cos(theta), m2*L^2];
    v = [m2*thetadot^2*sin(theta); -m2*g*L*sin(theta)];
    xxdot = [xdot; thetadot; M \ (u + v)];
end

%% Simple animation
function anim(t,xx,params,b)
    plot(xx(1)+params.L*sin(xx(2)), b/2-params.L*cos(xx(2)), 'ko')
    line([xx(1)-b/2; xx(1)+b/2], [0;0])
    line([xx(1)+b/2; xx(1)+b/2], [0,b])
    line([xx(1)+b/2; xx(1)-b/2], [b,b])
    line([xx(1)-b/2; xx(1)-b/2], [0,b])
    line([xx(1), xx(1)+params.L*sin(xx(2))], [b/2,b/2-params.L*cos(xx(2))])
end
{% endhighlight %}

### Testing the Model
First we test the behaviour of the system without any controller. We let the pendulum go from a horizontal position $$\theta=\pi/2$$
and simulate the system for 15 sec. We plot the response and look at the animation.
{% highlight MATLAB %}
clear; close all; clc;
% Parameter-struct
params = struct;
params.m1 = 5;
params.m2 = 2;
params.L = 2;
params.g = 9.81;
% Input vector (e.g. feedback control vector)
u = [0;0];
% Simulation settings
tspan = linspace(0,15,5000);
xx0 = [0; pi/2; 0; 0];
[t, xx] = ode45(@(t,xx) model(t, xx, u, params), tspan, xx0);

%% Plot results and animate
fig1 = figure(1);
subplot(211);
p1 = plot(t,xx(:, [1,3]));
title('Cart translation');
legend('x', 'xdot');
subplot(212);
p2 = plot(t, xx(:, [2,4]));
title('Pole rotation');
legend('theta', 'thetadot');
% Animate
fig3 = figure(3);
for i = 1:numel(t)
    if mod(i,10) == 0
        figure(fig3)
        clf
        anim(t(i), xx(i,:), params, 1)
        axis([-3,3,-3,3])
    end
end

{% endhighlight %}
![cart-pole-plot1]({{site.baseurl}}/images/cart-pole-plot1.jpg)
Keep in mind that there is no friction included in this model when looking at the animation.
From my personal intuition, it seems to match expected physical behaviour considering the length of the arm and the relative mass of the bodies.
### Implementing Feedback Control
Now we linearize the system and employ LQR to find the state feedback control $$\mathbf{u}=-\mathbf{K}(\mathbf{x}-\mathbf{x}_e)$$.
Recall that our problem is to apply a force on the cart such that the cart moves in such a way that the pendulum swings from $$\theta=0$$ to $$\theta=\pi$$. 
{% highlight MATLAB %}
%% Linearize using Symbolic Toolbox
xsym = sym('x', [4,1]);
Fsym = sym('F');
tsym = sym('t');
usym = [Fsym; 0];
A = jacobian(model(tsym, xsym, usym, params), xsym);
B = jacobian(model(tsym, xsym, usym, params), Fsym);
A = double(subs(A, xsym, [xsym(1); pi; 0; 0]));
B = double(subs(B, xsym, [xsym(1); pi; 0; 0]));
clear xsym Fsym tsym usym

%% State feedback control with LQR
Q = diag([500,50,1,1]);
R = 0.1;
K = lqr(A,B,Q,R);
% Stabilize about the (unstable) fixed-point [0; pi; 0; 0]
u = @(xx) - K * (xx - [0; pi; 0; 0]); 
xx0 = [0;0;0;0];
[t,xx] = ode45(@(t,xx) model(t, xx, u(xx), params), tspan, xx0);

%% Plot results and animate
fig2 = figure(2);
subplot(211);
p1 = plotyy(t,xx(:, 1), t, xx(:,3));
title('Cart translation');
legend('x', 'xdot');
subplot(212);
p2 = plotyy(t, xx(:,2), t, xx(:,4));
title('Pole rotation');
legend('theta', 'thetadot');
line([0,15],[pi,pi],'linestyle','--','color','black')
% Animate
fig4 = figure(4);
for i = 1:numel(t)
    if mod(i, 20) == 0
        figure(fig4)
        clf
        anim(t(i), xx(i,:), params, 1)
        axis([-5,5,-5,5])
        axis equal
        drawnow
    end
end


{% endhighlight %}

![cart-pole-plot2]({{site.baseurl}}/images/cart-pole-plot2.jpg)
Recall that the system is underactuated. I.e. can only reach the uncontrollable state $$\theta$$ indirectly via the cart-force $$F$$.
The cart accelerates to the right, overshooting the desired payload angle, and comes back to stabilize it at $$\theta=\pi$$ near the desired cart-position.
Note that the figures above have two vertical axes, with position along the left axis and velocity along the right.
