---
layout: page
title:  ""
categories: jekyll update
permalink: /note/
mathjax: true
---
Consider a three-phase inverter driving an AC synchronous motor, with input power $$P=IV$$ where $$I$$ is the input DC current and $$V$$ is the DC voltage magnitude of the DC link. Assuming no inverter losses, then (instantaneous) AC three-phase power can be written
$$p_a+p_b+p_c=v_ai_a+v_bi_b+v_ci_c$$
where $$a,b,c$$ are phases and $$v_a,v_b,v_c$$ are phase voltages and $$i_a,i_b,i_c$$ are phase currents. Now, these are AC quantities, so lets assume sinusoidal steady-state with frequency $$\omega$$ and a phase lag between $$v_a,i_a$$ of angle $$\phi$$. Then, we write
$$\begin{align}p_a+p_b+p_c&=\hat{V}\hat{I}\cos(\omega t)\cos(\omega t-\phi)+\hat{V}\hat{I}\cos(\omega t-120^{\circ})\cos(\omega t-120^{\circ}-\phi)\\&+\hat{V}\hat{I}\cos(\omega t-240^{\circ})\cos(\omega t-240^{\circ}-\phi)\\&=\frac{3\hat{V}\hat{I}\cos(\phi)}{2}\end{align}$$

where $$\hat{V},\hat{I}$$ are amplitudes of phase voltages and currents.

Now, with no inverter losses we have that  $$P=p_a+p_b+p_c$$. Hence

$$IV=\frac{3\hat{V}\hat{I}\cos(\phi)}{2}$$

Furthermore, assume that the phase voltage amplitude is controlled by PWM to produce a sinusoidal current .....

FINALLY I GOT WHAT I WAS MISINTERPRETING. THE PWM IS ALREADY REGULATING FOR THE COSINES !!!