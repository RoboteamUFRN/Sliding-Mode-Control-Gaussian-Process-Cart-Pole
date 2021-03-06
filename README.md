# Sliding-Mode-Control-Gaussian-Process-Cart-Pole
Sliding Mode Control with uncertainty compensation using Gaussian Process Regression for underactuated mechanical systems.

Gabriel S. Lima, gabriel.lima.095@ufrn.edu.br, 
Wallace M. Bessa, wmobes@utu.fi

In this project you can find the:
(a) codes used for implementating the proposed controller for a cart-pole system;
(b) and all the results obtained by this implementation considering the proposed and conventional approaches.

The columns of the results, results_smc, results_track, and results_smc_track files are divided as follows:
[1] time;
[2] x position;
[3] x desired position;
[4] angular position;
[5] angular desired position;
[6] control action;
[7] sliding variable;
[8] desired output for the GP distribution;
[9] mean of the GP distribution;
[10] variance of the GP distribution.

The columns of the results_dist_cart and results_dist_cart_track files are divided as follows:
[1] input of the distribution;
[2] posterior mean;
[3] posterior variance.

The Armadillo library [1] is needed for the implementation.

[1] Sanderson, C. and Curtin, R., 2016. Armadillo: a template-based C++ library for linear algebra. Journal of Open Source Software, 1(2), p.26.
