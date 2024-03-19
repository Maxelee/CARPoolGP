CARPoolGP Theory
=====

CARPoolGP builds upon this general framework of Gaussian process
regression (see https://arxiv.org/abs/2209.08940 for a review) while borrowing the key principle of the CARPool method to
reduce variance in a predicted quantity
(see https://arxiv.org/abs/2009.08970). 


CARPool achieves this through an experimental design that leverages correlations between ‘expensive’
*base* samples and ‘cheap’ *surrogate* samples. In CARPool, numerous
inexpensive surrogate samples can be generated at arbitrarily low costs
to estimate the surrogate quantity's mean accurately. The
correlation between surrogates and a few base samples is then leveraged
to apply this accurate estimate to the base quantity. A reduced variance
and accurate expression of some mean quantity can be constructed without
the need for many expensive samples. 

The novelty of CARPoolGP is in applying this reduced variance procedure
across an entire parameter space. In CARPoolGP, base samples provide
parameter space coverage at more parameter space locations but are
expected to be riddled with high variance. Surrogates, however, can be
generated with the same or different process as the base (e.g. different
cost), and are generated at an alternate set of locations that we call
*parameter islands*. Parameter islands contain multiple surrogates,
providing a reduced variance estimate at these locations. By correlating
isolated base samples to surrogates, the parameter space can be sampled
more densely while borrowing the variance reduction achieved at
parameter islands.

.. image:: CARPoolGP_cartoon.pdf
   :width: 600

We present a simplistic picture of the CARPoolGP sampling method In the above figure. 
We show three base samples
spread through a one-dimensional parameter space with some associated
noisy quantity :math:`\tilde{Q}(\theta_i)`. Surrogate samples live on
the parameter island, shown as the black bar, located at
:math:`\theta^S`. A base-surrogate pair, which has correlated sample
noise realizations, is shown by samples with matching numbers. The
colors show how surrogates from the parameter island can maintain some
conception of their variation at the parameter island (e.g., red is too
high, blue is too low) and broadcast this to the base samples.



