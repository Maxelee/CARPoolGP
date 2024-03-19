Welcome to CARPoolGP documentation! 
===================================

CARPoolGP is an sampling and regression technique developed in https://arxiv.org/abs/2403.10609 . The basic idea, is that when we can force correlations between samples in parameter space, we can reduce variance on emulated quantities. CARPoolGP leverages the CARPool method of https://arxiv.org/abs/2009.08970 and Gaussian process regression. 

CARPoolGP can be used:

    1. To emulate a quantity throughout some parameter space given preexisting samples
    2. Learn the best place in parameter space to generate new samples at (Active Learning)

We provide here a tutorial with a one dimensional toy example, an application using simulations from GZ here, and a an application to emulate profiles again using the simulations of:


If using in your own work, please :doc:`cite` our work! 



.. note::

   This project is under active development.

Contents
--------
See the :doc:`installation` section for details on getting started with CARPoolGP. To find a brief description of the theoretical framework for CARPoolGP see :doc:`theory`. We include tutorials in the :doc:`tutorial` section. 

.. toctree::

   installation
   theory
   tutorial
   contact
