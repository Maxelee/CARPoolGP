"""
This code is meant to develop the CARPool kernels that are associated with arxiv:..... 

The kernels interact generally with tinygp (Foreman-Mackey - ). This is the backend we use to write the kernels. 

We need a kernel V, for smooth variation in Q
We need a kernel W, for the smooth variation in R
We need a kernel Isigma, for the noise variations in Q
We need a kernel E, for the designed noise variations in R
We need a kernel X, for the cross correlation in Q and R
We need a kernel M, for the cross correlation in the noise variations between Q and R

We then want a block kernel [[V, X],[X^T, W]] for the smooth variation and 
We then want a block kernel [[Isigma, M], [M^T, E]] for the noise variations
"""
from tinygp import kernels
from tinygp.kernels.distance import Distance, L1Distance, L2Distance
import jax.numpy as jnp
import jax
jax.config.update("jax_enable_x64", True)


class VWKernel(kernels.Kernel):
    """
    Custom kernel for carpool that can take N-dimensional scale. This is realy just a squared 
    exponential kernel
    """
    def __init__(self, amp, scale):
        self.scale = jnp.atleast_1d(scale)
        self.amp   = jnp.atleast_1d(amp)

    def evaluate(self, X1, X2):
        x = jnp.atleast_1d(jnp.sqrt((X2 - X1)**2))
        # return jnp.prod(self.amp * jnp.exp(-0.5 * x**2 / self.scale**2))
        return jnp.prod(self.amp * jnp.exp(-0.5 * x**2/self.scale**2))
    
class WKernel(kernels.Kernel):
    """
    Custom kernel for carpool that can take N-dimensional scale. This is realy just a squared 
    exponential kernel
    """
    def __init__(self, amp, scale):
        self.scale = jnp.atleast_1d(scale)
        self.amp   = jnp.atleast_1d(amp)

    def evaluate(self, X1, X2):
        x = jnp.atleast_1d(jnp.sqrt((X2 - X1)**2))
        arg = jnp.sqrt(3) + x/self.scale
        return jnp.prod( (1 + arg)*jnp.exp(-arg))
    
class XKernel(kernels.Kernel):
    """
    Custom kernel for carpool that can take N-dimensional scale
    """
    def __init__(self, amp, scale, deltaP):
        self.scale   =jnp.atleast_1d(scale)
        self.deltaP  =jnp.atleast_1d(deltaP)
        self.amp     = jnp.atleast_1d(amp)

    def evaluate(self, X1, X2):
        x = jnp.atleast_1d(jnp.sqrt((X2 - X1)**2))
        # return jnp.prod(self.amp*jnp.exp(-0.5 * (x**2 + self.deltaP)/self.scale))
        return jnp.prod(self.amp * jnp.exp(-0.5 * (x**2 + self.deltaP)/self.scale**2))
    
# class XKernel(kernels.Kernel):
#     """
#     Custom kernel for carpool that can take N-dimensional scale
#     """
#     def __init__(self, amp, scale, deltaP):
#         self.scale   = jnp.atleast_1d(scale)
#         self.deltaP  = jnp.atleast_1d(deltaP)
#         self.amp     = jnp.atleast_1d(amp)

#     def evaluate(self, X1, X2):
#         x = jnp.atleast_1d(jnp.sqrt((X2 - X1)**2))
#         arg = jnp.sqrt(3) + (x**2 + self.deltaP)/self.scale**2
#         return jnp.prod(self.amp*(1 + arg)*jnp.exp(-arg))
    
class EKernel(kernels.Kernel):
    """
    Custom kernel for carpool that can take N-dimensional scale. This is realy just a linear 
    exponential kernel
    """
    def __init__(self, scale):
        self.scale = jnp.atleast_1d(scale)
        
    def evaluate(self, X1, X2):
        x = jnp.atleast_1d(jnp.sqrt((X2 - X1)**2))
        return jnp.prod(jnp.exp(-0.5 * x/ self.scale**2))
    

