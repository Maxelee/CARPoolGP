Tutorials
=====

.. container:: cell code

   .. code:: python

      import jax.numpy as jnp
      import numpy as np
      import matplotlib.pyplot as plt
      import scienceplots
      plt.style.use(['science', 'no-latex', 'notebook', 'grid'])

.. container:: cell markdown

   .. rubric:: Welcome to CARPoolGP!
      :name: welcome-to-carpoolgp

.. container:: cell markdown

   This is an emulator developed in **Zooming by in the CARPoolGP lane:
   new CAMELS-TNG simulations of zoomed-in massive halos**. The goal of
   CARPoolGP is to emulate a value of parameter space where we are
   limited to the number of samples we can obtain at each parameter
   space location. There is a general structure to CARPoolGP which I
   will outline below, we can use this as a guide throughout this
   notebook.

.. container:: cell markdown

   #. Generate a set of parameters, :math:`B`, at which to extract Base
      sample quantities :math:`\tilde{Q}`.
   #. Generate a set :math:`S` of Parameter Islands, :math:`\theta_{S}`
      at which to extract Surrogate sample quantities,
      :math:`\tilde{Q}^S`. Ensure that Base-Surrogate pairs have some
      level of correlation between them.
   #. Determine the noise kernels and associated hyperparameters for
      Base, Surrogate, and Combined quantities: :math:`C_{ij}`,
      :math:`D_{ij}`, :math:`X_{ij}`.
   #. Maximize the likelihood function to obtain the optimal set of
      hyperparameters: :math:`\hat{\tau}`.
   #. Emulate to find $ Q(\\theta_{p, i}^\\ast)$, and
      :math:`\sigma^2(\theta_{p, i}^\ast, \theta_{p,j}^\ast)`.

.. container:: cell markdown

   .. rubric:: A Toy model to get started
      :name: a-toy-model-to-get-started

.. container:: cell markdown

   Before even starting with CARPoolGP we need to outline some of the
   preliminary information... like the model that we are going to use.
   In practice we are not lucky enough to know the model, otherwise that
   would defeat the whole purpose... But here as a toy example to get us
   started we can make this simple.

   We consider some mean quantity, $ Q$, which has a functional form,

   \\begin{equation} \\begin{split} Q(\\theta_{i}) &= a \\theta_{i} +
   b\\theta_{i}^3\\sin(\\theta_{i})\\ \\tilde{Q}(\\theta_{i}) &=
   Q(\\theta_{i}) + \\epsilon, \\end{split} \\end{equation}

   Where a and b are some constants, and :math:`\theta_i` is our
   independent parameter, and :math:`\epsilon` is gaussian noise where
   we take :math:`\epsilon` to have :math:`\langle\epsilon\rangle=0` and
   :math:`\langle\epsilon^2\rangle=\sigma_Q^2(\theta_{i})`. We choose
   the dimensionality of our parameter to be 1. We define this model
   below

.. container:: cell code

   .. code:: python

      def get_Q(theta, A=-0.02, B=0.19):
          """
          This is the model which we ultimitely want to predict, represents the 
          smooth variation on some function. 
          """
          return (A * theta  +  theta**3*B**4 * np.sin(theta))

.. container:: cell markdown

   We need to define some global variables that we will use through out
   the toy model as well.

.. container:: cell code

   .. code:: python

      sigma_Q  = 0.1       # jitter ontop of smooth variation
      N        =50         # Number of Base data samples
      Domain   = [-10,10]  # Domain of parameters
      LR       = 1e-2      # Optimization learning rate 
      ITERS    = 5000      # maximum no of iteration

.. container:: cell markdown

   Lets make a set of "True" values that we can test our emulator
   against in the future. This will also show us how the functoin moves
   through spae

.. container:: cell code

   .. code:: python

      # underlying truth: 
      plt.figure(figsize=(12, 6))
      theta=np.linspace(*np.array(Domain)+0.001, 1001, endpoint=True)
      Y = get_Q(theta) 
      plt.plot(theta, Y, 'k--', label='Truth')
      plt.xlabel(r'$\theta$')
      plt.ylabel(r'$Q$');

   .. container:: output display_data

      .. image:: 15cd34a90a316fc60a9e4c9c95eaff9d3d2ce59f.png

.. container:: cell markdown

   .. rubric:: 1. Generate a set of parameters, :math:`B`, at which to
      extract Base sample quantities :math:`\tilde{Q}`
      :name: 1-generate-a-set-of-parameters-b-at-which-to-extract-base-sample-quantities-tildeq

.. container:: cell markdown

   We build the Base quantities by sampling a uniformly random
   distribution within the domain :math:`[-10, 10]` to generate a set of
   parameters :math:`B\equiv\{{\bf \theta}_{i}| i=1, 2, ..., N\}`, and
   quantities, :math:`\tilde{Q}({\theta}_i)`. We choose :math:`N=50` and
   draw :math:`\epsilon_{Q,i}` from a normal distribution with
   :math:`\langle\epsilon_{Q}\rangle = 0` and
   :math:`\langle\epsilon_Q^2\rangle = \sigma_Q^2`.

.. container:: cell code

   .. code:: python

      def model_data(theta, seed=1993, noise=None):
          """
          Args:
              theta (array) : the set of parameters B
              noise (array)  : an array of noise values to add to the raw Q
              seed (int) : seed used to generate random numbers

          Returns:
              \tilde{Q}: noisey data
              noise : noise added to data. 
          """
          Q = get_Q(theta)
          if noise is not None:
              return Q + noise
          np.random.seed(seed)
          noise = np.random.normal(0, sigma_Q, len(theta))
          return Q+noise, noise

.. container:: cell code

   .. code:: python

      # Sample Data
      np.random.seed(194)
      theta_B = np.random.uniform(Domain[0], Domain[1], N)
      Q_B, intrinsic_noise = model_data(theta_B, 194, None)
      plt.plot(theta, Y, 'k--', label='Truth')
      plt.plot(theta_B, Q_B, '.', label='Base')
      plt.legend();

   .. container:: output display_data

      .. image:: 8de8b253c36b46a7cbcd0f63c1aed40716a39cfe.png

.. container:: cell markdown

   .. rubric:: 2. Generate a set :math:`S` of Parameter Islands,
      :math:`\theta_{S}` at which to extract Surrogate sample
      quantities, :math:`\tilde{Q}^S`. Ensure that Base-Surrogate pairs
      have some level of correlation between them.
      :name: 2-generate-a-set-s-of-parameter-islands-theta_s-at-which-to-extract-surrogate-sample-quantities-tildeqs-ensure-that-base-surrogate-pairs-have-some-level-of-correlation-between-them

.. container:: cell markdown

   We then generate parameter islands in the set
   :math:`S\equiv\{\theta_{i}| i=1, 2, ..., N_S\}` by linearly spacing
   :math:`N_S=5` points in the range :math:`[-8, 8]` with the same
   process as defined above. For each base sample, the island closest to
   the parameter is identified, and a surrogate sample is drawn at this
   island location, :math:`\theta_i`, to generate
   :math:`\tilde{Q}^S(\theta_i)` where the noise, :math:`\epsilon_s`, is
   perfectly correlated with the noise of the base simulation (i.e., the
   same amplitude of the noise is used
   :math:`\epsilon_{s, i} = \epsilon{i}`).

.. container:: cell code

   .. code:: python

      def match_surrogates(theta_Q, intrinsic_noise, Groups, f=0.0):
          """
          Generate the correlated surrogate samples given the set of groups, the set of base samples and the noise

          Args:
              theta_Q (array): base samples
              intrinsic_noise (array): noise associated with tilde{Q}
              Groups (int): number of groups
              f (float: _description_. Defaults to 0.

          Returns:
              theta_S, and S
          """
          Surrogate_locs = np.linspace(-8, 8, Groups, endpoint=True)
          nearest_island = np.zeros_like(theta_Q)
          for i, pi in enumerate(theta_Q):
              nearest_island[i] = Surrogate_locs[np.argmin((pi - Surrogate_locs)**2)]
          S_raw = get_Q(nearest_island)
          S = S_raw + intrinsic_noise*(1-f) + f*np.random.normal(0, sigma_Q, len(S_raw))
          return nearest_island, S

.. container:: cell code

   .. code:: python

      theta_S, Q_S = match_surrogates(theta_B, intrinsic_noise, Groups=5, f=0)

.. container:: cell code

   .. code:: python

      fig, axs = plt.subplots(ncols=1, sharex=True, sharey=True, figsize=(12,6), gridspec_kw={'wspace':0.02})
      axs.plot(theta_B, Q_B, 'o', color='r', markersize=5, label='Base Samples')
      axs.plot(theta_S, Q_S, 's', color='b', label='Surrogate Samples')
      axs.plot(theta, Y, 'k', label='True Variation')
      axs.set_title('CARPoolGP Sampling')
      axs.set_xlabel(r'$\theta$')
      axs.set_ylabel('$Q$')
      axs.legend(ncol=2, loc='upper center');

   .. container:: output display_data

      .. image:: bf6f18a661b92e39ec304e5e3bddc88c4a6dcfa8.png

.. container:: cell markdown

   .. rubric:: 3. Determine the noise kernels and associated
      hyperparameters for Base, Surrogate, and Combined quantities:
      :math:`C_{ij}`, :math:`D_{ij}`, :math:`X_{ij}`.
      :name: 3-determine-the-noise-kernels-and-associated-hyperparameters-for-base-surrogate-and-combined-quantities-c_ij-d_ij-x_ij

.. container:: cell markdown

   For both base and surrogate samples, we use a radial basis function
   defined in Eq.~\\ref{eq:rbf} to describe the smooth varying component
   of the covariance. Base and surrogates are drawn from the same
   underlying process and with the same level of sample variance, so the
   hyperparameters, :math:`\bm{\tau}`, are shared across both matrices.

   \\begin{equation} \\begin{split} V_{ij} = \\alpha\\exp\\left(-\\gamma
   , d_E(\\theta_{i} - \\theta_{j})^2\\right)\\ W_{ij} =
   \\alpha\\exp\\left(-\\gamma , d_E(\\theta_{i} -
   \\theta_{j})^2\\right). \\end{split} \\end{equation}

   The only difference between the two matrices is the parameters that
   are used to generate them. :math:`V`, uses the base samples, while
   :math:`W` uses the surrogate samples. The full covariance for the
   base samples and the surrogate samples can be written following
   covariance functions

   \\begin{equation} \\begin{split} C_{ij} &=
   \\alpha\\exp\\left(-\\gamma , d_E(\\theta_{i} -
   \\theta_{j})^2\\right) + \\sigma_Q^2\\mathcal{I}\\ D_{ij} &=
   \\alpha\\exp\\left(-\\gamma , d_E(\\theta_{i} -
   \\theta_{j})^2\\right) + \\sigma_Q^2\\mathcal{I}. \\end{split}
   \\end{equation}

   We choose the kernel that describes the smooth covariance between the
   base and surrogate samples to be an RBF, but we set the additional
   parameter, :math:`\Delta q_{BS} =0`, as the processes between the
   base and surrogates are the same. We use the same scale and amplitude
   parameters for the :math:`V_{ij}` and :math:`W_{ij}` matrices to
   define the covariance between base and surrogate samples,

   | \\begin{equation} Y_{ij} = \\alpha\\exp\\left(-\\gamma ,
     \\left(d_E(\\theta_{i}, \\theta_{j})^2\\right)\\right).\\
   | \\end{equation}

   To relate the base samples to the surrogates, we use the fact that we
   have set a perfect correlation between the sample fluctuations and,
   therefore, set the :math:`M` matrix to \\begin{equation} M_{ij} =
   \\sigma_Q^2\\delta_{ij}, \\end{equation} where the
   :math:`\delta_{ij}` is a delta function that is :math:`1` at
   locations of base-surrogate pairs, and :math:`0` elsewhere. Recall
   that the distance between parameter space locations in :math:`Y_{ij}`
   and :math:`M_{ij}` are evaluated between base and surrogate samples.
   Following Eq.~\\ref{eq:cov_X}, we then have \\begin{equation}
   \\begin{split} X_{ij} = &\\alpha\\exp\\left(-\\gamma ,
   \\left(d_E(\\theta_{i} - \\theta_{j})^2\\right)\\right) +
   \\sigma_Q^2.\\ \\end{split} \\end{equation} We can now build the
   block covariance matrix containing all of these components following
   Eq.~\\ref{eq:sigma} where :math:`\bm{\tau}` is the vector of
   hyperparameters, :math:`\bm{\tau}=(\alpha, \gamma, \sigma^2_Q)`

.. container:: cell markdown

   All of these kernels are taken care of internally by CARPoolGP. So
   what we have to do is define the simulations and surrogates which we
   can do as follows:

.. container:: cell code

   .. code:: python

      from src import CARPoolSimulations
      from src import CARPoolEmulator


      # Create simulation objects and surrogate objects
      sims = CARPoolSimulations.Simulation()
      surrs = CARPoolSimulations.Simulation()

      # Set the parameters and quantities for these simulations
      sims.parameters  = theta_B  ;  sims.quantities  = Q_B
      surrs.parameters = theta_S  ;  surrs.quantities = Q_S

.. container:: cell markdown

   .. rubric:: 4. Maximize the likelihood function (minimize inverse
      Wishart function) to obtain the optimal set of hyperparameters:
      :math:`\hat{\tau}`.
      :name: 4-maximize-the-likelihood-function-minimize-inverse-wishart-function-to-obtain-the-optimal-set-of-hyperparameters-hattau

.. container:: cell markdown

   We use the Gaussian likelihood function as defined below and choose
   uninformative priors for :math:`\mu_B` and :math:`\mu_S`, but allow
   them to be learned as additional hyperparameters in the regression.
   We then minimize the negative log of the likelihood function to
   obtain an optimal set of hyperparameters, :math:`\hat{\tau}` using
   Stochastic Gradient Descent (SGD).

   \\begin{equation} \\begin{split} \\mathcal{L}(\\tau) =
   &\\frac{1}{(2\\pi)^{N/2}} \|\\Sigma(\\tau)|^{-1/2},\\times
   \\exp\\left(-\\frac{1}{2}\\begin{pmatrix} \\tilde{Q}-\\mu_Q\\
   \\tilde{Q}^S-\\mu_S \\end{pmatrix}^T
   \\Sigma(\\tau)^{-1}\\begin{pmatrix} \\tilde{Q}-\\mu_Q\\
   \\tilde{Q}^S-\\mu_S \\end{pmatrix}\\right) \\end{split}
   \\end{equation}

.. container:: cell code

   .. code:: python

      #Build an emulator object (this generates the kernels which you can find in the CARPoolKernels file) 
      emu = CARPoolEmulator.Emulator(sims, surrs)

      params = {"log_scaleV":3.0, "log_ampV":0.1,
                "log_scaleM":1.0,  "log_jitterV":-1.0, "log_mean":0.0}

      # Train the emulator
      best_params = emu.train(params, learning_rate=0.01, max_iterations=ITERS)

.. container:: cell code

   .. code:: python

      plt.semilogx(np.diff(emu.losses))
      plt.xlabel('Iterations')
      plt.ylabel(r'$\Delta\mathcal{L}$');

   .. container:: output display_data

      .. image:: 7354b0b696bff5df5cf064e142a2bfa6c51acf0d.png

.. container:: cell markdown

   .. rubric:: 5. Emulate to find $ Q(\\theta_{p, i}')$, and
      :math:`\sigma^2(\theta_{p, i}', \theta_{p,j}')`.
      :name: 5-emulate-to-find--qtheta_p-i-and-sigma2theta_p-i-theta_pj

.. container:: cell markdown

   We now have all we need to perform an emulation at sample points from
   the set :math:`T` using:

.. container:: cell markdown

   \\begin{equation} \\begin{split} Q(\\theta_{p, i}') & =
   \\text{K}\ *s(\\hat{\\tau}), \\Sigma^{-1}*\ {ij}(\\hat{\\tau})
   \\begin{pmatrix} \\tilde{Q}-\\mu_Q\\ \\tilde{Q}^S-\\mu_S
   \\end{pmatrix} + \\begin{pmatrix} \\mu_Q\\ \\mu_S \\end{pmatrix}\\
   \\sigma^2(\\theta_{p, i}', \\theta_{p, j}') & =
   \\text{K}_{tt}(\\hat{\\tau}) -
   \\text{K}\ *t(\\hat{\\tau})\\Sigma*\ {ij}^{-1}(\\hat{\\tau})\\text{K}_t^T(\\hat{\\tau})
   \\end{split} \\end{equation}

.. container:: cell markdown

   With

   \\begin{equation} \\begin{split} \\text{K}\ *t(\\hat{\\tau}) &=
   \\Sigma(\\theta*\ {p,i}', \\theta_{p,j} ; \\hat{\\tau})\\
   \\text{K}\ *{tt}(\\hat{\\tau}) &= \\begin{pmatrix}
   V(\\theta*\ {p,i}', \\theta_{p,j}' ; \\hat{\\tau}) ,,,
   Y(\\theta_{p,i}', \\theta_{p,j}' ; \\hat{\\tau}) \\
   Y^T(\\theta_{p,i}', \\theta_{p,j}' ; \\hat{\\tau}) ,
   W(\\theta_{p,i}', \\theta_{p,j}' ; \\hat{\\tau}) \\end{pmatrix}\\
   \\end{split} \\end{equation}

.. container:: cell markdown

   All of this is taken care by the package

.. container:: cell code

   .. code:: python

      # now emulate! 
      pred_mean, pred_var = emu.predict(theta)

.. container:: cell code

   .. code:: python

      fig, axs = plt.subplots(nrows=2, sharex=True, figsize=(12, 8))
      std = np.sqrt(np.diag(pred_var))
      axs[0].fill_between(theta, pred_mean - 2*std, pred_mean+2*std, label='98% Confidence interval')
      axs[0].plot(theta, Y, 'k', label='True Evolution')

      axs[1].fill_between(theta, (pred_mean-Y) - 2*std, (pred_mean-Y)+2*std, label='98% Confidence interval')
      axs[1].plot(theta, (pred_mean - Y), 'k', label='Residual')
      axs[0].set_ylabel('Q')
      axs[1].set_xlabel(r'$\theta$')
      axs[1].set_ylabel(r'$Q_{\rm pred} - Q_{\rm True}$')
      axs[0].legend();

   .. container:: output display_data

      .. image:: ca612b0fda49340e82b8858935a30a0151ae95e2.png

.. container:: cell markdown

   .. rubric:: Active learning with CARPoolGP
      :name: active-learning-with-carpoolgp

.. container:: cell markdown

   We introduce an active learning method to predict the best next
   places to sample in parameter space. This is all taken care of in
   CARPoolGP!

.. container:: cell code

   .. code:: python

      from src import CARPoolEmulator

      # Generate an active learning model
      model = CARPoolEmulator.ActiveLearning(sims, surrs, theta, Domain[0], Domain[1])

      # Initialize the training
      best_params = model.train(params, learning_rate=LR, max_iterations=ITERS)

.. container:: cell code

   .. code:: python

      # Run an active learning step to find the next state (Ngrid is for 2**N)
      num_new = 10 # Number of new points to sample
      Ngrid   = 7  # The number of locations to test at in base 2, (eg, 2^7)
      next_thetas, next_surrogates = model.active_learning_step(num_new=10, Ngrid=7, normalize=False)

.. container:: cell code

   .. code:: python

      print('Next base samples:', [i[0] for i in next_thetas])
      print('Next surrogate samples:', [i[0] for i in next_surrogates])

   .. container:: output stream stdout

      ::

         Next base samples: [-5.848567672073841, 9.901670515537262, 2.133758831769228, -6.054342966526747, -2.0871826633810997, -5.965835005044937, 2.1017765067517757, -2.064566109329462, -6.006976924836636, 5.986908171325922]
         Next surrogate samples: [-4.0, 8.0, 4.0, -8.0, -4.0, -4.0, 4.0, -4.0, -8.0, 4.0]
