"""
Gaussian Process wrapper using GPyTorch.

Provides a simplified interface for GP regression with RBF kernel,
automatic hyperparameter optimization, and prior specification.
"""
import os
import math
import torch


import gpytorch

from matplotlib import pyplot as plt


# Numerical-stability floor for the GP likelihood noise. By default the noise
# hyperparameter is fixed here and not learned (raw_noise.requires_grad is
# disabled below); pass learn_noise=True to GP() to let it be optimized instead.
NOISE_FLOOR = 1e-4

# Initial noise variance used when learn_noise=True. Must be strictly above
# NOISE_FLOOR: initializing exactly at the GreaterThan(NOISE_FLOOR) constraint
# boundary maps to raw_noise = -inf under the softplus transform, which has
# zero gradient everywhere and silently prevents the noise from ever moving
# even though requires_grad is True.
NOISE_INIT_LEARNABLE = 1e-2


class GP(gpytorch.models.ExactGP):
    """
    Gaussian Process regression model with RBF kernel.
    
    Wraps GPyTorch's ExactGP with convenient initialization and
    hyperparameter optimization methods. Supports ARD (Automatic
    Relevance Determination) and custom priors.
    
    Parameters
    ----------
    train_x : torch.Tensor
        Training inputs of shape (n_samples, n_features).
    train_y : torch.Tensor
        Training outputs of shape (n_samples,).
    initialization : dict, optional
        Initial hyperparameter values. Keys: 'likelihood.noise_covar.noise',
        'covar_module.base_kernel.lengthscale', 'covar_module.outputscale',
        'mean_module.constant'.
    prior : dict, optional
        Prior distributions. Keys: 'lengthscale', 'outputscale', 'noise_std'.
        Values should be GPyTorch prior objects (e.g., GammaPrior).
    ard : bool, default=True
        Use Automatic Relevance Determination (different lengthscale per dimension).
    fix_mean_at : float, optional
        Fix the mean function at a specific value.
    learn_noise : bool, default=False
        If True, the likelihood noise is optimized by optimize_hyperparameters
        instead of being frozen at NOISE_FLOOR.

    Attributes
    ----------
    mean_module : gpytorch.means.ConstantMean
        Constant mean function.
    covar_module : gpytorch.kernels.ScaleKernel
        Scaled RBF kernel with optional ARD.
    
    Notes
    -----
    - Noise variance constrained to be >= 1e-3 to avoid numerical issues
    - Lengthscale constrained to be >= 5e-2
    - Default initialization: noise=0.001, lengthscale=prior.mean, outputscale=prior.mean
    """
    def __init__(
        self,
        train_x,
        train_y,
        initialization=None,
        prior=None,
        ard=True,
        fix_mean_at=None,
        learn_noise=False,
    ):
        # if ard, ard_num_dims = dim, use the different lengthscales for different input dimensions
        # else, ard_num_dims = None, use the same lengthscale for all input dimensions
        # NOTE: the likelihood noise variance should be initialized with a small value!
        #       since if it is large, the GP tends to learn a constant function
        # initialization = {
        #     'likelihood.noise_covar.noise': torch.tensor(1.),
        #     'covar_module.base_kernel.lengthscale': torch.tensor(0.5),
        #     'covar_module.outputscale': torch.tensor(2.),
        #     'mean_module.constant': torch.tensor(2.)
        # }
        # prior = {'lengthscale': gpytorch.priors.GammaPrior(3.0, 6.0),
        #          'outputscale': gpytorch.priors.GammaPrior(2.0, 0.15),
        #          'noise_std': gpytorch.priors.NormalPrior(0., 1.)}
        likelihood = gpytorch.likelihoods.GaussianLikelihood(
            noise_constraint=gpytorch.constraints.GreaterThan(NOISE_FLOOR),
        )
        # print(
            # "gp.py: WARNING: avoid numerical issue by seeting noise constraint:",
            # likelihood.noise_covar.raw_noise_constraint,
        # )

        super(GP, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()

        input_dim = train_x.shape[1]
        assert input_dim > 0

        ard_num_dims = input_dim if ard else None

        if prior is None:
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(ard_num_dims=ard_num_dims)
            )
        else:
            self.likelihood.noise_covar.register_prior(
                "noise_std_prior",
                prior["noise_std"],
                lambda module: module.noise.sqrt(),
            )

            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(
                    ard_num_dims=ard_num_dims, lengthscale_prior=prior["lengthscale"]
                ),
                outputscale_prior=prior["outputscale"],
            )
            self.covar_module.base_kernel.register_constraint(
                "raw_lengthscale", gpytorch.constraints.GreaterThan(5e-2)
            )


        # Initialize lengthscale and outputscale to mean of priors
        if initialization is None:
            if ard:
                if prior:
                    if (
                        len(prior["lengthscale"].mean.shape) == 0
                        or prior["lengthscale"].mean.squeeze().shape[0] == 1
                    ):
                        init_lengthscale = torch.squeeze(
                            prior["lengthscale"].mean
                        ) * torch.ones(1, input_dim)
                    else:
                        init_lengthscale = prior["lengthscale"].mean
                else:
                    init_lengthscale = torch.ones(1, ard_num_dims)

                self.covar_module.base_kernel.lengthscale = init_lengthscale
            else:
                self.covar_module.base_kernel.lengthscale = (
                    prior["lengthscale"].mean if prior else 1.0
                )

            self.covar_module.outputscale = (
                prior["outputscale"].mean if prior is not None else 1.0
            )
            # Initialize the noise variance. When learning noise, start strictly
            # above NOISE_FLOOR (see NOISE_INIT_LEARNABLE) so gradients can flow;
            # otherwise pin it at the floor and freeze it.
            if learn_noise:
                self.likelihood.noise_covar.noise = NOISE_INIT_LEARNABLE
            else:
                self.likelihood.noise_covar.noise = NOISE_FLOOR
                # Freeze: treat the underlying function as deterministic.
                self.likelihood.noise_covar.raw_noise.requires_grad_(False)
            # Initialize the constant mean
            self.mean_module.constant = 0.0

            # # to fix the mean function, e.g., used in the constraint
            if fix_mean_at is not None:
                self.mean_module.constant = fix_mean_at
                self.mean_module.constant.requires_grad = False

        else:
            self.initialize(**initialization)
            if not learn_noise:
                self.likelihood.noise_covar.raw_noise.requires_grad_(False)
            elif not torch.isfinite(self.likelihood.noise_covar.raw_noise).all():
                # Loaded noise sat exactly on the constraint boundary (see
                # NOISE_INIT_LEARNABLE comment above) -- nudge off it so it's
                # actually learnable.
                self.likelihood.noise_covar.noise = NOISE_INIT_LEARNABLE

        print("All constraints:")
        for constraint_name, constraint in self.named_constraints():
            print(
                f"Constraint name: {constraint_name:55} constraint = {constraint}"
            )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    @staticmethod
    def get_default_hyperparameter_prior(function_name="gausscurve"):
        # large lengthscale (12,6) -> more correlated
        # small lengthscale (3,6) -> less correlated
        return {
            # "lengthscale": gpytorch.priors.GammaPrior(3.0, 6.0),  # (1.0,2.0)
            # "lengthscale": gpytorch.priors.GammaPrior(3.0, 9.0),
            "lengthscale": gpytorch.priors.GammaPrior(0.25, 0.5), # 0.0625, 0.25
            "outputscale": gpytorch.priors.GammaPrior(2.0, 0.15),
            "noise_std": gpytorch.priors.NormalPrior(0.0, 0.1),  # (0., 1.)
            # "noise_std": gpytorch.priors.NormalPrior(0.1, 0.001), # (0., 1.)
        }

    def save(self, path="model_state.pth"):
        torch.save(self.state_dict(), path)

    def plot1d(self, ax, x):
        assert x.shape[1] == 1

        f_preds = GP.predict_f(self, x)

        with torch.no_grad():
            f_means = f_preds.mean
            f_vars = f_preds.variance
            f_stds = torch.sqrt(f_vars)

        ax.fill_between(
            x.squeeze(),
            f_means - f_stds,
            f_means + f_stds,
            alpha=0.5,
        )
        ax.plot(x.squeeze(), f_means)

        ax.scatter(self.train_inputs[0].squeeze(), self.train_targets.squeeze())
        return ax

    @staticmethod
    def load(model, path="model_state.pth"):
        state_dict = torch.load(path)
        model.load_state_dict(state_dict)

    @staticmethod
    def optimize_hyperparameters(
        model, train_x, train_y, learning_rate=0.1, training_iter=50, verbose=True
    ):
        model.train()

        # Use the adam optimizer
        optimizer = torch.optim.Adam(
            model.parameters(), lr=learning_rate
        )  # Includes GaussianLikelihood parameters

        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)

        # Allow Cholesky to add larger jitter before giving up. Default is
        # start=1e-6, max_tries=3 → cap ~1e-4, which is too tight when the
        # frozen noise floor is also 1e-4 and ARD lengthscales shrink.
        with gpytorch.settings.cholesky_jitter(1e-4), gpytorch.settings.cholesky_max_tries(6):
            for i in range(training_iter):
                # Zero gradients from previous iteration
                optimizer.zero_grad()
                # Output from model
                output = model(train_x)
                # Calc loss and backprop gradients
                loss = -mll(output, train_y)
                loss.backward()
                if verbose:
                    print(
                        f"Iter {i+1}/{training_iter} - Loss: {loss.item():.3f} lengthscale: {model.covar_module.base_kernel.lengthscale}  noise: {model.likelihood.noise}"
                    )
                optimizer.step()

    @staticmethod
    def predict_f(model, test_x):
        with torch.no_grad(), gpytorch.settings.cholesky_jitter(1e-4), gpytorch.settings.cholesky_max_tries(6):
            model.eval()
            return model(test_x)

    @staticmethod
    def predict_y(model, test_x):
        with torch.no_grad(), gpytorch.settings.cholesky_jitter(1e-4), gpytorch.settings.cholesky_max_tries(6):
            model.eval()
            return model.likelihood(model(test_x))
