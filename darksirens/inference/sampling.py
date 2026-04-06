import numpy as np
import jax
import jax.numpy as jnp

def run_sampler(method, likelihood, prior_transform, labels,
                lower_bound, upper_bound, opts):
    """
    method: "jaxns", "dynesty", or "emcee"
    likelihood: function(coord) -> logL
    prior_transform: maps unit cube -> parameter space
    labels: list of parameter names
    lower_bound, upper_bound: arrays
    opts: argparse namespace

    Returns a dict:
        {
            "samples": array of shape (Nsamp, ndim),
            "logZ": float or None,
            "logZerr": float or None
        }
    """

    ndims = len(labels)

    # --------------------------------------------------------
    # JAXNS
    # --------------------------------------------------------
    if method == "jaxns":
        import tensorflow_probability.substrates.jax as tfp
        tfpd = tfp.distributions
        from jaxns import NestedSampler
        from jaxns.framework.model import Model
        from jaxns.framework.prior import Prior

        # Prior model: returns a vector theta of shape (ndim,)
        def prior_model():
            params = []
            for i, name in enumerate(labels):
                low = float(lower_bound[i])
                high = float(upper_bound[i])
                x = yield Prior(tfpd.Uniform(low=low, high=high), name=name)
                params.append(x)
            return jnp.stack(params)

        def log_likelihood(theta):
            return likelihood(jnp.asarray(theta))

        model = Model(
            prior_model=prior_model,
            log_likelihood=log_likelihood,
        )

        ns = NestedSampler(
            model=model,
            num_live_points=opts.nlive,
            max_samples=opts.max_samples,
            verbose=True,
        )

        key = jax.random.PRNGKey(opts.seed)
        term, state = ns(key)
        results = ns.to_results(term, state)

        posterior = results.samples  # dict of arrays
        samples = jnp.column_stack([posterior[name] for name in labels])

        return {
            "samples": np.asarray(samples),
            "logZ": None,        # JAXNS evidence not extracted here
            "logZerr": None
        }

    # --------------------------------------------------------
    # dynesty
    # --------------------------------------------------------
    elif method == "dynesty":
        from dynesty import NestedSampler
        from dynesty.utils import resample_equal

        sampler = NestedSampler(
            likelihood, prior_transform, ndims,
            bound="multi", sample="rwalk",
            nlive=opts.nlive
        )
        sampler.run_nested(dlogz=0.1)
        res = sampler.results

        # Posterior samples
        weights = np.exp(res["logwt"] - res["logz"][-1])
        samples = resample_equal(res.samples, weights)

        # Evidence
        logZ = float(res.logz[-1])
        logZerr = float(res.logzerr[-1])

        return {
            "samples": np.asarray(samples),
            "logZ": logZ,
            "logZerr": logZerr
        }

    # --------------------------------------------------------
    # emcee
    # --------------------------------------------------------
    elif method == "emcee":
        import emcee

        def log_prob(coord):
            if np.any(coord < lower_bound) or np.any(coord > upper_bound):
                return -np.inf
            return likelihood(coord)

        p0 = np.random.uniform(lower_bound, upper_bound,
                               size=(opts.nwalkers, ndims))

        sampler = emcee.EnsembleSampler(
            opts.nwalkers, ndims, log_prob,
            moves=[(emcee.moves.DEMove(), 0.8),
                   (emcee.moves.DESnookerMove(), 0.2)]
        )
        sampler.run_mcmc(p0, opts.nsteps, progress=True)

        chain = sampler.flatchain
        samples = chain[len(chain)//2:]

        return {
            "samples": np.asarray(samples),
            "logZ": None,        # emcee does not compute evidence
            "logZerr": None
        }

    else:
        raise ValueError(f"Unknown sampler: {method}")
