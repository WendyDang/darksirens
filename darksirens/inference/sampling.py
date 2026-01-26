# sampling.py
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
    """

    ndims = len(labels)
    samples = None

    # --------------------------------------------------------
    # JAXNS
    # --------------------------------------------------------
    if method == "jaxns":
        import tensorflow_probability.substrates.jax as tfp
        tfpd = tfp.distributions
        from jaxns import NestedSampler
        from jaxns.framework.model import Model
        from jaxns.framework.prior import Prior

        # Build prior model
        def prior_model():
            for i, name in enumerate(labels):
                low = lower_bound[i]
                high = upper_bound[i]
                yield Prior(tfpd.Uniform(low=low, high=high), name=name)

        # Likelihood wrapper
        def log_likelihood(*coords):
            coord = jnp.array(coords)
            return likelihood(coord)

        model = Model(prior_model=prior_model,
                      log_likelihood=log_likelihood)

        ns = NestedSampler(model=model,
                           num_live_points=opts.nlive,
                           max_samples=opts.max_samples,
                           verbose=True)

        key = jax.random.PRNGKey(opts.seed)
        term, state = ns(key)
        results = ns.to_results(term, state)

        posterior = results.samples_x
        samples = jnp.column_stack([posterior[name] for name in labels])

    # --------------------------------------------------------
    # dynesty
    # --------------------------------------------------------
    elif method == "dynesty":
        from dynesty import NestedSampler
        from dynesty.utils import resample_equal

        sampler = NestedSampler(likelihood, prior_transform, ndims,
                                bound="multi", sample="rwalk",
                                nlive=opts.nlive)
        sampler.run_nested(dlogz=0.1)
        res = sampler.results

        weights = np.exp(res["logwt"] - res["logz"][-1])
        samples = resample_equal(res.samples, weights)

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

    else:
        raise ValueError(f"Unknown sampler: {method}")

    return np.asarray(samples)
