import numpy as np
import jax
import jax.numpy as jnp

def run_sampler(method, likelihood, prior_transform, labels,
                lower_bound, upper_bound, opts):
    """
    method: "jaxns", "dynesty", or "emcee"
    likelihood: function(coord) -> logL (expects 1D array)
    prior_transform: maps unit cube -> parameter space (expects 1D array)
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

        # 1. JIT compile the single-point likelihood for maximum sequential speed.
        # We do NOT use vmap here because Dynesty evaluates points one at a time.
        fast_likelihood = jax.jit(likelihood)

        sampler = NestedSampler(
            fast_likelihood, 
            prior_transform, 
            ndims,
            bound="multi", 
            sample="rwalk",
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

        # Vectorize and JIT the likelihood for the GPU
        batched_likelihood = jax.jit(jax.vmap(likelihood))

        # --- NEW: Define a safe batch size to prevent GPU OOM ---
        # 8 is usually a safe sweet spot. If it still crashes, drop to 4 or 2.
        # If your GPU has lots of memory (e.g., 40GB A100), you can push it to 16.
        BATCH_SIZE = 32

        def batched_log_prob(coords):
            # 1. Find which walkers are out of bounds (boolean mask)
            out_of_bounds = np.any((coords < lower_bound) | (coords > upper_bound), axis=1)
            
            # 2. Evaluate likelihood in chunks to save GPU memory
            logl_list = []
            for i in range(0, len(coords), BATCH_SIZE):
                batch_coords = coords[i : i + BATCH_SIZE]
                batch_logl = batched_likelihood(batch_coords)
                logl_list.append(np.asarray(batch_logl))
            
            logl = np.concatenate(logl_list)
            
            # 3. Apply -inf to the out-of-bounds walkers
            logl[out_of_bounds] = -np.inf
            return logl

        p0 = np.random.uniform(lower_bound, upper_bound,
                               size=(opts.nwalkers, ndims))

        sampler = emcee.EnsembleSampler(
            opts.nwalkers, 
            ndims, 
            batched_log_prob,
            vectorize=True,  # Keep this True! We are handling the vectorization internally now.
            moves=[(emcee.moves.DEMove(), 0.8),
                   (emcee.moves.DESnookerMove(), 0.2)]
        )
        sampler.run_mcmc(p0, opts.nsteps, progress=True)

        chain = sampler.flatchain
        samples = chain[len(chain)//2:]

        return {
            "samples": np.asarray(samples),
            "logZ": None,
            "logZerr": None
        }

    else:
        raise ValueError(f"Unknown sampler: {method}")