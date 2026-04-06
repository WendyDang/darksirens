#!/usr/bin/env python3
import os
import json
import argparse

import numpy as np
import corner
from tqdm import tqdm

import jax
import jax.numpy as jnp

from darksirens.gw.populations import pop_model_parser

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['font.sans-serif'] = ['Bitstream Vera Sans']
matplotlib.rcParams['text.usetex'] = False
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['figure.figsize'] = (16.0, 10.0)
matplotlib.rcParams['axes.unicode_minus'] = False

import seaborn as sns
sns.set_context('talk')
sns.set_style('ticks')
sns.set_palette('colorblind')
c = sns.color_palette('colorblind')


# ------------------------------------------------------------
# I/O
# ------------------------------------------------------------
def load_run(run_dir):
    settings = json.load(open(os.path.join(run_dir, "settings.json")))
    results = np.load(os.path.join(run_dir, "samples.npy"), allow_pickle=True).item()

    samples = results["samples"]
    logZ = results.get("logZ", None)
    logZerr = results.get("logZerr", None)

    return settings, samples, logZ, logZerr


# ------------------------------------------------------------
# Evidence plotting
# ------------------------------------------------------------
def plot_model_evidences(labels, logZs, logZerrs, figsize=(10, 6)):
    fig, ax = plt.subplots(figsize=figsize)

    xs = np.arange(len(labels))
    ax.bar(xs, logZs, yerr=logZerrs, color=c[:len(labels)], alpha=0.8)

    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=18)
    ax.set_ylabel(r"$\log_{10} Z$", fontsize=24)
    ax.set_title("Model Evidences", fontsize=26)
    ax.tick_params(labelsize=18)

    fig.tight_layout()
    return fig


def print_bayes_factors(labels, logZs):
    print("\n=== Bayes Factors (log BF) ===")
    for i in range(len(labels)):
        for j in range(i+1, len(labels)):
            if logZs[i] is not None and logZs[j] is not None:
                print(f"{labels[i]} vs {labels[j]}:  log10 BF = {logZs[i] - logZs[j]:.3f}")


def plot_bayes_factor_matrix_pairwise(labels, log10Zs, log10Zerrs, figsize=(10, 10)):
    """
    Full pairwise log10 Bayes factor matrix.
    Color-coded by Jeffreys scale.
    """
    n = len(labels)
    fig, axes = plt.subplots(n, n, figsize=figsize)

    for i in range(n):
        for j in range(n):
            ax = axes[i, j]

            if i == j:
                ax.text(0.5, 0.5, labels[i], ha="center", va="center", fontsize=14)
                ax.set_xticks([]); ax.set_yticks([])
                continue

            if log10Zs[i] is not None and log10Zs[j] is not None:
                bf = log10Zs[i] - log10Zs[j]
                bf_err = np.sqrt(log10Zerrs[i]**2 + log10Zerrs[j]**2)
                color = jeffreys_color(bf)
                ax.set_facecolor(color)
                ax.text(0.5, 0.55, f"{bf:.2f}", ha="center", va="center", fontsize=14)
                ax.text(0.5, 0.25, f"±{bf_err:.2f}", ha="center", va="center", fontsize=10)
            else:
                ax.set_facecolor("lightgray")
                ax.text(0.5, 0.5, "—", ha="center", va="center", fontsize=14)

            ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle("Pairwise log10 Bayes Factors", fontsize=20)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def jeffreys_color(bf):
    """
    Color coding for log10 Bayes factors using Jeffreys scale.
    """
    if bf is None:
        return "lightgray"
    bf = abs(bf)
    if bf < 0.5:
        return "#d9d9d9"   # very weak
    elif bf < 1.0:
        return "#a6bddb"   # substantial
    elif bf < 2.0:
        return "#3690c0"   # strong
    else:
        return "#034e7b"   # decisive


# ------------------------------------------------------------
# JAX posterior predictive engine (batched, per-sample PPD)
# ------------------------------------------------------------
def make_single_theta_predictive(pop_model, mgrid, qgrid, zgrid):
    mgrid = jnp.asarray(mgrid)
    qgrid = jnp.asarray(qgrid)
    zgrid = jnp.asarray(zgrid)

    nm = mgrid.size

    m1_grid = mgrid[:, None, None]
    q_grid  = qgrid[None, :, None]
    z_grid  = zgrid[None, None, :]

    m2_vals = mgrid
    q_eval = m2_vals[None, :] / mgrid[:, None]
    valid = (q_eval >= qgrid[0]) & (q_eval <= qgrid[-1])
    jac = 1.0 / mgrid[:, None]

    @jax.jit
    def single_theta(theta):
        logp = pop_model(m1_grid, q_grid, z_grid, *theta)
        p = jnp.exp(logp)

        p_m1 = jnp.trapezoid(jnp.trapezoid(p, zgrid, axis=2), qgrid, axis=1)
        p_m1 /= jnp.trapezoid(p_m1, mgrid)

        p_q = jnp.trapezoid(jnp.trapezoid(p, zgrid, axis=2), mgrid, axis=0)
        p_q /= jnp.trapezoid(p_q, qgrid)

        p_z = jnp.trapezoid(jnp.trapezoid(p, qgrid, axis=1), mgrid, axis=0)
        p_z /= jnp.trapezoid(p_z, zgrid)

        p_m1q = jnp.trapezoid(p, zgrid, axis=2)

        p_interp = jax.vmap(
            lambda row, qev: jnp.interp(qev, qgrid, row, left=0.0, right=0.0)
        )(p_m1q, q_eval)

        p_m1m2 = p_interp * jac * valid

        norm_2d = jnp.trapezoid(
            jnp.trapezoid(p_m1m2, mgrid, axis=0), mgrid, axis=0
        )
        p_m1m2 = jnp.where(norm_2d > 0, p_m1m2 / norm_2d, p_m1m2)

        p_m2 = jnp.trapezoid(p_m1m2, mgrid, axis=0)
        p_m2 /= jnp.trapezoid(p_m2, mgrid)

        return p_m1, p_m2, p_q, p_z, p_m1m2

    return single_theta


def auto_batch_size(nsamples, nm, nq, nz, target_bytes=2e9):
    per_sample_bytes = nm * nq * nz * 8.0
    if per_sample_bytes == 0:
        return min(nsamples, 64)

    max_batch = int(target_bytes // per_sample_bytes)
    max_batch = max(1, min(max_batch, 256))
    return min(nsamples, max_batch)


def posterior_predictive_mass_distributions_jax(
    pop_model, samples, mgrid, qgrid, zgrid, batch_size=None
):
    samples = jnp.asarray(samples)
    mgrid = jnp.asarray(mgrid)
    qgrid = jnp.asarray(qgrid)
    zgrid = jnp.asarray(zgrid)

    ns = samples.shape[0]
    nm = mgrid.size
    nq = qgrid.size
    nz = zgrid.size

    if batch_size is None:
        batch_size = auto_batch_size(ns, nm, nq, nz)

    single_theta = make_single_theta_predictive(pop_model, mgrid, qgrid, zgrid)

    pm1_list   = []
    pm2_list   = []
    pq_list    = []
    pz_list    = []
    pm1m2_list = []

    n_batches = (ns + batch_size - 1) // batch_size

    for i in tqdm(range(n_batches), desc="Posterior predictive batches"):
        start = i * batch_size
        end = min((i + 1) * batch_size, ns)
        batch = samples[start:end]

        p1_batch, p2_batch, pq_batch, pz_batch, p2d_batch = jax.vmap(single_theta)(batch)

        pm1_list.append(p1_batch)
        pm2_list.append(p2_batch)
        pq_list.append(pq_batch)
        pz_list.append(pz_batch)
        pm1m2_list.append(p2d_batch)

    pm1_samples   = jnp.concatenate(pm1_list,   axis=0)
    pm2_samples   = jnp.concatenate(pm2_list,   axis=0)
    pq_samples    = jnp.concatenate(pq_list,    axis=0)
    pz_samples    = jnp.concatenate(pz_list,    axis=0)
    pm1m2_samples = jnp.concatenate(pm1m2_list, axis=0)

    return pm1_samples, pm2_samples, pq_samples, pz_samples, pm1m2_samples


# ------------------------------------------------------------
# PPD summarization + plotting
# ------------------------------------------------------------
def summarize_ppd(ppd_samples, limits=(5, 95)):
    lo, hi = limits
    median = jnp.median(ppd_samples, axis=0)
    lower = jnp.percentile(ppd_samples, lo, axis=0)
    upper = jnp.percentile(ppd_samples, hi, axis=0)
    return median, lower, upper


def plot_1d_spectrum(
    xgrid,
    ppd_list,
    labels,
    limits=(5, 95),
    xlabel="x",
    ylabel="p(x)",
    xlim=None,
    ylim=None,
    colors=None,
    figsize=(24, 10),
    logy=True,
):
    if colors is None:
        colors = plt.cm.tab10.colors

    xgrid = np.asarray(xgrid)
    fig, ax = plt.subplots(figsize=figsize)

    means = []
    for i, ppd in enumerate(ppd_list):
        ppd = jnp.asarray(ppd)
        median, lower, upper = summarize_ppd(ppd, limits)

        color = colors[i % len(colors)]

        ax.fill_between(
            xgrid, np.asarray(lower), np.asarray(upper),
            alpha=0.25,
            color=color,
            label=labels[i],
        )

        ax.plot(xgrid, np.asarray(median), color=color, lw=2.5)

        mean_curve = np.asarray(ppd.mean(axis=0))
        means.append(mean_curve)
        ax.plot(
            xgrid, mean_curve,
            color=color,
            linestyle="--",
            lw=2.0,
            alpha=0.9,
        )

    if xlim is None:
        xlim = (xgrid.min(), xgrid.max())
    if ylim is None:
        mean_arr = np.vstack(means)
        ymin = max(1e-6, float(mean_arr.min()))
        ymax = float(mean_arr.max()) * 1.2
        ylim = (ymin, ymax)

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)

    ax.set_xlabel(xlabel, fontsize=32)
    ax.set_ylabel(ylabel, fontsize=32)
    ax.legend(fontsize=24, frameon=False)
    ax.tick_params(labelsize=22)

    if logy:
        ax.set_yscale("log")

    fig.tight_layout()
    return fig


def summarize_ppd_2d(ppd2d_samples):
    return np.asarray(jnp.median(ppd2d_samples, axis=0))


def plot_joint_m1m2(
    mgrid,
    ppd2d_list,
    labels,
    colors=None,
    figsize=(12, 10),
    n_levels=5,
):
    if colors is None:
        colors = plt.cm.tab10.colors

    mgrid = np.asarray(mgrid)
    M1, M2 = np.meshgrid(mgrid, mgrid, indexing="ij")

    fig, ax = plt.subplots(figsize=figsize)

    legend_handles = []

    for i, ppd2d in enumerate(ppd2d_list):
        median = summarize_ppd_2d(ppd2d)
        color = colors[i % len(colors)]

        flat = median.ravel()
        flat = flat[flat > 0]
        qs = np.linspace(0.1, 0.9, n_levels)
        levels = np.quantile(flat, qs)

        cs = ax.contour(
            M1, M2, median,
            levels=levels,
            colors=[color],
            linewidths=2,
        )

        proxy = Line2D([0], [0], color=color, lw=2)
        legend_handles.append(proxy)

    ax.set_xlabel(r"$m_1$ [$M_\odot$]", fontsize=28)
    ax.set_ylabel(r"$m_2$ [$M_\odot$]", fontsize=28)
    ax.tick_params(labelsize=20)

    ax.legend(legend_handles, labels, fontsize=20, frameon=False, loc='upper left')

    ax.set_xlim(mgrid.min(), mgrid.max())
    ax.set_ylim(mgrid.min(), mgrid.max())

    fig.tight_layout()
    return fig


# ------------------------------------------------------------
# CLI / main
# ------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run_dirs",
        nargs="+",
        required=True,
        help="One or more run directories (PL1, PL2, Bump, etc.)"
    )
    parser.add_argument("--mmin", type=float, default=1.0)
    parser.add_argument("--mmax", type=float, default=100.0)
    parser.add_argument("--nm", type=int, default=300)
    parser.add_argument("--nq", type=int, default=200)
    parser.add_argument("--nz", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--cred_lo", type=float, default=5.0)
    parser.add_argument("--cred_hi", type=float, default=95.0)
    args = parser.parse_args()

    mgrid = np.linspace(args.mmin, args.mmax, args.nm)
    qgrid = np.linspace(0.01, 1.0, args.nq)
    zgrid = np.linspace(0.0, 2.0, args.nz)

    pm1_list   = []
    pm2_list   = []
    pq_list    = []
    pz_list    = []
    pm1m2_list = []
    labels     = []

    logZs      = []
    logZerrs   = []

    for run_dir in args.run_dirs:
        print(f"\n=== Processing model: {run_dir} ===")

        settings, samples, logZ, logZerr = load_run(run_dir)
        pop_model = pop_model_parser(settings["pop_model"])

        # Convert to log10 if available
        if logZ is not None:
            log10Z = logZ / np.log(10.0)
            log10Zerr = logZerr / np.log(10.0) if logZerr is not None else None
        else:
            log10Z = None
            log10Zerr = None

        logZs.append(log10Z)
        logZerrs.append(log10Zerr)

        pm1_samples, pm2_samples, pq_samples, pz_samples, pm1m2_samples = (
            posterior_predictive_mass_distributions_jax(
                pop_model, samples, mgrid, qgrid, zgrid, batch_size=args.batch_size
            )
        )

        pm1_list.append(pm1_samples)
        pm2_list.append(pm2_samples)
        pq_list.append(pq_samples)
        pz_list.append(pz_samples)
        pm1m2_list.append(pm1m2_samples)

        model_label = settings.get("model_name", os.path.basename(run_dir))
        labels.append(model_label)

    limits = (args.cred_lo, args.cred_hi)

    # p(m1)
    fig_m1 = plot_1d_spectrum(
        mgrid,
        pm1_list,
        labels,
        limits,
        xlabel=r"$m_1$ [$M_\odot$]",
        ylabel=r"$p(m_1)$ [$M_\odot^{-1}$]",
        logy=True,
        ylim=(1e-5, 1.0),
    )
    fig_m1.savefig("pm1_all_models.pdf")

    # p(m2)
    fig_m2 = plot_1d_spectrum(
        mgrid,
        pm2_list,
        labels,
        limits,
        xlabel=r"$m_2$ [$M_\odot$]",
        ylabel=r"$p(m_2)$ [$M_\odot^{-1}$]",
        logy=True,
        ylim=(1e-5, 1.0),
    )
    fig_m2.savefig("pm2_all_models.pdf")

    # p(q)
    fig_q = plot_1d_spectrum(
        qgrid,
        pq_list,
        labels,
        limits,
        xlabel=r"$q$",
        ylabel=r"$p(q)$",
        xlim=(0.0, 1.0),
        ylim=(1e-6, 10.0),
        logy=True,
    )
    fig_q.savefig("pq_all_models.pdf")

    # p(z)
    fig_z = plot_1d_spectrum(
        zgrid,
        pz_list,
        labels,
        limits,
        xlabel=r"$z$",
        ylabel=r"$p(z)$",
        xlim=(0.0, 2.0),
        ylim=(1e-2, 10.0),
        logy=True,
    )
    fig_z.savefig("pz_all_models.pdf")

    # joint p(m1,m2)
    fig_joint = plot_joint_m1m2(
        mgrid,
        ppd2d_list=pm1m2_list,
        labels=labels,
    )
    fig_joint.savefig("pm1m2_all_models.pdf")

    # ------------------------------------------------------------
    # Evidence comparison (separate figure)
    # ------------------------------------------------------------
    if any(z is not None for z in logZs):
        fig_ev = plot_model_evidences(labels, logZs, logZerrs)
        fig_ev.savefig("model_evidences.pdf")

        print("\n=== Model Evidences ===")
        for label, z, ze in zip(labels, logZs, logZerrs):
            print(f"{label:20s} log10Z = {z} ± {ze}")

        print_bayes_factors(labels, logZs)
    else:
        print("\nNo evidence information found in any run.")
        
    # ------------------------------------------------------------
    # Bayes factor matrices
    # ------------------------------------------------------------
    if any(z is not None for z in logZs):
        # Full pairwise matrix
        fig_pair = plot_bayes_factor_matrix_pairwise(labels, logZs, logZerrs)
        fig_pair.savefig("bayes_factors_pairwise.pdf")
        print("Saved bayes_factors_pairwise.pdf")

    else:
        print("\nNo evidence available to compute Bayes factors.")


if __name__ == "__main__":
    main()
