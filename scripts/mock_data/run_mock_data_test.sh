#!/usr/bin/env bash
# Generate a realistic low-redshift mock dark-sirens data set and verify that
# the inference data loaders can ingest it. Set RUN_INFERENCE=1 to launch a
# small dark-sirens sampler run using survey hyperparameters matched to the
# generated catalog.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUTDIR="${OUTDIR:-${ROOT_DIR}/data/mock_dark_sirens_test}"
SEED="${SEED:-1234}"

# Survey-generation defaults.  N0=1e-3 Mpc^-3 is intentionally inside the
# default inference prior log10n0 ∈ [-4, -1].  The low zmax keeps the fixture
# lightweight while still using a physically meaningful density normalization.
N0="${N0:-1e-3}"
ZMAX="${ZMAX:-0.08}"
SURVEY_Z50="${SURVEY_Z50:-0.75}"
SURVEY_WIDTH="${SURVEY_WIDTH:-0.12}"
GALAXY_DENSITY_DELTA="${GALAXY_DENSITY_DELTA:-0.0}"
LOG10N0="$(python -c 'import math, sys; print(math.log10(float(sys.argv[1])))' "${N0}")"

# Mock size/performance knobs.
NSIDE="${NSIDE:-8}"
NOBS="${NOBS:-3}"
NSAMP="${NSAMP:-128}"
NDRAW="${NDRAW:-50000}"
SELECTION_BATCH_SIZE="${SELECTION_BATCH_SIZE:-50000}"
SELECTION_PER_OBSERVATION_FACTOR="${SELECTION_PER_OBSERVATION_FACTOR:-6}"

# Fractional/absolute widths used to generate mock GW PE samples.
DL_FRAC_UNCERTAINTY="${DL_FRAC_UNCERTAINTY:-0.20}"
M1DET_FRAC_UNCERTAINTY="${M1DET_FRAC_UNCERTAINTY:-0.08}"
M2DET_FRAC_UNCERTAINTY="${M2DET_FRAC_UNCERTAINTY:-0.10}"
CHIEFF_UNCERTAINTY="${CHIEFF_UNCERTAINTY:-0.08}"
SKY_UNCERTAINTY_DEG="${SKY_UNCERTAINTY_DEG:-5.0}"

FIXED_SURVEY_JSON="${FIXED_SURVEY_JSON:-{\"log10n0\": ${LOG10N0}, \"z50\": ${SURVEY_Z50}, \"w\": ${SURVEY_WIDTH}, \"delta\": ${GALAXY_DENSITY_DELTA}, \"b_miss\": 1.0, \"alpha_miss\": 0.5}}"

cd "${ROOT_DIR}"
mkdir -p "${OUTDIR}"

python scripts/mock_data/generate_mock_data.py \
  --outdir "${OUTDIR}" \
  --seed "${SEED}" \
  --n0 "${N0}" \
  --zmax "${ZMAX}" \
  --survey-z50 "${SURVEY_Z50}" \
  --survey-width "${SURVEY_WIDTH}" \
  --galaxy-density-delta "${GALAXY_DENSITY_DELTA}" \
  --nobs "${NOBS}" \
  --nsamp "${NSAMP}" \
  --ndraw "${NDRAW}" \
  --selection-batch-size "${SELECTION_BATCH_SIZE}" \
  --selection-per-observation-factor "${SELECTION_PER_OBSERVATION_FACTOR}" \
  --dL-fractional-uncertainty "${DL_FRAC_UNCERTAINTY}" \
  --m1det-fractional-uncertainty "${M1DET_FRAC_UNCERTAINTY}" \
  --m2det-fractional-uncertainty "${M2DET_FRAC_UNCERTAINTY}" \
  --chieff-uncertainty "${CHIEFF_UNCERTAINTY}" \
  --sky-uncertainty-deg "${SKY_UNCERTAINTY_DEG}" \
  --nside "${NSIDE}"

export OUTDIR NSIDE NOBS NSAMP
python - <<'PY'
from argparse import Namespace
from pathlib import Path
import os
import numpy as np
from darksirens.inference.data import load_all_data

out = Path(os.environ["OUTDIR"])
nside = int(os.environ["NSIDE"])
nobs = int(os.environ["NOBS"])
nsamp = int(os.environ["NSAMP"])
opts = Namespace(
    universe_model="dark_sirens",
    survey_path=str(out / f"catalog_pixelated_nside_{nside}.h5"),
    gw_path=str(out / "mock_gw_events.h5"),
    gwselection_path=str(out / "mock_gw_selection.h5"),
    sigma_kernel=0.005,
    use_LSS=False,
    counterpart=None,
    counterpart_nside=1,
    counterpart_dz=1e-4,
)
data = load_all_data(opts)
assert int(data["nEvents"]) == nobs, data["nEvents"]
assert int(data["nsamp"]) == nsamp, data["nsamp"]
assert int(data["nside"]) == nside, data["nside"]
assert len(data["p_draw"]) > 5 * nobs, "too few detected selection samples"
assert np.isfinite(np.asarray(data["p_draw"])).all(), "non-finite p_draw values"
print("Ingestion smoke test passed.")
PY

if [[ "${RUN_INFERENCE:-0}" == "1" ]]; then
  python -m darksirens.tool.darksirens_inference \
    --gw_path "${OUTDIR}/mock_gw_events.h5" \
    --gwselection_path "${OUTDIR}/mock_gw_selection.h5" \
    --survey_path "${OUTDIR}/catalog_pixelated_nside_${NSIDE}.h5" \
    --sampler dynesty \
    --pop_model powerlaw+peak_shared_beta_spin \
    --universe_model dark_sirens \
    --fix_population True \
    --fix_survey True \
    --fixed_parameter_values "${FIXED_SURVEY_JSON}" \
    --nlive 100 \
    --dlogz 1.0 \
    --max_samples 2000 \
    --seed "${SEED}" \
    --show_progress False \
    --save_path "${OUTDIR}/inference_smoke"
fi

cat <<EOF
Mock data test complete.
Products are in: ${OUTDIR}
Generated survey density N0=${N0} Mpc^-3 and zmax=${ZMAX}; fixed inference survey JSON: ${FIXED_SURVEY_JSON}
Set RUN_INFERENCE=1 to run the optional darksirens_inference sampler smoke test.
EOF
