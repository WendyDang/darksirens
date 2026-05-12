#!/usr/bin/env bash
# Generate a small mock dark-sirens data set and verify it can be ingested by
# darksirens_inference's data loaders. Set RUN_INFERENCE=1 to also launch a
# tiny sampler smoke run.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUTDIR="${OUTDIR:-${ROOT_DIR}/data/mock_dark_sirens_test}"
NSIDE="${NSIDE:-8}"
N_GALAXIES="${N_GALAXIES:-12000}"
NOBS="${NOBS:-3}"
NSAMP="${NSAMP:-128}"
NDRAW="${NDRAW:-50000}"
SEED="${SEED:-1234}"
SELECTION_BATCH_SIZE="${SELECTION_BATCH_SIZE:-1000}"
SELECTION_PER_OBSERVATION_FACTOR="${SELECTION_PER_OBSERVATION_FACTOR:-6}"

cd "${ROOT_DIR}"
mkdir -p "${OUTDIR}"

python scripts/mock_data/generate_mock_data.py \
  --outdir "${OUTDIR}" \
  --seed "${SEED}" \
  --n-galaxies "${N_GALAXIES}" \
  --nobs "${NOBS}" \
  --nsamp "${NSAMP}" \
  --ndraw "${NDRAW}" \
  --selection-batch-size "${SELECTION_BATCH_SIZE}" \
  --selection-per-observation-factor "${SELECTION_PER_OBSERVATION_FACTOR}" \
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
    --nlive 50 \
    --dlogz 5.0 \
    --max_samples 500 \
    --seed "${SEED}" \
    --show_progress False \
    --save_path "${OUTDIR}/inference_smoke"
fi

cat <<EOF
Mock data test complete.
Products are in: ${OUTDIR}
Set RUN_INFERENCE=1 to run the optional tiny darksirens_inference sampler smoke test.
EOF
