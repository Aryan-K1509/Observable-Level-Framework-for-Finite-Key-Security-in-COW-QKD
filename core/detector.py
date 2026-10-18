"""
detector.py
===========
Layer 2 of the COW-QKD framework: classical detector physics and scaling.

Takes the raw click counts from observable.py (ObservableSnapshot) and:
    - applies dark counts (independent Poissonian false clicks) to the data line,
    - applies non-paralyzable dead-time thinning to the genuine-signal slots,
    - forms the Z-basis bit error rate Ez and the sifted count n_z,
    - forms the monitor-line decoy and signal-state counts/gains used by the
      phase-error bounds in security_bounds.py,
    - propagates Wilson 95% confidence intervals to every output.

Counts are scaled by N_target/N_sf. For a valid finite-key bound the caller
should use N_sf == N_target (scale factor 1); the scaling only matters for
diagnostic runs with a smaller sample.

Monitor-line counts from observable.py already include dark counts (Gao et
al.'s closed-form gains have p_d built in), so dark counts are NOT re-applied
to the monitor line here.

References:
    Gao et al., Opt. Express 30, 23783 (2022): Eq. (4) (E_z), App. A (gains).
    Li et al., Phys. Rev. Research 6, 013022 (2024): Sec. II.B, Eq. (9).
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy import stats

from observable import ObservableSnapshot


# ----------------------------------------------------------------------------
# Data container
# ----------------------------------------------------------------------------

@dataclass
class ScaledMetrics:
    """
    Detector-corrected observables, scaled to N_target, consumed by Layer 3.

    Data line:
        n_z, n_z_ci        : sifted Z-basis clicks and 95% CI (counts)
        Ez, Ez_ci          : bit error rate (Gao Eq. 4) and 95% CI

    Monitor line, decoy states:
        n_DM0_alphaalpha, n_DM1_alphaalpha : D_M0/D_M1 clicks for |alpha,alpha>
        n_DM0_00, n_DM1_00                 : D_M0/D_M1 clicks for |0,0>
        n_alphaalpha, n_00                 : number of each decoy sent

    Monitor line, signal states (Cao et al. 2026 notation n^{Di}_{0z/1z}):
        n_DM0_0z, n_DM1_0z, n_DM0_1z, n_DM1_1z

    n_0z_total, n_1z_total : number of |0z>/|1z> pulses sent (denominators
                             for the signal-state monitor gains, Gao Eq. 7)
    G_DM*_*                : monitor-line decoy gains (rates, not counts)
    mu, tB, eta, p_dark    : echoed operating parameters
    N_target, N_sf         : target experiment size and simulated pulses
    """

    n_z: float
    n_z_ci: float
    Ez: float
    Ez_ci: float

    n_DM0_alphaalpha: float
    n_DM1_alphaalpha: float
    n_DM0_00: float
    n_DM1_00: float
    n_alphaalpha: float
    n_00: float

    n_DM0_0z: float
    n_DM1_0z: float
    n_DM0_1z: float
    n_DM1_1z: float

    n_0z_total: float
    n_1z_total: float

    G_DM0_alphaalpha: float
    G_DM1_alphaalpha: float
    G_DM0_00: float
    G_DM1_00: float

    mu: float
    tB: float
    eta: float
    p_dark: float
    N_target: int
    N_sf: int
    effective_rep_rate_hz: float


# ----------------------------------------------------------------------------
# Dark count / dead time application
# ----------------------------------------------------------------------------

def _wilson_ci(successes: int, trials: int, confidence: float = 0.95):
    """
    Wilson score interval for a binomial proportion (robust for the very
    small click probabilities of the vacuum-decoy channel). Returns
    (p_centre, half_width) in probability units.
    """
    if trials == 0:
        return 0.0, 0.0
    z = stats.norm.ppf(0.5 + confidence / 2.0)
    p_hat = successes / trials
    denom = 1.0 + z**2 / trials
    centre = p_hat + z**2 / (2 * trials)
    half_width = z * np.sqrt(p_hat * (1 - p_hat) / trials + z**2 / (4 * trials**2))
    lower = (centre - half_width) / denom
    upper = (centre + half_width) / denom
    p_centre = (lower + upper) / 2.0
    half = (upper - lower) / 2.0
    return p_centre, half


def apply_dark_counts(
    clicks_signal: int,
    n_trials: int,
    dark_count_prob: float,
) -> tuple[float, float]:
    """
    Combine a quantum click probability with an independent dark-count
    process for a threshold detector:

        P(click) = 1 - (1 - P_quantum) * (1 - p_dark)

    Returns (corrected_click_probability, corrected_click_count).
    """
    if n_trials == 0:
        return 0.0, 0.0
    p_quantum = clicks_signal / n_trials
    p_total = 1.0 - (1.0 - p_quantum) * (1.0 - dark_count_prob)
    return p_total, p_total * n_trials


def apply_dead_time_thinning(click_probability_per_pulse: float, dead_time_s: float, rep_rate_hz: float) -> float:
    """
    Non-paralyzable dead-time correction of a per-pulse click probability
    (Knoll, "Radiation Detection and Measurement"):

        p_eff = p / (1 + p * rep_rate_hz * dead_time_s)

    Applied only to the genuine-signal slots, which dominate the click rate.
    With an SNSPD-scale dead time (~100 ns) the correction is ~1% at Cao et
    al.'s 25 km click rates and smaller at longer distances.
    """
    if dead_time_s <= 0.0 or rep_rate_hz <= 0.0:
        return click_probability_per_pulse
    p = float(click_probability_per_pulse)
    denom = 1.0 + p * rep_rate_hz * dead_time_s
    return p / denom if denom > 0.0 else p


def effective_rep_rate_hz(rep_rate_hz: float, dead_time_s: float) -> float:
    """
    Maximum sustainable gate rate given the dead time (reporting only; not
    used in any rate calculation).
    """
    if dead_time_s <= 0:
        return rep_rate_hz
    return 1.0 / (1.0 / rep_rate_hz + dead_time_s)


# ----------------------------------------------------------------------------
# Scaling helper (propagates Wilson CIs from the SF sample to N_target)
# ----------------------------------------------------------------------------

def _scale_count(successes: int, trials: int, n_target_trials: float):
    """
    Scale a click count observed in `trials` rounds to `n_target_trials`
    rounds, assuming a stationary click probability. Returns
    (scaled_count, scaled_95ci_halfwidth). The relative uncertainty is set by
    the simulated sample size, not by N_target.
    """
    p_centre, p_half = _wilson_ci(successes, trials)
    scaled = p_centre * n_target_trials
    scaled_ci = p_half * n_target_trials
    return scaled, scaled_ci


# ----------------------------------------------------------------------------
# Main entry point (the public interface of Layer 2)
# ----------------------------------------------------------------------------

def apply_detector_physics(
    obs: ObservableSnapshot,
    dark_count_rate_hz: float,
    dead_time_s: float,
    rep_rate_hz: float,
    N_target: int,
) -> ScaledMetrics:
    """
    Apply dark counts and dead time to an ObservableSnapshot and scale all
    estimates to N_target pulses with propagated error bars.

    Parameters
    ----------
    obs : ObservableSnapshot
        Raw click counts from Layer 1.
    dark_count_rate_hz : float
        Dark-count rate (Hz); per-pulse p_d = dark_count_rate_hz / rep_rate_hz.
    dead_time_s : float
        Detector dead time (s), e.g. ~100 ns for SNSPDs.
    rep_rate_hz : float
        Pulse repetition rate (Hz).
    N_target : int
        Number of pulses the experiment represents.

    Returns
    -------
    ScaledMetrics
    """
    p_dark = dark_count_rate_hz / rep_rate_hz

    n_sf = obs.num_pulses_simulated
    scale_factor = N_target / n_sf if n_sf > 0 else 0.0

    # ---- Data line: the four gains G^{tau_j}_{wz}, dark-count corrected ----
    p_0z_tau0, _ = apply_dark_counts(obs.clicks_data_0z_tau0, obs.n_0z, p_dark)
    p_0z_tau1, _ = apply_dark_counts(obs.clicks_data_0z_tau1, obs.n_0z, p_dark)
    p_1z_tau0, _ = apply_dark_counts(obs.clicks_data_1z_tau0, obs.n_1z, p_dark)
    p_1z_tau1, _ = apply_dark_counts(obs.clicks_data_1z_tau1, obs.n_1z, p_dark)

    # Wilson half-widths on the raw counts; the dark-count correction is
    # affine in p_quantum, so they propagate by the factor (1 - p_dark).
    _, _h_0z_tau0 = _wilson_ci(obs.clicks_data_0z_tau0, obs.n_0z)
    _, _h_0z_tau1 = _wilson_ci(obs.clicks_data_0z_tau1, obs.n_0z)
    _, _h_1z_tau0 = _wilson_ci(obs.clicks_data_1z_tau0, obs.n_1z)
    _, _h_1z_tau1 = _wilson_ci(obs.clicks_data_1z_tau1, obs.n_1z)
    h_0z_tau0 = (1.0 - p_dark) * _h_0z_tau0
    h_0z_tau1 = (1.0 - p_dark) * _h_0z_tau1
    h_1z_tau0 = (1.0 - p_dark) * _h_1z_tau0
    h_1z_tau1 = (1.0 - p_dark) * _h_1z_tau1

    # Dead-time thinning on the genuine-signal slots (tau1 for |0z>, tau0 for
    # |1z>). Since dp_eff/dp = (p_eff/p)^2, the half-widths scale by the same
    # factor.
    p_0z_tau1_pre_thinning = p_0z_tau1
    p_1z_tau0_pre_thinning = p_1z_tau0
    # The bright bin arrives once per two-bin round, so the effective
    # repetition rate for this slot's probability is rep_rate_hz / 2.
    f_round = rep_rate_hz / 2.0
    p_0z_tau1 = apply_dead_time_thinning(p_0z_tau1, dead_time_s, f_round)
    p_1z_tau0 = apply_dead_time_thinning(p_1z_tau0, dead_time_s, f_round)
    if p_0z_tau1_pre_thinning > 0.0:
        h_0z_tau1 *= (p_0z_tau1 / p_0z_tau1_pre_thinning) ** 2
    if p_1z_tau0_pre_thinning > 0.0:
        h_1z_tau0 *= (p_1z_tau0 / p_1z_tau0_pre_thinning) ** 2

    # Bit assignment follows Cao et al. 2026 (arXiv:2601.06772, Sec. III):
    # a click in the former slot -> bit 1, latter slot -> bit 0. Hence the
    # genuine-signal click is the correct term and the dark-count-only click
    # is the error term for both |0z> and |1z>.
    G_tau0_0z = p_0z_tau0          # dark-count-only slot for |0z> -> error term
    G_tau1_0z = p_0z_tau1          # genuine-signal slot for |0z> -> correct term
    G_tau0_1z = p_1z_tau0          # genuine-signal slot for |1z> -> correct term
    G_tau1_1z = p_1z_tau1          # dark-count-only slot for |1z> -> error term

    error_terms = G_tau0_0z + G_tau1_1z
    total_terms = G_tau0_0z + G_tau1_0z + G_tau0_1z + G_tau1_1z
    Ez = error_terms / total_terms if total_terms > 0 else 0.0

    # Ez CI via the delta method, treating the error and correct sums as
    # independent binomial contributions.
    correct_terms = G_tau1_0z + G_tau0_1z
    ci_error = float(np.sqrt(h_0z_tau0 ** 2 + h_1z_tau1 ** 2))
    ci_correct = float(np.sqrt(h_0z_tau1 ** 2 + h_1z_tau0 ** 2))
    if total_terms > 0:
        d_Ez_d_error = correct_terms / total_terms ** 2
        d_Ez_d_correct = -error_terms / total_terms ** 2
        Ez_half = float(np.sqrt(
            (d_Ez_d_error * ci_error) ** 2 + (d_Ez_d_correct * ci_correct) ** 2
        ))
    else:
        Ez_half = 0.0

    # n_z is a count: sum each state's gains times that state's pulse count
    # (no /2 averaging, which applies only at the gain level in Gao Eq. 2).
    n_0z_target = N_target * obs.n_0z / n_sf if n_sf > 0 else 0.0
    n_1z_target = N_target * obs.n_1z / n_sf if n_sf > 0 else 0.0
    n_z_from_0z = (G_tau0_0z + G_tau1_0z) * n_0z_target
    n_z_from_1z = (G_tau0_1z + G_tau1_1z) * n_1z_target
    n_z = n_z_from_0z + n_z_from_1z
    # Independent contributions combine in quadrature.
    n_z_from_0z_ci = float(np.sqrt(h_0z_tau0 ** 2 + h_0z_tau1 ** 2)) * n_0z_target
    n_z_from_1z_ci = float(np.sqrt(h_1z_tau0 ** 2 + h_1z_tau1 ** 2)) * n_1z_target
    n_z_ci = float(np.sqrt(n_z_from_0z_ci ** 2 + n_z_from_1z_ci ** 2))

    # ---- Monitor line: decoy gains (dark counts already included) ----
    p_DM0_aa = obs.clicks_mon_DM0_alphaalpha / obs.n_alphaalpha if obs.n_alphaalpha > 0 else 0.0
    p_DM1_aa = obs.clicks_mon_DM1_alphaalpha / obs.n_alphaalpha if obs.n_alphaalpha > 0 else 0.0
    p_DM0_00 = obs.clicks_mon_DM0_00 / obs.n_00 if obs.n_00 > 0 else 0.0
    p_DM1_00 = obs.clicks_mon_DM1_00 / obs.n_00 if obs.n_00 > 0 else 0.0

    n_alphaalpha_target = N_target * obs.n_alphaalpha / n_sf if n_sf > 0 else 0.0
    n_00_target = N_target * obs.n_00 / n_sf if n_sf > 0 else 0.0

    # ---- Monitor line: decoy counts, scaled with Wilson CI ----
    n_DM0_aa, n_DM0_aa_ci = _scale_count(obs.clicks_mon_DM0_alphaalpha, max(obs.n_alphaalpha, 1), n_alphaalpha_target)
    n_DM1_aa, n_DM1_aa_ci = _scale_count(obs.clicks_mon_DM1_alphaalpha, max(obs.n_alphaalpha, 1), n_alphaalpha_target)
    n_DM0_00, n_DM0_00_ci = _scale_count(obs.clicks_mon_DM0_00,         max(obs.n_00, 1),         n_00_target)
    n_DM1_00, n_DM1_00_ci = _scale_count(obs.clicks_mon_DM1_00,         max(obs.n_00, 1),         n_00_target)

    # ---- Monitor line: signal-state counts (dark counts already included) ----
    n_0z_target = N_target * obs.n_0z / n_sf if n_sf > 0 else 0.0
    n_1z_target = N_target * obs.n_1z / n_sf if n_sf > 0 else 0.0

    n_DM0_0z, _ = _scale_count(obs.clicks_mon_DM0_0z, max(obs.n_0z, 1), n_0z_target)
    n_DM1_0z, _ = _scale_count(obs.clicks_mon_DM1_0z, max(obs.n_0z, 1), n_0z_target)
    n_DM0_1z, _ = _scale_count(obs.clicks_mon_DM0_1z, max(obs.n_1z, 1), n_1z_target)
    n_DM1_1z, _ = _scale_count(obs.clicks_mon_DM1_1z, max(obs.n_1z, 1), n_1z_target)

    return ScaledMetrics(
        n_z=n_z,
        n_z_ci=n_z_ci,
        Ez=Ez,
        Ez_ci=Ez_half,
        n_DM0_alphaalpha=n_DM0_aa,
        n_DM1_alphaalpha=n_DM1_aa,
        n_DM0_00=n_DM0_00,
        n_DM1_00=n_DM1_00,
        n_alphaalpha=n_alphaalpha_target,
        n_00=n_00_target,
        n_DM0_0z=n_DM0_0z,
        n_DM1_0z=n_DM1_0z,
        n_DM0_1z=n_DM0_1z,
        n_DM1_1z=n_DM1_1z,
        n_0z_total=n_0z_target,
        n_1z_total=n_1z_target,
        G_DM0_alphaalpha=p_DM0_aa,
        G_DM1_alphaalpha=p_DM1_aa,
        G_DM0_00=p_DM0_00,
        G_DM1_00=p_DM1_00,
        mu=obs.mu,
        p_dark=p_dark,
        tB=obs.tB,
        eta=obs.eta,
        N_target=N_target,
        N_sf=n_sf,
        effective_rep_rate_hz=effective_rep_rate_hz(rep_rate_hz, dead_time_s),
    )
