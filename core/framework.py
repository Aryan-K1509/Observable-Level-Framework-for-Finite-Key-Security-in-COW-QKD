"""
framework.py
============
Layer 4 of the COW-QKD framework: orchestration.

    - COWParams: one experiment configuration (hardware, protocol, attacks).
    - run_experiment: observable.py -> detector.py -> security_bounds.py,
      with one Strawberry Fields simulation per configuration and every
      requested bound evaluated on the same record.
    - parameter_sweep, check_stationarity: sweep and sampling-stability tools.
    - optimize_mu_tB / optimize_sweep_distance: (mu, tB) optimization of the
      secure key rate with differential evolution.

Distance -> transmittance uses a paper-specific fibre-loss exponent:
    Gao et al. (2022): eta = 10**(-0.02 * L)   (0.2 dB/km)
    Li et al. (2024):  eta = 10**(-0.016 * L)  (0.16 dB/km)
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from observable import run_sf_simulation, ObservableSnapshot
from detector import apply_detector_physics, ScaledMetrics
from security_bounds import SecurityBound, KatoBound, AzumaBound, BoundResult, evaluate_all_bounds, binary_entropy


# ----------------------------------------------------------------------------
# Parameter container
# ----------------------------------------------------------------------------

@dataclass
class COWParams:
    """
    One experiment configuration. Defaults are illustrative; paper
    reproductions override them explicitly.

    mu                   : mean photon number of signal/decoy states.
    tB                   : Bob's beam-splitter transmittance to the data line.
    distance_km          : fibre length L (km).
    fibre_loss_exponent  : coefficient in 10**(-exponent * L).
    detector_eff         : data-line detector (D0) efficiency.
    detector_eff_monitor : monitor-line detector (D1/D2) efficiency; None
                           falls back to detector_eff. Cao et al. 2026 report
                           76.2% (D0) and 46.0% (D1/D2).
    insertion_loss_db    : extra data-line insertion loss (dB).
    monitor_insertion_loss_db : extra monitor-line loss (dB); 3 dB for Cao et
                           al.'s X-basis time-window selection.
    interference_visibility : monitor-line interferometer visibility V
                           (1.0 = perfect; Cao et al. report ~0.99).
    dark_count_rate_hz   : dark-count rate (Hz).
    dead_time_s          : detector dead time (s). Default 100 ns, a typical
                           SNSPD reset time (Cao et al. use SNSPDs but do not
                           report a value).
    rep_rate_hz          : pulse repetition rate (Hz).
    p_signal, p_decoy_alphaalpha, p_decoy_00 : state-preparation probabilities.
    phi                  : monitor interferometer phase (rad); pi gives
                           constructive interference at D_M0.
    qber_threshold       : abort threshold on Ez.
    correction_efficiency_f : error-correction inefficiency f.
    epsilon_cor, epsilon_sec : finite-key security parameters.
    monitor_gain_source  : "sf_physical" (SF-circuit-derived decoy monitor
                           gains with a physical visibility model) or
                           "closed_form" (Gao et al. 2022).
    use_sf_monitor_gains : also take the signal-state D_M0 click probability
                           from the SF circuit (legacy option).
    basis_mode           : "passive" (beam splitter, Cao et al.) or "active"
                           (optical switch, Gao et al.).

    Attack parameters (all off by default):
    ir_attack_strength   : intercept-resend probability per round, p_IR.
                           Leaves the QBER unchanged and drives the decoy
                           visibility to 1 - p_IR (Gisin et al. 2004,
                           Eqs. 19-26).
    bs_attack_strength   : beam-splitting attack strength s. Does not change
                           the observables; used only by the rate ceiling in
                           security_bounds.beam_splitting_rate_ceiling.
    usd3_strength        : Branciard et al. 2007 USD3 data-line bit-faking
                           fraction q1 (perturbs the data-line click
                           probability).
    usd4b_strength       : Branciard et al. 2007 USD4b decoy-pair faking
                           fraction q3 (perturbs the monitor-line gains).
    backflash_strength   : backflash leak probability (stand-in for P_b in
                           Singh et al. 2025). Does not change the
                           observables; used only by
                           security_bounds.backflash_rate_ceiling.
    """

    mu: float = 0.5
    tB: float = 0.90
    distance_km: float = 80.0
    fibre_loss_exponent: float = 0.02  # Gao et al. 2022 convention
    detector_eff: float = 0.20
    detector_eff_monitor: float | None = None  # None -> detector_eff
    insertion_loss_db: float = 0.0
    monitor_insertion_loss_db: float = 0.0
    interference_visibility: float = 1.0
    dark_count_rate_hz: float = 900.0
    dead_time_s: float = 100e-9
    rep_rate_hz: float = 500e6
    p_signal: float = 0.5
    p_decoy_alphaalpha: float = 0.25
    p_decoy_00: float = 0.25
    phi: float = np.pi
    qber_threshold: float = 0.05
    correction_efficiency_f: float = 1.1
    epsilon_cor: float = 1e-15
    epsilon_sec: float = 1e-10
    monitor_gain_source: str = "sf_physical"
    use_sf_monitor_gains: bool = False
    basis_mode: str = "passive"
    ir_attack_strength: float = 0.0
    bs_attack_strength: float = 0.0
    usd3_strength: float = 0.0
    usd4b_strength: float = 0.0
    backflash_strength: float = 0.0

    @property
    def t_fiber(self) -> float:
        """Channel transmission excluding detector efficiency (Eve's attack
        surface under the trusted-detector assumption)."""
        return 10.0 ** (-self.fibre_loss_exponent * self.distance_km) * \
            10.0 ** (-self.insertion_loss_db / 10.0)

    def beam_splitting_ceiling_bps(self, round_rate_hz: float) -> float:
        """Beam-splitting rate ceiling in bits/s at this operating point
        (+inf when bs_attack_strength = 0)."""
        from security_bounds import beam_splitting_rate_ceiling
        ceiling_per_pulse = beam_splitting_rate_ceiling(
            t_fiber=self.t_fiber,
            tB=self.tB,
            detector_eff=self.detector_eff,
            p_signal=self.p_signal,
            bs_attack_strength=self.bs_attack_strength,
        )
        if ceiling_per_pulse == float("inf"):
            return float("inf")
        return ceiling_per_pulse * round_rate_hz

    @property
    def eta(self) -> float:
        """Data-line (D0) transmittance, including detector efficiency and
        insertion loss."""
        base_eta = self.detector_eff * 10.0 ** (-self.fibre_loss_exponent * self.distance_km)
        insertion_transmittance = 10.0 ** (-self.insertion_loss_db / 10.0)
        return base_eta * insertion_transmittance

    @property
    def eta_monitor(self) -> float:
        """Monitor-line (D1/D2) transmittance: shared fibre loss, monitor
        detector efficiency and monitor insertion loss."""
        eff = self.detector_eff_monitor if self.detector_eff_monitor is not None else self.detector_eff
        base_eta = eff * 10.0 ** (-self.fibre_loss_exponent * self.distance_km)
        insertion_transmittance = 10.0 ** (-self.monitor_insertion_loss_db / 10.0)
        return base_eta * insertion_transmittance

    def create_variant(self, **overrides) -> "COWParams":
        """
        Return a copy with the given fields overridden:

            variant = COWParams().create_variant(distance_km=120.0)
        """
        from dataclasses import replace
        return replace(self, **overrides)


# ----------------------------------------------------------------------------
# Result container for one full experiment
# ----------------------------------------------------------------------------

@dataclass
class ExperimentResult:
    """Output of one full Layer 1 -> 2 -> 3 run for one configuration."""

    params: COWParams
    observable: ObservableSnapshot
    metrics: ScaledMetrics
    bounds: dict[str, BoundResult]


# ----------------------------------------------------------------------------
# Core orchestration: run the full pipeline once
# ----------------------------------------------------------------------------

def run_experiment(
    params: COWParams,
    num_sf_pulses: int | None = None,
    N_target: int = 100_000_000,
    bounds: list[SecurityBound] | None = None,
    seed: int | None = None,
    exact_expected_values: bool = False,
) -> ExperimentResult:
    """
    Run the full pipeline once: SF simulation and sampling (Layer 1),
    detector physics (Layer 2), and every requested bound on the same
    record (Layer 3).

    Parameters
    ----------
    params : COWParams
    num_sf_pulses : int or None
        Number of sampled pulses. Defaults to N_target, which is required for
        a statistically valid finite-key bound; use a smaller value only for
        diagnostics.
    N_target : int
        Pulse count the experiment represents.
    bounds : list of SecurityBound, optional
        Defaults to [KatoBound, AzumaBound] built from params.
    seed : int, optional
        RNG seed.
    exact_expected_values : bool
        Use expected counts instead of random samples.

    Returns
    -------
    ExperimentResult
    """
    if bounds is None:
        bounds = [
            KatoBound(
                epsilon_cor=params.epsilon_cor,
                epsilon_sec=params.epsilon_sec,
                qber_threshold=params.qber_threshold,
                correction_efficiency_f=params.correction_efficiency_f,
            ),
            AzumaBound(
                epsilon_cor=params.epsilon_cor,
                epsilon_sec=params.epsilon_sec,
                qber_threshold=params.qber_threshold,
                correction_efficiency_f=params.correction_efficiency_f,
            ),
        ]

    if num_sf_pulses is None:
        num_sf_pulses = N_target

    p_dark = params.dark_count_rate_hz / params.rep_rate_hz

    obs = run_sf_simulation(
        mu=params.mu,
        tB=params.tB,
        eta=params.eta,
        eta_for_monitor=params.eta_monitor,
        interference_visibility=params.interference_visibility,
        p_dark=p_dark,
        num_pulses=num_sf_pulses,
        p_signal=params.p_signal,
        p_decoy_alphaalpha=params.p_decoy_alphaalpha,
        p_decoy_00=params.p_decoy_00,
        phi=params.phi,
        basis_mode=params.basis_mode,
        ir_attack_strength=params.ir_attack_strength,
        usd3_strength=params.usd3_strength,
        usd4b_strength=params.usd4b_strength,
        use_sf_monitor_gains=params.use_sf_monitor_gains,
        monitor_gain_source=params.monitor_gain_source,
        seed=seed,
        exact_expected_values=exact_expected_values,
    )

    metrics = apply_detector_physics(
        obs,
        dark_count_rate_hz=params.dark_count_rate_hz,
        dead_time_s=params.dead_time_s,
        rep_rate_hz=params.rep_rate_hz,
        N_target=N_target,
    )

    bound_results = evaluate_all_bounds(metrics, bounds)

    return ExperimentResult(
        params=params,
        observable=obs,
        metrics=metrics,
        bounds=bound_results,
    )


# ----------------------------------------------------------------------------
# Parameter sweeps
# ----------------------------------------------------------------------------

def parameter_sweep(
    base_params: COWParams,
    param_name: str,
    param_values: list[float],
    num_sf_pulses: int = 1_000_000,
    N_target: int = 100_000_000,
    bounds: list[SecurityBound] | None = None,
    seed: int | None = None,
) -> list[ExperimentResult]:
    """
    Run `run_experiment` once per value of `param_name`, with a fresh
    simulation per point.

        results = parameter_sweep(COWParams(), "distance_km", [20, 40, 60])
    """
    results = []
    for value in param_values:
        variant = base_params.create_variant(**{param_name: value})
        result = run_experiment(
            variant,
            num_sf_pulses=num_sf_pulses,
            N_target=N_target,
            bounds=bounds,
            seed=seed,
        )
        results.append(result)
    return results


# ----------------------------------------------------------------------------
# Stationarity check
# ----------------------------------------------------------------------------

@dataclass
class StationarityReport:
    """Spread of Ez and G_DM0_00 across independent batches at fixed params."""

    Ez_values: list[float]
    Ez_mean: float
    Ez_std: float
    G_DM0_00_values: list[float]
    G_DM0_00_mean: float
    G_DM0_00_std: float
    n_batches: int
    pulses_per_batch: int


def check_stationarity(
    params: COWParams,
    n_batches: int = 5,
    pulses_per_batch: int = 1_000_000,
    base_seed: int = 0,
) -> StationarityReport:
    """
    Run `n_batches` independently seeded simulations at the same params and
    report the spread of Ez and G_DM0_00. A spread comparable to the
    per-batch Wilson CI confirms that the CI captures the sampling
    uncertainty.
    """
    Ez_values = []
    G_DM0_00_values = []
    p_dark = params.dark_count_rate_hz / params.rep_rate_hz

    for i in range(n_batches):
        obs = run_sf_simulation(
            mu=params.mu,
            tB=params.tB,
            eta=params.eta,
            p_dark=p_dark,
            num_pulses=pulses_per_batch,
            p_signal=params.p_signal,
            p_decoy_alphaalpha=params.p_decoy_alphaalpha,
            p_decoy_00=params.p_decoy_00,
            phi=params.phi,
            basis_mode=params.basis_mode,
            seed=base_seed + i,
        )
        metrics = apply_detector_physics(
            obs,
            dark_count_rate_hz=params.dark_count_rate_hz,
            dead_time_s=params.dead_time_s,
            rep_rate_hz=params.rep_rate_hz,
            N_target=pulses_per_batch,  # no scaling needed for this check
        )
        Ez_values.append(metrics.Ez)
        G_DM0_00_values.append(metrics.G_DM0_00)

    return StationarityReport(
        Ez_values=Ez_values,
        Ez_mean=float(np.mean(Ez_values)),
        Ez_std=float(np.std(Ez_values)),
        G_DM0_00_values=G_DM0_00_values,
        G_DM0_00_mean=float(np.mean(G_DM0_00_values)),
        G_DM0_00_std=float(np.std(G_DM0_00_values)),
        n_batches=n_batches,
        pulses_per_batch=pulses_per_batch,
    )


# ----------------------------------------------------------------------------
# (mu, tB) optimizer
# ----------------------------------------------------------------------------
#
# Published COW key-rate-vs-distance curves optimize (mu, tB) at every
# distance; a single fixed operating point generally saturates the
# phase-error bound.

@dataclass
class OptimizationResult:
    """Optimal (mu, tB) for one distance and one bound."""

    distance_km: float
    bound_name: str
    mu_optimal: float
    tB_optimal: float
    key_rate_per_pulse: float
    Ep_upper: float
    Ez: float
    secure: bool
    n_evaluations: int


def _evaluate_key_rate_at(
    mu: float,
    tB: float,
    base_params: COWParams,
    bound_class: type,
    num_sf_pulses: int,
    N_target: int,
    seed: int | None,
    exact_expected_values: bool = False,
) -> tuple[float, BoundResult]:
    """Run the pipeline once at (mu, tB); return (key rate per pulse,
    BoundResult)."""
    params = base_params.create_variant(mu=mu, tB=tB)
    bound = bound_class(
        epsilon_cor=params.epsilon_cor,
        epsilon_sec=params.epsilon_sec,
        qber_threshold=params.qber_threshold,
        correction_efficiency_f=params.correction_efficiency_f,
    )
    result = run_experiment(
        params,
        num_sf_pulses=num_sf_pulses,
        N_target=N_target,
        bounds=[bound],
        seed=seed,
        exact_expected_values=exact_expected_values,
    )
    bound_result = result.bounds[bound.name()]
    return bound_result.key_rate_per_pulse, bound_result


def optimize_mu_tB(
    base_params: COWParams,
    bound_class: type = KatoBound,
    mu_bounds: tuple[float, float] = (1e-4, 0.6),
    tB_bounds: tuple[float, float] = (0.05, 0.98),
    num_sf_pulses: int = 1_000_000,
    N_target: int = int(1e9),
    seed: int | None = 0,
    maxiter: int = 40,
    popsize: int = 12,
    verbose: bool = False,
    progress_every: int = 25,
) -> OptimizationResult:
    """
    Maximize the secure key rate per pulse over (mu, tB) at fixed distance
    and hardware, using scipy's differential_evolution (global, robust to the
    flat zero-rate plateaus of insecure regions).

    The search uses expected-value counts at N_target with a fixed seed so
    the objective is deterministic; the reported result at the optimum uses
    genuine binomial sampling. mu is searched in log10 space.

    If `verbose`, prints progress every `progress_every` evaluations.
    """
    from scipy.optimize import differential_evolution
    import time as _time

    eval_count = 0
    best_rate_so_far = 0.0
    t_start = _time.time()
    # Fixed seed for every search evaluation keeps the objective deterministic.
    search_seed = seed if seed is not None else 0

    def objective(x):
        nonlocal eval_count, best_rate_so_far
        log10_mu, tB = x
        mu = 10.0 ** log10_mu
        eval_count += 1

        # Search phase: N_target pulses with exact expected values.
        rate, bound_result = _evaluate_key_rate_at(
            mu, tB, base_params, bound_class, N_target, N_target, search_seed, exact_expected_values=True
        )

        if rate > best_rate_so_far:
            best_rate_so_far = rate
        if verbose and eval_count % progress_every == 0:
            elapsed = _time.time() - t_start
            print(
                f"    [search] eval {eval_count:>5d}  elapsed {elapsed:>6.1f}s  "
                f"best rate so far: {best_rate_so_far:.4e}",
                flush=True,
            )
        if rate <= 0.0:
            # An all-zero population has zero spread and makes
            # differential_evolution stop early. For insecure points return
            # a small smooth signal, the key-rate bracket 1 - h(Ep) - f h(Ez),
            # scaled by 1e-9 so it never outranks a genuine positive rate.
            f_ec = base_params.correction_efficiency_f
            bracket = 1.0 - binary_entropy(bound_result.Ep_upper) - f_ec * binary_entropy(bound_result.Ez)
            return -1e-9 * bracket
        return -rate  # differential_evolution minimises

    if verbose:
        print(f"  Starting search (popsize={popsize}, maxiter={maxiter}, "
              f"~{popsize * 2} evals/generation)...", flush=True)

    # Search log10(mu): mu_bounds spans several decades and the secure region
    # is narrow, so linear sampling would badly undersample small mu.
    log10_mu_bounds = (np.log10(mu_bounds[0]), np.log10(mu_bounds[1]))

    res = differential_evolution(
        objective,
        bounds=[log10_mu_bounds, tB_bounds],
        maxiter=maxiter,
        popsize=popsize,
        seed=seed,
        # tol=0 disables the relative-spread early stop, which triggers
        # falsely when the whole initial population misses the secure region.
        tol=0,
        polish=True,
        workers=1,
    )

    if verbose:
        elapsed = _time.time() - t_start
        print(f"  Search done: {eval_count} evaluations in {elapsed:.1f}s. "
              f"Refining best point...", flush=True)

    mu_opt, tB_opt = float(10.0 ** res.x[0]), float(res.x[1])

    # Final phase: genuine binomial sampling at N_target for the reported result.
    final_rate, final_bound_result = _evaluate_key_rate_at(
        mu_opt, tB_opt, base_params, bound_class, N_target, N_target, seed, exact_expected_values=False
    )

    return OptimizationResult(
        distance_km=base_params.distance_km,
        bound_name=bound_class(epsilon_cor=1e-15, epsilon_sec=1e-10).name(),
        mu_optimal=mu_opt,
        tB_optimal=tB_opt,
        key_rate_per_pulse=final_rate,
        Ep_upper=final_bound_result.Ep_upper,
        Ez=final_bound_result.Ez,
        secure=final_bound_result.secure,
        n_evaluations=eval_count,
    )


def optimize_then_refine(
    base_params: COWParams,
    bound_class: type = KatoBound,
    mu_bounds: tuple[float, float] = (1e-4, 0.6),
    tB_bounds: tuple[float, float] = (0.05, 0.98),
    num_sf_pulses: int = 1_000_000,
    N_target: int = int(1e9),
    seed: int | None = 0,
    maxiter: int = 40,
    popsize: int = 12,
    verbose: bool = False,
    progress_every: int = 25,
) -> OptimizationResult:
    """Wrapper around `optimize_mu_tB` that also reports total time when
    verbose (kept for backward compatibility)."""
    import time as _time
    t_start = _time.time()

    result = optimize_mu_tB(
        base_params, bound_class, mu_bounds, tB_bounds,
        num_sf_pulses=num_sf_pulses, N_target=N_target,
        seed=seed, maxiter=maxiter, popsize=popsize,
        verbose=verbose, progress_every=progress_every,
    )

    if verbose:
        elapsed = _time.time() - t_start
        print(f"  Done. Total time for this distance: {elapsed:.1f}s", flush=True)

    return result


def optimize_sweep_distance(
    base_params: COWParams,
    distances_km: list[float],
    bound_class: type = KatoBound,
    mu_bounds: tuple[float, float] = (1e-4, 0.6),
    tB_bounds: tuple[float, float] = (0.05, 0.98),
    num_sf_pulses: int = 1_000_000,
    N_target: int = int(1e9),
    seed: int | None = 0,
    maxiter: int = 40,
    popsize: int = 12,
    verbose: bool = False,
    progress_every: int = 25,
) -> list[OptimizationResult]:
    """
    Optimize (mu, tB) independently at every distance in `distances_km`
    (key rate vs distance, as in Gao et al. Figs. 2-5).
    """
    import time as _time
    t_sweep_start = _time.time()

    results = []
    for i, d in enumerate(distances_km):
        if verbose:
            elapsed = _time.time() - t_sweep_start
            print(
                f"\n--- Distance {i+1}/{len(distances_km)}: L={d:.0f}km "
                f"(sweep elapsed: {elapsed:.1f}s) ---", flush=True
            )
        variant = base_params.create_variant(distance_km=d)
        result = optimize_then_refine(
            variant, bound_class, mu_bounds, tB_bounds,
            num_sf_pulses, N_target,
            seed, maxiter, popsize, verbose=verbose, progress_every=progress_every,
        )
        if verbose:
            print(
                f"  -> mu={result.mu_optimal:.6f}  tB={result.tB_optimal:.4f}  "
                f"rate={result.key_rate_per_pulse:.4e}  secure={result.secure}",
                flush=True,
            )
        results.append(result)

    if verbose:
        total_elapsed = _time.time() - t_sweep_start
        print(f"\nFull sweep complete in {total_elapsed:.1f}s.", flush=True)

    return results
