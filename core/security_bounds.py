"""
security_bounds.py
===================
Layer 3 of the COW-QKD framework: security bound calculation.

Bounds implemented, all acting on the same ScaledMetrics record:
  - KatoBound          : Cao et al. 2026 finite-key bound with Kato's
                         inequality, Eqs. (1)-(20).
  - AzumaBound         : same pipeline with the Azuma-Hoeffding inequality.
  - CaoAsymptoticBound : Cao et al.'s Eq. (18) with no finite-key correction.
  - GaoAsymptoticBound : Gao et al. 2022 asymptotic bound, Eqs. (6),(7),(12).

The count-level Eq. (6)/(7) bounds (n_D1_0x_lower, n_D2_0x_upper) are exactly
N times Gao's gain-level Eq. (12), obtained by substituting Q_w = n_w/(N P_w).

Notation: Cao et al.'s monitor detectors D1 (bright port) and D2 (dark port)
are DM0 and DM1 in this codebase.

References:
    Cao et al., Sci. Adv. (2026), arXiv:2601.06772.
    Gao et al., Opt. Express 30, 23783 (2022).
    Li et al., Phys. Rev. Research 6, 013022 (2024).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np

from detector import ScaledMetrics


# ----------------------------------------------------------------------------
# Small math helpers
# ----------------------------------------------------------------------------

def binary_entropy(x: float) -> float:
    """Shannon binary entropy h(x)."""
    x = float(np.clip(x, 0.0, 1.0))
    if x <= 0.0 or x >= 1.0:
        return 0.0
    return -x * np.log2(x) - (1.0 - x) * np.log2(1.0 - x)


def normalisation_factors(mu: float) -> tuple[float, float]:
    """N+ = 2(1+e^-mu), N- = 2(1-e^-mu) (Cao et al. Eq. 2)."""
    n_plus = 2.0 * (1.0 + np.exp(-mu))
    n_minus = 2.0 * (1.0 - np.exp(-mu))
    return n_plus, n_minus


# ----------------------------------------------------------------------------
# Eq. (6)-(7): bounds on n^{D1}_{0x}, n^{D2}_{0x} from decoy-state counts
# ----------------------------------------------------------------------------

def n_D1_0x_lower(mu: float, n_D1_aa: float, n_D1_00: float, P_aa: float, P_00: float, N: float) -> float:
    """
    Eq. (6)/(16): lower bound on n^{D1}_{0x}.

        n_0x^D1 = -sqrt(N) (N-/N+) (e^mu sqrt(n_aa/P_aa) + sqrt(n_00/P_00))
                  + (1/N+) (e^mu n_aa/P_aa + e^-mu n_00/P_00
                            - 2 sqrt(n_00/P_00 * n_aa/P_aa))
    """
    n_plus, n_minus = normalisation_factors(mu)

    term_sqrtN = -np.sqrt(N) * (n_minus / n_plus) * (
        np.exp(mu) * np.sqrt(max(n_D1_aa / P_aa, 0.0))
        + np.sqrt(max(n_D1_00 / P_00, 0.0))
    )
    term_rest = (1.0 / n_plus) * (
        np.exp(mu) * n_D1_aa / P_aa
        + np.exp(-mu) * n_D1_00 / P_00
        - 2.0 * np.sqrt(max(n_D1_00 / P_00, 0.0) * max(n_D1_aa / P_aa, 0.0))
    )
    return term_sqrtN + term_rest


def n_D2_0x_upper(mu: float, n_D2_aa: float, n_D2_00: float, P_aa: float, P_00: float, N: float) -> float:
    """
    Eq. (7)/(17): upper bound on n^{D2}_{0x}.

        n_0x^D2 = sqrt(N) (N-/N+) (e^mu sqrt(n_aa/P_aa) + sqrt(n_00/P_00))
                  + N (N-)^2 e^mu / (4 N+)
                  + (1/N+) (e^mu n_aa/P_aa + e^-mu n_00/P_00
                            + 2 sqrt(n_00/P_00 * n_aa/P_aa))
    """
    n_plus, n_minus = normalisation_factors(mu)

    term_sqrtN = np.sqrt(N) * (n_minus / n_plus) * (
        np.exp(mu) * np.sqrt(max(n_D2_aa / P_aa, 0.0))
        + np.sqrt(max(n_D2_00 / P_00, 0.0))
    )
    term_floor = N * (n_minus ** 2) * np.exp(mu) / (4.0 * n_plus)
    term_rest = (1.0 / n_plus) * (
        np.exp(mu) * n_D2_aa / P_aa
        + np.exp(-mu) * n_D2_00 / P_00
        + 2.0 * np.sqrt(max(n_D2_00 / P_00, 0.0) * max(n_D2_aa / P_aa, 0.0))
    )
    return term_sqrtN + term_floor + term_rest


# ----------------------------------------------------------------------------
# Eq. (8)-(13): Kato's inequality -- asymmetric upper/lower deviation
# ----------------------------------------------------------------------------

def _kato_C(k: float, Gamma_k: float, eps_a: float) -> float:
    """C, defined just below Eq. (13)."""
    ln_eps = np.log(eps_a)
    inner = -(k ** 2) * ln_eps * (9.0 * Gamma_k * (k - Gamma_k) - 2.0 * k * ln_eps)
    return float(np.sqrt(max(inner, 0.0)))


def kato_a1_b1(k: float, Gamma_k: float, eps_a: float) -> tuple[float, float]:
    """Eq. (10)-(11): a1, b1 for the upper deviation Delta1."""
    if k <= 0:
        return 0.0, 0.0
    ln_eps = np.log(eps_a)
    C = _kato_C(k, Gamma_k, eps_a)
    denom = 4.0 * (9.0 * k - 8.0 * ln_eps) * (9.0 * Gamma_k * (k - Gamma_k) - 2.0 * k * ln_eps)
    if denom == 0 or not np.isfinite(denom):
        a1 = 0.0
    else:
        numer = (
            3.0 * (72.0 * np.sqrt(k) * Gamma_k * (k - Gamma_k) * ln_eps
                   - 16.0 * k ** 1.5 * ln_eps ** 2
                   + 9.0 * np.sqrt(2.0) * (k - 2.0 * Gamma_k) * C)
        )
        a1 = numer / denom
        if not np.isfinite(a1):
            a1 = 0.0
    inner_b1 = 18.0 * a1 ** 2 * k - (16.0 * a1 ** 2 + 24.0 * a1 * np.sqrt(k) + 9.0 * k) * ln_eps
    b1 = float(np.sqrt(max(inner_b1, 0.0))) / (3.0 * np.sqrt(2.0 * k)) if k > 0 else 0.0
    return a1, b1


def kato_a2_b2(k: float, Gamma_k: float, eps_a: float) -> tuple[float, float]:
    """Eq. (12)-(13): a2, b2 for the lower deviation Delta2."""
    if k <= 0:
        return 0.0, 0.0
    ln_eps = np.log(eps_a)
    C = _kato_C(k, Gamma_k, eps_a)
    denom = 4.0 * (9.0 * k - 8.0 * ln_eps) * (9.0 * Gamma_k * (k - Gamma_k) - 2.0 * k * ln_eps)
    if denom == 0 or not np.isfinite(denom):
        a2 = 0.0
    else:
        numer = (
            -3.0 * (72.0 * np.sqrt(k) * Gamma_k * (k - Gamma_k) * ln_eps
                    - 16.0 * k ** 1.5 * ln_eps ** 2
                    - 9.0 * np.sqrt(2.0) * (k - 2.0 * Gamma_k) * C)
        )
        a2 = numer / denom
        if not np.isfinite(a2):
            a2 = 0.0
    inner_b2 = 18.0 * a2 ** 2 * k - (16.0 * a2 ** 2 - 24.0 * a2 * np.sqrt(k) + 9.0 * k) * ln_eps
    b2 = float(np.sqrt(max(inner_b2, 0.0))) / (3.0 * np.sqrt(2.0 * k)) if k > 0 else 0.0
    return a2, b2


def kato_delta_upper(k: float, Gamma_k: float, eps_a: float) -> float:
    """Delta1 = [b1 + a1 (2 Gamma_k/k - 1)] sqrt(k), Eq. (8),(10)-(11)."""
    if k <= 0:
        return 0.0
    a1, b1 = kato_a1_b1(k, Gamma_k, eps_a)
    d = (b1 + a1 * (2.0 * Gamma_k / k - 1.0)) * np.sqrt(k)
    return float(d) if np.isfinite(d) else float(np.sqrt(0.5 * k * np.log(1.0 / eps_a)))


def kato_delta_lower(k: float, Gamma_k: float, eps_a: float) -> float:
    """Delta2 = [b2 + a2 (2 Gamma_k/k - 1)] sqrt(k), Eq. (9),(12)-(13)."""
    if k <= 0:
        return 0.0
    a2, b2 = kato_a2_b2(k, Gamma_k, eps_a)
    d = (b2 + a2 * (2.0 * Gamma_k / k - 1.0)) * np.sqrt(k)
    return float(d) if np.isfinite(d) else float(np.sqrt(0.5 * k * np.log(1.0 / eps_a)))


# ----------------------------------------------------------------------------
# Eq. (18)-(20): phase error rate
# ----------------------------------------------------------------------------

def expected_phase_error(
    mu: float,
    n_D2_0x_bound: float,
    n_D1_0x_bound: float,
    n_D1_0z: float,
    n_D1_1z: float,
    n_D2_0z: float,
    n_D2_1z: float,
) -> float:
    """
    Eq. (18):
        E*_p = (n^{D2*}_0x - n^{D1*}_0x + n^{D1}_0z + n^{D1}_1z)
               / (n^{D1}_0z + n^{D2}_0z + n^{D1}_1z + n^{D2}_1z)

    n^{Di}_{0z}, n^{Di}_{1z} are measured (here, simulated) observables.
    """
    numerator = n_D2_0x_bound - n_D1_0x_bound + n_D1_0z + n_D1_1z
    denominator = n_D1_0z + n_D2_0z + n_D1_1z + n_D2_1z
    if denominator <= 0:
        return 0.499
    return float(np.clip(numerator / denominator, 0.0, 0.499))


def observed_phase_error_bound(E_p_expected: float, n_z: float, eps2: float) -> float:
    """
    Eq. (19)-(20): E_p = (n_z E*_p + Delta_p) / n_z with
    Delta_p = sqrt(0.5 n_z ln(1/eps2)).
    """
    if n_z <= 0:
        return 0.499
    n_p_expected = n_z * E_p_expected
    delta_p = np.sqrt(0.5 * n_z * np.log(1.0 / eps2))
    n_p_upper = n_p_expected + delta_p
    return float(np.clip(n_p_upper / n_z, 0.0, 0.499))


# ----------------------------------------------------------------------------
# Result container
# ----------------------------------------------------------------------------

@dataclass
class BoundResult:
    """Output of a single SecurityBound.evaluate() call."""

    bound_name: str
    Ep_upper: float
    Ez: float
    leak_EC: float
    key_length: float
    key_rate_per_pulse: float
    secure: bool
    epsilon_cor: float
    epsilon_sec: float
    n_z: float
    n_z_ci: float


# ----------------------------------------------------------------------------
# Abstract base class
# ----------------------------------------------------------------------------

class SecurityBound(ABC):
    """
    Interface for a security-bound method. Subclasses supply the decoy-count
    deviation terms; the Eq. (6)-(20) pipeline is implemented once in
    `evaluate()`.
    """

    def __init__(
        self,
        epsilon_cor: float,
        epsilon_sec: float,
        qber_threshold: float = 0.05,
        correction_efficiency_f: float = 1.1,
    ):
        self.epsilon_cor = epsilon_cor
        self.epsilon_sec = epsilon_sec
        self.qber_threshold = qber_threshold
        self.f = correction_efficiency_f

    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def _decoy_deviation_upper(self, k_total: float, Gamma_k: float, eps_a: float) -> float:
        """Deviation term for the upper bound on a decoy count."""
        ...

    @abstractmethod
    def _decoy_deviation_lower(self, k_total: float, Gamma_k: float, eps_a: float) -> float:
        """Deviation term for the lower bound on a decoy count."""
        ...

    def _apply_observed_value_correction(self, E_p_expected: float, n_z: float, eps2: float) -> float:
        """Eq. (19)-(20) expected-to-observed correction (finite-key default)."""
        return observed_phase_error_bound(E_p_expected, n_z, eps2)

    def _finite_key_overhead(self) -> float:
        """Eq. (1): log2(2/eps_cor) + 2 log2(5/eps_sec) (finite-key default)."""
        return np.log2(2.0 / self.epsilon_cor) + 2.0 * np.log2(5.0 / self.epsilon_sec)

    def evaluate(self, metrics: ScaledMetrics) -> BoundResult:
        """
        Cao et al. 2026 pipeline, count-level throughout:
        widen decoy counts (Eq. 8-9, 14-17) -> Eq. (6)/(7) -> Eq. (18)
        phase-error ratio -> Eq. (19)-(20) observed-value correction ->
        Eq. (1) key length.

        Note: Cao's Eq. (18) and Gao's Eq. (9) have different denominators,
        so this Ep need not match GaoAsymptoticBound's Ep on the same data.
        """
        mu = metrics.mu
        eps_a = self.epsilon_sec / 10.0  # eps1 = eps_sec/10
        eps2 = self.epsilon_sec / 10.0

        P_aa = metrics.n_alphaalpha / metrics.N_target if metrics.N_target > 0 else 0.0
        P_00 = metrics.n_00 / metrics.N_target if metrics.N_target > 0 else 0.0
        N = metrics.N_target

        k_aa = max(metrics.n_alphaalpha, 1.0)
        k_00 = max(metrics.n_00, 1.0)

        delta_aa_up = self._decoy_deviation_upper(k_aa, metrics.n_DM1_alphaalpha, eps_a)
        delta_aa_lo = self._decoy_deviation_lower(k_aa, metrics.n_DM0_alphaalpha, eps_a)
        delta_00_up = self._decoy_deviation_upper(k_00, metrics.n_DM1_00, eps_a)
        delta_00_lo = self._decoy_deviation_lower(k_00, metrics.n_DM0_00, eps_a)

        n_D2_aa_bound = min(metrics.n_DM1_alphaalpha + delta_aa_up, k_aa)
        n_D1_aa_bound = max(metrics.n_DM0_alphaalpha - delta_aa_lo, 0.0)
        n_D2_00_bound = min(metrics.n_DM1_00 + delta_00_up, k_00)
        n_D1_00_bound = max(metrics.n_DM0_00 - delta_00_lo, 0.0)

        # Eq. (6)/(7) evaluated on the widened decoy counts.
        n_D1_0x_bound = n_D1_0x_lower(mu, n_D1_aa_bound, n_D1_00_bound, P_aa, P_00, N)
        n_D2_0x_bound = n_D2_0x_upper(mu, n_D2_aa_bound, n_D2_00_bound, P_aa, P_00, N)
        n_D1_0x_bound = max(n_D1_0x_bound, 0.0)  # a count cannot be negative

        E_p_expected = expected_phase_error(
            mu,
            n_D2_0x_bound, n_D1_0x_bound,
            metrics.n_DM0_0z, metrics.n_DM0_1z,
            metrics.n_DM1_0z, metrics.n_DM1_1z,
        )

        Ep_upper = self._apply_observed_value_correction(E_p_expected, metrics.n_z, eps2)

        leak_EC = self.f * metrics.n_z * binary_entropy(metrics.Ez)

        key_length = (
            metrics.n_z * (1.0 - binary_entropy(Ep_upper))
            - leak_EC
            - self._finite_key_overhead()
        )

        secure = (
            key_length > 0
            and metrics.Ez <= self.qber_threshold
            and Ep_upper < 0.499
        )

        return BoundResult(
            bound_name=self.name(),
            Ep_upper=Ep_upper,
            Ez=metrics.Ez,
            leak_EC=leak_EC,
            key_length=key_length if secure else 0.0,
            key_rate_per_pulse=(key_length / metrics.N_target) if secure else 0.0,
            secure=secure,
            epsilon_cor=self.epsilon_cor,
            epsilon_sec=self.epsilon_sec,
            n_z=metrics.n_z,
            n_z_ci=metrics.n_z_ci,
        )


# ----------------------------------------------------------------------------
# Kato's inequality (Cao et al. 2026 Eq. 8-13)
# ----------------------------------------------------------------------------

class KatoBound(SecurityBound):
    """Kato's inequality, as in Cao et al. 2026 Eq. (8)-(13)."""

    def __init__(
        self,
        epsilon_cor: float = 1e-15,
        epsilon_sec: float = 1e-10,
        qber_threshold: float = 0.05,
        correction_efficiency_f: float = 1.1,
    ):
        super().__init__(epsilon_cor, epsilon_sec, qber_threshold, correction_efficiency_f)

    def name(self) -> str:
        return "Kato"

    def _decoy_deviation_upper(self, k_total: float, Gamma_k: float, eps_a: float) -> float:
        return kato_delta_upper(k_total, Gamma_k, eps_a)

    def _decoy_deviation_lower(self, k_total: float, Gamma_k: float, eps_a: float) -> float:
        return kato_delta_lower(k_total, Gamma_k, eps_a)


# ----------------------------------------------------------------------------
# Azuma-Hoeffding inequality
# ----------------------------------------------------------------------------

class AzumaBound(SecurityBound):
    """
    Azuma-Hoeffding inequality: deviation = sqrt(0.5 k_total ln(1/eps_a)) for
    both directions. Unlike Kato it ignores the observed count Gamma_k, so it
    is loosest for the sparse vacuum-decoy counts (Li et al. 2024).
    """

    def __init__(
        self,
        epsilon_cor: float = 1e-15,
        epsilon_sec: float = 1e-10,
        qber_threshold: float = 0.05,
        correction_efficiency_f: float = 1.1,
    ):
        super().__init__(epsilon_cor, epsilon_sec, qber_threshold, correction_efficiency_f)

    def name(self) -> str:
        return "Azuma"

    @staticmethod
    def _azuma_deviation(k_total: float, eps_a: float) -> float:
        if k_total <= 0:
            return 0.0
        return float(np.sqrt(0.5 * k_total * np.log(1.0 / eps_a)))

    def _decoy_deviation_upper(self, k_total: float, Gamma_k: float, eps_a: float) -> float:
        return self._azuma_deviation(k_total, eps_a)

    def _decoy_deviation_lower(self, k_total: float, Gamma_k: float, eps_a: float) -> float:
        return self._azuma_deviation(k_total, eps_a)


# ----------------------------------------------------------------------------
# Cao et al. 2026 asymptotic (infinite-data) limit
# ----------------------------------------------------------------------------

class CaoAsymptoticBound(SecurityBound):
    """
    Cao et al.'s Eq. (18) pipeline with no finite-key correction: no decoy
    widening, no Delta_p, no epsilon overhead. The error-correction leakage
    f n_z h(Ez) is kept. Shares Kato's normalization, so it is the
    like-for-like asymptotic reference for KatoBound.
    """

    def __init__(
        self,
        epsilon_cor: float = 1e-15,
        epsilon_sec: float = 1e-10,
        qber_threshold: float = 0.05,
        correction_efficiency_f: float = 1.1,
    ):
        super().__init__(epsilon_cor, epsilon_sec, qber_threshold, correction_efficiency_f)

    def name(self) -> str:
        return "CaoAsymptotic"

    def _decoy_deviation_upper(self, k_total: float, Gamma_k: float, eps_a: float) -> float:
        return 0.0

    def _decoy_deviation_lower(self, k_total: float, Gamma_k: float, eps_a: float) -> float:
        return 0.0

    def _apply_observed_value_correction(self, E_p_expected: float, n_z: float, eps2: float) -> float:
        return float(np.clip(E_p_expected, 0.0, 0.499))

    def _finite_key_overhead(self) -> float:
        return 0.0


# ----------------------------------------------------------------------------
# Gao et al. 2022 Eq. (6),(7),(12), pure asymptotic, gain level
# ----------------------------------------------------------------------------

def _gao_eq12_bounds(mu: float, G_M1_aa: float, G_M1_00: float,
                      G_M0_aa: float, G_M0_00: float) -> tuple[float, float]:
    """Gao Eq. (12): Cauchy-Schwarz bounds on the X-basis gains Q_0x."""
    n_plus, n_minus = normalisation_factors(mu)

    Q_M1_0x_upper = (1.0 / n_plus) * (
        np.exp(mu / 2.0) * np.sqrt(max(G_M1_aa, 0.0))
        + np.exp(-mu / 2.0) * np.sqrt(max(G_M1_00, 0.0))
    ) ** 2 + (n_minus / n_plus) * (
        np.exp(mu) * n_minus / 4.0
        + np.exp(mu) * np.sqrt(max(G_M1_aa, 0.0))
        + np.sqrt(max(G_M1_00, 0.0))
    )

    Q_M0_0x_lower = (1.0 / n_plus) * (
        np.exp(mu) * G_M0_aa + np.exp(-mu) * G_M0_00
        - 2.0 * np.sqrt(max(G_M0_00, 0.0) * max(G_M0_aa, 0.0))
    ) - (n_minus / n_plus) * (
        np.exp(mu) * np.sqrt(max(G_M0_aa, 0.0)) + np.sqrt(max(G_M0_00, 0.0))
    )
    # A lower bound on a click probability is floored at 0; a negative value
    # would artificially inflate the Eq. (7) numerator.
    Q_M0_0x_lower = max(Q_M0_0x_lower, 0.0)

    return Q_M1_0x_upper, Q_M0_0x_lower


class GaoAsymptoticBound(SecurityBound):
    """
    Gao et al. 2022, Eq. (6)-(7),(12), asymptotic case with no finite-key
    correction. Used as a reference against Gao's published curves.
    """

    def name(self) -> str:
        return "GaoAsymptotic"

    def _decoy_deviation_upper(self, k_total: float, Gamma_k: float, eps_a: float) -> float:
        return 0.0

    def _decoy_deviation_lower(self, k_total: float, Gamma_k: float, eps_a: float) -> float:
        return 0.0

    def evaluate(self, metrics: ScaledMetrics) -> BoundResult:
        mu = metrics.mu

        # Eq. (12) on gains.
        Q_M1_0x, Q_M0_0x = _gao_eq12_bounds(
            mu,
            metrics.G_DM1_alphaalpha, metrics.G_DM1_00,
            metrics.G_DM0_alphaalpha, metrics.G_DM0_00,
        )
        n_plus, _ = normalisation_factors(mu)

        # Eq. (7): signal-state monitor gains use the number of |0z>/|1z>
        # pulses sent as denominator.
        G_M0_0z = metrics.n_DM0_0z / metrics.n_0z_total if metrics.n_0z_total > 0 else 0.0
        G_M1_0z = metrics.n_DM1_0z / metrics.n_0z_total if metrics.n_0z_total > 0 else 0.0
        G_M0_1z = metrics.n_DM0_1z / metrics.n_1z_total if metrics.n_1z_total > 0 else 0.0
        G_M1_1z = metrics.n_DM1_1z / metrics.n_1z_total if metrics.n_1z_total > 0 else 0.0

        numerator = n_plus * (Q_M1_0x - Q_M0_0x) + 2.0 * (G_M0_0z + G_M0_1z)
        denominator = 2.0 * (G_M0_0z + G_M1_0z + G_M0_1z + G_M1_1z)
        Ep_upper = float(np.clip(numerator / denominator, 0.0, 0.499)) if denominator > 0 else 0.499

        # Eq. (6): R = Q [1 - h(Ep) - f h(Ez)].
        Q = metrics.n_z / metrics.N_target if metrics.N_target > 0 else 0.0
        key_rate_per_pulse = Q * (1.0 - binary_entropy(Ep_upper) - self.f * binary_entropy(metrics.Ez))
        key_rate_per_pulse = max(key_rate_per_pulse, 0.0)

        secure = (
            key_rate_per_pulse > 0
            and metrics.Ez <= self.qber_threshold
            and Ep_upper < 0.499
        )

        return BoundResult(
            bound_name=self.name(),
            Ep_upper=Ep_upper,
            Ez=metrics.Ez,
            leak_EC=self.f * metrics.n_z * binary_entropy(metrics.Ez),
            key_length=key_rate_per_pulse * metrics.N_target if secure else 0.0,
            key_rate_per_pulse=key_rate_per_pulse if secure else 0.0,
            secure=secure,
            epsilon_cor=self.epsilon_cor,
            epsilon_sec=self.epsilon_sec,
            n_z=metrics.n_z,
            n_z_ci=metrics.n_z_ci,
        )


# ----------------------------------------------------------------------------
# Convenience: run several bounds on the same ScaledMetrics
# ----------------------------------------------------------------------------

def evaluate_all_bounds(
    metrics: ScaledMetrics, bounds: list[SecurityBound]
) -> dict[str, BoundResult]:
    """Evaluate every bound on the same ScaledMetrics record."""
    return {bound.name(): bound.evaluate(metrics) for bound in bounds}


# ----------------------------------------------------------------------------
# Beam-splitting attack: theoretical rate ceiling
# ----------------------------------------------------------------------------

def beam_splitting_rate_ceiling(
    t_fiber: float,
    tB: float,
    detector_eff: float,
    p_signal: float,
    bs_attack_strength: float,
) -> float:
    """
    Beam-splitting attack rate ceiling, Gisin et al., arXiv:quant-ph/0411022
    (2004), Sec. III.C.1, Eqs. 15-18.

    The attack leaves Bob's statistics unchanged, so it is expressed as a
    ceiling on the achievable rate rather than as a simulation perturbation:

        R_opt(s) = t tB eta_d p_signal / (4 s (1-t))

    s = bs_attack_strength is the fraction of channel loss attributed to Eve
    (s = 1 recovers Gisin Eq. 18; s < 1 is an extension for sensitivity
    studies). t_fiber is the channel transmission excluding detector
    efficiency. p_signal plays the role of Gisin's (1-f) (fraction of rounds
    carrying key bits).

    The ceiling is a necessary condition only: a framework rate above it
    indicates a bound error. Returns +inf when s = 0 or t = 1.
    """
    s = float(bs_attack_strength)
    t = float(t_fiber)
    if s <= 0.0 or t >= 1.0:
        return float("inf")
    denom = 4.0 * s * (1.0 - t)
    if denom <= 0.0:
        return float("inf")
    return t * tB * detector_eff * p_signal / denom


def beam_splitting_mu_opt(t_fiber: float, bs_attack_strength: float) -> float:
    """mu_opt(s) = 1 / (2 s (1-t)); see beam_splitting_rate_ceiling."""
    s = float(bs_attack_strength)
    t = float(t_fiber)
    if s <= 0.0 or t >= 1.0:
        return float("inf")
    denom = 2.0 * s * (1.0 - t)
    if denom <= 0.0:
        return float("inf")
    return 1.0 / denom


# ----------------------------------------------------------------------------
# Backflash attack: theoretical rate ceiling
# ----------------------------------------------------------------------------

def backflash_rate_ceiling(
    honest_rate_per_pulse: float,
    honest_n_z: float,
    N_target: float,
    backflash_strength: float,
) -> float:
    """
    Backflash attack rate ceiling, Singh et al., arXiv:2502.04081 (2025).

    Backflash photons from Bob's data-line detector leak his bit to Eve
    without disturbing the received state, so no observable changes. The
    leak is subtracted as an absolute per-pulse information rate from the
    honest (backflash_strength = 0) rate:

        ceiling = max(0, honest_rate - (honest_n_z / N_target) * s)

    following the paper's P_learn = P_b * P_sift (Eqs. 5-6). Since no bound
    can observe this leak, any s > 0 is expected to violate the ceiling.

    Parameters
    ----------
    honest_rate_per_pulse : float
        Key rate per pulse at backflash_strength = 0.
    honest_n_z : float
        Sifted count n_z of the same honest run, scaled to N_target.
    N_target : float
        Total pulse count.
    backflash_strength : float
        Leak probability in [0, 1] (stand-in for the paper's P_b).

    Returns
    -------
    Ceiling on the secure rate per pulse.
    """
    s = float(backflash_strength)
    if s <= 0.0 or N_target <= 0:
        return float(honest_rate_per_pulse)
    s = min(s, 1.0)
    Q_sift_honest = float(honest_n_z) / float(N_target)
    return max(0.0, float(honest_rate_per_pulse) - Q_sift_honest * s)
