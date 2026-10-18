"""
observable.py
=============
Layer 1 of the COW-QKD framework: quantum simulation.

    - Prepares the four COW states:
          |0z> = |0>|alpha>      (bit 0)
          |1z> = |alpha>|0>      (bit 1)
          |alpha,alpha>          (decoy, coherence probe)
          |0,0>                  (vacuum decoy)
    - Propagates them with the Strawberry Fields Gaussian backend (exact for
      coherent states and linear loss) through Bob's beam splitter (tB), the
      lossy channel, and the monitor-line one-bit-delay interferometer.
    - Converts output amplitudes to threshold-detector click probabilities
      and samples raw click counts for `num_pulses` rounds.

SF is run once per configuration: coherent states stay coherent under every
operation here, so the output amplitudes are deterministic. The per-round
randomness (state choice and click outcome) is sampled in
ObservableCollector.run().

Monitor-line gains are per-round quantities (a decoy round is a pulse pair).
The SF interferometer feeds each pulse's full amplitude into one recombining
beam splitter at phi = pi; this reproduces Gao et al.'s closed-form decoy
gain Q^{M0}_{alphaalpha} and signal gain c1 to <0.02% at Cao et al.'s
operating points, and matches Cao et al.'s measured Table S2 counts.

Dark counts: the data line here is dark-count-free (detector.py adds them).
The monitor-line gains include p_d, since Gao et al.'s closed forms have it
built in.

References:
    Gao et al., Opt. Express 30, 23783 (2022), arXiv:2107.09329v5, App. A.
    Cao et al., Sci. Adv. (2026), arXiv:2601.06772.
    Gisin et al., arXiv:quant-ph/0411022 (2004) (intercept-resend).
    Branciard et al., QIC 7, 639 (2007), arXiv:quant-ph/0609090 (USD).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

import strawberryfields as sf
from strawberryfields import ops


# ----------------------------------------------------------------------------
# Data container
# ----------------------------------------------------------------------------

@dataclass
class ObservableSnapshot:
    """
    Raw click counts from one simulated run of `num_pulses` COW rounds. Each
    count is a binomial sample of an exact closed-form click probability, so
    it carries genuine sampling noise (significant at small mu).

    Data line (bit assignment per Cao et al. 2026: former slot -> bit 1,
    latter slot -> bit 0):
        clicks_data_0z_tau0 : |0z> sent, click at tau0 (dark-count-only, error)
        clicks_data_0z_tau1 : |0z> sent, click at tau1 (signal, correct)
        clicks_data_1z_tau0 : |1z> sent, click at tau0 (signal, correct)
        clicks_data_1z_tau1 : |1z> sent, click at tau1 (dark-count-only, error)

    Monitor line, decoy states:
        clicks_mon_DM0_alphaalpha, clicks_mon_DM1_alphaalpha
        clicks_mon_DM0_00, clicks_mon_DM1_00

    Monitor line, signal states (measured observables in Cao et al. 2026,
    since passive basis choice also routes signal pulses to the monitor):
        clicks_mon_DM0_0z, clicks_mon_DM1_0z
        clicks_mon_DM0_1z, clicks_mon_DM1_1z

    Rounds sent of each type:
        n_0z, n_1z, n_alphaalpha, n_00
    """

    n_0z: int
    n_1z: int
    n_alphaalpha: int
    n_00: int

    clicks_data_0z_tau0: int
    clicks_data_0z_tau1: int
    clicks_data_1z_tau0: int
    clicks_data_1z_tau1: int

    clicks_mon_DM0_alphaalpha: int
    clicks_mon_DM1_alphaalpha: int
    clicks_mon_DM0_00: int
    clicks_mon_DM1_00: int

    clicks_mon_DM0_0z: int
    clicks_mon_DM1_0z: int
    clicks_mon_DM0_1z: int
    clicks_mon_DM1_1z: int

    num_pulses_simulated: int = field(default=0)
    mu: float = field(default=0.0)
    tB: float = field(default=0.0)
    eta: float = field(default=0.0)
    seed: int | None = field(default=None)


# ----------------------------------------------------------------------------
# Core SF physics helpers
# ----------------------------------------------------------------------------

def _click_probability_from_amplitude(beta: complex, cutoff_threshold: float = 1e-12) -> float:
    """
    Threshold-detector click probability for a coherent state |beta>
    (efficiency already folded into beta):

        P(click) = 1 - exp(-|beta|^2)
    """
    p_click = 1.0 - np.exp(-np.abs(beta) ** 2)
    return float(np.clip(p_click, 0.0, 1.0))


# 12-node Gauss-Hermite quadrature for averaging over Gaussian phase jitter
# (imperfect visibility) in monitor_line_gains_sf_physical.
_GH_NODES_RAW, _GH_WEIGHTS_RAW = np.polynomial.hermite.hermgauss(12)
_GH_WEIGHTS = _GH_WEIGHTS_RAW / np.sqrt(np.pi)


def _visibility_to_phase_sigma(interference_visibility: float) -> float:
    """
    Fringe visibility V -> standard deviation of Gaussian relative-phase
    noise, from V = exp(-sigma^2 / 2):

        sigma = sqrt(-2 ln V)
    """
    V = float(interference_visibility)
    if V >= 1.0:
        return 0.0
    return float(np.sqrt(-2.0 * np.log(V)))


# ----------------------------------------------------------------------------
# Zero-error attack (Branciard, Gisin, Luetkenhaus, Scarani, QIC 7, 639 (2007),
# arXiv:quant-ph/0609090): USD3 and USD4b conclusive-discrimination
# probabilities. f is the alpha-alpha decoy probability.
# ----------------------------------------------------------------------------

def usd3_conclusive_prob(mu: float, f: float) -> float:
    """
    Eq. 12: probability that USD3 conclusively identifies |0-alpha-0>:
        ((1-f)/2)^2 (1 - e^{-mu})^2
    """
    return ((1.0 - f) / 2.0) ** 2 * (1.0 - np.exp(-mu)) ** 2


def usd4b_conclusive_prob(mu: float, f: float) -> float:
    """
    Eq. 26: probability that USD4b conclusively identifies |0:alpha-alpha:0>:
        f ((1-f)/2)^2 (1-e^{-mu})^3 / (1+e^{-mu})
    """
    return f * ((1.0 - f) / 2.0) ** 2 * (1.0 - np.exp(-mu)) ** 3 / (1.0 + np.exp(-mu))


class COWQuantumChannel:
    """
    Strawberry Fields Gaussian-backend model of:
        Alice's state -> lossy channel -> Bob's beam splitter (tB data /
        1-tB monitor) -> monitor-line one-bit-delay interferometer -> D_M0/D_M1.

    Each distinct configuration is run through a real SF circuit once and the
    output amplitude cached.
    """

    def __init__(self, mu: float, tB: float, eta: float, phi: float = np.pi,
                 basis_mode: str = "passive", eta_for_monitor: float | None = None):
        """
        Parameters
        ----------
        mu  : mean photon number, |alpha|^2 = mu.
        tB  : beam-splitter transmittance to the data line (passive mode).
              The (1-tB) factor in the monitor-line exponents is the same in
              both basis modes.
        eta : data-line (D0) transmittance including detector efficiency.
        eta_for_monitor : monitor-line (D1/D2) transmittance; None uses eta.
              Cao et al. 2026: D0 76.2%, D1/D2 46.0%, plus 3 dB monitor loss.
        phi : interferometer phase; pi gives constructive interference at D_M0.
        basis_mode : "passive" (beam splitter: every pulse reaches both lines,
              so monitor gains carry the joint "data detector did not click"
              factor) or "active" (optical switch: each round goes to one
              line, no data-line attenuation). Gao et al. App. A.
        """
        self.mu = float(mu)
        self.alpha = np.sqrt(self.mu)  # real amplitude (COW convention)
        self.tB = float(tB)
        self.eta = float(eta)
        self.eta_for_monitor = float(eta_for_monitor) if eta_for_monitor is not None else None
        self.phi = float(phi)
        if basis_mode not in ("passive", "active"):
            raise ValueError(f"basis_mode must be 'passive' or 'active', got {basis_mode!r}")
        self.basis_mode = basis_mode
        self._amplitude_cache: dict[tuple, complex] = {}

    # -- SF circuit execution (run once per distinct input amplitude) --

    def _run_sf_data_line_circuit(self, alpha_in: complex) -> complex:
        """
        SF circuit for the data-line amplitude.

        Passive: Dgate(alpha_in) -> BSgate (T = tB) -> LossChannel(eta).
        Active:  Dgate(alpha_in) -> LossChannel(eta) (full amplitude reaches
                 the data line).
        """
        prog = sf.Program(2)
        eng = sf.Engine("gaussian")

        with prog.context as q:
            ops.Dgate(np.abs(alpha_in), np.angle(alpha_in)) | q[0]
            ops.Vac | q[1]
            if self.basis_mode == "passive":
                theta = np.arccos(np.sqrt(self.tB))
                ops.BSgate(theta, 0.0) | (q[0], q[1])
            ops.LossChannel(self.eta) | q[0]

        result = eng.run(prog)
        mu_vec = result.state.means()
        n_modes = 2
        x0, p0 = mu_vec[0], mu_vec[0 + n_modes]
        amp_out = (x0 + 1j * p0) / 2.0
        return complex(amp_out)

    def _data_line_amplitude(self, alpha_in: complex) -> complex:
        """Data-line output amplitude (cached per input amplitude and mode)."""
        key = (complex(alpha_in), self.basis_mode)
        if key not in self._amplitude_cache:
            self._amplitude_cache[key] = self._run_sf_data_line_circuit(alpha_in)
        return self._amplitude_cache[key]

    def _monitor_input_amplitude(self, alpha_in: complex) -> complex:
        """
        Amplitude entering the monitor interferometer:
            sqrt(eta_monitor (1-tB)) * alpha_in
        (same in passive and active mode).
        """
        eta = self.eta_for_monitor if self.eta_for_monitor is not None else self.eta
        return np.sqrt(eta * (1.0 - self.tB)) * alpha_in

    def _run_sf_monitor_circuit(self, beta_prev: complex, beta_curr: complex) -> complex:
        """
        D_M0 (bright port) amplitude of the monitor interferometer.

        The earlier pulse's full amplitude, phase-shifted by phi, and the later
        pulse's full amplitude enter one recombining 50:50 beam splitter. This
        per-round accounting gives, at phi = pi:
            decoy |alpha,alpha> : whole round budget 2|beta|^2 at D_M0, zero
                                  at D_M1 (Gao's Q^{M0}_{alphaalpha} exponent)
            signal |0z>/|1z>    : |beta|^2/2 at each port (Gao's c1 exponent)

        Under perfect interference D_M1 is dark-count-only; see
        `_run_sf_monitor_circuit_both_ports` for imperfect visibility.
        """
        amp_DM0, _amp_DM1 = self._run_sf_monitor_circuit_both_ports(beta_prev, beta_curr)
        return amp_DM0

    def _run_sf_monitor_circuit_both_ports(
        self, beta_prev: complex, beta_curr: complex, phase_offset: float = 0.0
    ) -> tuple[complex, complex]:
        """
        Same circuit as `_run_sf_monitor_circuit`, returning both port
        amplitudes (D_M0, D_M1) with an extra phase offset on the delayed arm.
        A nonzero offset leaks flux into D_M1; total power is conserved
        because the beam splitter is unitary.
        """
        prog = sf.Program(2)
        eng = sf.Engine("gaussian")

        # Delayed (earlier) pulse carries the interferometer phase phi plus
        # the phase-jitter sample.
        prev_delayed = beta_prev * np.exp(1j * (self.phi + phase_offset))
        curr = beta_curr

        with prog.context as q:
            ops.Dgate(np.abs(prev_delayed), np.angle(prev_delayed)) | q[0]
            ops.Dgate(np.abs(curr), np.angle(curr)) | q[1]
            ops.BSgate(np.pi / 4, 0.0) | (q[0], q[1])

        result = eng.run(prog)
        mu_vec = result.state.means()
        # means() is ordered [x0, x1, p0, p1].
        n_modes = 2
        x0, p0 = mu_vec[0], mu_vec[0 + n_modes]
        x1, p1 = mu_vec[1], mu_vec[1 + n_modes]
        amp_DM0 = complex((x0 + 1j * p0) / 2.0)
        amp_DM1 = complex((x1 + 1j * p1) / 2.0)
        return amp_DM0, amp_DM1

    def monitor_line_quantum_click_probs(self) -> dict:
        """
        Dark-count-free D_M0 click probabilities for the decoy states from the
        SF circuit. Returns {"P_DM0_alphaalpha", "P_DM0_00"} (the latter is 0).
        """
        mon_in_alpha = self._monitor_input_amplitude(self.alpha)
        mon_in_zero = self._monitor_input_amplitude(0.0)

        amp_DM0_aa = self._run_sf_monitor_circuit(mon_in_alpha, mon_in_alpha)
        P_DM0_aa = _click_probability_from_amplitude(amp_DM0_aa)

        amp_DM0_00 = self._run_sf_monitor_circuit(mon_in_zero, mon_in_zero)
        P_DM0_00 = _click_probability_from_amplitude(amp_DM0_00)

        return {
            "P_DM0_alphaalpha": P_DM0_aa,
            "P_DM0_00": P_DM0_00,
        }

    def monitor_line_gains_sf_physical(self, p_dark: float, interference_visibility: float = 1.0,
                                         ir_attack_strength: float = 0.0,
                                         usd4b_strength: float = 0.0,
                                         p_decoy_alphaalpha: float = 0.0) -> dict:
        """
        Decoy-state monitor gains computed from the SF interferometer, with
        imperfect visibility modelled physically as Gaussian phase jitter
        (sigma from `_visibility_to_phase_sigma`) averaged by 12-node
        Gauss-Hermite quadrature over SF circuit runs. The dark port receives
        genuine leaked flux at each node.

        Dark counts are combined as independent processes,
        P = 1 - (1 - P_quantum)(1 - p_dark).

        Against Cao et al.'s Table S2, the bright-port prediction agrees to
        0.9-2.4% at all four distances. V = 0.9964 was fitted to the dark-port
        counts, so dark-port agreement is not independent evidence.

        The intercept-resend and USD4b attacks are applied with the same
        mixtures as in `monitor_line_gains_closed_form`; both are identity
        maps at zero strength.

        Returns {"G_M0_alphaalpha", "G_M1_alphaalpha", "G_M0_00", "G_M1_00"}.
        """
        pd = p_dark
        sigma = _visibility_to_phase_sigma(interference_visibility)
        if sigma == 0.0:
            deltas = np.array([0.0])
            weights = np.array([1.0])
        else:
            deltas = sigma * np.sqrt(2.0) * _GH_NODES_RAW
            weights = _GH_WEIGHTS

        def _quadrature_gains(beta_prev: complex, beta_curr: complex) -> tuple[float, float]:
            P_bright = 0.0
            P_dark = 0.0
            for delta, w in zip(deltas, weights):
                amp0, amp1 = self._run_sf_monitor_circuit_both_ports(
                    beta_prev, beta_curr, phase_offset=delta
                )
                Pq0 = _click_probability_from_amplitude(amp0)
                Pq1 = _click_probability_from_amplitude(amp1)
                P_bright += w * (1.0 - (1.0 - Pq0) * (1.0 - pd))
                P_dark += w * (1.0 - (1.0 - Pq1) * (1.0 - pd))
            return float(P_bright), float(P_dark)

        mon_in_alpha = self._monitor_input_amplitude(self.alpha)
        mon_in_zero = self._monitor_input_amplitude(0.0)

        G_M0_aa, G_M1_aa = _quadrature_gains(mon_in_alpha, mon_in_alpha)

        # Intercept-resend (Gisin et al. 2004, Eqs. 19-26): flux-conserving
        # 50/50 mixture on the attacked fraction of rounds.
        p_ir = float(ir_attack_strength)
        if p_ir > 0.0:
            total_flux_alphaalpha = G_M0_aa + G_M1_aa
            G_M0_aa = (1.0 - p_ir) * G_M0_aa + p_ir * (total_flux_alphaalpha / 2.0)
            G_M1_aa = (1.0 - p_ir) * G_M1_aa + p_ir * (total_flux_alphaalpha / 2.0)

        # USD4b (Branciard et al. 2007, Eqs. 26, 29-31).
        q3 = float(usd4b_strength)
        f_decoy = float(p_decoy_alphaalpha)
        if q3 > 0.0 and f_decoy > 0.0:
            eta_mon = self.eta_for_monitor if self.eta_for_monitor is not None else self.eta
            p_concl_usd4b = usd4b_conclusive_prob(self.mu, f_decoy)
            usd4b_bright = 0.5 * p_concl_usd4b * ((1.0 - self.tB) * eta_mon / 2.0)
            usd4b_dark = 0.0  # Eq. 30
            G_M0_aa = (1.0 - q3) * G_M0_aa + q3 * usd4b_bright
            G_M1_aa = (1.0 - q3) * G_M1_aa + q3 * usd4b_dark

        # Vacuum decoy: no photons, dark counts only; unaffected by the attacks.
        G_M0_00 = float(pd)
        G_M1_00 = float(pd)

        return {
            "G_M0_alphaalpha": float(np.clip(G_M0_aa, 0.0, 1.0)),
            "G_M1_alphaalpha": float(np.clip(G_M1_aa, 0.0, 1.0)),
            "G_M0_00": G_M0_00,
            "G_M1_00": G_M1_00,
        }

    def monitor_line_gains_closed_form(self, p_dark: float, interference_visibility: float = 1.0,
                                         ir_attack_strength: float = 0.0,
                                         usd4b_strength: float = 0.0,
                                         p_decoy_alphaalpha: float = 0.0) -> dict:
        """
        Decoy-state monitor gains including dark counts, Gao et al. 2022
        (arXiv:2107.09329v5) Appendix A, passive mode:

            Q^{M0}_{alphaalpha} = (1-pd)^3 [1-(1-pd) exp(-2 mu (1-tB) eta)] c5
            Q^{M1}_{alphaalpha} = pd (1-pd)^3 exp(-2 mu (1-tB) eta) c5
            Q^{Mi}_{00}         = pd (1-pd)^3
            c5 = exp(-2 tB mu eta)   (corrected form in v5)

        Active mode uses (1-pd)^1 and no c5 factor (applied by analogy with
        the paper's stated active-mode pattern).

        eta is the monitor-line transmittance.

        interference_visibility V < 1 adds a (1-V)/(1+V) fraction of the
        bright-port gain to the dark port (a small-perturbation correction,
        not part of Gao's formula).

        ir_attack_strength and usd4b_strength apply the intercept-resend and
        USD4b attacks (see inline comments).

        Returns {"G_M0_alphaalpha", "G_M1_alphaalpha", "G_M0_00", "G_M1_00"}.
        """
        pd = p_dark
        mu, tB = self.mu, self.tB
        eta = self.eta_for_monitor if self.eta_for_monitor is not None else self.eta

        c5 = np.exp(-2.0 * tB * mu * eta)

        if self.basis_mode == "passive":
            pd_power = (1.0 - pd) ** 3
            decoy_factor = c5
        else:  # active
            pd_power = (1.0 - pd) ** 1
            decoy_factor = 1.0

        G_M0_coh = pd_power * (1.0 - (1.0 - pd) * np.exp(-2.0 * mu * (1.0 - tB) * eta)) * decoy_factor
        G_M1_coh = pd * pd_power * np.exp(-2.0 * mu * (1.0 - tB) * eta) * decoy_factor

        # Hardware visibility leakage (valid near V ~ 1; not flux-conserving).
        V = float(interference_visibility)
        leak_fraction = (1.0 - V) / (1.0 + V) if V < 1.0 else 0.0
        G_M1_coh = G_M1_coh + leak_fraction * G_M0_coh

        # Intercept-resend attack (Gisin et al., arXiv:quant-ph/0411022,
        # Sec. III.C.2, Eqs. 19-26). p_IR is a per-round probability; an
        # attacked round loses all coherence and its flux splits equally
        # between the two ports. The mixture conserves total flux and gives
        # V_obs = (1 - p_IR) V_baseline. The data line is unaffected (Eq. 19).
        p_ir = float(ir_attack_strength)
        total_flux_alphaalpha = G_M0_coh + G_M1_coh
        G_M0_alphaalpha = (1.0 - p_ir) * G_M0_coh + p_ir * (total_flux_alphaalpha / 2.0)
        G_M1_alphaalpha = (1.0 - p_ir) * G_M1_coh + p_ir * (total_flux_alphaalpha / 2.0)

        # USD4b zero-error attack (Branciard et al., arXiv:quant-ph/0609090,
        # Sec. III.D, Eqs. 26, 29-31). On a fraction q3 of alpha-alpha decoy
        # rounds Eve attempts USD on |0:alpha-alpha:0>; when conclusive she
        # forwards a coherent pair landing only in the bright port
        # (Eqs. 29-30), otherwise nothing. f = p_decoy_alphaalpha. USD4a is
        # not modelled (it needs a cross-round signal coherence statistic).
        q3 = float(usd4b_strength)
        f_decoy = float(p_decoy_alphaalpha)
        eta_mon = eta
        if q3 > 0.0 and f_decoy > 0.0:
            p_concl_usd4b = usd4b_conclusive_prob(mu, f_decoy)
            # Pi(p) = p for a forwarded single photon, p = (1-tB) eta_mon / 2.
            usd4b_bright = 0.5 * p_concl_usd4b * ((1.0 - tB) * eta_mon / 2.0)
            usd4b_dark = 0.0  # Eq. 30
            G_M0_alphaalpha = (1.0 - q3) * G_M0_alphaalpha + q3 * usd4b_bright
            G_M1_alphaalpha = (1.0 - q3) * G_M1_alphaalpha + q3 * usd4b_dark

        # Vacuum decoy: dark counts only; unaffected by the attacks (nothing
        # for Eve to detect).
        G_M0_00 = pd * pd_power
        G_M1_00 = pd * pd_power

        return {
            "G_M0_alphaalpha": float(np.clip(G_M0_alphaalpha, 0.0, 1.0)),
            "G_M1_alphaalpha": float(np.clip(G_M1_alphaalpha, 0.0, 1.0)),
            "G_M0_00": float(np.clip(G_M0_00, 0.0, 1.0)),
            "G_M1_00": float(np.clip(G_M1_00, 0.0, 1.0)),
        }

    def monitor_line_signal_quantum_click_probs(self) -> dict:
        """
        Dark-count-free D_M0 click probabilities for the signal states from
        the SF circuit (one input vacuum, the other the monitor amplitude).
        Returns {"P_DM0_0z", "P_DM0_1z"}.
        """
        mon_in_alpha = self._monitor_input_amplitude(self.alpha)
        mon_in_zero = self._monitor_input_amplitude(0.0)

        # |0z> = |0>_tau0 |alpha>_tau1 -> (beta_prev, beta_curr) = (0, alpha).
        P_DM0_0z = _click_probability_from_amplitude(
            self._run_sf_monitor_circuit(mon_in_zero, mon_in_alpha)
        )

        # |1z> = |alpha>_tau0 |0>_tau1: reversed.
        P_DM0_1z = _click_probability_from_amplitude(
            self._run_sf_monitor_circuit(mon_in_alpha, mon_in_zero)
        )

        return {
            "P_DM0_0z": P_DM0_0z,
            "P_DM0_1z": P_DM0_1z,
        }

    def monitor_line_signal_gains_closed_form(self, p_dark: float) -> dict:
        """
        Signal-state monitor gains, Gao et al. 2022 Appendix A:

            passive: Q^{Mi}_{0z} = Q^{Mi}_{1z} = (1-pd)^2 exp(-tB mu eta) c1 (1-c1)
            active:  Q^{Mi}_{0z} = Q^{Mi}_{1z} = c1 (1-c1)
            c1 = (1-pd) exp(-(1-tB) mu eta / 2)

        Identical for both detectors: a lone signal pulse has no interference
        partner, so neither port is favoured.

        Returns {"G_M0_0z", "G_M1_0z", "G_M0_1z", "G_M1_1z"} (all equal).
        """
        pd = p_dark
        mu, tB = self.mu, self.tB
        eta = self.eta_for_monitor if self.eta_for_monitor is not None else self.eta

        c1 = (1.0 - pd) * np.exp(-(1.0 - tB) * mu * eta / 2.0)
        if self.basis_mode == "passive":
            Q_signal = (1.0 - pd) ** 2 * np.exp(-tB * mu * eta) * c1 * (1.0 - c1)
        else:  # active
            Q_signal = c1 * (1.0 - c1)
        Q_signal = float(np.clip(Q_signal, 0.0, 1.0))

        pd_power = (1.0 - pd) ** 3 if self.basis_mode == "passive" else (1.0 - pd) ** 1
        G_M1_signal = pd * pd_power

        return {
            "G_M0_0z": Q_signal,
            "G_M1_0z": Q_signal,
            "G_M0_1z": Q_signal,
            "G_M1_1z": Q_signal,
        }

    # -- SF amplitude check (verification only) --

    def sf_verify_beamsplitter_amplitude(self, alpha_in: complex) -> complex:
        """
        Passive data-line amplitude from an explicit SF BSgate + LossChannel
        circuit, for verification.
        """
        prog = sf.Program(2)
        eng = sf.Engine("gaussian")

        with prog.context as q:
            ops.Dgate(np.abs(alpha_in), np.angle(alpha_in)) | q[0]
            ops.Vac | q[1]
            # asymmetric beamsplitter: BSgate(theta, varphi), T = cos^2(theta)
            theta = np.arccos(np.sqrt(self.tB))
            ops.BSgate(theta, 0.0) | (q[0], q[1])
            ops.LossChannel(self.eta) | q[0]

        result = eng.run(prog)
        mu_vec = result.state.means()
        # means() is ordered [x0, x1, p0, p1]; with hbar = 2, x = 2 Re(alpha),
        # p = 2 Im(alpha).
        n_modes = 2
        x0, p0 = mu_vec[0], mu_vec[0 + n_modes]
        amp_out = (x0 + 1j * p0) / 2.0
        return complex(amp_out)


# ----------------------------------------------------------------------------
# Observable collector (the public entry point of Layer 1)
# ----------------------------------------------------------------------------

class ObservableCollector:
    """
    Computes per-round click probabilities from the SF circuits once, then
    samples raw click counts for `num_pulses` rounds in `run()`. One snapshot
    is shared by all security bounds.

    Round probabilities: |0z> and |1z> each p_signal/2, |alpha,alpha>
    p_decoy_alphaalpha, |0,0> p_decoy_00.
    """

    def __init__(
        self,
        mu: float,
        tB: float,
        eta: float,
        p_dark: float,
        p_signal: float = 0.5,
        p_decoy_alphaalpha: float = 0.25,
        p_decoy_00: float = 0.25,
        phi: float = np.pi,
        seed: int | None = None,
        basis_mode: str = "passive",
        eta_for_monitor: float | None = None,
        interference_visibility: float = 1.0,
        ir_attack_strength: float = 0.0,
        usd3_strength: float = 0.0,
        usd4b_strength: float = 0.0,
        use_sf_monitor_gains: bool = False,
        monitor_gain_source: str = "sf_physical",
    ):
        """
        Parameters
        ----------
        mu, tB, eta, phi, basis_mode, eta_for_monitor : see COWQuantumChannel.
        p_dark : dark-count probability per pulse window.
        p_signal : probability of a signal round (|0z> or |1z>, split 50/50).
        p_decoy_alphaalpha, p_decoy_00 : decoy-state probabilities.
        seed : RNG seed.
        interference_visibility : monitor interferometer visibility V.
        monitor_gain_source : "sf_physical" (monitor_line_gains_sf_physical)
            or "closed_form" (monitor_line_gains_closed_form) for the decoy
            monitor gains. With "sf_physical" the closed form is kept as
            `_monitor_gains_crosscheck`.
        use_sf_monitor_gains : also take the signal-state D_M0 probabilities
            from the SF circuit (legacy option).
        ir_attack_strength : intercept-resend probability per round.
        usd3_strength, usd4b_strength : Branciard et al. 2007 USD3 (q1,
            data-line) and USD4b (q3, decoy monitor) attack fractions.
        """
        if not np.isclose(p_signal + p_decoy_alphaalpha + p_decoy_00, 1.0):
            raise ValueError(
                f"Probabilities must sum to 1: got p_signal={p_signal}, "
                f"p_d1={p_decoy_alphaalpha}, p_d2={p_decoy_00}, "
                f"sum={p_signal + p_decoy_alphaalpha + p_decoy_00}"
            )

        self.channel = COWQuantumChannel(mu=mu, tB=tB, eta=eta, phi=phi, basis_mode=basis_mode,
                                          eta_for_monitor=eta_for_monitor)
        self.p_dark = p_dark
        self.p_signal = p_signal
        self.p_decoy_alphaalpha = p_decoy_alphaalpha
        self.p_decoy_00 = p_decoy_00
        self.seed = seed
        self._rng = np.random.default_rng(seed)

        # Data line: SF amplitude -> click probability (dark-count-free).
        alpha = self.channel.alpha
        self._data_amp_alpha = self.channel._data_line_amplitude(alpha)
        self._data_amp_zero  = self.channel._data_line_amplitude(0.0)
        self._p_data_alpha = _click_probability_from_amplitude(self._data_amp_alpha)
        self._p_data_zero  = _click_probability_from_amplitude(self._data_amp_zero)

        # USD3 zero-error attack (Branciard et al. 2007, Sec. III.B, Eqs. 12-13).
        # On a fraction q1 of signal rounds Eve attempts USD on |0-alpha-0>;
        # when conclusive she forwards a single photon into the correct slot
        # (click probability tB*eta), otherwise nothing. No errors and no
        # fake decoy clicks are introduced.
        q1 = float(usd3_strength)
        if q1 > 0.0:
            p_concl_usd3 = usd3_conclusive_prob(mu, p_decoy_alphaalpha)
            usd3_click = p_concl_usd3 * (tB * eta)
            self._p_data_alpha = (1.0 - q1) * self._p_data_alpha + q1 * usd3_click

        # Monitor line: decoy gains from the selected source, signal gains
        # from the closed form.
        gains_closed_form = self.channel.monitor_line_gains_closed_form(
            p_dark, interference_visibility=interference_visibility,
            ir_attack_strength=ir_attack_strength,
            usd4b_strength=usd4b_strength,
            p_decoy_alphaalpha=p_decoy_alphaalpha,
        )
        signal_gains_with_dark = self.channel.monitor_line_signal_gains_closed_form(p_dark)

        self.monitor_gain_source = str(monitor_gain_source)

        if self.monitor_gain_source == "sf_physical":
            gains_primary = self.channel.monitor_line_gains_sf_physical(
                p_dark, interference_visibility=interference_visibility,
                ir_attack_strength=ir_attack_strength,
                usd4b_strength=usd4b_strength,
                p_decoy_alphaalpha=p_decoy_alphaalpha,
            )
            self._monitor_gains_crosscheck = gains_closed_form
        elif self.monitor_gain_source == "closed_form":
            gains_primary = gains_closed_form
            self._monitor_gains_crosscheck = None
        else:
            raise ValueError(f"monitor_gain_source must be 'sf_physical' or 'closed_form', got {monitor_gain_source!r}")

        self._p_M0_aa  = float(gains_primary["G_M0_alphaalpha"])
        self._p_M1_aa  = float(gains_primary["G_M1_alphaalpha"])
        self._p_M0_00  = float(gains_primary["G_M0_00"])
        self._p_M1_00  = float(gains_primary["G_M1_00"])
        self._p_M0_0z  = float(signal_gains_with_dark["G_M0_0z"])
        self._p_M1_0z  = float(signal_gains_with_dark["G_M1_0z"])
        self._p_M0_1z  = float(signal_gains_with_dark["G_M0_1z"])
        self._p_M1_1z  = float(signal_gains_with_dark["G_M1_1z"])

        # Legacy option: signal-state D_M0 from the SF circuit.
        self.use_sf_monitor_gains = bool(use_sf_monitor_gains)
        if self.use_sf_monitor_gains:
            sf_signal_probs = self.channel.monitor_line_signal_quantum_click_probs()

            def _combine(p_quantum: float) -> float:
                return float(1.0 - (1.0 - p_quantum) * (1.0 - p_dark))

            self._p_M0_0z = _combine(sf_signal_probs["P_DM0_0z"])
            self._p_M0_1z = _combine(sf_signal_probs["P_DM0_1z"])

    def run(self, num_pulses: int, exact_expected_values: bool = False) -> ObservableSnapshot:
        """
        Sample `num_pulses` independent rounds (multinomial round types,
        binomial clicks) and return the raw tallies. With
        exact_expected_values=True, return rounded expected counts instead
        (deterministic, used during optimization).
        """
        p_data_alpha = self._p_data_alpha
        p_data_zero  = self._p_data_zero
        p_M0_aa, p_M1_aa = self._p_M0_aa, self._p_M1_aa
        p_M0_00, p_M1_00 = self._p_M0_00, self._p_M1_00
        p_M0_0z, p_M1_0z = self._p_M0_0z, self._p_M1_0z
        p_M0_1z, p_M1_1z = self._p_M0_1z, self._p_M1_1z

        round_probs = [
            self.p_signal / 2,        # |0z>
            self.p_signal / 2,        # |1z>
            self.p_decoy_alphaalpha,  # |alpha,alpha>
            self.p_decoy_00,          # |0,0>
        ]

        if exact_expected_values:
            n_0z = int(round(num_pulses * round_probs[0]))
            n_1z = int(round(num_pulses * round_probs[1]))
            n_alphaalpha = int(round(num_pulses * round_probs[2]))
            n_00 = int(round(num_pulses * round_probs[3]))

            clicks_data_0z_tau0 = int(round(n_0z * p_data_zero))
            clicks_data_0z_tau1 = int(round(n_0z * p_data_alpha))
            clicks_data_1z_tau0 = int(round(n_1z * p_data_alpha))
            clicks_data_1z_tau1 = int(round(n_1z * p_data_zero))

            clicks_mon_DM0_alphaalpha = int(round(n_alphaalpha * p_M0_aa))
            clicks_mon_DM1_alphaalpha = int(round(n_alphaalpha * p_M1_aa))
            clicks_mon_DM0_00 = int(round(n_00 * p_M0_00))
            clicks_mon_DM1_00 = int(round(n_00 * p_M1_00))
            clicks_mon_DM0_0z = int(round(n_0z * p_M0_0z))
            clicks_mon_DM1_0z = int(round(n_0z * p_M1_0z))
            clicks_mon_DM0_1z = int(round(n_1z * p_M0_1z))
            clicks_mon_DM1_1z = int(round(n_1z * p_M1_1z))

        else:
            # Sample in batches to bound memory use.
            BATCH = int(5e7)
            n_0z = n_1z = n_alphaalpha = n_00 = 0
            clicks_data_0z_tau0 = clicks_data_0z_tau1 = 0
            clicks_data_1z_tau0 = clicks_data_1z_tau1 = 0
            clicks_mon_DM0_alphaalpha = clicks_mon_DM1_alphaalpha = 0
            clicks_mon_DM0_00 = clicks_mon_DM1_00 = 0
            clicks_mon_DM0_0z = clicks_mon_DM1_0z = 0
            clicks_mon_DM0_1z = clicks_mon_DM1_1z = 0

            remaining = num_pulses
            while remaining > 0:
                batch_size = min(BATCH, remaining)
                remaining -= batch_size

                counts = self._rng.multinomial(batch_size, round_probs)
                b_0z, b_1z, b_alphaalpha, b_00 = int(counts[0]), int(counts[1]), int(counts[2]), int(counts[3])

                n_0z += b_0z
                n_1z += b_1z
                n_alphaalpha += b_alphaalpha
                n_00 += b_00

                if b_0z > 0:
                    clicks_data_0z_tau0 += int(self._rng.binomial(b_0z, p_data_zero))
                    clicks_data_0z_tau1 += int(self._rng.binomial(b_0z, p_data_alpha))
                if b_1z > 0:
                    clicks_data_1z_tau0 += int(self._rng.binomial(b_1z, p_data_alpha))
                    clicks_data_1z_tau1 += int(self._rng.binomial(b_1z, p_data_zero))

                if b_alphaalpha > 0:
                    clicks_mon_DM0_alphaalpha += int(self._rng.binomial(b_alphaalpha, p_M0_aa))
                    clicks_mon_DM1_alphaalpha += int(self._rng.binomial(b_alphaalpha, p_M1_aa))
                if b_00 > 0:
                    clicks_mon_DM0_00 += int(self._rng.binomial(b_00, p_M0_00))
                    clicks_mon_DM1_00 += int(self._rng.binomial(b_00, p_M1_00))
                if b_0z > 0:
                    clicks_mon_DM0_0z += int(self._rng.binomial(b_0z, p_M0_0z))
                    clicks_mon_DM1_0z += int(self._rng.binomial(b_0z, p_M1_0z))
                if b_1z > 0:
                    clicks_mon_DM0_1z += int(self._rng.binomial(b_1z, p_M0_1z))
                    clicks_mon_DM1_1z += int(self._rng.binomial(b_1z, p_M1_1z))

        return ObservableSnapshot(
            n_0z=n_0z,
            n_1z=n_1z,
            n_alphaalpha=n_alphaalpha,
            n_00=n_00,
            clicks_data_0z_tau0=clicks_data_0z_tau0,
            clicks_data_0z_tau1=clicks_data_0z_tau1,
            clicks_data_1z_tau0=clicks_data_1z_tau0,
            clicks_data_1z_tau1=clicks_data_1z_tau1,
            clicks_mon_DM0_alphaalpha=clicks_mon_DM0_alphaalpha,
            clicks_mon_DM1_alphaalpha=clicks_mon_DM1_alphaalpha,
            clicks_mon_DM0_00=clicks_mon_DM0_00,
            clicks_mon_DM1_00=clicks_mon_DM1_00,
            clicks_mon_DM0_0z=clicks_mon_DM0_0z,
            clicks_mon_DM1_0z=clicks_mon_DM1_0z,
            clicks_mon_DM0_1z=clicks_mon_DM0_1z,
            clicks_mon_DM1_1z=clicks_mon_DM1_1z,
            num_pulses_simulated=num_pulses,
            mu=self.channel.mu,
            tB=self.channel.tB,
            eta=self.channel.eta,
            seed=self.seed,
        )


def run_sf_simulation(
    mu: float,
    tB: float,
    eta: float,
    p_dark: float,
    num_pulses: int = 1_000_000,
    p_signal: float = 0.5,
    p_decoy_alphaalpha: float = 0.25,
    p_decoy_00: float = 0.25,
    phi: float = np.pi,
    seed: int | None = None,
    basis_mode: str = "passive",
    eta_for_monitor: float | None = None,
    interference_visibility: float = 1.0,
    ir_attack_strength: float = 0.0,
    usd3_strength: float = 0.0,
    usd4b_strength: float = 0.0,
    use_sf_monitor_gains: bool = False,
    monitor_gain_source: str = "sf_physical",
    exact_expected_values: bool = False,
) -> ObservableSnapshot:
    """
    Layer 1 entry point: build an ObservableCollector and run it once.
    Arguments are as in ObservableCollector.
    """
    collector = ObservableCollector(
        mu=mu,
        tB=tB,
        eta=eta,
        eta_for_monitor=eta_for_monitor,
        p_dark=p_dark,
        p_signal=p_signal,
        p_decoy_alphaalpha=p_decoy_alphaalpha,
        p_decoy_00=p_decoy_00,
        phi=phi,
        seed=seed,
        basis_mode=basis_mode,
        interference_visibility=interference_visibility,
        ir_attack_strength=ir_attack_strength,
        usd3_strength=usd3_strength,
        usd4b_strength=usd4b_strength,
        use_sf_monitor_gains=use_sf_monitor_gains,
        monitor_gain_source=monitor_gain_source,
    )
    return collector.run(num_pulses, exact_expected_values=exact_expected_values)
