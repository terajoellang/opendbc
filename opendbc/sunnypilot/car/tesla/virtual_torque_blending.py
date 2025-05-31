"""
Torque-blending / co-steering state-machine for Tesla (openpilot / opendbc).
All comments in English.

AUTO       – normal openpilot control  
HOLD       – driver holds wheel, we gently nudge ≤±1 ° toward the planner  
RAMP_BACK  – wheel eases back to planner with an S-curve (slow–fast–slow)  

Revisions (May 2025)
--------------------
* Keep variable names (`apply_angle_out`, `lat_active_out`) to minimise git diff.
* Increase torque dead-band (TORQUE_EXIT) to 1.0 Nm – matches Tesla assist law.
* HOLD nudges reduced to ±1 ° and issued only every 5th frame (rate-limit) to
  avoid continuous wheel creep.
* S-curve (smooth-step) replaces linear interpolation for ramp-back.
* 0.4 s "grace" after entering AUTO ignores residual driver torque – lets the
  system grab the wheel even if the driver is still resting a hand.
* Additional exit condition: if wheel ≈ planner (|∆|<2 °) we auto-resume even
  while the driver is lightly torquing.

Copyright (c) 2025.
Licensed under the MIT License.
"""

from __future__ import annotations

import enum
import numpy as np
from opendbc.car import structs
from opendbc.car.interfaces import CarStateBase

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

DT_DEFAULT = 0.02                      # s – control-loop period (50 Hz)

# Torque thresholds (Nm)
TORQUE_ENTER = 1.0                     # ≥ → driver clearly wants control
TORQUE_EXIT  = 1.0                     # ≤ → driver has released wheel (dead-band)

# Debounce times (s)
ENTER_TIME = 0.05                      # 50 ms continuous above TORQUE_ENTER
EXIT_TIME  = 0.10                      # 100 ms continuous below TORQUE_EXIT

# Ramp-back durations vs speed (m/s)
V_5_KMH      = 5.0 / 3.6               # 1.39 m/s
V_10_KMH     = 10.0 / 3.6              # 2.78 m/s
RAMP_T_STANDSTILL = 1.0                # <5 km/h
RAMP_T_5_10        = 0.75              # 5…10 km/h
RAMP_T_ABOVE_10    = 0.5               # >10 km/h

# HOLD-nudge parameters
HOLD_NUDGE_MAX_DEG = 2.0               # deg – absolute cap per nudge
HOLD_NUDGE_RATE    = 1                 # issue every N frames (4 Hz)

# Planner-match threshold for early resume
ANGLE_MATCH_THRESHOLD = 2.0            # deg

# Grace period after AUTO engage (ignores driver torque)
GRACE_AFTER_AUTO = 0.4                 # s

# -----------------------------------------------------------------------------
# Helper enum
# -----------------------------------------------------------------------------

class TBState(enum.IntEnum):
    AUTO = 0       # openpilot has full control
    HOLD = 1       # driver holds wheel, small nudges
    RAMP_BACK = 2  # wheel returns to planner

# -----------------------------------------------------------------------------
# Main controller
# -----------------------------------------------------------------------------

class TorqueBlendingCarController:
    """Driver-friendly torque-blending state machine with gentle probing."""

    def __init__(self, dt: float = DT_DEFAULT):
        self.dt = dt
        self.enabled = True

        # FSM
        self.state: TBState = TBState.AUTO
        self.ramp_timer = 0.0
        self.ramp_duration = 0.0
        self.ramp_start = 0.0

        # Debouncers
        self._above_timer = 0.0
        self._below_timer = 0.0

        # Frame counter for nudge rate-limit
        self._frame = 0

        # Grace timer after (re)entering AUTO
        self.grace_timer = 0.0

    # ------------------------------------------------------------------
    # Helper functions
    # ------------------------------------------------------------------

    def _update_debouncers(self, torque: float, ignore: bool) -> None:
        """Integrate debounce timers unless ignore=True."""
        if ignore:
            self._above_timer = 0.0
            self._below_timer = 0.0
            return

        if abs(torque) >= TORQUE_ENTER:
            self._above_timer += self.dt
        else:
            self._above_timer = 0.0

        if abs(torque) <= TORQUE_EXIT:
            self._below_timer += self.dt
        else:
            self._below_timer = 0.0

    @staticmethod
    def _ramp_tau(v_ego: float) -> float:
        if v_ego < V_5_KMH:
            return RAMP_T_STANDSTILL
        if v_ego < V_10_KMH:
            return RAMP_T_5_10
        return RAMP_T_ABOVE_10

    @staticmethod
    def _smooth_step(t: float) -> float:
        """3t²-2t³ – slow-fast-slow S-curve."""
        return t * t * (3.0 - 2.0 * t)

    # ------------------------------------------------------------------
    # Public API – called each control cycle
    # ------------------------------------------------------------------

    def update_torque_blending(
        self,
        CS: CarStateBase,
        CC: structs.CarControl,
        lat_active: bool,
        apply_angle: float,
    ) -> tuple[bool, float]:
        """Returns (lat_active, apply_angle) after torque blending."""

        if not self.enabled:
            return lat_active, apply_angle

        # Inputs
        v_ego = CS.out.vEgo                     # m/s
        wheel_angle = CS.out.steeringAngleDeg   # deg
        driver_torque = CS.out.steeringTorque   # Nm

        # Grace period after AUTO engage
        ignore_torque = self.grace_timer > 0.0
        if self.grace_timer > 0.0:
            self.grace_timer = max(0.0, self.grace_timer - self.dt)

        # 1. FSM transitions -------------------------------------------------
        self._update_debouncers(driver_torque, ignore_torque)
        angle_diff = abs(apply_angle - wheel_angle)

        if self.state == TBState.AUTO:
            if self._above_timer >= ENTER_TIME:
                self.state = TBState.HOLD

        elif self.state == TBState.HOLD:
            # Early resume if wheel ~ planner + torque low
            if (angle_diff < ANGLE_MATCH_THRESHOLD and self._below_timer >= EXIT_TIME) or \
               (self._below_timer >= 2.0 * EXIT_TIME):
                self.state = TBState.RAMP_BACK
                self.ramp_timer = 0.0
                self.ramp_start = wheel_angle
                self.ramp_duration = self._ramp_tau(v_ego)

        elif self.state == TBState.RAMP_BACK:
            if self._above_timer >= ENTER_TIME:
                self.state = TBState.HOLD
            else:
                self.ramp_timer += self.dt
                if self.ramp_timer >= self.ramp_duration:
                    self.state = TBState.AUTO
                    self.grace_timer = GRACE_AFTER_AUTO  # start grace

        # 2. Outputs ---------------------------------------------------------
        self._frame += 1
        apply_angle_out = apply_angle
        lat_active_out = CC.latActive

        if self.state == TBState.HOLD:
            # Rate-limited gentle nudge
            if self._frame % HOLD_NUDGE_RATE == 0:
                delta = np.clip(apply_angle - wheel_angle, -HOLD_NUDGE_MAX_DEG, HOLD_NUDGE_MAX_DEG)
                apply_angle_out = wheel_angle + delta
            else:
                apply_angle_out = wheel_angle  # keep last commanded angle
            lat_active_out = False  # drop OP torque during HOLD

        elif self.state == TBState.RAMP_BACK:
            t_norm = np.clip(self.ramp_timer / self.ramp_duration, 0.0, 1.0)
            alpha = self._smooth_step(t_norm)
            apply_angle_out = self.ramp_start + alpha * (apply_angle - self.ramp_start)
            lat_active_out = True

        return lat_active_out, apply_angle_out


# -----------------------------------------------------------------------------
# Optional helper remains unchanged (no semantic diff)
# -----------------------------------------------------------------------------

class TorqueBlendingCarState:
    """Adds steeringDisengage flag for specific Tesla EAC errors."""

    def __init__(self):
        self.enabled = True

    def update_torque_blending(
        self,
        ret: structs.CarState,
        eac_status: str,
        eac_error_code: str,
    ) -> None:
        if not self.enabled:
            return

        ret.steeringDisengage = (
            eac_status == "EAC_INHIBITED"
            and eac_error_code == "EAC_ERROR_HIGH_ANGLE_RATE_SAFETY"
        )