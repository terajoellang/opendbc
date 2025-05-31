"""
Torque-blending / co-steering state-machine for Tesla (openpilot / opendbc).
All comments in English.

Revision – May 2025
-------------------
* HOLD_NUDGE_MAX_DEG → **2.0°** (stronger feel)
* HOLD_NUDGE_RATE    → every frame (1) for more immediate feedback
* RAMP_T_5_10 & RAMP_T_ABOVE_10 → **1 s** (slower ramp at >5 km/h)
* Grace window (0.4 s) now fires **only when the user manually enables AP**
  (transition lat_active False → True). It is **not** applied on automatic
  re-entry from HOLD/RAMP_BACK.

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
TORQUE_EXIT  = 1.0                     # ≤ → driver has released wheel

# Debounce times (s)
ENTER_TIME = 0.05                      # 50 ms continuous above TORQUE_ENTER
EXIT_TIME  = 0.10                      # 100 ms continuous below TORQUE_EXIT

# Ramp-back durations vs speed (m/s)
V_5_KMH      = 5.0 / 3.6               # 1.39 m/s
V_10_KMH     = 10.0 / 3.6              # 2.78 m/s
RAMP_T_STANDSTILL = 1.0                # <5 km/h
RAMP_T_5_10        = 1.0               # 5…10 km/h  ← updated
RAMP_T_ABOVE_10    = 1.0               # >10 km/h   ← updated

# HOLD-nudge parameters
HOLD_NUDGE_MAX_DEG = 2.0               # deg – absolute cap per nudge  ← updated
HOLD_NUDGE_RATE    = 1                 # issue every frame  ← updated

# Planner-match threshold for early resume
ANGLE_MATCH_THRESHOLD = 1.5            # deg

# Grace window after manual enable
GRACE_AFTER_ENABLE   = 0.4  # s
GRACE_TORQUE_BREAK   = 5.0  # Nm – override threshold during grace

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
    """Driver-friendly torque-blending state machine."""

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

        # Grace timer only after manual enable
        self.grace_timer = 0.0
        self._prev_lat_active = False  # track transition for grace

    # ------------------------------------------------------------------
    # Helper functions
    # ------------------------------------------------------------------

    def _update_debouncers(self, torque: float, ignore: bool) -> None:
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
        return t * t * (3.0 - 2.0 * t)  # 3t²-2t³

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update_torque_blending(
        self,
        CS: CarStateBase,
        CC: structs.CarControl,
        lat_active: bool,
        apply_angle: float,
    ) -> tuple[bool, float]:
        """Returns (lat_active, apply_angle) after torque blending."""

        # Detect manual enable transition for grace
        if CC.latActive and not self._prev_lat_active:
            self.grace_timer = GRACE_AFTER_ENABLE
        self._prev_lat_active = CC.latActive

        if not self.enabled:
            return lat_active, apply_angle

        v_ego = CS.out.vEgo
        wheel_angle = CS.out.steeringAngleDeg
        driver_torque = CS.out.steeringTorque

        # Grace handling
        ignore_torque = (self.grace_timer > 0.0) and (abs(driver_torque) < GRACE_TORQUE_BREAK)
        if self.grace_timer > 0.0:
            self.grace_timer = max(0.0, self.grace_timer - self.dt)

        # FSM transitions
        self._update_debouncers(driver_torque, ignore_torque)
        angle_diff = abs(apply_angle - wheel_angle)

        if self.state == TBState.AUTO:
            if self._above_timer >= ENTER_TIME:
                self.state = TBState.HOLD

        elif self.state == TBState.HOLD:
            if (angle_diff < ANGLE_MATCH_THRESHOLD and self._below_timer >= EXIT_TIME) or \
               (self._below_timer >= 2 * EXIT_TIME):
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

        # Outputs
        self._frame += 1
        apply_angle_out = apply_angle
        lat_active_out = CC.latActive

        if self.state == TBState.HOLD:
            if self._frame % HOLD_NUDGE_RATE == 0:
                delta = np.clip(apply_angle - wheel_angle, -HOLD_NUDGE_MAX_DEG, HOLD_NUDGE_MAX_DEG)
                apply_angle_out = wheel_angle + delta
            else:
                apply_angle_out = wheel_angle
            lat_active_out = False  # cut OP torque

        elif self.state == TBState.RAMP_BACK:
            t_norm = np.clip(self.ramp_timer / self.ramp_duration, 0.0, 1.0)
            alpha = self._smooth_step(t_norm)
            apply_angle_out = self.ramp_start + alpha * (apply_angle - self.ramp_start)
            lat_active_out = True

        return lat_active_out, apply_angle_out


# -----------------------------------------------------------------------------
# Helper class (unchanged)
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
            eac_status == "EAC_INHIBITED" and eac_error_code == "EAC_ERROR_HIGH_ANGLE_RATE_SAFETY"
        )