"""
Torque-blending / co-steering state-machine for Tesla (openpilot / opendbc).
All comments in English.

Key behaviour (May 2025)
------------------------
AUTO       – normal openpilot control  
HOLD       – driver holds wheel, gentle nudge ≤ 2 °/frame  
RAMP_BACK  – wheel eases back to planner with S-curve (slow-fast-slow)  

Latest tweaks
-------------
* **Dynamic breakout thresholds** – easier to override when wheel ≠ planner.
* **Grace window (0.4 s)** fires only on *manual* AP enable (latActive ↑).
* **Emergency breakout** ≥ 5 Nm still works during grace.
* **Ramp durations** ≥ 5 km/h = 1 s.  
  Stand-still remains 1 s (parking manoeuvres feel natural).

Copyright (c) 2025. MIT-licensed.
"""

from __future__ import annotations

import enum
import numpy as np
from opendbc.car import structs
from opendbc.car.interfaces import CarStateBase

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

DT = 0.02                               # s – control-loop period (50 Hz)

# Base torque thresholds (Nm)
TORQUE_ENTER_BASE = 1.0                 # driver takes control ≥
TORQUE_ENTER_MIN  = 0.6                 # lower bound after scaling
TORQUE_EXIT       = 1.0                 # driver released ≤
TORQUE_ANGLE_SCALE = 0.05               # Nm per deg of |planner-wheel|

# Debounce (s)
ENTER_TIME = 0.05                       # 50 ms
EXIT_TIME  = 0.10                       # 100 ms

# Ramp-back τ vs speed (m/s)
V_5  = 5.0 / 3.6                        # 1.39
V_10 = 10.0 / 3.6                       # 2.78
TAU_STAND    = 1.0                      # <5 km/h
TAU_5_10     = 1.0                      # 5…10 km/h
TAU_ABOVE_10 = 1.0                      # >10 km/h

# HOLD nudge
HOLD_NUDGE_MAX = 2.0                   # deg/frame (absolute cap)

# Early resume if angle & torque small
ANGLE_MATCH = 1.5                       # deg

# Grace window after manual enable
GRACE_TIME  = 0.4                       # s
GRACE_BREAK = 5.0                       # Nm – break grace

# -----------------------------------------------------------------------------
# FSM
# -----------------------------------------------------------------------------

class TBState(enum.IntEnum):
    AUTO = 0
    HOLD = 1
    RAMP_BACK = 2

# -----------------------------------------------------------------------------
# Controller
# -----------------------------------------------------------------------------

class TorqueBlendingCarController:
    """Three-state driver-override controller with dynamic thresholds."""

    def __init__(self, dt: float = DT):
        self.dt = dt
        self.enabled = True

        # FSM vars
        self.state = TBState.AUTO
        self.ramp_timer = 0.0
        self.ramp_dur = 0.0
        self.ramp_start = 0.0

        # Debouncers
        self._above_t = 0.0
        self._below_t = 0.0

        # Misc
        self._frame = 0
        self._prev_lat_active = False  # for grace detection
        self.grace = 0.0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _smooth_step(self, x: float) -> float:
        """3x²-2x³ S-curve."""
        return x * x * (3.0 - 2.0 * x)

    def _tau_for_speed(self, v: float) -> float:
        if v < V_5:
            return TAU_STAND
        if v < V_10:
            return TAU_5_10
        return TAU_ABOVE_10

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
        """Returns updated (lat_active, apply_angle)."""
        if not self.enabled:
            return lat_active, apply_angle

        v_ego = CS.out.vEgo
        wheel = CS.out.steeringAngleDeg
        torque = CS.out.steeringTorque
        angle_err = abs(apply_angle - wheel)

        # Detect manual enable rising edge for grace window
        if lat_active and not self._prev_lat_active:
            self.grace = GRACE_TIME
        self._prev_lat_active = lat_active

        # Grace countdown
        in_grace = self.grace > 0.0
        if in_grace:
            self.grace = max(0.0, self.grace - self.dt)
        ignore_small_torque = in_grace and abs(torque) < GRACE_BREAK

        # Dynamic enter threshold (easier in curves)
        dyn_enter = max(TORQUE_ENTER_BASE - TORQUE_ANGLE_SCALE * angle_err, TORQUE_ENTER_MIN)

        # ----------------- Debouncers -----------------
        if ignore_small_torque:
            self._above_t = 0.0
            self._below_t = 0.0
        else:
            self._above_t = self._above_t + self.dt if abs(torque) >= dyn_enter else 0.0
            self._below_t = self._below_t + self.dt if abs(torque) <= TORQUE_EXIT else 0.0

        # ----------------- FSM transitions -----------
        if self.state == TBState.AUTO:
            if self._above_t >= ENTER_TIME:
                self.state = TBState.HOLD

        elif self.state == TBState.HOLD:
            if (angle_err < ANGLE_MATCH and self._below_t >= EXIT_TIME) or self._below_t >= 2 * EXIT_TIME:
                self.state = TBState.RAMP_BACK
                self.ramp_timer = 0.0
                self.ramp_start = wheel
                self.ramp_dur = self._tau_for_speed(v_ego)

        elif self.state == TBState.RAMP_BACK:
            if self._above_t >= ENTER_TIME:  # driver re-grabs
                self.state = TBState.HOLD
            else:
                self.ramp_timer += self.dt
                if self.ramp_timer >= self.ramp_dur:
                    self.state = TBState.AUTO

        # ----------------- Outputs -------------------
        self._frame += 1
        out_lat = CC.latActive
        out_angle = apply_angle

        if self.state == TBState.HOLD:
            if self._frame % 1 == 0:  # every frame now
                delta = np.clip(apply_angle - wheel, -HOLD_NUDGE_MAX, HOLD_NUDGE_MAX)
                out_angle = wheel + delta
            else:
                out_angle = wheel
            out_lat = False  # disable torque

        elif self.state == TBState.RAMP_BACK:
            alpha = self._smooth_step(np.clip(self.ramp_timer / self.ramp_dur, 0.0, 1.0))
            out_angle = self.ramp_start + alpha * (apply_angle - self.ramp_start)
            out_lat = True

        return out_lat, out_angle


# -----------------------------------------------------------------------------
# CarState helper (unchanged semantic)
# -----------------------------------------------------------------------------

class TorqueBlendingCarState:
    """Maps Tesla EAC errors to steeringDisengage."""

    def __init__(self):
        self.enabled = True

    def update_torque_blending(self, ret: structs.CarState, eac_status: str, eac_error_code: str) -> None:
        if not self.enabled:
            return
        ret.steeringDisengage = (
            eac_status == "EAC_INHIBITED" and eac_error_code == "EAC_ERROR_HIGH_ANGLE_RATE_SAFETY"
        )