"""
Torque‑blending / co‑steering state‑machine for Tesla (openpilot / opendbc).
All comments in English.

Behaviour summary
-----------------
AUTO       – normal openpilot control  
HOLD       – driver holds wheel, we gently "nudge" ≤ ±1.5 ° toward planner
RAMP_BACK  – interpolate wheel angle back to planner over a speed‑based τ

Copyright (c) 2025.
Licensed under the MIT License.
"""

from __future__ import annotations

import enum
import numpy as np
from opendbc.car import structs
from opendbc.car.interfaces import CarStateBase

# -----------------------------------------------------------------------------
# Constants – tune to taste, keep semantic units.
# -----------------------------------------------------------------------------

DT_DEFAULT = 0.02                      # s – control‑loop period (50 Hz)

# Torque thresholds (Nm)
TORQUE_ENTER = 0.5                     # ≥ → driver clearly wants control
TORQUE_EXIT  = 1                     # ≤ → driver has released wheel

# Debounce times (s)
ENTER_TIME = 0.05                      # 50 ms continuous above TORQUE_ENTER
EXIT_TIME  = 0.10                      # 100 ms continuous below TORQUE_EXIT

# Ramp‑back durations as a function of speed – thresholds in m/s
V_5_KMH      = 5.0 / 3.6               # 5 km/h ≈ 1.39 m/s
V_10_KMH     = 10.0 / 3.6              # 10 km/h ≈ 2.78 m/s

RAMP_T_STANDSTILL = 1.0                # s – very gentle below 5 km/h
RAMP_T_5_10        = 0.75              # s – 5 … 10 km/h
RAMP_T_ABOVE_10    = 0.5               # s – anything faster

# HOLD‑nudge parameters
HOLD_NUDGE_MAX_DEG = 0.5               # deg – maximum offset commanded in HOLD

# -----------------------------------------------------------------------------
# Helper enum
# -----------------------------------------------------------------------------

class TBState(enum.IntEnum):
    """Three‑state co‑steering machine."""
    AUTO = 0       # openpilot has full control
    HOLD = 1       # driver holds wheel; we gently probe
    RAMP_BACK = 2  # wheel returns smoothly to planner

# -----------------------------------------------------------------------------
# Main controller
# -----------------------------------------------------------------------------

class TorqueBlendingCarController:
    """Driver‑friendly torque‑blending state machine with small HOLD nudge."""

    def __init__(self, dt: float = DT_DEFAULT):
        self.dt: float = dt
        self.enabled: bool = True

        # State machine variables
        self.state: TBState = TBState.AUTO
        self.ramp_timer: float = 0.0
        self.ramp_duration: float = 0.0
        self.ramp_start: float = 0.0

        # Debounce timers
        self._above_timer: float = 0.0
        self._below_timer: float = 0.0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _update_debouncers(self, torque: float) -> None:
        """Integrate timers for ≥ TORQUE_ENTER and ≤ TORQUE_EXIT."""
        if abs(torque) >= TORQUE_ENTER:
            self._above_timer += self.dt
        else:
            self._above_timer = 0.0

        if abs(torque) <= TORQUE_EXIT:
            self._below_timer += self.dt
        else:
            self._below_timer = 0.0

    @staticmethod
    def _ramp_duration_for_speed(v_ego: float) -> float:
        """Piece‑wise constant τ depending on speed (m/s)."""
        if v_ego < V_5_KMH:
            return RAMP_T_STANDSTILL
        if v_ego < V_10_KMH:
            return RAMP_T_5_10
        return RAMP_T_ABOVE_10

    # ------------------------------------------------------------------
    # Public API – called once per control cycle
    # ------------------------------------------------------------------

    def update_torque_blending(
        self,
        CS: CarStateBase,
        CC: structs.CarControl,
        lat_active: bool,
        apply_angle: float,
    ) -> tuple[bool, float]:
        """Update function called by CarController each cycle."""

        # Early‑out if disabled
        if not self.enabled:
            return lat_active, apply_angle

        v_ego = CS.out.vEgo                    # m/s
        wheel_angle = CS.out.steeringAngleDeg  # deg – real wheel position
        driver_torque = CS.out.steeringTorque  # Nm – driver effort

        # 1. State transitions ------------------------------------------------
        self._update_debouncers(driver_torque)

        if self.state == TBState.AUTO:
            if self._above_timer >= ENTER_TIME:
                self.state = TBState.HOLD

        elif self.state == TBState.HOLD:
            if self._below_timer >= EXIT_TIME:
                self.state = TBState.RAMP_BACK
                self.ramp_timer = 0.0
                self.ramp_start = wheel_angle
                self.ramp_duration = self._ramp_duration_for_speed(v_ego)

        elif self.state == TBState.RAMP_BACK:
            if self._above_timer >= ENTER_TIME:  # driver grabbed again
                self.state = TBState.HOLD
            else:
                self.ramp_timer += self.dt
                if self.ramp_timer >= self.ramp_duration:
                    self.state = TBState.AUTO

        # 2. Output according to current state --------------------------------
        if self.state == TBState.AUTO:
            lat_active_out = CC.latActive
            apply_out = apply_angle

        elif self.state == TBState.HOLD:
            # Nudge a small capped offset toward planner
            delta = apply_angle - wheel_angle
            delta_clamped = np.clip(delta, -HOLD_NUDGE_MAX_DEG, HOLD_NUDGE_MAX_DEG)
            apply_out = wheel_angle + delta_clamped
            lat_active_out = False

        else:  # RAMP_BACK
            alpha = np.clip(self.ramp_timer / self.ramp_duration, 0.0, 1.0)
            apply_out = self.ramp_start + alpha * (apply_angle - self.ramp_start)
            lat_active_out = True

        return lat_active_out, apply_out


# -----------------------------------------------------------------------------
# Optional helper – copy‑unchanged from previous version
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
