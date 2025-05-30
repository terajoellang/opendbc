"""
Torque‑blending / co‑steering state‑machine for Tesla (openpilot / opendbc).
All comments in English.

Copyright (c) 2025.
Licensed under the MIT License.
"""

from __future__ import annotations

import enum
import numpy as np
from opendbc.car import structs
from opendbc.car.interfaces import CarStateBase

# -----------------------------------------------------------------------------
# Constants – feel free to tune, but keep semantic units.
# -----------------------------------------------------------------------------

DT_DEFAULT = 0.02                      # s – control‑loop period (50 Hz)

# Torque thresholds (Nm)
TORQUE_ENTER = 2.0                     # ≥ → driver clearly wants control
TORQUE_EXIT  = 0.3                     # ≤ → driver has released wheel

# Debounce times (s)
ENTER_TIME = 0.05                      # 50 ms continuous above TORQUE_ENTER
EXIT_TIME  = 0.30                      # 300 ms continuous below TORQUE_EXIT

# Ramp‑back durations as a function of speed – thresholds in m/s
V_STANDSTILL = 0.0                     # 0 km/h
V_5_KMH      = 5.0 / 3.6               # 5 km/h ≈ 1.3889 m/s
V_10_KMH     = 10.0 / 3.6              # 10 km/h ≈ 2.7778 m/s

RAMP_T_STANDSTILL = 1.5                # s – very gentle below 5 km/h
RAMP_T_5_10        = 1.0               # s – 5 … 10 km/h
RAMP_T_ABOVE_10    = 0.5               # s – anything faster

# Openpilot treats vEgo in m/s, so we keep all speed thresholds in m/s.

# -----------------------------------------------------------------------------
# Helper enum
# -----------------------------------------------------------------------------

class TBState(enum.IntEnum):
    """Three‑state co‑steering machine."""
    AUTO = 0       # openpilot has full control
    HOLD = 1       # driver holds wheel, OP torque clamped
    RAMP_BACK = 2  # wheel returns gently to planner angle

# -----------------------------------------------------------------------------
# Main controller
# -----------------------------------------------------------------------------

class TorqueBlendingCarController:
    """Implements a driver‑friendly torque‑blending state machine.

    Behaviour summary:
      • AUTO      – normal openpilot control
      • HOLD      – freeze wheel at current hardware angle, disable lateral
      • RAMP_BACK – interpolate wheel angle back to planner over a speed‑based τ
    """

    # ---------------------------------------------------------------------
    # Construction
    # ---------------------------------------------------------------------

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

    # ---------------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------------

    def _update_debouncers(self, torque: float) -> None:
        """Integrate timers for conditions ≥ TORQUE_ENTER and ≤ TORQUE_EXIT."""
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

    # ---------------------------------------------------------------------
    # Public API – called once per control cycle
    # ---------------------------------------------------------------------

    def update_torque_blending(
        self,
        CS: CarStateBase,
        CC: structs.CarControl,
        lat_active: bool,
        apply_angle: float,
    ) -> tuple[bool, float]:
        """Main entry point for the Sunnypilot CarController.

        Args:
            CS:          Current CarState (openpilot object).
            CC:          Previous CarControl.
            lat_active:  Lateral‑active flag before blending.
            apply_angle: Planner steering angle before blending.
        Returns:
            (lat_active, apply_angle) updated for torque‑blending state.
        """

        # Early exit if disabled – keep previous behaviour untouched
        if not self.enabled:
            return lat_active, apply_angle

        v_ego = CS.out.vEgo                      # m/s
        wheel_angle_deg = CS.out.steeringAngleDeg
        driver_torque = CS.out.steeringTorque    # Nm

        # ------------------------------------------------------------------
        # 1. Update debounce timers and decide state transitions
        # ------------------------------------------------------------------
        self._update_debouncers(driver_torque)

        if self.state == TBState.AUTO:
            if self._above_timer >= ENTER_TIME:
                # Driver clearly takes over → HOLD
                self.state = TBState.HOLD

        elif self.state == TBState.HOLD:
            # Stay frozen until driver lets go long enough
            if self._below_timer >= EXIT_TIME:
                self.state = TBState.RAMP_BACK
                self.ramp_timer = 0.0
                self.ramp_start = wheel_angle_deg
                self.ramp_duration = self._ramp_duration_for_speed(v_ego)
            # If driver torques again strongly we just stay in HOLD (above timer resets)

        elif self.state == TBState.RAMP_BACK:
            # If driver grabs again → back to HOLD
            if self._above_timer >= ENTER_TIME:
                self.state = TBState.HOLD
            else:
                self.ramp_timer += self.dt
                if self.ramp_timer >= self.ramp_duration:
                    self.state = TBState.AUTO  # fade finished

        # ------------------------------------------------------------------
        # 2. Produce resulting apply_angle and lat_active according to state
        # ------------------------------------------------------------------
        if self.state == TBState.AUTO:
            # Full OP control
            lat_active_out = CC.latActive  # keep original
            apply_angle_out = apply_angle

        elif self.state == TBState.HOLD:
            # Freeze wheel; disable lateral so planner stops integrating error
            lat_active_out = False
            apply_angle_out = wheel_angle_deg

        else:  # RAMP_BACK
            alpha = np.clip(self.ramp_timer / self.ramp_duration, 0.0, 1.0)
            apply_angle_out = self.ramp_start + alpha * (apply_angle - self.ramp_start)
            lat_active_out = True  # lateral back on – we're guiding to planner

        return lat_active_out, apply_angle_out


# -----------------------------------------------------------------------------
# Optional helper that flags steeringDisengage from EAC errors.
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
