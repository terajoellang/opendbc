"""
Enhanced torque‑blending implementation for Tesla (openpilot / opendbc).
All code comments are in English as requested.

Copyright (c) 2025.
Licensed under the MIT License.
"""

import numpy as np
from opendbc.car import structs
from opendbc.car.interfaces import CarStateBase

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

LOW_SPEED_ALLOWED = True
LOW_SPEED_CUTOFF = 13.5  # m/s ≈ 30 mph – cut‑over speed for low‑speed handling
LOW_SPEED_FADE_THRESHOLD = 5.0  # m/s – below this speed we use a slower fade‑out

TORQUE_DEADZONE_BASE = 0.5        # Nm – always masked out
TORQUE_DEADZONE_EXTRA = 0.15      # Nm – additional dead‑zone at very low speed
TORQUE_TO_ANGLE_CLIP = 10.0       # Nm – absolute safety clamp in case of EPS glitches
TORQUE_LINEAR_SCALE = 6.0         # Nm – tanh(x) reaches 76 % at ±6 Nm
MAX_OFFSET_DEG = 12.0             # deg – maximum wheel offset produced by blending

OVERRIDE_CONTINUE_ANGLE = 10.0    # deg – hysteresis for staying in override
OVERRIDE_TORQUE_THRESHOLD = 1.0   # Nm – cancels fade‑out instantly when driver re‑grabs

# Controller loop period; change if your control cycle differs
DEFAULT_DT = 0.02  # s (50 Hz)

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

def torque_blending_allowed(speed_mps: float) -> bool:
    """True if torque‑blending should run at this vehicle speed."""
    return (LOW_SPEED_ALLOWED and speed_mps < LOW_SPEED_CUTOFF) or speed_mps >= LOW_SPEED_CUTOFF


def speed_factor(v_ego: float) -> float:
    """Gain factor that scales linearly from 0.2 (standstill) to 1.0 (≥ LOW_SPEED_CUTOFF)."""
    return np.clip(v_ego / LOW_SPEED_CUTOFF, 0.2, 1.0)


# -----------------------------------------------------------------------------
# Main controller
# -----------------------------------------------------------------------------

class TorqueBlendingCarController:
    """Applies driver‑torque blending plus smooth fade‑out when the driver releases the wheel."""

    def __init__(self, dt: float = DEFAULT_DT):
        self.enabled: bool = True
        self.steering_override: bool = False
        self.fade_offset: float = 0.0  # current angle offset used for fade‑out (deg)
        self.dt: float = dt            # control loop period (s)

    # ---------------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------------

    def _torque_blended_angle(self, planner_angle: float, driver_torque: float, v_ego: float) -> float:
        """Returns the blended steering angle while the driver is actively overriding."""
        k_speed = speed_factor(v_ego)

        # Dynamic dead‑zone increases slightly at very low speed so that simply resting
        # a hand on the wheel does not trigger blending.
        deadzone = TORQUE_DEADZONE_BASE + TORQUE_DEADZONE_EXTRA * (1.0 - k_speed)
        if abs(driver_torque) < deadzone:
            return planner_angle + self.fade_offset

        # Remove the dead‑zone and compress torque via tanh() into ±1.
        # Remove the dead‑zone and clamp to ±TORQUE_TO_ANGLE_CLIP in case the EPS
        # sends out‑of‑range values. This retains the original "safety guard" behaviour.
        torque_eff = driver_torque - np.sign(driver_torque) * deadzone
        torque_eff = np.clip(torque_eff, -TORQUE_TO_ANGLE_CLIP, TORQUE_TO_ANGLE_CLIP)
        torque_norm = np.tanh(torque_eff / TORQUE_LINEAR_SCALE)

        # Convert normalised torque into a wheel offset (deg) proportional to speed.
        offset_deg = torque_norm * MAX_OFFSET_DEG * k_speed

        # Save for the upcoming fade‑out and return the blended angle.
        self.fade_offset = offset_deg
        return planner_angle + offset_deg

    def _fade_time_constant(self, v_ego: float) -> float:
        """Returns the fade‑out time constant τ (s) as a function of speed."""
        # 1.5 s at standstill → 0.5 s at cutoff speed
        return np.interp(v_ego,
                         [0.0, LOW_SPEED_FADE_THRESHOLD, LOW_SPEED_CUTOFF],
                         [1.5, 1.5, 0.5])

    # ---------------------------------------------------------------------
    # Public API – called once per control cycle
    # ---------------------------------------------------------------------

    def update_torque_blending(self,
                               CS: CarStateBase,
                               CC: structs.CarControl,
                               lat_active: bool,
                               apply_angle: float) -> tuple[bool, float]:
        """Main entry point.

        Args:
            CS:          Latest CarState instance.
            CC:          Last CarControl sent (contains latActive).
            lat_active:  Previous lateral‑active state – may be overridden here.
            apply_angle: Planner steering angle before torque blending.

        Returns:
            (lat_active, apply_angle) with updated values.
        """

        # ------------------------------------------------------------------
        # 1. Early exit when disabled or below speed threshold
        # ------------------------------------------------------------------
        if not self.enabled or not torque_blending_allowed(CS.out.vEgo):
            self.fade_offset = 0.0
            return lat_active, apply_angle

        planner_angle = apply_angle
        driver_torque = CS.out.steeringTorque

        # ------------------------------------------------------------------
        # 2. Detect driver override
        # ------------------------------------------------------------------
        hands_on = getattr(CS, "hands_on_level", 0) >= 3
        angle_diff = abs(CS.out.steeringAngleDeg - planner_angle)

        self.steering_override = hands_on or (
            CS.out.steeringPressed and
            angle_diff > OVERRIDE_CONTINUE_ANGLE and
            not CS.out.standstill
        )

        # Planner itself inactive → clear override
        if not CC.latActive:
            self.steering_override = False

        # ------------------------------------------------------------------
        # 3. Apply either active blending or fade‑out
        # ------------------------------------------------------------------
        if self.steering_override:
            # Active torque blending while the driver is exerting torque
            apply_angle = self._torque_blended_angle(planner_angle, driver_torque, CS.out.vEgo)
        else:
            # Exponential fade‑out of any residual offset
            if self.fade_offset != 0.0:
                tau = self._fade_time_constant(CS.out.vEgo)
                alpha = np.exp(-self.dt / tau)
                self.fade_offset *= alpha
                if abs(self.fade_offset) < 0.01:
                    self.fade_offset = 0.0

            apply_angle = planner_angle + self.fade_offset

            # Cancel fade‑out immediately if the driver torques the wheel again
            if abs(driver_torque) > OVERRIDE_TORQUE_THRESHOLD:
                self.fade_offset = 0.0

        # ------------------------------------------------------------------
        # 4. Final lateral‑active flag
        # ------------------------------------------------------------------
        lat_active = CC.latActive and not self.steering_override
        return lat_active, apply_angle


# -----------------------------------------------------------------------------
# Optional helper to map EAC errors to steeringDisengage flag
# -----------------------------------------------------------------------------

class TorqueBlendingCarState:
    """Augments the CarState with steering disengage information for sunny‑pilot."""

    def __init__(self):
        self.enabled: bool = True

    def update_torque_blending(self,
                               ret: structs.CarState,
                               eac_status: str,
                               eac_error_code: str) -> None:
        if not self.enabled or not torque_blending_allowed(ret.vEgo):
            return

        ret.steeringDisengage = (
            eac_status == "EAC_INHIBITED" and
            eac_error_code == "EAC_ERROR_HIGH_ANGLE_RATE_SAFETY"
        )
