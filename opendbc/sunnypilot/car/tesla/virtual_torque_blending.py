"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and other contributors.
This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for further details.
"""

import time
import numpy as np
from opendbc.car import structs
from opendbc.car.interfaces import CarStateBase

# ---------------- Constants ----------------
TORQUE_TO_ANGLE_DEADZONE = 0.5      # Nm. Ignore torque below this.
TORQUE_TO_ANGLE_CLIP = 10.0         # Nm. Maximum effective torque.
BLENDING_RAMP_DURATION = 0.5        # seconds over which the blend is ramped.
# (Turn signal logic has been removed.)

# ---------------- Nonlinear Blending Multiplier ----------------
def nonlinear_blending_multiplier(vehicle_speed: float) -> float:
    """
    Returns the base multiplier (in deg/Nm) that converts effective driver torque to a steering angle offset.
    Speed-dependent (vehicle_speed in km/h):
      0–15 km/h:   12.0   (highest sensitivity)
      15–30 km/h:  9.0
      30–50 km/h:  6.5
      50–70 km/h:  5.0
      70–140 km/h: 3.0   (lowest sensitivity)
    """
    if vehicle_speed < 15:
        return 12.0
    elif vehicle_speed < 30:
        return 9.0
    elif vehicle_speed < 50:
        return 6.5
    elif vehicle_speed < 70:
        return 5.0
    else:
        return 3.0

# ---------------- Natural Torque Adjustment with Logistic Scaling ----------------
def natural_torque_adjustment(apply_angle: float, torsion_bar_torque: float, vehicle_speed: float) -> float:
    """
    Computes the target steering angle command (in degrees) by blending the openpilot (SP) command with the driver’s torque.
    Steps:
      1. If driver's torque is below the deadzone, return the SP command.
      2. Otherwise, subtract the deadzone (preserving sign) and clip the torque.
      3. Choose a base multiplier from nonlinear_blending_multiplier()--boost it by 1.5 if the driver is opposing the SP command.
      4. Use a logistic (sigmoid) scaling function to softly saturate the influence.
      5. Return the modified (target) steering angle.
    """
    if abs(torsion_bar_torque) < TORQUE_TO_ANGLE_DEADZONE:
        return apply_angle

    effective_torque = torsion_bar_torque - np.sign(torsion_bar_torque) * TORQUE_TO_ANGLE_DEADZONE
    effective_torque = np.clip(effective_torque, -TORQUE_TO_ANGLE_CLIP, TORQUE_TO_ANGLE_CLIP)
    
    if apply_angle * torsion_bar_torque >= 0:
        base_multiplier = nonlinear_blending_multiplier(vehicle_speed)
    else:
        base_multiplier = nonlinear_blending_multiplier(vehicle_speed) * 1.5

    k = 1.0  # Logistic scaling steepness
    scale = 1 / (1 + np.exp(-k * (abs(effective_torque) - (TORQUE_TO_ANGLE_CLIP / 2))))
    
    target_angle = apply_angle + effective_torque * base_multiplier * scale
    return target_angle

# ---------------- Torque Blending Car Controller ----------------
class TorqueBlendingCarController:
    def __init__(self):
        self.enabled = True
        self.steering_override = False
        # Removed turn-signal state logic.
        self.blending_start_time = None
        self.previous_angle = 0.0  # in degrees

    def torque_blended_angle(self, apply_angle: float, torsion_bar_torque: float, vehicle_speed: float) -> float:
        """
        Computes the new steering angle using natural torque adjustment and then blends the result with the
        previous command over BLENDING_RAMP_DURATION seconds.
        """
        target_angle = natural_torque_adjustment(apply_angle, torsion_bar_torque, vehicle_speed)
        
        # Start a new ramp if no ramp is active.
        if self.blending_start_time is None:
            self.blending_start_time = time.time()
        
        elapsed_time = time.time() - self.blending_start_time
        blend_ratio = min(elapsed_time / BLENDING_RAMP_DURATION, 1.0)
        smoothed_angle = self.previous_angle * (1 - blend_ratio) + target_angle * blend_ratio
        return smoothed_angle

    def update_torque_blending(self, CS: CarStateBase, CC: structs.CarControl, lat_active: bool, apply_angle: float) -> tuple[bool, float]:
        """
        Updates the steering angle command by blending the SP desired angle with driver torque.
        Uses traditional override logic: if the driver is actively providing manual input (via
        hands_on_level or steeringPressed), the override flag remains. When manual input stops,
        the override is cleared and the blending timer is reset so that SP control can quickly resync.
        
        Returns:
            (lat_active, blended_angle), where blended_angle is in degrees.
        """
        if not self.enabled:
            return lat_active, apply_angle

        # Convert vehicle speed from m/s to km/h.
        vehicle_speed = getattr(CS.out, 'vEgo', 0.0) * 3.6

        # Traditional manual override:
        override_threshold = 10.0  # degrees
        if not self.steering_override:
            self.steering_override = CS.hands_on_level >= 3 or (
                CS.out.steeringPressed and abs(CS.out.steeringAngleDeg - apply_angle) > override_threshold and not CS.out.standstill
            )
        
        # When the driver stops providing manual input, we want SP to resume.
        if not CS.out.steeringPressed and CS.hands_on_level < 3:
            # Clear the override flag.
            if self.steering_override:
                self.steering_override = False
                # Reset the blending timer and resynchronize to the SP desired angle.
                self.blending_start_time = None
                self.previous_angle = apply_angle

        if not CC.latActive:
            self.steering_override = False

        lat_active = CC.latActive and not self.steering_override

        blended_angle = self.torque_blended_angle(apply_angle, CS.out.steeringTorque, vehicle_speed)
        self.previous_angle = blended_angle

        return lat_active, blended_angle

# ---------------- Torque Blending Car State ----------------
class TorqueBlendingCarState:
    def __init__(self):
        self.enabled = True

    def update_torque_blending(self, ret: structs.CarState, eac_status: str, eac_error_code: str) -> None:
        if not self.enabled:
            return
        ret.steeringDisengage = (eac_status == "EAC_INHIBITED" and 
                                  eac_error_code == "EAC_ERROR_HIGH_ANGLE_RATE_SAFETY")