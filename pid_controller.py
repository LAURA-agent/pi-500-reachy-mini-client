"""Simple PID Controller for smooth face tracking.

A PID controller continuously calculates an error value as the difference between
a desired setpoint and a measured process variable, and applies a correction based
on proportional, integral, and derivative terms.

- P (Proportional): Responds to current error
- I (Integral): Responds to accumulated past errors
- D (Derivative): Responds to rate of change (damping)
"""

import time


class PIDController:
    """Simple PID controller with tunable gains."""

    def __init__(self, kp: float = 1.0, ki: float = 0.0, kd: float = 0.0, setpoint: float = 0.0):
        """Initialize PID controller.

        Args:
            kp: Proportional gain (responsiveness)
            ki: Integral gain (eliminates steady-state error)
            kd: Derivative gain (damping/smoothing)
            setpoint: Target value (typically 0 for centering)
        """
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.setpoint = setpoint

        # State variables
        self._last_error = 0.0
        self._integral = 0.0
        self._last_time = None

    def reset(self) -> None:
        """Reset controller state (call when target changes or tracking restarts)."""
        self._last_error = 0.0
        self._integral = 0.0
        self._last_time = None

    def update(self, measurement: float, dt: float | None = None) -> float:
        """Calculate PID output based on current measurement.

        Args:
            measurement: Current process variable (e.g., pixel error from center)
            dt: Time step in seconds (auto-calculated if None)

        Returns:
            Control output to apply to the system
        """
        # Calculate time step if not provided
        current_time = time.time()
        if dt is None:
            if self._last_time is None:
                dt = 0.0  # First call, no derivative
            else:
                dt = current_time - self._last_time
        self._last_time = current_time

        # Calculate error
        error = self.setpoint - measurement

        # Proportional term
        p_term = self.kp * error

        # Integral term (with basic anti-windup)
        if dt > 0:
            self._integral += error * dt
            # Simple anti-windup: clamp integral
            max_integral = 100.0  # Prevent integral windup
            self._integral = max(-max_integral, min(max_integral, self._integral))
        i_term = self.ki * self._integral

        # Derivative term (rate of change)
        if dt > 0:
            derivative = (error - self._last_error) / dt
        else:
            derivative = 0.0
        d_term = self.kd * derivative

        # Update state
        self._last_error = error

        # Return combined output
        return p_term + i_term + d_term

    def set_gains(self, kp: float | None = None, ki: float | None = None, kd: float | None = None) -> None:
        """Update PID gains (useful for runtime tuning).

        Args:
            kp: New proportional gain (None = keep current)
            ki: New integral gain (None = keep current)
            kd: New derivative gain (None = keep current)
        """
        if kp is not None:
            self.kp = kp
        if ki is not None:
            self.ki = ki
        if kd is not None:
            self.kd = kd
