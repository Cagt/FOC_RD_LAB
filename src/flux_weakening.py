import numpy as np
import json
import os

class FluxWeakeningController:
    """
    Flux Weakening Controller for PMSM
    Implements flux weakening control to extend the speed range beyond base speed
    """
    
    def __init__(self, config_file="../config/motor_params.json"):
        """Initialize flux weakening controller with parameters"""
        self.load_parameters(config_file)
        self.initialize_controller()
        self.reset()
        
    def load_parameters(self, config_file):
        """Load controller parameters from config file"""
        with open(os.path.join(os.path.dirname(__file__), config_file), 'r') as f:
            params = json.load(f)
            
        self.poles = params['poles']
        self.Rs = params['Rs']
        self.Ld = params['Ld']
        self.Lq = params['Lq']
        self.flux_linkage = params['flux_linkage']
        self.max_current = params['max_current']
        self.max_voltage = params['max_voltage']
        self.dc_bus_voltage = params['dc_bus_voltage']
        self.sample_time = params['sample_time']
        
        # Flux weakening parameters
        self.base_speed = self.calculate_base_speed()
        self.fw_kp = 0.5  # Flux weakening proportional gain
        self.fw_ki = 10.0  # Flux weakening integral gain
        self.voltage_margin = 0.95  # Voltage margin for flux weakening (0-1)
        
    def calculate_base_speed(self):
        """Calculate the base speed (no-load speed at rated voltage)"""
        # Base speed occurs when back-EMF equals maximum available voltage
        v_max = self.dc_bus_voltage / np.sqrt(3)
        base_speed_e = v_max / self.flux_linkage  # Electrical angular velocity
        base_speed_m = base_speed_e / (self.poles / 2)  # Mechanical angular velocity
        return base_speed_m
    
    def initialize_controller(self):
        """Initialize PI controller for flux weakening"""
        # Flux weakening controller (negative d-axis current)
        current_limit = [-self.max_current, 0]  # Only negative d-axis current for flux weakening
        self.fw_controller = PIController(self.fw_kp, self.fw_ki, current_limit)
        
    def reset(self):
        """Reset flux weakening controller"""
        self.fw_controller.reset()
        self.id_fw = 0.0  # Flux weakening d-axis current
        self.in_flux_weakening = False
        
    def calculate_voltage_limit_ellipse(self, wr):
        """
        Calculate voltage limit ellipse parameters
        Returns the maximum d-q currents for a given speed
        """
        v_max = self.dc_bus_voltage / np.sqrt(3)
        
        # Voltage limit equation: (Rs*id - wr*Lq*iq)^2 + (Rs*iq + wr*Ld*id + wr*flux)^2 <= v_max^2
        
        # For simplicity, ignore resistance at high speeds
        if wr > 10:  # High speed approximation
            # Voltage limit ellipse center
            id_center = -self.flux_linkage / self.Ld
            iq_center = 0
            
            # Voltage limit ellipse radii
            a = v_max / (wr * self.Ld)  # d-axis radius
            b = v_max / (wr * self.Lq)  # q-axis radius
            
            return id_center, iq_center, a, b
        else:
            # Low speed: no flux weakening needed
            return 0, 0, self.max_current, self.max_current
    
    def calculate_current_limit_circle(self):
        """
        Calculate current limit circle parameters
        Returns the maximum d-q currents based on current limit
        """
        # Current limit equation: id^2 + iq^2 <= Imax^2
        return 0, 0, self.max_current  # Center at origin, radius = Imax
    
    def voltage_based_flux_weakening(self, wr, vd, vq):
        """
        Voltage-based flux weakening control
        Adjusts d-axis current based on voltage error
        """
        # Calculate voltage magnitude
        v_mag = np.sqrt(vd**2 + vq**2)
        
        # Voltage limit with margin
        v_limit = self.dc_bus_voltage / np.sqrt(3) * self.voltage_margin
        
        # Voltage error
        v_error = v_mag - v_limit
        
        # Only apply flux weakening if voltage exceeds limit
        if v_error > 0:
            self.in_flux_weakening = True
            # PI controller to generate negative d-axis current
            self.id_fw = self.fw_controller.update(v_error, self.sample_time)
        else:
            self.in_flux_weakening = False
            # Gradually reduce flux weakening current
            self.id_fw *= 0.95
            if abs(self.id_fw) < 0.01:
                self.id_fw = 0.0
                self.fw_controller.reset()
        
        return self.id_fw
    
    def speed_based_flux_weakening(self, wr, iq_ref):
        """
        Speed-based flux weakening control
        Calculates required d-axis current based on speed and q-axis current
        """
        # Check if we're above base speed
        if wr > self.base_speed:
            self.in_flux_weakening = True
            
            # Calculate required d-axis current for flux weakening
            # Using simplified equation: id = -(flux + Lq*iq) / Ld
            id_fw = -(self.flux_linkage + self.Lq * iq_ref) / self.Ld
            
            # Apply current limits
            id_fw = np.clip(id_fw, -self.max_current, 0)
            
            self.id_fw = id_fw
        else:
            self.in_flux_weakening = False
            self.id_fw = 0.0
            self.fw_controller.reset()
        
        return self.id_fw
    
    def lookup_table_flux_weakening(self, wr, iq_ref):
        """
        Lookup table-based flux weakening control
        Uses pre-calculated lookup table for optimal d-axis current
        """
        # This would typically use a 2D lookup table (wr, iq_ref) -> id_ref
        # For simplicity, we'll use a simplified calculation
        
        # Speed ratio (normalized to base speed)
        speed_ratio = wr / self.base_speed
        
        if speed_ratio <= 1.0:
            # Below base speed: no flux weakening
            return 0.0
        else:
            # Above base speed: apply flux weakening
            # Simplified calculation
            over_speed_factor = speed_ratio - 1.0
            
            # Calculate required d-axis current
            id_fw = -over_speed_factor * self.flux_linkage / self.Ld
            
            # Consider q-axis current effect
            id_fw -= self.Lq * iq_ref / self.Ld
            
            # Apply current limits
            id_fw = np.clip(id_fw, -self.max_current, 0)
            
            return id_fw
    
    def get_optimal_current_references(self, wr, torque_ref):
        """
        Calculate optimal d-q current references for maximum torque per ampere (MTPA)
        or flux weakening operation
        """
        # For PMSM with Ld = Lq (surface-mounted PMSM)
        if abs(self.Ld - self.Lq) < 1e-6:
            # Surface-mounted PMSM: MTPA is id = 0
            if wr <= self.base_speed:
                # Below base speed: MTPA operation
                id_ref = 0.0
                iq_ref = torque_ref / (1.5 * self.poles * self.flux_linkage)
            else:
                # Above base speed: flux weakening
                id_ref = self.speed_based_flux_weakening(wr, torque_ref / (1.5 * self.poles * self.flux_linkage))
                iq_ref = torque_ref / (1.5 * self.poles * (self.flux_linkage + self.Ld * id_ref))
        else:
            # Interior PMSM with saliency (Ld != Lq)
            # MTPA calculation for interior PMSM
            if wr <= self.base_speed:
                # MTPA operation
                # Simplified MTPA calculation
                iq_ref = np.sqrt(abs(torque_ref) / (1.5 * self.poles * (self.Ld - self.Lq)))
                id_ref = -self.flux_linkage / (2 * (self.Ld - self.Lq)) - np.sqrt(
                    (self.flux_linkage / (2 * (self.Ld - self.Lq)))**2 + iq_ref**2
                )
            else:
                # Flux weakening operation
                # This is more complex and requires solving a set of equations
                # For simplicity, we'll use the speed-based approach
                iq_ref = torque_ref / (1.5 * self.poles * self.flux_linkage)
                id_ref = self.speed_based_flux_weakening(wr, iq_ref)
        
        # Apply current limits
        i_mag = np.sqrt(id_ref**2 + iq_ref**2)
        if i_mag > self.max_current:
            scale = self.max_current / i_mag
            id_ref *= scale
            iq_ref *= scale
        
        return id_ref, iq_ref
    
    def update(self, wr, vd, vq, iq_ref, method='voltage'):
        """
        Update flux weakening controller
        method: 'voltage', 'speed', or 'lookup'
        """
        if method == 'voltage':
            return self.voltage_based_flux_weakening(wr, vd, vq)
        elif method == 'speed':
            return self.speed_based_flux_weakening(wr, iq_ref)
        elif method == 'lookup':
            return self.lookup_table_flux_weakening(wr, iq_ref)
        else:
            return 0.0


class PIController:
    """PI Controller implementation (duplicate from foc_control.py for standalone use)"""
    
    def __init__(self, kp, ki, output_limit=None):
        self.kp = kp  # Proportional gain
        self.ki = ki  # Integral gain
        self.output_limit = output_limit  # Output saturation limits
        self.integral = 0.0
        self.prev_error = 0.0
        
    def reset(self):
        """Reset controller state"""
        self.integral = 0.0
        self.prev_error = 0.0
        
    def update(self, error, dt):
        """Update PI controller"""
        # Proportional term
        p_term = self.kp * error
        
        # Integral term with anti-windup
        self.integral += error * dt
        i_term = self.ki * self.integral
        
        # Calculate output
        output = p_term + i_term
        
        # Apply output limits if specified
        if self.output_limit is not None:
            if output > self.output_limit[1]:
                output = self.output_limit[1]
                # Anti-windup: prevent integral from growing further
                if error > 0:
                    self.integral -= error * dt
            elif output < self.output_limit[0]:
                output = self.output_limit[0]
                # Anti-windup: prevent integral from growing further
                if error < 0:
                    self.integral -= error * dt
        
        return output