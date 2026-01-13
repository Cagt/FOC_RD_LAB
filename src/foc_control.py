import numpy as np
import json
import os

class PIController:
    """PI Controller implementation"""
    
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

class FOCController:
    """
    Field-Oriented Control (FOC) Controller for PMSM
    Implements the complete FOC algorithm including Clarke and Park transformations
    """
    
    def __init__(self, config_file="../config/motor_params.json"):
        """Initialize FOC controller with parameters"""
        self.load_parameters(config_file)
        self.initialize_controllers()
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
        
        # Controller gains (can be tuned)
        self.id_kp = 2.0 * self.Ld
        self.id_ki = self.Rs
        self.iq_kp = 2.0 * self.Lq
        self.iq_ki = self.Rs
        self.speed_kp = 0.5
        self.speed_ki = 5.0
        
    def initialize_controllers(self):
        """Initialize PI controllers"""
        # Current controllers
        voltage_limit = [-self.max_voltage, self.max_voltage]
        self.id_controller = PIController(self.id_kp, self.id_ki, voltage_limit)
        self.iq_controller = PIController(self.iq_kp, self.iq_ki, voltage_limit)
        
        # Speed controller
        current_limit = [-self.max_current, self.max_current]
        self.speed_controller = PIController(self.speed_kp, self.speed_ki, current_limit)
        
    def reset(self):
        """Reset all controllers"""
        self.id_controller.reset()
        self.iq_controller.reset()
        self.speed_controller.reset()
        
        # Reset references
        self.id_ref = 0.0
        self.iq_ref = 0.0
        self.speed_ref = 0.0
        
        # Reset outputs
        self.vd = 0.0
        self.vq = 0.0
        
    def clarke_transform(self, ia, ib, ic):
        """Clarke transformation: three-phase to alpha-beta"""
        i_alpha = ia
        i_beta = (ia + 2 * ib) / np.sqrt(3)
        return i_alpha, i_beta
    
    def inverse_clarke_transform(self, v_alpha, v_beta):
        """Inverse Clarke transformation: alpha-beta to three-phase"""
        va = v_alpha
        vb = (-v_alpha + np.sqrt(3) * v_beta) / 2
        vc = (-v_alpha - np.sqrt(3) * v_beta) / 2
        return va, vb, vc
    
    def park_transform(self, i_alpha, i_beta, theta_e):
        """Park transformation: alpha-beta to d-q"""
        id = i_alpha * np.cos(theta_e) + i_beta * np.sin(theta_e)
        iq = -i_alpha * np.sin(theta_e) + i_beta * np.cos(theta_e)
        return id, iq
    
    def inverse_park_transform(self, vd, vq, theta_e):
        """Inverse Park transformation: d-q to alpha-beta"""
        v_alpha = vd * np.cos(theta_e) - vq * np.sin(theta_e)
        v_beta = vd * np.sin(theta_e) + vq * np.cos(theta_e)
        return v_alpha, v_beta
    
    def space_vector_modulation(self, v_alpha, v_beta):
        """
        Space Vector PWM modulation
        Converts alpha-beta voltages to three-phase duty cycles
        """
        # Sector determination
        sector = self.get_sector(v_alpha, v_beta)
        
        # Calculate duty cycles based on sector
        if sector == 1:
            t1 = np.sqrt(3) * v_beta - v_alpha
            t2 = 2 * v_alpha
        elif sector == 2:
            t1 = np.sqrt(3) * v_beta + v_alpha
            t2 = -np.sqrt(3) * v_beta + v_alpha
        elif sector == 3:
            t1 = 2 * v_beta
            t2 = -np.sqrt(3) * v_beta - v_alpha
        elif sector == 4:
            t1 = -np.sqrt(3) * v_beta - v_alpha
            t2 = -2 * v_alpha
        elif sector == 5:
            t1 = -np.sqrt(3) * v_beta + v_alpha
            t2 = np.sqrt(3) * v_beta + v_alpha
        else:  # sector == 6
            t1 = -2 * v_beta
            t2 = np.sqrt(3) * v_beta - v_alpha
        
        # Normalize to DC bus voltage
        v_dc = self.dc_bus_voltage
        t1 = t1 / v_dc
        t2 = t2 / v_dc
        
        # Calculate duty cycles
        t0 = (1 - t1 - t2) / 2
        
        # Generate three-phase duty cycles based on sector
        if sector == 1:
            duty_a = t1 + t2 + t0
            duty_b = t2 + t0
            duty_c = t0
        elif sector == 2:
            duty_a = t1 + t0
            duty_b = t1 + t2 + t0
            duty_c = t0
        elif sector == 3:
            duty_a = t0
            duty_b = t1 + t2 + t0
            duty_c = t2 + t0
        elif sector == 4:
            duty_a = t0
            duty_b = t1 + t0
            duty_c = t1 + t2 + t0
        elif sector == 5:
            duty_a = t2 + t0
            duty_b = t0
            duty_c = t1 + t2 + t0
        else:  # sector == 6
            duty_a = t1 + t2 + t0
            duty_b = t0
            duty_c = t1 + t0
        
        return duty_a, duty_b, duty_c
    
    def get_sector(self, v_alpha, v_beta):
        """Determine the sector of the reference voltage vector"""
        angle = np.arctan2(v_beta, v_alpha)
        if angle < 0:
            angle += 2 * np.pi
        
        sector = int(angle / (np.pi / 3)) + 1
        if sector > 6:
            sector = 6
            
        return sector
    
    def speed_control(self, speed_ref, speed_actual):
        """Speed control loop"""
        speed_error = speed_ref - speed_actual
        iq_ref = self.speed_controller.update(speed_error, self.sample_time)
        return iq_ref
    
    def current_control(self, id_ref, iq_ref, id_actual, iq_actual, theta_e):
        """Current control loop"""
        # Current errors
        id_error = id_ref - id_actual
        iq_error = iq_ref - iq_actual
        
        # PI controllers
        self.vd = self.id_controller.update(id_error, self.sample_time)
        self.vq = self.iq_controller.update(iq_error, self.sample_time)
        
        # Apply voltage limits
        v_mag = np.sqrt(self.vd**2 + self.vq**2)
        v_max = self.dc_bus_voltage / np.sqrt(3)
        
        if v_mag > v_max:
            scale = v_max / v_mag
            self.vd *= scale
            self.vq *= scale
        
        return self.vd, self.vq
    
    def update(self, ia, ib, ic, speed_ref, speed_actual, theta_e):
        """
        Complete FOC control update
        Returns three-phase duty cycles for inverter control
        """
        # Clarke transformation
        i_alpha, i_beta = self.clarke_transform(ia, ib, ic)
        
        # Park transformation
        id_actual, iq_actual = self.park_transform(i_alpha, i_beta, theta_e)
        
        # Speed control (generates iq reference)
        self.iq_ref = self.speed_control(speed_ref, speed_actual)
        
        # Current control (generates vd, vq)
        self.vd, self.vq = self.current_control(self.id_ref, self.iq_ref, 
                                                id_actual, iq_actual, theta_e)
        
        # Inverse Park transformation
        v_alpha, v_beta = self.inverse_park_transform(self.vd, self.vq, theta_e)
        
        # Space vector modulation
        duty_a, duty_b, duty_c = self.space_vector_modulation(v_alpha, v_beta)
        
        return {
            'duty_a': duty_a,
            'duty_b': duty_b,
            'duty_c': duty_c,
            'id_actual': id_actual,
            'iq_actual': iq_actual,
            'id_ref': self.id_ref,
            'iq_ref': self.iq_ref,
            'vd': self.vd,
            'vq': self.vq
        }
    
    def set_id_reference(self, id_ref):
        """Set d-axis current reference"""
        self.id_ref = np.clip(id_ref, -self.max_current, self.max_current)
    
    def set_speed_reference(self, speed_ref):
        """Set speed reference in rad/s"""
        self.speed_ref = speed_ref