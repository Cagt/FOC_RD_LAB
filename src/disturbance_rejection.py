import numpy as np
import json
import os
from collections import deque

class DisturbanceObserver:
    """
    Disturbance Observer for estimating and compensating external disturbances
    """
    
    def __init__(self, config_file="../config/motor_params.json"):
        """Initialize disturbance observer"""
        self.load_parameters(config_file)
        self.initialize_observer()
        self.reset()
        
    def load_parameters(self, config_file):
        """Load motor parameters from config file"""
        with open(os.path.join(os.path.dirname(__file__), config_file), 'r') as f:
            params = json.load(f)
            
        self.J = params['J']
        self.B = params['B']
        self.poles = params['poles']
        self.sample_time = params['sample_time']
        
        # Observer parameters
        self.observer_bandwidth = 100  # Observer bandwidth (rad/s)
        self.observer_gain = self.observer_bandwidth * self.J
        
    def initialize_observer(self):
        """Initialize disturbance observer parameters"""
        # Disturbance observer state variables
        self.disturbance_estimate = 0.0
        self.speed_estimate = 0.0
        self.observed_torque = 0.0
        
    def reset(self):
        """Reset observer state"""
        self.disturbance_estimate = 0.0
        self.speed_estimate = 0.0
        self.observed_torque = 0.0
        
    def update(self, Te, wr_actual, load_torque_estimate=0.0):
        """
        Update disturbance observer
        Te: Electromagnetic torque
        wr_actual: Actual mechanical speed
        load_torque_estimate: Initial estimate of load torque
        """
        # Calculate speed error
        speed_error = self.speed_estimate - wr_actual
        
        # Update disturbance estimate
        self.disturbance_estimate += self.observer_gain * speed_error * self.sample_time
        
        # Update speed estimate
        self.speed_estimate += ((Te - self.B * wr_actual - self.disturbance_estimate) / self.J) * self.sample_time
        
        # Calculate observed total disturbance torque
        self.observed_torque = self.disturbance_estimate + load_torque_estimate
        
        return self.observed_torque


class AdaptiveController:
    """
    Adaptive Controller for adjusting control parameters based on operating conditions
    """
    
    def __init__(self, config_file="../config/motor_params.json"):
        """Initialize adaptive controller"""
        self.load_parameters(config_file)
        self.initialize_controller()
        self.reset()
        
    def load_parameters(self, config_file):
        """Load motor parameters from config file"""
        with open(os.path.join(os.path.dirname(__file__), config_file), 'r') as f:
            params = json.load(f)
            
        self.Rs = params['Rs']
        self.Ld = params['Ld']
        self.Lq = params['Lq']
        self.flux_linkage = params['flux_linkage']
        self.J = params['J']
        self.B = params['B']
        self.sample_time = params['sample_time']
        
        # Adaptive controller parameters
        self.adaptation_rate = 0.01  # Adaptation rate
        self.min_gain = 0.1  # Minimum controller gain
        self.max_gain = 10.0  # Maximum controller gain
        
    def initialize_controller(self):
        """Initialize adaptive controller parameters"""
        # Adaptive gains
        self.id_kp_adaptive = 2.0 * self.Ld
        self.id_ki_adaptive = self.Rs
        self.iq_kp_adaptive = 2.0 * self.Lq
        self.iq_ki_adaptive = self.Rs
        self.speed_kp_adaptive = 0.5
        self.speed_ki_adaptive = 5.0
        
        # Performance metrics
        self.id_error_integral = 0.0
        self.iq_error_integral = 0.0
        self.speed_error_integral = 0.0
        
    def reset(self):
        """Reset adaptive controller"""
        self.id_error_integral = 0.0
        self.iq_error_integral = 0.0
        self.speed_error_integral = 0.0
        
    def update_gains(self, id_error, iq_error, speed_error, wr):
        """
        Update controller gains based on performance metrics
        """
        # Update error integrals
        self.id_error_integral += id_error**2 * self.sample_time
        self.iq_error_integral += iq_error**2 * self.sample_time
        self.speed_error_integral += speed_error**2 * self.sample_time
        
        # Adapt gains based on performance
        # Increase gains if performance is poor, decrease if good
        if self.id_error_integral > 0.1:
            self.id_kp_adaptive *= (1 + self.adaptation_rate)
            self.id_ki_adaptive *= (1 + self.adaptation_rate)
        else:
            self.id_kp_adaptive *= (1 - self.adaptation_rate * 0.5)
            self.id_ki_adaptive *= (1 - self.adaptation_rate * 0.5)
        
        if self.iq_error_integral > 0.1:
            self.iq_kp_adaptive *= (1 + self.adaptation_rate)
            self.iq_ki_adaptive *= (1 + self.adaptation_rate)
        else:
            self.iq_kp_adaptive *= (1 - self.adaptation_rate * 0.5)
            self.iq_ki_adaptive *= (1 - self.adaptation_rate * 0.5)
        
        if self.speed_error_integral > 1.0:
            self.speed_kp_adaptive *= (1 + self.adaptation_rate)
            self.speed_ki_adaptive *= (1 + self.adaptation_rate)
        else:
            self.speed_kp_adaptive *= (1 - self.adaptation_rate * 0.5)
            self.speed_ki_adaptive *= (1 - self.adaptation_rate * 0.5)
        
        # Apply gain limits
        self.id_kp_adaptive = np.clip(self.id_kp_adaptive, self.min_gain, self.max_gain)
        self.id_ki_adaptive = np.clip(self.id_ki_adaptive, self.min_gain, self.max_gain)
        self.iq_kp_adaptive = np.clip(self.iq_kp_adaptive, self.min_gain, self.max_gain)
        self.iq_ki_adaptive = np.clip(self.iq_ki_adaptive, self.min_gain, self.max_gain)
        self.speed_kp_adaptive = np.clip(self.speed_kp_adaptive, self.min_gain, self.max_gain)
        self.speed_ki_adaptive = np.clip(self.speed_ki_adaptive, self.min_gain, self.max_gain)
        
        # Reset integrals periodically
        if int(wr) % 100 == 0:
            self.id_error_integral *= 0.9
            self.iq_error_integral *= 0.9
            self.speed_error_integral *= 0.9
    
    def get_adaptive_gains(self):
        """Return current adaptive gains"""
        return {
            'id_kp': self.id_kp_adaptive,
            'id_ki': self.id_ki_adaptive,
            'iq_kp': self.iq_kp_adaptive,
            'iq_ki': self.iq_ki_adaptive,
            'speed_kp': self.speed_kp_adaptive,
            'speed_ki': self.speed_ki_adaptive
        }


class RobustController:
    """
    Robust Controller with H-infinity and sliding mode control techniques
    """
    
    def __init__(self, config_file="../config/motor_params.json"):
        """Initialize robust controller"""
        self.load_parameters(config_file)
        self.initialize_controller()
        self.reset()
        
    def load_parameters(self, config_file):
        """Load motor parameters from config file"""
        with open(os.path.join(os.path.dirname(__file__), config_file), 'r') as f:
            params = json.load(f)
            
        self.Rs = params['Rs']
        self.Ld = params['Ld']
        self.Lq = params['Lq']
        self.flux_linkage = params['flux_linkage']
        self.J = params['J']
        self.B = params['B']
        self.sample_time = params['sample_time']
        
        # Robust control parameters
        self.sliding_gain = 10.0  # Sliding mode control gain
        self.boundary_layer = 0.1  # Boundary layer thickness for chattering reduction
        self.h_infinity_gamma = 1.0  # H-infinity performance level
        
    def initialize_controller(self):
        """Initialize robust controller parameters"""
        # Sliding mode variables
        self.sliding_surface = 0.0
        self.sliding_surface_prev = 0.0
        self.control_output_prev = 0.0
        
        # H-infinity controller state
        self.hinf_state = np.zeros((2, 1))  # State vector for H-infinity controller
        
    def reset(self):
        """Reset robust controller"""
        self.sliding_surface = 0.0
        self.sliding_surface_prev = 0.0
        self.control_output_prev = 0.0
        self.hinf_state = np.zeros((2, 1))
        
    def sliding_mode_control(self, error, error_dot, disturbance_estimate=0.0):
        """
        Sliding mode control for disturbance rejection
        error: Tracking error
        error_dot: Derivative of tracking error
        disturbance_estimate: Estimated disturbance
        """
        # Define sliding surface: s = error_dot + lambda * error
        lambda_smc = 10.0  # Sliding surface parameter
        self.sliding_surface = error_dot + lambda_smc * error
        
        # Calculate control law with disturbance compensation
        equivalent_control = -error_dot - lambda_smc * error
        switching_control = -self.sliding_gain * np.sign(self.sliding_surface)
        
        # Add boundary layer to reduce chattering
        if abs(self.sliding_surface) < self.boundary_layer:
            switching_control = -self.sliding_gain * self.sliding_surface / self.boundary_layer
        
        # Total control with disturbance compensation
        control_output = equivalent_control + switching_control - disturbance_estimate
        
        return control_output
    
    def h_infinity_control(self, state, reference, disturbance):
        """
        H-infinity control for robust performance
        state: System state vector [position, velocity]
        reference: Reference signal
        disturbance: Disturbance signal
        """
        # Simplified H-infinity state feedback controller
        # In practice, this would be designed using proper H-infinity synthesis methods
        
        # State feedback gains (would be calculated using H-infinity synthesis)
        K = np.array([[5.0, 2.0]])  # State feedback gain matrix
        
        # Calculate control input
        error = reference - state[1]  # Speed error
        control_output = K @ state + 10.0 * error
        
        return float(control_output)


class DisturbanceRejectionController:
    """
    Combined Disturbance Rejection Controller
    Integrates disturbance observer, adaptive control, and robust control techniques
    """
    
    def __init__(self, config_file="../config/motor_params.json"):
        """Initialize disturbance rejection controller"""
        self.disturbance_observer = DisturbanceObserver(config_file)
        self.adaptive_controller = AdaptiveController(config_file)
        self.robust_controller = RobustController(config_file)
        
        # Control mode selection
        self.control_mode = 'observer'  # 'observer', 'adaptive', 'robust', 'combined'
        
        # Performance monitoring
        self.performance_metrics = {
            'speed_error_rms': 0.0,
            'torque_ripple': 0.0,
            'disturbance_rejection_ratio': 0.0
        }
        
        # Data buffers for performance analysis
        self.speed_error_buffer = deque(maxlen=1000)
        self.torque_buffer = deque(maxlen=1000)
        self.disturbance_buffer = deque(maxlen=1000)
        
    def reset(self):
        """Reset all controllers"""
        self.disturbance_observer.reset()
        self.adaptive_controller.reset()
        self.robust_controller.reset()
        
        # Clear data buffers
        self.speed_error_buffer.clear()
        self.torque_buffer.clear()
        self.disturbance_buffer.clear()
        
    def set_control_mode(self, mode):
        """Set control mode"""
        if mode in ['observer', 'adaptive', 'robust', 'combined']:
            self.control_mode = mode
        else:
            raise ValueError("Invalid control mode. Use 'observer', 'adaptive', 'robust', or 'combined'")
    
    def update(self, Te, wr_actual, wr_ref, id_actual, iq_actual, id_ref, iq_ref):
        """
        Update disturbance rejection controller
        Returns modified current references or voltage references
        """
        # Calculate errors
        speed_error = wr_ref - wr_actual
        id_error = id_ref - id_actual
        iq_error = iq_ref - iq_actual
        
        # Update disturbance observer
        disturbance_torque = self.disturbance_observer.update(Te, wr_actual)
        
        # Store data for performance analysis
        self.speed_error_buffer.append(speed_error)
        self.torque_buffer.append(Te)
        self.disturbance_buffer.append(disturbance_torque)
        
        # Calculate control action based on selected mode
        if self.control_mode == 'observer':
            # Use disturbance observer for feedforward compensation
            compensation = disturbance_torque / (1.5 * self.poles * self.flux_linkage)
            iq_ref_modified = iq_ref + compensation
            
        elif self.control_mode == 'adaptive':
            # Use adaptive controller to adjust gains
            self.adaptive_controller.update_gains(id_error, iq_error, speed_error, wr_actual)
            iq_ref_modified = iq_ref  # Gains are modified, not references
            
        elif self.control_mode == 'robust':
            # Use sliding mode control for robust disturbance rejection
            speed_error_dot = 0.0  # Would need proper calculation
            compensation = self.robust_controller.sliding_mode_control(speed_error, speed_error_dot, disturbance_torque)
            iq_ref_modified = iq_ref + compensation / 10.0  # Scale factor
            
        elif self.control_mode == 'combined':
            # Combine all techniques
            compensation = disturbance_torque / (1.5 * self.poles * self.flux_linkage)
            self.adaptive_controller.update_gains(id_error, iq_error, speed_error, wr_actual)
            speed_error_dot = 0.0  # Would need proper calculation
            robust_compensation = self.robust_controller.sliding_mode_control(speed_error, speed_error_dot, disturbance_torque)
            iq_ref_modified = iq_ref + compensation + robust_compensation / 20.0
            
        else:
            iq_ref_modified = iq_ref
        
        # Update performance metrics
        self.update_performance_metrics()
        
        return {
            'id_ref_modified': id_ref,
            'iq_ref_modified': iq_ref_modified,
            'disturbance_estimate': disturbance_torque,
            'adaptive_gains': self.adaptive_controller.get_adaptive_gains(),
            'performance_metrics': self.performance_metrics
        }
    
    def update_performance_metrics(self):
        """Update performance metrics"""
        if len(self.speed_error_buffer) > 10:
            # RMS speed error
            speed_errors = np.array(list(self.speed_error_buffer))
            self.performance_metrics['speed_error_rms'] = np.sqrt(np.mean(speed_errors**2))
            
            # Torque ripple (peak-to-peak / average)
            torques = np.array(list(self.torque_buffer))
            if len(torques) > 0 and np.mean(np.abs(torques)) > 0:
                torque_ripple = (np.max(torques) - np.min(torques)) / (2 * np.mean(np.abs(torques)))
                self.performance_metrics['torque_ripple'] = torque_ripple
            
            # Disturbance rejection ratio
            disturbances = np.array(list(self.disturbance_buffer))
            if len(disturbances) > 0 and np.max(np.abs(disturbances)) > 0:
                rejection_ratio = self.performance_metrics['speed_error_rms'] / np.max(np.abs(disturbances))
                self.performance_metrics['disturbance_rejection_ratio'] = rejection_ratio