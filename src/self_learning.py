import numpy as np
import json
import os
from collections import deque
import pickle

class MotorParameterIdentification:
    """
    Motor Parameter Identification and Self-Learning System
    Implements online parameter estimation for PMSM motors
    """
    
    def __init__(self, config_file="../config/motor_params.json"):
        """Initialize parameter identification system"""
        self.load_parameters(config_file)
        self.initialize_estimators()
        self.reset()
        
    def load_parameters(self, config_file):
        """Load initial motor parameters from config file"""
        with open(os.path.join(os.path.dirname(__file__), config_file), 'r') as f:
            params = json.load(f)
            
        self.poles = params['poles']
        self.Rs_nominal = params['Rs']
        self.Ld_nominal = params['Ld']
        self.Lq_nominal = params['Lq']
        self.flux_nominal = params['flux_linkage']
        self.J_nominal = params['J']
        self.B_nominal = params['B']
        self.max_current = params['max_current']
        self.sample_time = params['sample_time']
        
        # Identification parameters
        self.identification_enabled = True
        self.forgetting_factor = 0.99  # Forgetting factor for RLS
        self.min_speed_for_identification = 10  # Minimum speed for identification
        self.data_buffer_size = 1000  # Size of data buffer for offline analysis
        
    def initialize_estimators(self):
        """Initialize parameter estimators"""
        # Resistance estimator (RLS - Recursive Least Squares)
        self.Rs_estimator = {
            'theta': np.array([self.Rs_nominal]),  # Parameter vector [Rs]
            'P': np.eye(1) * 1000,  # Covariance matrix
            'lambda': self.forgetting_factor
        }
        
        # Inductance estimator (RLS)
        self.L_estimator = {
            'theta': np.array([self.Ld_nominal, self.Lq_nominal]),  # [Ld, Lq]
            'P': np.eye(2) * 1000,
            'lambda': self.forgetting_factor
        }
        
        # Flux linkage estimator (RLS)
        self.flux_estimator = {
            'theta': np.array([self.flux_nominal]),  # [flux_linkage]
            'P': np.eye(1) * 1000,
            'lambda': self.forgetting_factor
        }
        
        # Mechanical parameters estimator (RLS)
        self.mech_estimator = {
            'theta': np.array([self.J_nominal, self.B_nominal]),  # [J, B]
            'P': np.eye(2) * 1000,
            'lambda': self.forgetting_factor
        }
        
    def reset(self):
        """Reset all estimators and data buffers"""
        self.initialize_estimators()
        
        # Data buffers for offline analysis
        self.voltage_buffer = deque(maxlen=self.data_buffer_size)
        self.current_buffer = deque(maxlen=self.data_buffer_size)
        self.speed_buffer = deque(maxlen=self.data_buffer_size)
        self.torque_buffer = deque(maxlen=self.data_buffer_size)
        
        # Identified parameters
        self.Rs_identified = self.Rs_nominal
        self.Ld_identified = self.Ld_nominal
        self.Lq_identified = self.Lq_nominal
        self.flux_identified = self.flux_nominal
        self.J_identified = self.J_nominal
        self.B_identified = self.B_nominal
        
        # Identification status
        self.identification_count = 0
        self.identification_complete = False
        
    def rls_update(self, estimator, phi, y):
        """
        Recursive Least Squares update
        estimator: Dictionary with theta, P, lambda
        phi: Regression vector
        y: Measured output
        """
        theta = estimator['theta']
        P = estimator['P']
        lambda_factor = estimator['lambda']
        
        # Calculate gain vector
        denominator = lambda_factor + phi.T @ P @ phi
        if abs(denominator) > 1e-10:
            K = P @ phi / denominator
        else:
            K = np.zeros_like(theta)
        
        # Update parameter estimate
        error = y - phi.T @ theta
        theta_new = theta + K * error
        
        # Update covariance matrix
        P_new = (P - np.outer(K, phi.T @ P)) / lambda_factor
        
        # Update estimator
        estimator['theta'] = theta_new
        estimator['P'] = P_new
        
        return estimator, error
    
    def identify_resistance(self, vd, id, vq, iq, wr):
        """
        Identify stator resistance using stationary tests
        Works best at zero or very low speed
        """
        if abs(wr) < self.min_speed_for_identification:
            # At low speed, voltage equations simplify to:
            # vd ≈ Rs * id, vq ≈ Rs * iq
            
            # Use d-axis for identification
            if abs(id) > 0.1:  # Avoid division by very small current
                phi = np.array([id])
                y = vd
                self.Rs_estimator, error = self.rls_update(self.Rs_estimator, phi, y)
                self.Rs_identified = self.Rs_estimator['theta'][0]
                
            # Use q-axis for identification
            elif abs(iq) > 0.1:
                phi = np.array([iq])
                y = vq
                self.Rs_estimator, error = self.rls_update(self.Rs_estimator, phi, y)
                self.Rs_identified = self.Rs_estimator['theta'][0]
    
    def identify_inductance(self, vd, id, vq, iq, wr, did_dt, diq_dt):
        """
        Identify d-q axis inductances
        Requires dynamic operation and current derivatives
        """
        if abs(wr) > self.min_speed_for_identification:
            # Voltage equations in d-q frame:
            # vd = Rs*id - wr*Lq*iq + Ld*did/dt
            # vq = Rs*iq + wr*Ld*id + wr*flux + Lq*diq/dt
            
            # Rearrange for Ld and Lq identification
            # For Ld: use q-axis equation
            if abs(diq_dt) > 1e-6:
                phi_Ld = np.array([wr * id, diq_dt])
                y_Ld = vq - self.Rs_identified * iq - wr * self.flux_identified
                self.L_estimator['theta'][0] = self.Ld_identified
                self.L_estimator, error_Ld = self.rls_update(self.L_estimator, phi_Ld, y_Ld)
                self.Ld_identified = self.L_estimator['theta'][0]
            
            # For Lq: use d-axis equation
            if abs(did_dt) > 1e-6:
                phi_Lq = np.array([-wr * iq, did_dt])
                y_Lq = vd - self.Rs_identified * id
                temp_theta = np.array([self.Lq_identified, self.Ld_identified])
                temp_estimator = {'theta': temp_theta, 'P': self.L_estimator['P'], 'lambda': self.L_estimator['lambda']}
                temp_estimator, error_Lq = self.rls_update(temp_estimator, phi_Lq, y_Lq)
                self.Lq_identified = temp_estimator['theta'][0]
    
    def identify_flux_linkage(self, vd, id, vq, iq, wr):
        """
        Identify permanent magnet flux linkage
        Works best at medium to high speeds
        """
        if abs(wr) > self.min_speed_for_identification and abs(iq) > 0.1:
            # From q-axis voltage equation:
            # vq = Rs*iq + wr*Ld*id + wr*flux + Lq*diq/dt
            
            # Assuming diq/dt ≈ 0 for steady-state operation
            phi_flux = np.array([wr])
            y_flux = vq - self.Rs_identified * iq - wr * self.Ld_identified * id
            self.flux_estimator, error = self.rls_update(self.flux_estimator, phi_flux, y_flux)
            self.flux_identified = self.flux_estimator['theta'][0]
    
    def identify_mechanical_parameters(self, Te, wr, dwr_dt):
        """
        Identify mechanical parameters (J and B)
        Te: Electromagnetic torque
        wr: Mechanical angular velocity
        dwr_dt: Angular acceleration
        """
        # Mechanical equation: Te = J*dwr/dt + B*wr + TL
        # Assuming load torque TL is known or can be estimated
        
        if abs(dwr_dt) > 1e-6:
            phi_mech = np.array([dwr_dt, wr])
            y_mech = Te  # Assuming no load torque for simplicity
            self.mech_estimator, error = self.rls_update(self.mech_estimator, phi_mech, y_mech)
            self.J_identified = self.mech_estimator['theta'][0]
            self.B_identified = self.mech_estimator['theta'][1]
    
    def inject_test_signal(self, signal_type='current', amplitude=1.0, frequency=10.0):
        """
        Generate test signals for parameter identification
        signal_type: 'current', 'voltage', or 'speed'
        amplitude: Signal amplitude
        frequency: Signal frequency in Hz
        """
        t = np.arange(0, 1, self.sample_time)
        
        if signal_type == 'current':
            # Inject small AC current perturbation
            id_test = amplitude * np.sin(2 * np.pi * frequency * t)
            iq_test = amplitude * np.cos(2 * np.pi * frequency * t)
            return id_test, iq_test
        
        elif signal_type == 'voltage':
            # Inject small AC voltage perturbation
            vd_test = amplitude * np.sin(2 * np.pi * frequency * t)
            vq_test = amplitude * np.cos(2 * np.pi * frequency * t)
            return vd_test, vq_test
        
        elif signal_type == 'speed':
            # Speed reference perturbation
            speed_test = amplitude * np.sin(2 * np.pi * frequency * t)
            return speed_test
        
        return None
    
    def update(self, vd, id, vq, iq, wr, Te=None, load_torque=0.0):
        """
        Update parameter identification
        Should be called at each control interval
        """
        if not self.identification_enabled:
            return
        
        # Store data in buffers
        self.voltage_buffer.append((vd, vq))
        self.current_buffer.append((id, iq))
        self.speed_buffer.append(wr)
        if Te is not None:
            self.torque_buffer.append(Te)
        
        # Calculate derivatives using finite differences
        if len(self.current_buffer) >= 2:
            id_prev, iq_prev = self.current_buffer[-2]
            did_dt = (id - id_prev) / self.sample_time
            diq_dt = (iq - iq_prev) / self.sample_time
        else:
            did_dt = 0.0
            diq_dt = 0.0
        
        if len(self.speed_buffer) >= 2:
            wr_prev = self.speed_buffer[-2]
            dwr_dt = (wr - wr_prev) / self.sample_time
        else:
            dwr_dt = 0.0
        
        # Perform parameter identification
        self.identify_resistance(vd, id, vq, iq, wr)
        self.identify_inductance(vd, id, vq, iq, wr, did_dt, diq_dt)
        self.identify_flux_linkage(vd, id, vq, iq, wr)
        
        if Te is not None:
            self.identify_mechanical_parameters(Te, wr, dwr_dt)
        
        self.identification_count += 1
        
        # Check if identification is complete (based on number of samples)
        if self.identification_count > 100:
            self.identification_complete = True
    
    def get_identified_parameters(self):
        """Return the identified motor parameters"""
        return {
            'Rs': self.Rs_identified,
            'Ld': self.Ld_identified,
            'Lq': self.Lq_identified,
            'flux_linkage': self.flux_identified,
            'J': self.J_identified,
            'B': self.B_identified,
            'identification_count': self.identification_count,
            'identification_complete': self.identification_complete
        }
    
    def save_identified_parameters(self, filename="identified_params.json"):
        """Save identified parameters to file"""
        params = self.get_identified_parameters()
        
        with open(os.path.join(os.path.dirname(__file__), "../config/" + filename), 'w') as f:
            json.dump(params, f, indent=4)
    
    def load_identified_parameters(self, filename="identified_params.json"):
        """Load previously identified parameters from file"""
        try:
            with open(os.path.join(os.path.dirname(__file__), "../config/" + filename), 'r') as f:
                params = json.load(f)
            
            self.Rs_identified = params['Rs']
            self.Ld_identified = params['Ld']
            self.Lq_identified = params['Lq']
            self.flux_identified = params['flux_linkage']
            self.J_identified = params['J']
            self.B_identified = params['B']
            self.identification_complete = params['identification_complete']
            
            return True
        except FileNotFoundError:
            return False
    
    def export_data_for_analysis(self, filename="identification_data.pkl"):
        """Export collected data for offline analysis"""
        data = {
            'voltage': list(self.voltage_buffer),
            'current': list(self.current_buffer),
            'speed': list(self.speed_buffer),
            'torque': list(self.torque_buffer),
            'parameters': self.get_identified_parameters()
        }
        
        with open(os.path.join(os.path.dirname(__file__), "../data/" + filename), 'wb') as f:
            pickle.dump(data, f)
    
    def enable_identification(self):
        """Enable parameter identification"""
        self.identification_enabled = True
    
    def disable_identification(self):
        """Disable parameter identification"""
        self.identification_enabled = False