import numpy as np
import json
from scipy.integrate import odeint
import os

class PMSMModel:
    """
    PMSM (Permanent Magnet Synchronous Motor) Model
    Implements the mathematical model of a PMSM motor in d-q reference frame
    """
    
    def __init__(self, config_file="../config/motor_params.json"):
        """Initialize motor model with parameters from config file"""
        self.load_parameters(config_file)
        self.reset_state()
        
    def load_parameters(self, config_file):
        """Load motor parameters from JSON config file"""
        with open(os.path.join(os.path.dirname(__file__), config_file), 'r') as f:
            params = json.load(f)
            
        self.poles = params['poles']
        self.Rs = params['Rs']  # Stator resistance (Ohm)
        self.Ld = params['Ld']  # d-axis inductance (H)
        self.Lq = params['Lq']  # q-axis inductance (H)
        self.flux_linkage = params['flux_linkage']  # Permanent magnet flux linkage (Wb)
        self.J = params['J']  # Moment of inertia (kg.m^2)
        self.B = params['B']  # Viscous friction coefficient (N.m.s)
        self.max_current = params['max_current']  # Maximum current (A)
        self.max_voltage = params['max_voltage']  # Maximum voltage (V)
        self.dc_bus_voltage = params['dc_bus_voltage']  # DC bus voltage (V)
        self.sample_time = params['sample_time']  # Sample time (s)
        
    def reset_state(self):
        """Reset motor state to initial conditions"""
        self.id = 0.0  # d-axis current
        self.iq = 0.0  # q-axis current
        self.wr = 0.0  # Electrical angular velocity (rad/s)
        self.theta_e = 0.0  # Electrical angle (rad)
        self.load_torque = 0.0  # Load torque (N.m)
        
    def electrical_dynamics(self, state, t, vd, vq, load_torque):
        """
        Electrical dynamics of PMSM in d-q reference frame
        state = [id, iq, wr, theta_e]
        """
        id, iq, wr, theta_e = state
        
        # Voltage equations in d-q reference frame
        did_dt = (vd - self.Rs * id + wr * self.Lq * iq) / self.Ld
        diq_dt = (vq - self.Rs * iq - wr * self.Ld * id - wr * self.flux_linkage) / self.Lq
        
        # Electromagnetic torque
        Te = 1.5 * self.poles * (self.flux_linkage * iq + (self.Ld - self.Lq) * id * iq)
        
        # Mechanical dynamics
        dwr_dt = (Te - load_torque - self.B * wr) / self.J
        
        # Electrical angle
        dtheta_dt = wr
        
        return [did_dt, diq_dt, dwr_dt, dtheta_dt]
    
    def update(self, vd, vq, load_torque=0.0):
        """
        Update motor state for one time step
        vd, vq: d-q axis voltages
        load_torque: External load torque
        """
        # Current state
        state = [self.id, self.iq, self.wr, self.theta_e]
        
        # Time points for integration
        t = [0, self.sample_time]
        
        # Solve ODE for one time step
        solution = odeint(self.electrical_dynamics, state, t, args=(vd, vq, load_torque))
        
        # Update state
        self.id = solution[-1, 0]
        self.iq = solution[-1, 1]
        self.wr = solution[-1, 2]
        self.theta_e = solution[-1, 3]
        
        # Keep electrical angle in [0, 2*pi]
        self.theta_e = self.theta_e % (2 * np.pi)
        
        # Store load torque
        self.load_torque = load_torque
        
        # Calculate electromagnetic torque
        Te = 1.5 * self.poles * (self.flux_linkage * self.iq + (self.Ld - self.Lq) * self.id * self.iq)
        
        return {
            'id': self.id,
            'iq': self.iq,
            'wr': self.wr,
            'theta_e': self.theta_e,
            'Te': Te,
            'speed_rpm': self.wr * 60 / (2 * np.pi * self.poles / 2)
        }
    
    def get_three_phase_currents(self):
        """Convert d-q currents to three-phase currents"""
        # Clarke transformation (d-q to alpha-beta)
        i_alpha = self.id * np.cos(self.theta_e) - self.iq * np.sin(self.theta_e)
        i_beta =  self.id * np.sin(self.theta_e) + self.iq * np.cos(self.theta_e)
        
        # Park transformation (alpha-beta to three-phase)
        ia = i_alpha
        ib = -0.5 * i_alpha + np.sqrt(3)/2 * i_beta
        ic = -0.5 * i_alpha - np.sqrt(3)/2 * i_beta
        
        return ia, ib, ic
    
    def get_dc_bus_current(self):
        """Calculate DC bus current from three-phase currents"""
        ia, ib, ic = self.get_three_phase_currents()
        # Simplified calculation - actual implementation would consider inverter switching
        return abs(ia) + abs(ib) + abs(ic)
    
    def apply_voltage_limits(self, vd, vq):
        """Apply voltage limits based on DC bus voltage"""
        # Calculate voltage magnitude
        v_mag = np.sqrt(vd**2 + vq**2)
        
        # Maximum available voltage (considering modulation index)
        v_max = self.dc_bus_voltage / np.sqrt(3)
        
        if v_mag > v_max:
            # Scale voltages to fit within limit
            scale = v_max / v_mag
            vd *= scale
            vq *= scale
            
        return vd, vq
    
    def apply_current_limits(self, id_ref, iq_ref):
        """Apply current limits"""
        # Calculate current magnitude
        i_mag = np.sqrt(id_ref**2 + iq_ref**2)
        
        if i_mag > self.max_current:
            # Scale currents to fit within limit
            scale = self.max_current / i_mag
            id_ref *= scale
            iq_ref *= scale
            
        return id_ref, iq_ref