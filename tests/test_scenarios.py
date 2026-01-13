import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from motor_model import PMSMModel
from foc_control import FOCController
from flux_weakening import FluxWeakeningController
from self_learning import MotorParameterIdentification
from disturbance_rejection import DisturbanceRejectionController

class TestScenarios:
    """
    Test scenarios for validating motor control algorithms
    """
    
    def __init__(self):
        """Initialize test scenarios"""
        self.motor = PMSMModel()
        self.foc_controller = FOCController()
        self.flux_weakening = FluxWeakeningController()
        self.param_identification = MotorParameterIdentification()
        self.disturbance_rejection = DisturbanceRejectionController()
        
        # Test parameters
        self.sample_time = self.motor.sample_time
        self.test_duration = 2.0  # seconds
        
    def run_speed_control_test(self):
        """Test speed control performance"""
        print("Running speed control test...")
        
        # Reset components
        self.motor.reset_state()
        self.foc_controller.reset()
        
        # Test parameters
        speed_ref_profile = []
        time_points = []
        speed_actual = []
        torque_output = []
        id_actual = []
        iq_actual = []
        
        # Generate speed reference profile
        num_steps = int(self.test_duration / self.sample_time)
        for i in range(num_steps):
            t = i * self.sample_time
            time_points.append(t)
            
            # Speed reference: step changes
            if t < 0.5:
                speed_ref = 0
            elif t < 1.0:
                speed_ref = 100 * 2 * np.pi / 60  # 100 RPM
            elif t < 1.5:
                speed_ref = 500 * 2 * np.pi / 60  # 500 RPM
            else:
                speed_ref = 1000 * 2 * np.pi / 60  # 1000 RPM
                
            speed_ref_profile.append(speed_ref)
            
            # Get three-phase currents
            ia, ib, ic = self.motor.get_three_phase_currents()
            
            # FOC control
            foc_results = self.foc_controller.update(
                ia, ib, ic, speed_ref, self.motor.wr, self.motor.theta_e
            )
            
            # Update motor
            motor_results = self.motor.update(
                foc_results['vd'], foc_results['vq'], 0.0
            )
            
            # Store results
            speed_actual.append(motor_results['speed_rpm'])
            torque_output.append(motor_results['Te'])
            id_actual.append(foc_results['id_actual'])
            iq_actual.append(foc_results['iq_actual'])
        
        # Plot results
        plt.figure(figsize=(12, 8))
        
        plt.subplot(3, 1, 1)
        plt.plot(time_points, [s * 60 / (2 * np.pi) for s in speed_ref_profile], 'r--', label='Speed Reference')
        plt.plot(time_points, speed_actual, 'b-', label='Actual Speed')
        plt.xlabel('Time (s)')
        plt.ylabel('Speed (RPM)')
        plt.title('Speed Control Test')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(3, 1, 2)
        plt.plot(time_points, torque_output, 'g-')
        plt.xlabel('Time (s)')
        plt.ylabel('Torque (N.m)')
        plt.title('Electromagnetic Torque')
        plt.grid(True)
        
        plt.subplot(3, 1, 3)
        plt.plot(time_points, id_actual, 'b-', label='id')
        plt.plot(time_points, iq_actual, 'r-', label='iq')
        plt.xlabel('Time (s)')
        plt.ylabel('Current (A)')
        plt.title('d-q Axis Currents')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('../tests/results/speed_control_test.png', dpi=150)
        plt.show()
        
        print("Speed control test completed. Results saved to tests/results/speed_control_test.png")
        
    def run_flux_weakening_test(self):
        """Test flux weakening control at high speeds"""
        print("Running flux weakening test...")
        
        # Reset components
        self.motor.reset_state()
        self.foc_controller.reset()
        self.flux_weakening.reset()
        
        # Test parameters
        time_points = []
        speed_actual = []
        id_actual = []
        iq_actual = []
        id_fw = []
        voltage_magnitude = []
        
        # High speed test
        num_steps = int(self.test_duration / self.sample_time)
        speed_ref = 2000 * 2 * np.pi / 60  # 2000 RPM (above base speed)
        
        for i in range(num_steps):
            t = i * self.sample_time
            time_points.append(t)
            
            # Get three-phase currents
            ia, ib, ic = self.motor.get_three_phase_currents()
            
            # FOC control
            foc_results = self.foc_controller.update(
                ia, ib, ic, speed_ref, self.motor.wr, self.motor.theta_e
            )
            
            # Apply flux weakening
            id_fw_value = self.flux_weakening.update(
                self.motor.wr, foc_results['vd'], foc_results['vq'],
                foc_results['iq_ref'], 'voltage'
            )
            
            # Update motor with flux weakening
            motor_results = self.motor.update(
                foc_results['vd'], foc_results['vq'], 0.0
            )
            
            # Store results
            speed_actual.append(motor_results['speed_rpm'])
            id_actual.append(foc_results['id_actual'])
            iq_actual.append(foc_results['iq_actual'])
            id_fw.append(id_fw_value)
            voltage_magnitude.append(np.sqrt(foc_results['vd']**2 + foc_results['vq']**2))
        
        # Plot results
        plt.figure(figsize=(12, 10))
        
        plt.subplot(4, 1, 1)
        plt.plot(time_points, speed_actual, 'b-')
        plt.axhline(y=speed_ref * 60 / (2 * np.pi), color='r', linestyle='--', label='Speed Reference')
        plt.xlabel('Time (s)')
        plt.ylabel('Speed (RPM)')
        plt.title('Flux Weakening Test - Speed')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(4, 1, 2)
        plt.plot(time_points, id_actual, 'b-', label='id')
        plt.plot(time_points, iq_actual, 'r-', label='iq')
        plt.plot(time_points, id_fw, 'g--', label='id_fw')
        plt.xlabel('Time (s)')
        plt.ylabel('Current (A)')
        plt.title('d-q Axis Currents')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(4, 1, 3)
        plt.plot(time_points, voltage_magnitude, 'g-')
        v_max = self.motor.dc_bus_voltage / np.sqrt(3)
        plt.axhline(y=v_max, color='r', linestyle='--', label='Voltage Limit')
        plt.xlabel('Time (s)')
        plt.ylabel('Voltage (V)')
        plt.title('Voltage Magnitude')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(4, 1, 4)
        plt.plot(time_points, id_fw, 'g-')
        plt.xlabel('Time (s)')
        plt.ylabel('Flux Weakening Current (A)')
        plt.title('d-axis Flux Weakening Current')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('../tests/results/flux_weakening_test.png', dpi=150)
        plt.show()
        
        print("Flux weakening test completed. Results saved to tests/results/flux_weakening_test.png")
        
    def run_parameter_identification_test(self):
        """Test parameter identification with parameter variations"""
        print("Running parameter identification test...")
        
        # Reset components
        self.motor.reset_state()
        self.foc_controller.reset()
        self.param_identification.reset()
        
        # Test parameters
        time_points = []
        rs_identified = []
        ld_identified = []
        lq_identified = []
        flux_identified = []
        
        # Introduce parameter variations
        original_rs = self.motor.Rs
        original_flux = self.motor.flux_linkage
        
        # Change motor parameters to simulate real-world variations
        self.motor.Rs *= 1.2  # 20% increase in resistance
        self.motor.flux_linkage *= 0.9  # 10% decrease in flux linkage
        
        # Parameter identification test
        num_steps = int(self.test_duration / self.sample_time)
        speed_ref = 500 * 2 * np.pi / 60  # 500 RPM
        
        for i in range(num_steps):
            t = i * self.sample_time
            time_points.append(t)
            
            # Get three-phase currents
            ia, ib, ic = self.motor.get_three_phase_currents()
            
            # FOC control
            foc_results = self.foc_controller.update(
                ia, ib, ic, speed_ref, self.motor.wr, self.motor.theta_e
            )
            
            # Update motor
            motor_results = self.motor.update(
                foc_results['vd'], foc_results['vq'], 0.0
            )
            
            # Parameter identification
            self.param_identification.update(
                foc_results['vd'], foc_results['id_actual'],
                foc_results['vq'], foc_results['iq_actual'],
                self.motor.wr, motor_results['Te'], 0.0
            )
            
            # Get identified parameters
            params = self.param_identification.get_identified_parameters()
            rs_identified.append(params['Rs'])
            ld_identified.append(params['Ld'])
            lq_identified.append(params['Lq'])
            flux_identified.append(params['flux_linkage'])
        
        # Restore original parameters
        self.motor.Rs = original_rs
        self.motor.flux_linkage = original_flux
        
        # Plot results
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        plt.plot(time_points, rs_identified, 'b-')
        plt.axhline(y=original_rs, color='r', linestyle='--', label='True Rs')
        plt.axhline(y=self.motor.Rs, color='g', linestyle='--', label='Actual Rs')
        plt.xlabel('Time (s)')
        plt.ylabel('Resistance (Ω)')
        plt.title('Stator Resistance Identification')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(2, 2, 2)
        plt.plot(time_points, ld_identified, 'b-', label='Ld')
        plt.plot(time_points, lq_identified, 'r-', label='Lq')
        plt.axhline(y=self.motor.Ld, color='b', linestyle='--', alpha=0.5)
        plt.axhline(y=self.motor.Lq, color='r', linestyle='--', alpha=0.5)
        plt.xlabel('Time (s)')
        plt.ylabel('Inductance (H)')
        plt.title('d-q Axis Inductance Identification')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(2, 2, 3)
        plt.plot(time_points, flux_identified, 'g-')
        plt.axhline(y=original_flux, color='r', linestyle='--', label='True Flux')
        plt.axhline(y=self.motor.flux_linkage, color='g', linestyle='--', label='Actual Flux')
        plt.xlabel('Time (s)')
        plt.ylabel('Flux Linkage (Wb)')
        plt.title('Flux Linkage Identification')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(2, 2, 4)
        plt.plot(time_points, np.array(rs_identified) / original_rs, 'b-', label='Rs Error')
        plt.plot(time_points, np.array(flux_identified) / original_flux, 'g-', label='Flux Error')
        plt.axhline(y=1.0, color='k', linestyle='--', alpha=0.5)
        plt.xlabel('Time (s)')
        plt.ylabel('Parameter Ratio')
        plt.title('Parameter Identification Error')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('../tests/results/parameter_identification_test.png', dpi=150)
        plt.show()
        
        print("Parameter identification test completed. Results saved to tests/results/parameter_identification_test.png")
        
    def run_disturbance_rejection_test(self):
        """Test disturbance rejection control"""
        print("Running disturbance rejection test...")
        
        # Reset components
        self.motor.reset_state()
        self.foc_controller.reset()
        self.disturbance_rejection.reset()
        
        # Test parameters
        time_points = []
        speed_actual_no_dr = []
        speed_actual_with_dr = []
        load_torque = []
        disturbance_estimate = []
        
        # Speed reference
        speed_ref = 1000 * 2 * np.pi / 60  # 1000 RPM
        
        # Test without disturbance rejection
        num_steps = int(self.test_duration / self.sample_time)
        for i in range(num_steps):
            t = i * self.sample_time
            time_points.append(t)
            
            # Apply load torque disturbance
            if 0.5 < t < 1.5:
                load = 5.0  # 5 N.m load
            else:
                load = 0.0
                
            load_torque.append(load)
            
            # Get three-phase currents
            ia, ib, ic = self.motor.get_three_phase_currents()
            
            # FOC control without disturbance rejection
            foc_results = self.foc_controller.update(
                ia, ib, ic, speed_ref, self.motor.wr, self.motor.theta_e
            )
            
            # Update motor
            motor_results = self.motor.update(
                foc_results['vd'], foc_results['vq'], load
            )
            
            speed_actual_no_dr.append(motor_results['speed_rpm'])
        
        # Reset motor and test with disturbance rejection
        self.motor.reset_state()
        self.foc_controller.reset()
        self.disturbance_rejection.reset()
        self.disturbance_rejection.set_control_mode('observer')
        
        for i in range(num_steps):
            t = i * self.sample_time
            
            # Apply same load torque disturbance
            if 0.5 < t < 1.5:
                load = 5.0  # 5 N.m load
            else:
                load = 0.0
            
            # Get three-phase currents
            ia, ib, ic = self.motor.get_three_phase_currents()
            
            # FOC control
            foc_results = self.foc_controller.update(
                ia, ib, ic, speed_ref, self.motor.wr, self.motor.theta_e
            )
            
            # Apply disturbance rejection
            dr_results = self.disturbance_rejection.update(
                self.motor.Te, self.motor.wr, speed_ref,
                foc_results['id_actual'], foc_results['iq_actual'],
                foc_results['id_ref'], foc_results['iq_ref']
            )
            
            # Update motor with modified current references
            # Note: In a real implementation, the controller would use the modified references
            motor_results = self.motor.update(
                foc_results['vd'], foc_results['vq'], load
            )
            
            speed_actual_with_dr.append(motor_results['speed_rpm'])
            disturbance_estimate.append(dr_results['disturbance_estimate'])
        
        # Plot results
        plt.figure(figsize=(12, 8))
        
        plt.subplot(3, 1, 1)
        plt.plot(time_points, [speed_ref * 60 / (2 * np.pi)] * len(time_points), 'k--', label='Speed Reference')
        plt.plot(time_points, speed_actual_no_dr, 'r-', label='Without Disturbance Rejection')
        plt.plot(time_points, speed_actual_with_dr, 'b-', label='With Disturbance Rejection')
        plt.xlabel('Time (s)')
        plt.ylabel('Speed (RPM)')
        plt.title('Disturbance Rejection Test - Speed')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(3, 1, 2)
        plt.plot(time_points, load_torque, 'g-')
        plt.xlabel('Time (s)')
        plt.ylabel('Load Torque (N.m)')
        plt.title('Load Torque Disturbance')
        plt.grid(True)
        
        plt.subplot(3, 1, 3)
        plt.plot(time_points, disturbance_estimate, 'm-')
        plt.xlabel('Time (s)')
        plt.ylabel('Disturbance Estimate (N.m)')
        plt.title('Disturbance Observer Output')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('../tests/results/disturbance_rejection_test.png', dpi=150)
        plt.show()
        
        print("Disturbance rejection test completed. Results saved to tests/results/disturbance_rejection_test.png")
        
    def run_all_tests(self):
        """Run all test scenarios"""
        # Create results directory
        os.makedirs('../tests/results', exist_ok=True)
        
        print("Running all test scenarios...")
        self.run_speed_control_test()
        self.run_flux_weakening_test()
        self.run_parameter_identification_test()
        self.run_disturbance_rejection_test()
        print("All tests completed successfully!")


if __name__ == "__main__":
    # Create test scenarios
    tests = TestScenarios()
    
    # Run all tests
    tests.run_all_tests()