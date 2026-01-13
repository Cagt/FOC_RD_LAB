import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.animation as animation
from collections import deque
import tkinter as tk
from tkinter import ttk

class MotorControlVisualization:
    """
    Real-time visualization system for motor control
    Provides plots for DC bus voltage, id, iq currents, and other parameters
    """
    
    def __init__(self, master=None, buffer_size=1000):
        """Initialize visualization system"""
        self.master = master
        self.buffer_size = buffer_size
        
        # Data buffers
        self.time_buffer = deque(maxlen=buffer_size)
        self.id_buffer = deque(maxlen=buffer_size)
        self.iq_buffer = deque(maxlen=buffer_size)
        self.id_ref_buffer = deque(maxlen=buffer_size)
        self.iq_ref_buffer = deque(maxlen=buffer_size)
        self.speed_buffer = deque(maxlen=buffer_size)
        self.speed_ref_buffer = deque(maxlen=buffer_size)
        self.dc_bus_voltage_buffer = deque(maxlen=buffer_size)
        self.dc_bus_current_buffer = deque(maxlen=buffer_size)
        self.torque_buffer = deque(maxlen=buffer_size)
        self.vd_buffer = deque(maxlen=buffer_size)
        self.vq_buffer = deque(maxlen=buffer_size)
        
        # Create matplotlib figure
        self.fig = Figure(figsize=(14, 8), dpi=100)
        self.fig.subplots_adjust(hspace=0.4, wspace=0.3)
        
        # Create subplots
        self.ax_current = self.fig.add_subplot(2, 3, 1)
        self.ax_voltage = self.fig.add_subplot(2, 3, 2)
        self.ax_speed = self.fig.add_subplot(2, 3, 3)
        self.ax_dc_bus = self.fig.add_subplot(2, 3, 4)
        self.ax_torque = self.fig.add_subplot(2, 3, 5)
        self.ax_phasor = self.fig.add_subplot(2, 3, 6, projection='polar')
        
        # Initialize plots
        self.init_plots()
        
        # Create canvas if master is provided
        if master:
            self.canvas = FigureCanvasTkAgg(self.fig, master=master)
            self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Animation
        self.animation = None
        self.is_animating = False
        
    def init_plots(self):
        """Initialize all plots"""
        # Current plot
        self.ax_current.set_title('d-q Axis Currents')
        self.ax_current.set_xlabel('Time (s)')
        self.ax_current.set_ylabel('Current (A)')
        self.ax_current.grid(True)
        self.id_line, = self.ax_current.plot([], [], 'b-', label='id')
        self.iq_line, = self.ax_current.plot([], [], 'r-', label='iq')
        self.id_ref_line, = self.ax_current.plot([], [], 'b--', label='id_ref')
        self.iq_ref_line, = self.ax_current.plot([], [], 'r--', label='iq_ref')
        self.ax_current.legend(loc='upper right')
        
        # Voltage plot
        self.ax_voltage.set_title('d-q Axis Voltages')
        self.ax_voltage.set_xlabel('Time (s)')
        self.ax_voltage.set_ylabel('Voltage (V)')
        self.ax_voltage.grid(True)
        self.vd_line, = self.ax_voltage.plot([], [], 'b-', label='vd')
        self.vq_line, = self.ax_voltage.plot([], [], 'r-', label='vq')
        self.ax_voltage.legend(loc='upper right')
        
        # Speed plot
        self.ax_speed.set_title('Motor Speed')
        self.ax_speed.set_xlabel('Time (s)')
        self.ax_speed.set_ylabel('Speed (RPM)')
        self.ax_speed.grid(True)
        self.speed_line, = self.ax_speed.plot([], [], 'g-', label='Speed')
        self.speed_ref_line, = self.ax_speed.plot([], [], 'g--', label='Speed Ref')
        self.ax_speed.legend(loc='upper right')
        
        # DC bus plot
        self.ax_dc_bus.set_title('DC Bus')
        self.ax_dc_bus.set_xlabel('Time (s)')
        self.ax_dc_bus.set_ylabel('Voltage (V) / Current (A)')
        self.ax_dc_bus.grid(True)
        self.dc_voltage_line, = self.ax_dc_bus.plot([], [], 'b-', label='Voltage')
        self.dc_current_line, = self.ax_dc_bus.plot([], [], 'r-', label='Current')
        self.ax_dc_bus.legend(loc='upper right')
        
        # Torque plot
        self.ax_torque.set_title('Electromagnetic Torque')
        self.ax_torque.set_xlabel('Time (s)')
        self.ax_torque.set_ylabel('Torque (N.m)')
        self.ax_torque.grid(True)
        self.torque_line, = self.ax_torque.plot([], [], 'm-', label='Torque')
        self.ax_torque.legend(loc='upper right')
        
        # Phasor diagram
        self.ax_phasor.set_title('Current Phasor Diagram')
        self.ax_phasor.grid(True)
        self.current_phasor, = self.ax_phasor.plot([], [], 'ro', markersize=8)
        self.current_trajectory, = self.ax_phasor.plot([], [], 'b-', alpha=0.3)
        
    def update_data(self, data):
        """Update data buffers with new measurements"""
        current_time = data.get('time', 0)
        
        # Update buffers
        self.time_buffer.append(current_time)
        self.id_buffer.append(data.get('id', 0))
        self.iq_buffer.append(data.get('iq', 0))
        self.id_ref_buffer.append(data.get('id_ref', 0))
        self.iq_ref_buffer.append(data.get('iq_ref', 0))
        self.speed_buffer.append(data.get('speed_rpm', 0))
        self.speed_ref_buffer.append(data.get('speed_ref_rpm', 0))
        self.dc_bus_voltage_buffer.append(data.get('dc_bus_voltage', 0))
        self.dc_bus_current_buffer.append(data.get('dc_bus_current', 0))
        self.torque_buffer.append(data.get('torque', 0))
        self.vd_buffer.append(data.get('vd', 0))
        self.vq_buffer.append(data.get('vq', 0))
        
    def update_plots(self, frame=None):
        """Update all plots with current data"""
        if len(self.time_buffer) == 0:
            return []
        
        time_array = np.array(self.time_buffer)
        
        # Update current plot
        self.id_line.set_data(time_array, np.array(self.id_buffer))
        self.iq_line.set_data(time_array, np.array(self.iq_buffer))
        self.id_ref_line.set_data(time_array, np.array(self.id_ref_buffer))
        self.iq_ref_line.set_data(time_array, np.array(self.iq_ref_buffer))
        self.ax_current.relim()
        self.ax_current.autoscale_view()
        
        # Update voltage plot
        self.vd_line.set_data(time_array, np.array(self.vd_buffer))
        self.vq_line.set_data(time_array, np.array(self.vq_buffer))
        self.ax_voltage.relim()
        self.ax_voltage.autoscale_view()
        
        # Update speed plot
        self.speed_line.set_data(time_array, np.array(self.speed_buffer))
        self.speed_ref_line.set_data(time_array, np.array(self.speed_ref_buffer))
        self.ax_speed.relim()
        self.ax_speed.autoscale_view()
        
        # Update DC bus plot
        self.dc_voltage_line.set_data(time_array, np.array(self.dc_bus_voltage_buffer))
        self.dc_current_line.set_data(time_array, np.array(self.dc_bus_current_buffer))
        self.ax_dc_bus.relim()
        self.ax_dc_bus.autoscale_view()
        
        # Update torque plot
        self.torque_line.set_data(time_array, np.array(self.torque_buffer))
        self.ax_torque.relim()
        self.ax_torque.autoscale_view()
        
        # Update phasor diagram
        if len(self.id_buffer) > 0:
            # Current phasor
            current_magnitude = np.sqrt(self.id_buffer[-1]**2 + self.iq_buffer[-1]**2)
            current_angle = np.arctan2(self.iq_buffer[-1], self.id_buffer[-1])
            self.current_phasor.set_data([current_angle], [current_magnitude])
            
            # Current trajectory (last N points)
            trajectory_length = min(100, len(self.id_buffer))
            if trajectory_length > 1:
                id_traj = np.array(list(self.id_buffer)[-trajectory_length:])
                iq_traj = np.array(list(self.iq_buffer)[-trajectory_length:])
                traj_magnitude = np.sqrt(id_traj**2 + iq_traj**2)
                traj_angle = np.arctan2(iq_traj, id_traj)
                self.current_trajectory.set_data(traj_angle, traj_magnitude)
            
            self.ax_phasor.relim()
            self.ax_phasor.autoscale_view()
        
        # Redraw canvas if available
        if hasattr(self, 'canvas'):
            self.canvas.draw()
        
        return [self.id_line, self.iq_line, self.id_ref_line, self.iq_ref_line,
                self.vd_line, self.vq_line, self.speed_line, self.speed_ref_line,
                self.dc_voltage_line, self.dc_current_line, self.torque_line,
                self.current_phasor, self.current_trajectory]
    
    def start_animation(self, interval=50):
        """Start real-time animation"""
        if not self.is_animating:
            self.animation = animation.FuncAnimation(
                self.fig, self.update_plots, interval=interval, blit=False
            )
            self.is_animating = True
    
    def stop_animation(self):
        """Stop real-time animation"""
        if self.animation:
            self.animation.event_source.stop()
            self.is_animating = False
    
    def clear_data(self):
        """Clear all data buffers"""
        self.time_buffer.clear()
        self.id_buffer.clear()
        self.iq_buffer.clear()
        self.id_ref_buffer.clear()
        self.iq_ref_buffer.clear()
        self.speed_buffer.clear()
        self.speed_ref_buffer.clear()
        self.dc_bus_voltage_buffer.clear()
        self.dc_bus_current_buffer.clear()
        self.torque_buffer.clear()
        self.vd_buffer.clear()
        self.vq_buffer.clear()
        
        # Clear plots
        self.update_plots()
    
    def save_figure(self, filename):
        """Save current figure to file"""
        self.fig.savefig(filename, dpi=150, bbox_inches='tight')


class ParameterVisualization:
    """
    Visualization system for parameter identification and adaptive control
    """
    
    def __init__(self, master=None):
        """Initialize parameter visualization"""
        self.master = master
        
        # Data buffers
        self.param_buffers = {
            'Rs': deque(maxlen=1000),
            'Ld': deque(maxlen=1000),
            'Lq': deque(maxlen=1000),
            'flux': deque(maxlen=1000),
            'J': deque(maxlen=1000),
            'B': deque(maxlen=1000)
        }
        self.time_buffer = deque(maxlen=1000)
        
        # Create matplotlib figure
        self.fig = Figure(figsize=(12, 8), dpi=100)
        self.fig.subplots_adjust(hspace=0.4, wspace=0.3)
        
        # Create subplots
        self.ax_resistance = self.fig.add_subplot(2, 3, 1)
        self.ax_inductance = self.fig.add_subplot(2, 3, 2)
        self.ax_flux = self.fig.add_subplot(2, 3, 3)
        self.ax_mechanical = self.fig.add_subplot(2, 3, 4)
        self.ax_gains = self.fig.add_subplot(2, 3, 5)
        self.ax_performance = self.fig.add_subplot(2, 3, 6)
        
        # Initialize plots
        self.init_plots()
        
        # Create canvas if master is provided
        if master:
            self.canvas = FigureCanvasTkAgg(self.fig, master=master)
            self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def init_plots(self):
        """Initialize parameter plots"""
        # Resistance plot
        self.ax_resistance.set_title('Stator Resistance')
        self.ax_resistance.set_xlabel('Time (s)')
        self.ax_resistance.set_ylabel('Rs (Ω)')
        self.ax_resistance.grid(True)
        self.rs_line, = self.ax_resistance.plot([], [], 'b-')
        
        # Inductance plot
        self.ax_inductance.set_title('d-q Axis Inductances')
        self.ax_inductance.set_xlabel('Time (s)')
        self.ax_inductance.set_ylabel('Inductance (H)')
        self.ax_inductance.grid(True)
        self.ld_line, = self.ax_inductance.plot([], [], 'b-', label='Ld')
        self.lq_line, = self.ax_inductance.plot([], [], 'r-', label='Lq')
        self.ax_inductance.legend(loc='upper right')
        
        # Flux linkage plot
        self.ax_flux.set_title('Flux Linkage')
        self.ax_flux.set_xlabel('Time (s)')
        self.ax_flux.set_ylabel('Flux (Wb)')
        self.ax_flux.grid(True)
        self.flux_line, = self.ax_flux.plot([], [], 'g-')
        
        # Mechanical parameters plot
        self.ax_mechanical.set_title('Mechanical Parameters')
        self.ax_mechanical.set_xlabel('Time (s)')
        self.ax_mechanical.set_ylabel('Parameter Value')
        self.ax_mechanical.grid(True)
        self.j_line, = self.ax_mechanical.plot([], [], 'b-', label='J (kg.m²)')
        self.b_line, = self.ax_mechanical.plot([], [], 'r-', label='B (N.m.s)')
        self.ax_mechanical.legend(loc='upper right')
        
        # Adaptive gains plot
        self.ax_gains.set_title('Adaptive Gains')
        self.ax_gains.set_xlabel('Time (s)')
        self.ax_gains.set_ylabel('Gain Value')
        self.ax_gains.grid(True)
        
        # Performance metrics plot
        self.ax_performance.set_title('Performance Metrics')
        self.ax_performance.set_xlabel('Time (s)')
        self.ax_performance.set_ylabel('Metric Value')
        self.ax_performance.grid(True)
    
    def update_parameter_data(self, time, parameters):
        """Update parameter data"""
        self.time_buffer.append(time)
        
        for param, value in parameters.items():
            if param in self.param_buffers:
                self.param_buffers[param].append(value)
    
    def update_plots(self):
        """Update all parameter plots"""
        if len(self.time_buffer) == 0:
            return
        
        time_array = np.array(self.time_buffer)
        
        # Update resistance plot
        if len(self.param_buffers['Rs']) > 0:
            self.rs_line.set_data(time_array, np.array(self.param_buffers['Rs']))
            self.ax_resistance.relim()
            self.ax_resistance.autoscale_view()
        
        # Update inductance plot
        if len(self.param_buffers['Ld']) > 0 and len(self.param_buffers['Lq']) > 0:
            self.ld_line.set_data(time_array, np.array(self.param_buffers['Ld']))
            self.lq_line.set_data(time_array, np.array(self.param_buffers['Lq']))
            self.ax_inductance.relim()
            self.ax_inductance.autoscale_view()
        
        # Update flux linkage plot
        if len(self.param_buffers['flux']) > 0:
            self.flux_line.set_data(time_array, np.array(self.param_buffers['flux']))
            self.ax_flux.relim()
            self.ax_flux.autoscale_view()
        
        # Update mechanical parameters plot
        if len(self.param_buffers['J']) > 0 and len(self.param_buffers['B']) > 0:
            self.j_line.set_data(time_array, np.array(self.param_buffers['J']))
            self.b_line.set_data(time_array, np.array(self.param_buffers['B']))
            self.ax_mechanical.relim()
            self.ax_mechanical.autoscale_view()
        
        # Redraw canvas if available
        if hasattr(self, 'canvas'):
            self.canvas.draw()
    
    def clear_data(self):
        """Clear all data buffers"""
        self.time_buffer.clear()
        for buffer in self.param_buffers.values():
            buffer.clear()
        
        # Clear plots
        self.update_plots()