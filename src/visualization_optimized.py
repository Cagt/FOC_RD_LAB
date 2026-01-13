import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.animation as animation
from collections import deque
import tkinter as tk
from tkinter import ttk

class OptimizedMotorControlVisualization:
    """
    Optimized visualization system for motor control with performance options
    """
    
    def __init__(self, master=None, buffer_size=500, update_interval=0.1, enable_animation=False, fixed_axis=False):
        """Initialize optimized visualization system"""
        self.master = master
        self.buffer_size = buffer_size
        self.update_interval = update_interval
        self.enable_animation = enable_animation
        self.fixed_axis = fixed_axis
        
        # Fixed axis ranges for better visualization
        self.current_limits = [-25, 25]  # Current range in Amperes
        self.voltage_limits = [-50, 50]  # Voltage range in Volts
        self.speed_limits = [-2000, 2000]  # Speed range in RPM
        self.torque_limits = [-15, 15]  # Torque range in N.m
        
        # Data buffers (smaller size for performance)
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
        
        # Performance tracking
        self.last_update_time = 0
        self.update_counter = 0
        self.fps = 0
        
        # Create matplotlib figure with optimized settings
        plt.style.use('default')  # Use default style for better performance
        self.fig = Figure(figsize=(10, 6), dpi=80)  # Reduced DPI for performance
        self.fig.subplots_adjust(hspace=0.3, wspace=0.3)
        
        # Create subplots
        self.ax_current = self.fig.add_subplot(2, 2, 1)
        self.ax_voltage = self.fig.add_subplot(2, 2, 2)
        self.ax_speed = self.fig.add_subplot(2, 2, 3)
        self.ax_torque = self.fig.add_subplot(2, 2, 4)
        
        # Initialize plots
        self.init_plots()
        
        # Create canvas if master is provided
        if master:
            self.canvas = FigureCanvasTkAgg(self.fig, master=master)
            self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Animation (optional)
        self.animation = None
        self.is_animating = False
        
    def init_plots(self):
        """Initialize all plots with performance optimizations"""
        # Current plot
        self.ax_current.set_title('d-q Axis Currents')
        self.ax_current.set_xlabel('Time (s)')
        self.ax_current.set_ylabel('Current (A)')
        self.ax_current.grid(True, alpha=0.5)  # Reduced grid opacity
        self.id_line, = self.ax_current.plot([], [], 'b-', label='id', linewidth=1)
        self.iq_line, = self.ax_current.plot([], [], 'r-', label='iq', linewidth=1)
        self.id_ref_line, = self.ax_current.plot([], [], 'b--', label='id_ref', linewidth=1, alpha=0.5)
        self.iq_ref_line, = self.ax_current.plot([], [], 'r--', label='iq_ref', linewidth=1, alpha=0.5)
        self.ax_current.legend(loc='upper right', fontsize=8)
        
        # Voltage plot
        self.ax_voltage.set_title('d-q Axis Voltages')
        self.ax_voltage.set_xlabel('Time (s)')
        self.ax_voltage.set_ylabel('Voltage (V)')
        self.ax_voltage.grid(True, alpha=0.5)
        self.vd_line, = self.ax_voltage.plot([], [], 'b-', label='vd', linewidth=1)
        self.vq_line, = self.ax_voltage.plot([], [], 'r-', label='vq', linewidth=1)
        self.ax_voltage.legend(loc='upper right', fontsize=8)
        
        # Speed plot
        self.ax_speed.set_title('Motor Speed')
        self.ax_speed.set_xlabel('Time (s)')
        self.ax_speed.set_ylabel('Speed (RPM)')
        self.ax_speed.grid(True, alpha=0.5)
        self.speed_line, = self.ax_speed.plot([], [], 'g-', label='Speed', linewidth=1)
        self.speed_ref_line, = self.ax_speed.plot([], [], 'g--', label='Speed Ref', linewidth=1, alpha=0.5)
        self.ax_speed.legend(loc='upper right', fontsize=8)
        
        # Torque plot
        self.ax_torque.set_title('Electromagnetic Torque')
        self.ax_torque.set_xlabel('Time (s)')
        self.ax_torque.set_ylabel('Torque (N.m)')
        self.ax_torque.grid(True, alpha=0.5)
        self.torque_line, = self.ax_torque.plot([], [], 'm-', label='Torque', linewidth=1)
        self.ax_torque.legend(loc='upper right', fontsize=8)
        
    def update_data(self, data):
        """Update data buffers with new measurements"""
        current_time = data.get('time', 0)
        
        # Only update if enough time has passed (for performance)
        if current_time - self.last_update_time < self.update_interval:
            return False
        
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
        
        self.last_update_time = current_time
        self.update_counter += 1
        
        return True
        
    def update_plots(self, frame=None):
        """Update all plots with current data"""
        if len(self.time_buffer) == 0:
            return []
        
        # Calculate FPS for performance monitoring
        import time
        current_time = time.time()
        if hasattr(self, 'last_fps_time'):
            dt = current_time - self.last_fps_time
            if dt > 0:
                self.fps = self.update_counter / dt
        self.last_fps_time = current_time
        self.update_counter = 0
        
        time_array = np.array(self.time_buffer)
        
        # Update current plot
        self.id_line.set_data(time_array, np.array(self.id_buffer))
        self.iq_line.set_data(time_array, np.array(self.iq_buffer))
        self.id_ref_line.set_data(time_array, np.array(self.id_ref_buffer))
        self.iq_ref_line.set_data(time_array, np.array(self.iq_ref_buffer))
        
        if self.fixed_axis:
            self.ax_current.set_ylim(self.current_limits)
        else:
            self.ax_current.relim()
            self.ax_current.autoscale_view()
        
        # Update voltage plot
        self.vd_line.set_data(time_array, np.array(self.vd_buffer))
        self.vq_line.set_data(time_array, np.array(self.vq_buffer))
        
        if self.fixed_axis:
            self.ax_voltage.set_ylim(self.voltage_limits)
        else:
            self.ax_voltage.relim()
            self.ax_voltage.autoscale_view()
        
        # Update speed plot
        self.speed_line.set_data(time_array, np.array(self.speed_buffer))
        self.speed_ref_line.set_data(time_array, np.array(self.speed_ref_buffer))
        
        if self.fixed_axis:
            self.ax_speed.set_ylim(self.speed_limits)
        else:
            self.ax_speed.relim()
            self.ax_speed.autoscale_view()
        
        # Update torque plot
        self.torque_line.set_data(time_array, np.array(self.torque_buffer))
        
        if self.fixed_axis:
            self.ax_torque.set_ylim(self.torque_limits)
        else:
            self.ax_torque.relim()
            self.ax_torque.autoscale_view()
        
        # Redraw canvas if available
        if hasattr(self, 'canvas'):
            self.canvas.draw_idle()  # Use draw_idle for better performance
        
        return [self.id_line, self.iq_line, self.id_ref_line, self.iq_ref_line,
                self.vd_line, self.vq_line, self.speed_line, self.speed_ref_line,
                self.torque_line]
    
    def set_fixed_axis(self, fixed=True):
        """Enable or disable fixed axis mode"""
        self.fixed_axis = fixed
    
    def set_axis_limits(self, current=None, voltage=None, speed=None, torque=None):
        """Set custom axis limits"""
        if current is not None:
            self.current_limits = current
        if voltage is not None:
            self.voltage_limits = voltage
        if speed is not None:
            self.speed_limits = speed
        if torque is not None:
            self.torque_limits = torque
    
    def start_animation(self, interval=100):
        """Start real-time animation (if enabled)"""
        if self.enable_animation and not self.is_animating:
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
        self.fig.savefig(filename, dpi=100, bbox_inches='tight')
    
    def get_performance_info(self):
        """Get performance information"""
        return {
            'fps': self.fps,
            'buffer_size': len(self.time_buffer),
            'max_buffer_size': self.buffer_size
        }


class MinimalVisualization:
    """
    Minimal visualization for maximum performance
    """
    
    def __init__(self, master=None, buffer_size=200):
        """Initialize minimal visualization"""
        self.master = master
        self.buffer_size = buffer_size
        
        # Data buffers (minimal size)
        self.time_buffer = deque(maxlen=buffer_size)
        self.id_buffer = deque(maxlen=buffer_size)
        self.iq_buffer = deque(maxlen=buffer_size)
        self.speed_buffer = deque(maxlen=buffer_size)
        
        # Create matplotlib figure with minimal settings
        self.fig = Figure(figsize=(8, 4), dpi=60)
        self.fig.subplots_adjust(hspace=0.4, wspace=0.3)
        
        # Create only essential subplots
        self.ax_current = self.fig.add_subplot(1, 2, 1)
        self.ax_speed = self.fig.add_subplot(1, 2, 2)
        
        # Initialize plots
        self.init_plots()
        
        # Create canvas if master is provided
        if master:
            self.canvas = FigureCanvasTkAgg(self.fig, master=master)
            self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def init_plots(self):
        """Initialize minimal plots"""
        # Current plot
        self.ax_current.set_title('d-q Currents')
        self.ax_current.set_xlabel('Time (s)')
        self.ax_current.set_ylabel('Current (A)')
        self.ax_current.grid(True, alpha=0.3)
        self.id_line, = self.ax_current.plot([], [], 'b-', label='id', linewidth=1)
        self.iq_line, = self.ax_current.plot([], [], 'r-', label='iq', linewidth=1)
        self.ax_current.legend(loc='upper right', fontsize=8)
        
        # Speed plot
        self.ax_speed.set_title('Speed')
        self.ax_speed.set_xlabel('Time (s)')
        self.ax_speed.set_ylabel('Speed (RPM)')
        self.ax_speed.grid(True, alpha=0.3)
        self.speed_line, = self.ax_speed.plot([], [], 'g-', label='Speed', linewidth=1)
        self.ax_speed.legend(loc='upper right', fontsize=8)
    
    def update_data(self, data):
        """Update data buffers"""
        current_time = data.get('time', 0)
        
        self.time_buffer.append(current_time)
        self.id_buffer.append(data.get('id', 0))
        self.iq_buffer.append(data.get('iq', 0))
        self.speed_buffer.append(data.get('speed_rpm', 0))
    
    def update_plots(self):
        """Update minimal plots"""
        if len(self.time_buffer) == 0:
            return
        
        time_array = np.array(self.time_buffer)
        
        # Update current plot
        self.id_line.set_data(time_array, np.array(self.id_buffer))
        self.iq_line.set_data(time_array, np.array(self.iq_buffer))
        self.ax_current.relim()
        self.ax_current.autoscale_view()
        
        # Update speed plot
        self.speed_line.set_data(time_array, np.array(self.speed_buffer))
        self.ax_speed.relim()
        self.ax_speed.autoscale_view()
        
        # Redraw canvas
        if hasattr(self, 'canvas'):
            self.canvas.draw_idle()
    
    def clear_data(self):
        """Clear all data buffers"""
        self.time_buffer.clear()
        self.id_buffer.clear()
        self.iq_buffer.clear()
        self.speed_buffer.clear()
        self.update_plots()