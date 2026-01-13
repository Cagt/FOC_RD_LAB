import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import time
import threading
import json
import os

# Import our modules
from motor_model import PMSMModel
from foc_control import FOCController
from flux_weakening import FluxWeakeningController
from visualization_optimized import OptimizedMotorControlVisualization

class LiteMotorControlSimulation:
    """
    Lightweight version of motor control simulation for low-performance computers
    """
    
    def __init__(self, config_file="../config/motor_params_lite.json"):
        """Initialize lightweight simulation"""
        # Load configuration
        with open(os.path.join(os.path.dirname(__file__), config_file), 'r') as f:
            self.config = json.load(f)
        
        # Initialize components
        self.motor = PMSMModel(config_file)
        self.foc_controller = FOCController(config_file)
        self.flux_weakening = FluxWeakeningController(config_file)
        
        # Simulation parameters
        self.simulation_time = 0.0
        self.sample_time = self.config['sample_time']
        self.is_running = False
        self.simulation_speed = 1.0
        
        # Visualization parameters
        self.visualization_update_interval = self.config['visualization_update_interval']
        self.data_buffer_size = self.config['data_buffer_size']
        self.enable_animation = self.config['enable_animation']
        self.fixed_axis = False  # Default to auto-scaling
        
        # Control references
        self.speed_ref = 0.0  # rad/s
        self.load_torque = 0.0  # N.m
        
        # Control modes
        self.enable_flux_weakening = False
        self.flux_weakening_method = 'voltage'
        
        # Data storage (limited size for performance)
        self.data_history = {
            'time': [],
            'id': [],
            'iq': [],
            'speed': [],
            'torque': [],
            'vd': [],
            'vq': []
        }
        
        # Last visualization update time
        self.last_viz_update = 0.0
        
        # Create GUI
        self.create_gui()
        
    def create_gui(self):
        """Create simplified GUI"""
        self.root = tk.Tk()
        self.root.title("FOC BLDC/PMSM 仿真实验室 (轻量版)")
        self.root.geometry("1000x700")
        
        # Create main frames
        self.create_menu()
        self.create_control_panel()
        self.create_visualization_panel()
        self.create_status_panel()
        
    def create_menu(self):
        """Create menu bar"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="文件", menu=file_menu)
        file_menu.add_command(label="加载参数", command=self.load_parameters)
        file_menu.add_command(label="保存参数", command=self.save_parameters)
        file_menu.add_separator()
        file_menu.add_command(label="导出数据", command=self.export_data)
        file_menu.add_separator()
        file_menu.add_command(label="退出", command=self.root.quit)
        
        # Simulation menu
        sim_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="仿真", menu=sim_menu)
        sim_menu.add_command(label="开始", command=self.start_simulation)
        sim_menu.add_command(label="暂停", command=self.pause_simulation)
        sim_menu.add_command(label="重置", command=self.reset_simulation)
        
        # View menu
        view_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="视图", menu=view_menu)
        view_menu.add_command(label="刷新图表", command=self.update_plots)
        
    def create_control_panel(self):
        """Create simplified control panel"""
        control_frame = ttk.LabelFrame(self.root, text="控制面板", padding=10)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        
        # Speed control
        speed_frame = ttk.LabelFrame(control_frame, text="速度控制", padding=5)
        speed_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(speed_frame, text="速度参考 (RPM):").grid(row=0, column=0, sticky=tk.W)
        self.speed_ref_var = tk.DoubleVar(value=0)
        speed_scale = ttk.Scale(speed_frame, from_=-2000, to=2000, variable=self.speed_ref_var, 
                                orient=tk.HORIZONTAL, length=150)
        speed_scale.grid(row=0, column=1, padx=5)
        self.speed_label = ttk.Label(speed_frame, text="0")
        self.speed_label.grid(row=0, column=2)
        speed_scale.config(command=self.update_speed_label)
        
        # Load torque control
        load_frame = ttk.LabelFrame(control_frame, text="负载转矩", padding=5)
        load_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(load_frame, text="负载转矩 (N.m):").grid(row=0, column=0, sticky=tk.W)
        self.load_torque_var = tk.DoubleVar(value=0)
        load_scale = ttk.Scale(load_frame, from_=0, to=5, variable=self.load_torque_var,
                              orient=tk.HORIZONTAL, length=150)
        load_scale.grid(row=0, column=1, padx=5)
        self.load_label = ttk.Label(load_frame, text="0.0")
        self.load_label.grid(row=0, column=2)
        load_scale.config(command=self.update_load_label)
        
        # Control options
        options_frame = ttk.LabelFrame(control_frame, text="控制选项", padding=5)
        options_frame.pack(fill=tk.X, pady=5)
        
        self.flux_weakening_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(options_frame, text="弱磁控制", 
                       variable=self.flux_weakening_var).pack(anchor=tk.W)
        
        # Flux weakening method
        fw_method_frame = ttk.LabelFrame(control_frame, text="弱磁控制方法", padding=5)
        fw_method_frame.pack(fill=tk.X, pady=5)
        
        self.fw_method_var = tk.StringVar(value="voltage")
        ttk.Radiobutton(fw_method_frame, text="电压法", variable=self.fw_method_var,
                       value="voltage").pack(anchor=tk.W)
        ttk.Radiobutton(fw_method_frame, text="速度法", variable=self.fw_method_var,
                       value="speed").pack(anchor=tk.W)
        
        # Simulation control
        sim_frame = ttk.LabelFrame(control_frame, text="仿真控制", padding=5)
        sim_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(sim_frame, text="开始", command=self.start_simulation).pack(side=tk.LEFT, padx=2)
        ttk.Button(sim_frame, text="暂停", command=self.pause_simulation).pack(side=tk.LEFT, padx=2)
        ttk.Button(sim_frame, text="重置", command=self.reset_simulation).pack(side=tk.LEFT, padx=2)
        
        # Simulation speed
        ttk.Label(sim_frame, text="仿真速度:").pack(side=tk.LEFT, padx=5)
        self.sim_speed_var = tk.DoubleVar(value=1.0)
        sim_speed_scale = ttk.Scale(sim_frame, from_=0.1, to=2.0, variable=self.sim_speed_var,
                                   orient=tk.HORIZONTAL, length=80)
        sim_speed_scale.pack(side=tk.LEFT)
        
        # Performance settings
        perf_frame = ttk.LabelFrame(control_frame, text="性能设置", padding=5)
        perf_frame.pack(fill=tk.X, pady=5)
        
        self.viz_interval_var = tk.DoubleVar(value=self.visualization_update_interval)
        ttk.Label(perf_frame, text="更新间隔(s):").grid(row=0, column=0, sticky=tk.W)
        viz_interval_scale = ttk.Scale(perf_frame, from_=0.05, to=0.5, variable=self.viz_interval_var,
                                      orient=tk.HORIZONTAL, length=100)
        viz_interval_scale.grid(row=0, column=1, padx=5)
        
        self.buffer_size_var = tk.IntVar(value=self.data_buffer_size)
        ttk.Label(perf_frame, text="缓冲区大小:").grid(row=1, column=0, sticky=tk.W)
        buffer_size_scale = ttk.Scale(perf_frame, from_=100, to=1000, variable=self.buffer_size_var,
                                    orient=tk.HORIZONTAL, length=100)
        buffer_size_scale.grid(row=1, column=1, padx=5)
        
        # Fixed axis option
        self.fixed_axis_var = tk.BooleanVar(value=self.fixed_axis)
        ttk.Checkbutton(perf_frame, text="固定Y轴范围",
                       variable=self.fixed_axis_var,
                       command=self.toggle_fixed_axis).grid(row=2, column=0, columnspan=2, sticky=tk.W, pady=5)
        
        # Motor parameters display
        params_frame = ttk.LabelFrame(control_frame, text="电机参数", padding=5)
        params_frame.pack(fill=tk.X, pady=5)
        
        self.params_text = tk.Text(params_frame, height=8, width=25)
        self.params_text.pack()
        self.update_params_display()
        
    def create_visualization_panel(self):
        """Create simplified visualization panel"""
        viz_frame = ttk.LabelFrame(self.root, text="可视化", padding=5)
        viz_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create matplotlib figure with fewer subplots
        self.fig, self.axes = plt.subplots(2, 2, figsize=(10, 6))
        self.fig.subplots_adjust(hspace=0.3, wspace=0.3)
        
        # Initialize plots
        self.ax_current = self.axes[0, 0]
        self.ax_speed = self.axes[0, 1]
        self.ax_voltage = self.axes[1, 0]
        self.ax_torque = self.axes[1, 1]
        
        # Configure plots
        self.ax_current.set_title('d-q Axis Currents')
        self.ax_current.set_xlabel('Time (s)')
        self.ax_current.set_ylabel('Current (A)')
        self.ax_current.grid(True)
        
        self.ax_speed.set_title('Motor Speed')
        self.ax_speed.set_xlabel('Time (s)')
        self.ax_speed.set_ylabel('Speed (RPM)')
        self.ax_speed.grid(True)
        
        self.ax_voltage.set_title('d-q Axis Voltages')
        self.ax_voltage.set_xlabel('Time (s)')
        self.ax_voltage.set_ylabel('Voltage (V)')
        self.ax_voltage.grid(True)
        
        self.ax_torque.set_title('Electromagnetic Torque')
        self.ax_torque.set_xlabel('Time (s)')
        self.ax_torque.set_ylabel('Torque (N.m)')
        self.ax_torque.grid(True)
        
        # Embed in tkinter
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        self.canvas = FigureCanvasTkAgg(self.fig, master=viz_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
    def create_status_panel(self):
        """Create status panel"""
        status_frame = ttk.LabelFrame(self.root, text="状态信息", padding=5)
        status_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=5)
        
        self.status_label = ttk.Label(status_frame, text="就绪")
        self.status_label.pack(side=tk.LEFT, padx=5)
        
        self.time_label = ttk.Label(status_frame, text="时间: 0.00 s")
        self.time_label.pack(side=tk.LEFT, padx=20)
        
        self.performance_label = ttk.Label(status_frame, text="性能: 正常")
        self.performance_label.pack(side=tk.LEFT, padx=20)
        
    def update_speed_label(self, value):
        """Update speed label"""
        self.speed_label.config(text=f"{float(value):.0f}")
        self.speed_ref = float(value) * 2 * np.pi / 60  # Convert RPM to rad/s
        
    def update_load_label(self, value):
        """Update load torque label"""
        self.load_label.config(text=f"{float(value):.1f}")
        self.load_torque = float(value)
        
    def update_params_display(self):
        """Update motor parameters display"""
        params = f"极对数: {self.motor.poles}\n"
        params += f"定子电阻: {self.motor.Rs:.3f} Ω\n"
        params += f"d轴电感: {self.motor.Ld:.4f} H\n"
        params += f"q轴电感: {self.motor.Lq:.4f} H\n"
        params += f"磁链: {self.motor.flux_linkage:.3f} Wb\n"
        params += f"母线电压: {self.motor.dc_bus_voltage:.1f} V\n"
        params += f"采样时间: {self.motor.sample_time:.4f} s"
        
        self.params_text.delete(1.0, tk.END)
        self.params_text.insert(1.0, params)
        
    def simulation_step(self):
        """Execute one simulation step"""
        # Get current settings
        self.enable_flux_weakening = self.flux_weakening_var.get()
        self.flux_weakening_method = self.fw_method_var.get()
        self.visualization_update_interval = self.viz_interval_var.get()
        self.data_buffer_size = int(self.buffer_size_var.get())
        
        # Get three-phase currents from motor
        ia, ib, ic = self.motor.get_three_phase_currents()
        
        # FOC control
        foc_results = self.foc_controller.update(
            ia, ib, ic, 
            self.speed_ref, self.motor.wr, 
            self.motor.theta_e
        )
        
        # Apply flux weakening if enabled
        if self.enable_flux_weakening:
            id_fw = self.flux_weakening.update(
                self.motor.wr, 
                foc_results['vd'], 
                foc_results['vq'],
                foc_results['iq_ref'],
                self.flux_weakening_method
            )
            foc_results['id_ref'] = id_fw
        
        # Update motor with voltages
        motor_results = self.motor.update(
            foc_results['vd'], 
            foc_results['vq'], 
            self.load_torque
        )
        
        # Store data (with limited buffer size)
        self.data_history['time'].append(self.simulation_time)
        self.data_history['id'].append(foc_results['id_actual'])
        self.data_history['iq'].append(foc_results['iq_actual'])
        self.data_history['speed'].append(motor_results['speed_rpm'])
        self.data_history['torque'].append(motor_results['Te'])
        self.data_history['vd'].append(foc_results['vd'])
        self.data_history['vq'].append(foc_results['vq'])
        
        # Limit buffer size
        for key in self.data_history:
            if len(self.data_history[key]) > self.data_buffer_size:
                self.data_history[key] = self.data_history[key][-self.data_buffer_size:]
        
        # Update visualization only at specified interval
        if self.simulation_time - self.last_viz_update >= self.visualization_update_interval:
            self.update_plots()
            self.last_viz_update = self.simulation_time
        
        # Update time
        self.simulation_time += self.sample_time
        self.time_label.config(text=f"时间: {self.simulation_time:.2f} s")
        
    def toggle_fixed_axis(self):
        """Toggle fixed axis mode"""
        self.fixed_axis = self.fixed_axis_var.get()
        self.update_plots()
        
    def update_plots(self):
        """Update plots with current data"""
        if len(self.data_history['time']) == 0:
            return
        
        # Clear plots
        self.ax_current.clear()
        self.ax_speed.clear()
        self.ax_voltage.clear()
        self.ax_torque.clear()
        
        # Reconfigure plots
        self.ax_current.set_title('d-q Axis Currents')
        self.ax_current.set_xlabel('Time (s)')
        self.ax_current.set_ylabel('Current (A)')
        self.ax_current.grid(True)
        
        self.ax_speed.set_title('Motor Speed')
        self.ax_speed.set_xlabel('Time (s)')
        self.ax_speed.set_ylabel('Speed (RPM)')
        self.ax_speed.grid(True)
        
        self.ax_voltage.set_title('d-q Axis Voltages')
        self.ax_voltage.set_xlabel('Time (s)')
        self.ax_voltage.set_ylabel('Voltage (V)')
        self.ax_voltage.grid(True)
        
        self.ax_torque.set_title('Electromagnetic Torque')
        self.ax_torque.set_xlabel('Time (s)')
        self.ax_torque.set_ylabel('Torque (N.m)')
        self.ax_torque.grid(True)
        
        # Set fixed axis ranges if enabled
        if self.fixed_axis:
            self.ax_current.set_ylim(-25, 25)
            self.ax_speed.set_ylim(-2000, 2000)
            self.ax_voltage.set_ylim(-50, 50)
            self.ax_torque.set_ylim(-15, 15)
        
        # Plot data
        self.ax_current.plot(self.data_history['time'], self.data_history['id'], 'b-', label='id')
        self.ax_current.plot(self.data_history['time'], self.data_history['iq'], 'r-', label='iq')
        self.ax_current.legend()
        
        self.ax_speed.plot(self.data_history['time'], self.data_history['speed'], 'g-', label='Speed')
        speed_ref_rpm = self.speed_ref * 60 / (2 * np.pi)
        self.ax_speed.axhline(y=speed_ref_rpm, color='g', linestyle='--', label='Ref')
        self.ax_speed.legend()
        
        self.ax_voltage.plot(self.data_history['time'], self.data_history['vd'], 'b-', label='vd')
        self.ax_voltage.plot(self.data_history['time'], self.data_history['vq'], 'r-', label='vq')
        self.ax_voltage.legend()
        
        self.ax_torque.plot(self.data_history['time'], self.data_history['torque'], 'm-', label='Torque')
        self.ax_torque.legend()
        
        # Redraw canvas
        self.canvas.draw()
        
    def simulation_loop(self):
        """Main simulation loop with performance monitoring"""
        last_time = time.time()
        frame_count = 0
        performance_check_interval = 1.0  # Check performance every second
        
        while self.is_running:
            start_time = time.time()
            
            # Execute simulation step
            self.simulation_step()
            
            # Control simulation speed
            elapsed = time.time() - start_time
            sleep_time = max(0, self.sample_time / self.sim_speed_var.get() - elapsed)
            time.sleep(sleep_time)
            
            # Performance monitoring
            frame_count += 1
            current_time = time.time()
            if current_time - last_time >= performance_check_interval:
                fps = frame_count / (current_time - last_time)
                if fps < 10:
                    self.performance_label.config(text=f"性能: 低 ({fps:.1f} FPS)")
                elif fps < 20:
                    self.performance_label.config(text=f"性能: 中 ({fps:.1f} FPS)")
                else:
                    self.performance_label.config(text=f"性能: 高 ({fps:.1f} FPS)")
                
                frame_count = 0
                last_time = current_time
            
    def start_simulation(self):
        """Start the simulation"""
        if not self.is_running:
            self.is_running = True
            self.status_label.config(text="运行中")
            
            # Start simulation thread
            self.sim_thread = threading.Thread(target=self.simulation_loop)
            self.sim_thread.daemon = True
            self.sim_thread.start()
            
    def pause_simulation(self):
        """Pause the simulation"""
        self.is_running = False
        self.status_label.config(text="暂停")
        
    def reset_simulation(self):
        """Reset the simulation"""
        self.is_running = False
        self.status_label.config(text="就绪")
        
        # Reset time and data
        self.simulation_time = 0.0
        self.last_viz_update = 0.0
        for key in self.data_history:
            self.data_history[key].clear()
        
        # Reset components
        self.motor.reset_state()
        self.foc_controller.reset()
        self.flux_weakening.reset()
        
        # Clear visualization
        self.update_plots()
        
        # Update display
        self.time_label.config(text="时间: 0.00 s")
        self.performance_label.config(text="性能: 正常")
        
    def load_parameters(self):
        """Load parameters from file"""
        filename = filedialog.askopenfilename(
            title="加载参数",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                with open(filename, 'r') as f:
                    params = json.load(f)
                
                # Update motor parameters
                self.motor.load_parameters(filename)
                self.foc_controller.load_parameters(filename)
                self.flux_weakening.load_parameters(filename)
                
                # Update display
                self.update_params_display()
                
                messagebox.showinfo("成功", "参数加载成功")
            except Exception as e:
                messagebox.showerror("错误", f"参数加载失败: {str(e)}")
                
    def save_parameters(self):
        """Save parameters to file"""
        filename = filedialog.asksaveasfilename(
            title="保存参数",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                # Get current parameters
                params = {
                    'motor_type': 'PMSM',
                    'poles': self.motor.poles,
                    'Rs': self.motor.Rs,
                    'Ld': self.motor.Ld,
                    'Lq': self.motor.Lq,
                    'flux_linkage': self.motor.flux_linkage,
                    'J': self.motor.J,
                    'B': self.motor.B,
                    'max_current': self.motor.max_current,
                    'max_voltage': self.motor.max_voltage,
                    'dc_bus_voltage': self.motor.dc_bus_voltage,
                    'sample_time': self.motor.sample_time,
                    'visualization_update_interval': self.visualization_update_interval,
                    'data_buffer_size': self.data_buffer_size
                }
                
                with open(filename, 'w') as f:
                    json.dump(params, f, indent=4)
                
                messagebox.showinfo("成功", "参数保存成功")
            except Exception as e:
                messagebox.showerror("错误", f"参数保存失败: {str(e)}")
                
    def export_data(self):
        """Export simulation data"""
        filename = filedialog.asksaveasfilename(
            title="导出数据",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                import pandas as pd
                
                # Create DataFrame
                df = pd.DataFrame(self.data_history)
                
                # Save to CSV
                df.to_csv(filename, index=False)
                
                messagebox.showinfo("成功", "数据导出成功")
            except Exception as e:
                messagebox.showerror("错误", f"数据导出失败: {str(e)}")
        
    def run(self):
        """Run the GUI application"""
        self.root.mainloop()


if __name__ == "__main__":
    # Create and run lightweight simulation
    sim = LiteMotorControlSimulation()
    sim.run()