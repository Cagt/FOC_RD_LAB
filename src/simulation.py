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
from self_learning import MotorParameterIdentification
from disturbance_rejection import DisturbanceRejectionController
from visualization import MotorControlVisualization, ParameterVisualization
from visualization_optimized import OptimizedMotorControlVisualization

class MotorControlSimulation:
    """
    Main simulation class that integrates all motor control components
    """
    
    def __init__(self):
        """Initialize simulation"""
        # Initialize components
        self.motor = PMSMModel()
        self.foc_controller = FOCController()
        self.flux_weakening = FluxWeakeningController()
        self.param_identification = MotorParameterIdentification()
        self.disturbance_rejection = DisturbanceRejectionController()
        
        # Simulation parameters
        self.simulation_time = 0.0
        self.sample_time = self.motor.sample_time
        self.is_running = False
        self.simulation_speed = 1.0  # Real-time factor
        
        # Control references
        self.speed_ref = 0.0  # rad/s
        self.load_torque = 0.0  # N.m
        
        # Control modes
        self.enable_flux_weakening = False
        self.enable_parameter_identification = False
        self.enable_disturbance_rejection = False
        self.flux_weakening_method = 'voltage'
        
        # Data storage
        self.data_history = {
            'time': [],
            'id': [],
            'iq': [],
            'id_ref': [],
            'iq_ref': [],
            'speed': [],
            'speed_ref': [],
            'torque': [],
            'vd': [],
            'vq': [],
            'dc_bus_voltage': [],
            'dc_bus_current': [],
            'theta_e': []
        }
        
        # Create GUI
        self.create_gui()
        
    def create_gui(self):
        """Create the main GUI window"""
        self.root = tk.Tk()
        self.root.title("FOC BLDC/PMSM 仿真实验室")
        self.root.geometry("1400x900")
        
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
        view_menu.add_command(label="主界面", command=self.show_main_view)
        view_menu.add_command(label="参数视图", command=self.show_parameter_view)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="帮助", menu=help_menu)
        help_menu.add_command(label="关于", command=self.show_about)
        
    def create_control_panel(self):
        """Create control panel"""
        control_frame = ttk.LabelFrame(self.root, text="控制面板", padding=10)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        
        # Speed control
        speed_frame = ttk.LabelFrame(control_frame, text="速度控制", padding=5)
        speed_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(speed_frame, text="速度参考 (RPM):").grid(row=0, column=0, sticky=tk.W)
        self.speed_ref_var = tk.DoubleVar(value=0)
        speed_scale = ttk.Scale(speed_frame, from_=-3000, to=3000, variable=self.speed_ref_var, 
                                orient=tk.HORIZONTAL, length=200)
        speed_scale.grid(row=0, column=1, padx=5)
        self.speed_label = ttk.Label(speed_frame, text="0")
        self.speed_label.grid(row=0, column=2)
        speed_scale.config(command=self.update_speed_label)
        
        # Load torque control
        load_frame = ttk.LabelFrame(control_frame, text="负载转矩", padding=5)
        load_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(load_frame, text="负载转矩 (N.m):").grid(row=0, column=0, sticky=tk.W)
        self.load_torque_var = tk.DoubleVar(value=0)
        load_scale = ttk.Scale(load_frame, from_=0, to=10, variable=self.load_torque_var,
                              orient=tk.HORIZONTAL, length=200)
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
        
        self.param_identification_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(options_frame, text="参数自学习", 
                       variable=self.param_identification_var).pack(anchor=tk.W)
        
        self.disturbance_rejection_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(options_frame, text="抗干扰控制", 
                       variable=self.disturbance_rejection_var).pack(anchor=tk.W)
        
        # Flux weakening method
        fw_method_frame = ttk.LabelFrame(control_frame, text="弱磁控制方法", padding=5)
        fw_method_frame.pack(fill=tk.X, pady=5)
        
        self.fw_method_var = tk.StringVar(value="voltage")
        ttk.Radiobutton(fw_method_frame, text="电压法", variable=self.fw_method_var,
                       value="voltage").pack(anchor=tk.W)
        ttk.Radiobutton(fw_method_frame, text="速度法", variable=self.fw_method_var,
                       value="speed").pack(anchor=tk.W)
        ttk.Radiobutton(fw_method_frame, text="查表法", variable=self.fw_method_var,
                       value="lookup").pack(anchor=tk.W)
        
        # Simulation control
        sim_frame = ttk.LabelFrame(control_frame, text="仿真控制", padding=5)
        sim_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(sim_frame, text="开始", command=self.start_simulation).pack(side=tk.LEFT, padx=2)
        ttk.Button(sim_frame, text="暂停", command=self.pause_simulation).pack(side=tk.LEFT, padx=2)
        ttk.Button(sim_frame, text="重置", command=self.reset_simulation).pack(side=tk.LEFT, padx=2)
        
        # Simulation speed
        ttk.Label(sim_frame, text="仿真速度:").pack(side=tk.LEFT, padx=5)
        self.sim_speed_var = tk.DoubleVar(value=1.0)
        sim_speed_scale = ttk.Scale(sim_frame, from_=0.1, to=5.0, variable=self.sim_speed_var,
                                   orient=tk.HORIZONTAL, length=100)
        sim_speed_scale.pack(side=tk.LEFT)
        
        # Motor parameters display
        params_frame = ttk.LabelFrame(control_frame, text="电机参数", padding=5)
        params_frame.pack(fill=tk.X, pady=5)
        
        self.params_text = tk.Text(params_frame, height=10, width=30)
        self.params_text.pack()
        self.update_params_display()
        
    def create_visualization_panel(self):
        """Create visualization panel"""
        viz_frame = ttk.LabelFrame(self.root, text="可视化", padding=5)
        viz_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create notebook for multiple views
        self.notebook = ttk.Notebook(viz_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Main visualization tab
        main_frame = ttk.Frame(self.notebook)
        self.notebook.add(main_frame, text="主界面")
        self.main_viz = MotorControlVisualization(main_frame)
        
        # Parameter visualization tab
        param_frame = ttk.Frame(self.notebook)
        self.notebook.add(param_frame, text="参数视图")
        self.param_viz = ParameterVisualization(param_frame)
        
    def create_status_panel(self):
        """Create status panel"""
        status_frame = ttk.LabelFrame(self.root, text="状态信息", padding=5)
        status_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=5)
        
        self.status_label = ttk.Label(status_frame, text="就绪")
        self.status_label.pack(side=tk.LEFT, padx=5)
        
        self.time_label = ttk.Label(status_frame, text="时间: 0.00 s")
        self.time_label.pack(side=tk.LEFT, padx=20)
        
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
        params += f"转动惯量: {self.motor.J:.4f} kg.m²\n"
        params += f"摩擦系数: {self.motor.B:.4f} N.m.s\n"
        params += f"母线电压: {self.motor.dc_bus_voltage:.1f} V\n"
        params += f"最大电流: {self.motor.max_current:.1f} A"
        
        self.params_text.delete(1.0, tk.END)
        self.params_text.insert(1.0, params)
        
    def simulation_step(self):
        """Execute one simulation step"""
        # Get current references
        self.enable_flux_weakening = self.flux_weakening_var.get()
        self.enable_parameter_identification = self.param_identification_var.get()
        self.enable_disturbance_rejection = self.disturbance_rejection_var.get()
        self.flux_weakening_method = self.fw_method_var.get()
        
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
        
        # Apply disturbance rejection if enabled
        if self.enable_disturbance_rejection:
            dr_results = self.disturbance_rejection.update(
                self.motor.Te, 
                self.motor.wr, 
                self.speed_ref,
                foc_results['id_actual'],
                foc_results['iq_actual'],
                foc_results['id_ref'],
                foc_results['iq_ref']
            )
            foc_results['id_ref'] = dr_results['id_ref_modified']
            foc_results['iq_ref'] = dr_results['iq_ref_modified']
        
        # Update motor with voltages
        motor_results = self.motor.update(
            foc_results['vd'], 
            foc_results['vq'], 
            self.load_torque
        )
        
        # Parameter identification if enabled
        if self.enable_parameter_identification:
            self.param_identification.update(
                foc_results['vd'], 
                foc_results['id_actual'],
                foc_results['vq'], 
                foc_results['iq_actual'],
                self.motor.wr,
                motor_results['Te'],
                self.load_torque
            )
            
            # Update parameter visualization
            if self.simulation_time % 0.1 < self.sample_time:  # Update every 0.1 seconds
                params = self.param_identification.get_identified_parameters()
                self.param_viz.update_parameter_data(self.simulation_time, params)
                self.param_viz.update_plots()
        
        # Store data
        self.data_history['time'].append(self.simulation_time)
        self.data_history['id'].append(foc_results['id_actual'])
        self.data_history['iq'].append(foc_results['iq_actual'])
        self.data_history['id_ref'].append(foc_results['id_ref'])
        self.data_history['iq_ref'].append(foc_results['iq_ref'])
        self.data_history['speed'].append(motor_results['speed_rpm'])
        self.data_history['speed_ref'].append(self.speed_ref * 60 / (2 * np.pi))
        self.data_history['torque'].append(motor_results['Te'])
        self.data_history['vd'].append(foc_results['vd'])
        self.data_history['vq'].append(foc_results['vq'])
        self.data_history['dc_bus_voltage'].append(self.motor.dc_bus_voltage)
        self.data_history['dc_bus_current'].append(self.motor.get_dc_bus_current())
        self.data_history['theta_e'].append(self.motor.theta_e)
        
        # Update visualization
        data = {
            'time': self.simulation_time,
            'id': foc_results['id_actual'],
            'iq': foc_results['iq_actual'],
            'id_ref': foc_results['id_ref'],
            'iq_ref': foc_results['iq_ref'],
            'speed_rpm': motor_results['speed_rpm'],
            'speed_ref_rpm': self.speed_ref * 60 / (2 * np.pi),
            'torque': motor_results['Te'],
            'vd': foc_results['vd'],
            'vq': foc_results['vq'],
            'dc_bus_voltage': self.motor.dc_bus_voltage,
            'dc_bus_current': self.motor.get_dc_bus_current()
        }
        
        self.main_viz.update_data(data)
        if self.simulation_time % 0.05 < self.sample_time:  # Update every 0.05 seconds
            self.main_viz.update_plots()
        
        # Update time
        self.simulation_time += self.sample_time
        self.time_label.config(text=f"时间: {self.simulation_time:.2f} s")
        
    def simulation_loop(self):
        """Main simulation loop"""
        while self.is_running:
            start_time = time.time()
            
            # Execute simulation step
            self.simulation_step()
            
            # Control simulation speed
            elapsed = time.time() - start_time
            sleep_time = max(0, self.sample_time / self.sim_speed_var.get() - elapsed)
            time.sleep(sleep_time)
            
    def start_simulation(self):
        """Start the simulation"""
        if not self.is_running:
            self.is_running = True
            self.status_label.config(text="运行中")
            
            # Start simulation thread
            self.sim_thread = threading.Thread(target=self.simulation_loop)
            self.sim_thread.daemon = True
            self.sim_thread.start()
            
            # Start visualization animation
            self.main_viz.start_animation()
            
    def pause_simulation(self):
        """Pause the simulation"""
        self.is_running = False
        self.status_label.config(text="暂停")
        self.main_viz.stop_animation()
        
    def reset_simulation(self):
        """Reset the simulation"""
        self.is_running = False
        self.status_label.config(text="就绪")
        self.main_viz.stop_animation()
        
        # Reset time and data
        self.simulation_time = 0.0
        for key in self.data_history:
            self.data_history[key].clear()
        
        # Reset components
        self.motor.reset_state()
        self.foc_controller.reset()
        self.flux_weakening.reset()
        self.param_identification.reset()
        self.disturbance_rejection.reset()
        
        # Clear visualization
        self.main_viz.clear_data()
        self.param_viz.clear_data()
        
        # Update display
        self.time_label.config(text="时间: 0.00 s")
        
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
                self.param_identification.load_parameters(filename)
                self.disturbance_rejection.disturbance_observer.load_parameters(filename)
                self.disturbance_rejection.adaptive_controller.load_parameters(filename)
                self.disturbance_rejection.robust_controller.load_parameters(filename)
                
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
                    'sample_time': self.motor.sample_time
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
                
    def show_main_view(self):
        """Show main visualization view"""
        self.notebook.select(0)
        
    def show_parameter_view(self):
        """Show parameter visualization view"""
        self.notebook.select(1)
        
    def show_about(self):
        """Show about dialog"""
        messagebox.showinfo(
            "关于",
            "FOC BLDC/PMSM 仿真实验室\n\n"
            "一个用于研究FOC控制算法的仿真平台\n"
            "支持弱磁控制、参数自学习、抗干扰控制等功能\n\n"
            "版本: 1.0"
        )
        
    def run(self):
        """Run the GUI application"""
        self.root.mainloop()


if __name__ == "__main__":
    # Create and run simulation
    sim = MotorControlSimulation()
    sim.run()