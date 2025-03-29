#!/usr/bin/env python3
"""
SRIM Vacancy Defect Concentration Calculator (GUI Version)

This program provides a graphical user interface for calculating defect concentrations
from SRIM vacancy data files. It allows users to specify multiple layers with different
atoms and calculates defect concentrations based on vacancy data and fluence.

Created by O.Z
"""

import numpy as np
import os
import sys
import re
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image, ImageTk, ImageDraw, ImageFont

class DefectCalculatorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("SRIM Vacancy Defect Concentration Calculator - O.Z")
        self.root.geometry("900x700")
        self.root.minsize(900, 700)
        
        # Variables
        self.num_layers = tk.IntVar(value=1)
        self.atoms_per_layer = []
        self.layer_limits = []
        self.fluence = tk.DoubleVar(value=1.0e14)
        self.vacancy_file = tk.StringVar(value="VACANCY.txt")
        
        # Create the main frame
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create tabs
        self.input_tab = ttk.Frame(self.notebook)
        self.results_tab = ttk.Frame(self.notebook)
        self.plot_tab = ttk.Frame(self.notebook)
        self.knockons_tab = ttk.Frame(self.notebook)
        self.dpa_tab = ttk.Frame(self.notebook)
        
        self.notebook.add(self.input_tab, text="Input Parameters")
        self.notebook.add(self.results_tab, text="Results")
        self.notebook.add(self.plot_tab, text="Vacancy Plots")
        self.notebook.add(self.knockons_tab, text="Knock-Ons")
        self.notebook.add(self.dpa_tab, text="DPA")
        
        # Setup input tab
        self.setup_input_tab()
        
        # Setup results tab
        self.setup_results_tab()
        
        # Setup plot tab
        self.setup_plot_tab()
        
        # Setup knock-ons tab
        self.setup_knockons_tab()
        
        # Setup DPA tab
        self.setup_dpa_tab()
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        self.status_var.set("Ready")
        
        # Add watermark
        self.add_watermark()
        
        # Initialize layer frames
        self.layer_frames = []
        self.create_layer_frames()
        
        # Store atom names
        self.atom_names = []
        
        # Store layer densities
        self.layer_densities = []
    
    def add_watermark(self):
        """Add a watermark to the GUI"""
        # Create a watermark label
        watermark_frame = ttk.Frame(self.root)
        watermark_frame.place(relx=1.0, rely=1.0, anchor="se", x=-10, y=-30)
        
        watermark_label = ttk.Label(watermark_frame, text="Created by O.Z", 
                                   font=("Arial", 10, "italic"), foreground="gray")
        watermark_label.pack()
        
    def setup_input_tab(self):
        # Create frames
        input_frame = ttk.LabelFrame(self.input_tab, text="Input Parameters", padding="10")
        input_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Number of layers
        ttk.Label(input_frame, text="Number of Layers (1-5):").grid(row=0, column=0, sticky=tk.W, pady=5)
        layer_spinbox = ttk.Spinbox(input_frame, from_=1, to=5, textvariable=self.num_layers, width=5)
        layer_spinbox.grid(row=0, column=1, sticky=tk.W, pady=5)
        
        # Bind the spinbox to update layers only when value changes
        self.prev_num_layers = self.num_layers.get()
        layer_spinbox.bind("<FocusOut>", self.check_layer_change)
        layer_spinbox.bind("<<Increment>>", self.check_layer_change)
        layer_spinbox.bind("<<Decrement>>", self.check_layer_change)
        
        # Vacancy file
        ttk.Label(input_frame, text="Vacancy File:").grid(row=1, column=0, sticky=tk.W, pady=5)
        file_frame = ttk.Frame(input_frame)
        file_frame.grid(row=1, column=1, columnspan=2, sticky=tk.W, pady=5)
        ttk.Entry(file_frame, textvariable=self.vacancy_file, width=30).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(file_frame, text="Browse...", command=self.browse_file).pack(side=tk.LEFT)
        
        # Fluence
        ttk.Label(input_frame, text="Fluence (particles/cm²):").grid(row=2, column=0, sticky=tk.W, pady=5)
        ttk.Entry(input_frame, textvariable=self.fluence, width=15).grid(row=2, column=1, sticky=tk.W, pady=5)
        
        # Separator
        ttk.Separator(input_frame, orient=tk.HORIZONTAL).grid(row=3, column=0, columnspan=3, sticky=tk.EW, pady=10)
        
        # Layer configuration frame
        self.layer_config_frame = ttk.LabelFrame(input_frame, text="Layer Configuration", padding="10")
        self.layer_config_frame.grid(row=4, column=0, columnspan=3, sticky=tk.NSEW, pady=5)
        input_frame.columnconfigure(2, weight=1)
        input_frame.rowconfigure(4, weight=1)
        
        # Calculate button
        ttk.Button(input_frame, text="Calculate Defect Concentrations", command=self.calculate).grid(row=5, column=0, columnspan=3, pady=10)
        
        # Add watermark to input tab
        watermark_label = ttk.Label(input_frame, text="Created by O.Z", 
                                   font=("Arial", 8, "italic"), foreground="gray")
        watermark_label.grid(row=6, column=2, sticky=tk.E, pady=5)
        
    def check_layer_change(self, event=None):
        """Check if the number of layers has changed and update only if needed"""
        current_layers = self.num_layers.get()
        if current_layers != self.prev_num_layers:
            self.prev_num_layers = current_layers
            self.update_layer_frames()
    
    def setup_results_tab(self):
        # Create results text area
        self.results_text = scrolledtext.ScrolledText(self.results_tab, wrap=tk.WORD, width=80, height=30)
        self.results_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.results_text.insert(tk.END, "Results will appear here after calculation.")
        self.results_text.config(state=tk.DISABLED)
        
        # Add watermark to results tab
        watermark_frame = ttk.Frame(self.results_tab)
        watermark_frame.place(relx=1.0, rely=1.0, anchor="se", x=-15, y=-15)
        
        watermark_label = ttk.Label(watermark_frame, text="Created by O.Z", 
                                   font=("Arial", 8, "italic"), foreground="gray")
        watermark_label.pack()
        
    def setup_plot_tab(self):
        # Create plot frame
        self.plot_frame = ttk.Frame(self.plot_tab)
        self.plot_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Initial message
        ttk.Label(self.plot_frame, text="Plots will appear here after calculation.").pack(pady=20)
        
        # Add watermark to plot tab
        watermark_frame = ttk.Frame(self.plot_tab)
        watermark_frame.place(relx=1.0, rely=1.0, anchor="se", x=-15, y=-15)
        
        watermark_label = ttk.Label(watermark_frame, text="Created by O.Z", 
                                   font=("Arial", 8, "italic"), foreground="gray")
        watermark_label.pack()
    
    def setup_knockons_tab(self):
        # Create knock-ons frame
        self.knockons_frame = ttk.Frame(self.knockons_tab)
        self.knockons_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create text area for knock-ons results
        self.knockons_text = scrolledtext.ScrolledText(self.knockons_frame, wrap=tk.WORD, width=80, height=15)
        self.knockons_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.knockons_text.insert(tk.END, "Knock-Ons results will appear here after calculation.")
        self.knockons_text.config(state=tk.DISABLED)
        
        # Create frame for knock-ons plot
        self.knockons_plot_frame = ttk.Frame(self.knockons_frame)
        self.knockons_plot_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Add watermark to knock-ons tab
        watermark_frame = ttk.Frame(self.knockons_tab)
        watermark_frame.place(relx=1.0, rely=1.0, anchor="se", x=-15, y=-15)
        
        watermark_label = ttk.Label(watermark_frame, text="Created by O.Z", 
                                   font=("Arial", 8, "italic"), foreground="gray")
        watermark_label.pack()
    
    def setup_dpa_tab(self):
        # Create DPA frame
        self.dpa_frame = ttk.Frame(self.dpa_tab)
        self.dpa_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create text area for DPA results
        self.dpa_text = scrolledtext.ScrolledText(self.dpa_frame, wrap=tk.WORD, width=80, height=15)
        self.dpa_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.dpa_text.insert(tk.END, "Displacements Per Atom (DPA) results will appear here after calculation.")
        self.dpa_text.config(state=tk.DISABLED)
        
        # Create frame for DPA plot
        self.dpa_plot_frame = ttk.Frame(self.dpa_frame)
        self.dpa_plot_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Add watermark to DPA tab
        watermark_frame = ttk.Frame(self.dpa_tab)
        watermark_frame.place(relx=1.0, rely=1.0, anchor="se", x=-15, y=-15)
        
        watermark_label = ttk.Label(watermark_frame, text="Created by O.Z", 
                                   font=("Arial", 8, "italic"), foreground="gray")
        watermark_label.pack()
        
    def create_layer_frames(self):
        """Create initial layer frames based on default number of layers"""
        num_layers = self.num_layers.get()
        
        # Initialize variables for each layer
        for i in range(num_layers):
            atoms_var = tk.IntVar(value=2)  # Default to 2 atoms
            self.atoms_per_layer.append(atoms_var)
            
            limit_var = tk.DoubleVar(value=0.5)  # Default thickness
            self.layer_limits.append(limit_var)
        
        # Create the frames
        self.update_layer_frames()
    
    def update_layer_frames(self):
        """Update layer frames when number of layers changes"""
        # Clear existing layer frames
        for frame in self.layer_frames:
            frame.destroy()
        self.layer_frames = []
        
        # Get current number of layers
        num_layers = self.num_layers.get()
        
        # Adjust atoms_per_layer and layer_limits lists to match current number of layers
        # Add new variables if needed
        while len(self.atoms_per_layer) < num_layers:
            atoms_var = tk.IntVar(value=2)  # Default to 2 atoms
            self.atoms_per_layer.append(atoms_var)
            
            limit_var = tk.DoubleVar(value=0.5)  # Default thickness
            self.layer_limits.append(limit_var)
        
        # Remove extra variables if needed (but keep the values for potential undo)
        # We don't actually remove them, just use the first num_layers elements
        
        # Create new layer frames
        for i in range(num_layers):
            layer_frame = ttk.LabelFrame(self.layer_config_frame, text=f"Layer {i+1}", padding="5")
            layer_frame.pack(fill=tk.X, expand=True, pady=5)
            self.layer_frames.append(layer_frame)
            
            # Number of atoms
            ttk.Label(layer_frame, text="Number of Atoms (1-5):").grid(row=0, column=0, sticky=tk.W, pady=2)
            ttk.Spinbox(layer_frame, from_=1, to=5, textvariable=self.atoms_per_layer[i], width=5).grid(row=0, column=1, sticky=tk.W, pady=2)
            
            # Layer thickness
            ttk.Label(layer_frame, text="Layer Thickness (μm):").grid(row=1, column=0, sticky=tk.W, pady=2)
            ttk.Entry(layer_frame, textvariable=self.layer_limits[i], width=10).grid(row=1, column=1, sticky=tk.W, pady=2)
    
    def browse_file(self):
        filename = filedialog.askopenfilename(
            title="Select Vacancy File",
            filetypes=(("Text files", "*.txt"), ("All files", "*.*"))
        )
        if filename:
            self.vacancy_file.set(filename)
    
    def extract_layer_densities(self, file_path):
        """
        Extract atomic densities for each layer from the vacancy file.
        
        Args:
            file_path (str): Path to the vacancy file
            
        Returns:
            list: List of atomic densities (atoms/cm³) for each layer
        """
        densities = []
        
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()
                
                # Look for density information in the file
                for line in lines:
                    if "Density" in line and "atoms/cm3" in line:
                        # Extract density value using regex
                        match = re.search(r'Density\s*=\s*(\d+\.\d+E\d+)\s*atoms/cm3', line)
                        if match:
                            density_str = match.group(1)
                            density = float(density_str)
                            densities.append(density)
            
            if not densities:
                # If no densities found, use default values
                self.status_var.set("Warning: Could not extract layer densities from file. Using default values.")
                densities = [5.0e22] * 5  # Default density for all layers
                
            return densities
                
        except Exception as e:
            messagebox.showerror("Error", f"Error extracting layer densities: {str(e)}")
            return [5.0e22] * 5  # Default density for all layers
            
    def extract_atom_names(self, file_path):
        """
        Extract atom names from the vacancy file header.
        
        Args:
            file_path (str): Path to the vacancy file
            
        Returns:
            list: List of atom names
        """
        atom_names = []
        
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()
                
                # Find the header line with "DEPTH" to determine data start
                header_line_idx = None
                for i, line in enumerate(lines):
                    if "DEPTH" in line:
                        header_line_idx = i
                        break
                
                if header_line_idx is None:
                    raise ValueError("Could not find header line in vacancy file")
                
                # Extract atom names from the header line
                header_parts = lines[header_line_idx].split()
                
                # Skip the first column (DEPTH) and second column (H Knock-Ons)
                for i in range(2, len(header_parts)):
                    atom_names.append(header_parts[i])
                
                # Remove duplicates while preserving order
                seen = set()
                unique_atom_names = []
                for name in atom_names:
                    if name not in seen:
                        seen.add(name)
                        unique_atom_names.append(name)
                
                return atom_names  # Return all atom names to preserve column mapping
                
        except Exception as e:
            messagebox.showerror("Error", f"Error extracting atom names: {str(e)}")
            return ["Unknown"] * 9  # Default to unknown names
            
    def read_vacancy_file(self, file_path):
        """
        Read the SRIM vacancy file and extract depth, knock-ons, and vacancy data.
        
        Args:
            file_path (str): Path to the vacancy file
            
        Returns:
            tuple: (depths, knockons_data, vacancy_data)
                - depths: numpy array of depth values in Angstroms
                - knockons_data: numpy array of knock-ons values
                - vacancy_data: numpy array of vacancy values
        """
        depths = []
        knockons_data = []
        vacancy_data = []
        
        try:
            # First extract atom names and layer densities
            self.atom_names = self.extract_atom_names(file_path)
            self.layer_densities = self.extract_layer_densities(file_path)
            
            with open(file_path, 'r') as f:
                lines = f.readlines()
                
                # Find the header line with "DEPTH" to determine data start
                header_line_idx = None
                for i, line in enumerate(lines):
                    if "DEPTH" in line:
                        header_line_idx = i
                        break
                
                if header_line_idx is None:
                    raise ValueError("Could not find header line in vacancy file")
                
                # Skip the header and separator lines
                data_start_idx = header_line_idx + 2
                
                # Process data lines
                for line in lines[data_start_idx:]:
                    if not line.strip() or "To convert to Energy Lost" in line:
                        break
                        
                    parts = line.split()
                    if len(parts) < 3:  # Need at least depth and some vacancy data
                        continue
                    
                    # Skip separator lines with dashes
                    if '-----------' in line:
                        continue
                        
                    # Extract depth (first column)
                    try:
                        depth_str = parts[0]
                        # Handle scientific notation format like "443010.E-03"
                        if 'E' in depth_str:
                            base, exp = depth_str.split('E')
                            depth = float(base) * (10 ** float(exp))
                        else:
                            depth = float(depth_str)
                        depths.append(depth)
                        
                        # Extract knock-ons data (second column)
                        knockons_str = parts[1]
                        if 'E' in knockons_str:
                            base, exp = knockons_str.split('E')
                            knockons = float(base) * (10 ** float(exp))
                        else:
                            knockons = float(knockons_str)
                        knockons_data.append(knockons)
                        
                        # Extract vacancy data (starting from column 3)
                        vac_values = []
                        for val_str in parts[2:]:  # Start from the third column
                            if 'E' in val_str:
                                base, exp = val_str.split('E')
                                val = float(base) * (10 ** float(exp))
                            else:
                                val = float(val_str)
                            vac_values.append(val)
                        
                        vacancy_data.append(vac_values)
                    except (ValueError, IndexError) as e:
                        self.status_var.set(f"Warning: Could not parse line: {line.strip()}")
                        continue
            
            # Convert to numpy arrays
            depths = np.array(depths)
            knockons_data = np.array(knockons_data)
            vacancy_data = np.array(vacancy_data)
            
            # Convert depths from Angstroms to micrometers
            depths = depths / 10000.0
            
            return depths, knockons_data, vacancy_data
            
        except Exception as e:
            messagebox.showerror("Error", f"Error reading vacancy file: {str(e)}")
            return None, None, None
    
    def find_layer_indices(self, depths, layer_limits):
        """
        Find the indices in the depth array that correspond to each layer's boundaries.
        
        Args:
            depths (numpy.ndarray): Array of depth values in micrometers
            layer_limits (list): List of layer thickness limits in micrometers
            
        Returns:
            list: List of tuples (start_idx, end_idx) for each layer
        """
        layer_indices = []
        layer_boundaries = []  # Store actual depth values at boundaries
        current_depth = 0.0
        
        for i, limit in enumerate(layer_limits):
            # Find the start index for this layer
            start_idx = 0
            if i > 0:
                # For layers after the first one, start from the previous layer's end
                start_idx = layer_indices[i-1][1]
            
            # Find the end index for this layer
            # Look for the smallest depth that is >= the layer limit
            end_depth = current_depth + limit
            end_idx = start_idx
            
            while end_idx < len(depths) and depths[end_idx] < end_depth:
                end_idx += 1
                
            # If we reached the end of the depths array, use the last index
            if end_idx >= len(depths):
                end_idx = len(depths) - 1
            
            # Store the actual depth values at layer boundaries
            start_depth = depths[start_idx] if start_idx < len(depths) else 0
            end_depth_actual = depths[end_idx] if end_idx < len(depths) else depths[-1]
            layer_boundaries.append((start_depth, end_depth_actual))
            
            layer_indices.append((start_idx, end_idx))
            current_depth = end_depth
            
        return layer_indices, layer_boundaries
    
    def calculate_defect_concentration(self, vacancy_data, layer_indices, atoms_per_layer, fluence):
        """
        Calculate defect concentration for each layer based on vacancy data and fluence.
        
        Args:
            vacancy_data (numpy.ndarray): Array of vacancy values
            layer_indices (list): List of tuples (start_idx, end_idx) for each layer
            atoms_per_layer (list): List of number of atoms for each layer
            fluence (float): Fluence value in particles/cm²
            
        Returns:
            list: List of defect concentrations for each layer and atom
        """
        defect_concentrations = []
        atom_indices = []  # To track which atom (column) is used for each layer/atom
        
        for layer_idx, (start_idx, end_idx) in enumerate(layer_indices):
            layer_defects = []
            layer_atom_indices = []
            num_atoms = atoms_per_layer[layer_idx]
            
            # Calculate for each atom in the layer
            for atom_idx in range(num_atoms):
                # Column index in vacancy_data
                # For layer 1: columns 0 to num_atoms-1
                # For layer 2: columns num_atoms to 2*num_atoms-1
                # And so on...
                col_idx = sum(atoms_per_layer[:layer_idx]) + atom_idx
                layer_atom_indices.append(col_idx)
                
                if col_idx >= vacancy_data.shape[1]:
                    # If column index is out of bounds, add zero concentration
                    layer_defects.append(0.0)
                    continue
                    
                # Extract vacancy data for this atom in this layer
                atom_vacancies = vacancy_data[start_idx:end_idx+1, col_idx]
                
                # Calculate average vacancy per Angstrom-Ion
                avg_vacancy = np.mean(atom_vacancies)
                
                # Convert to defect concentration (defects/cm³)
                # Vacancy units are vacancies/(Angstrom-Ion)
                # 1 Angstrom = 1e-8 cm
                # So vacancies/(Angstrom-Ion) * fluence(ions/cm²) / 1e-8 = defects/cm³
                defect_conc = avg_vacancy * fluence / 1e-8
                
                layer_defects.append(defect_conc)
                
            defect_concentrations.append(layer_defects)
            atom_indices.append(layer_atom_indices)
            
        return defect_concentrations, atom_indices
    
    def calculate_knockons_concentration(self, knockons_data, layer_indices, fluence):
        """
        Calculate knock-ons concentration for each layer based on knock-ons data and fluence.
        
        Args:
            knockons_data (numpy.ndarray): Array of knock-ons values
            layer_indices (list): List of tuples (start_idx, end_idx) for each layer
            fluence (float): Fluence value in particles/cm²
            
        Returns:
            list: List of knock-ons concentrations for each layer
        """
        knockons_concentrations = []
        
        for layer_idx, (start_idx, end_idx) in enumerate(layer_indices):
            # Extract knock-ons data for this layer
            layer_knockons = knockons_data[start_idx:end_idx+1]
            
            # Calculate average knock-ons per Angstrom-Ion
            avg_knockons = np.mean(layer_knockons)
            
            # Convert to knock-ons concentration (knock-ons/cm³)
            # Knock-ons units are knock-ons/(Angstrom-Ion)
            # 1 Angstrom = 1e-8 cm
            # So knock-ons/(Angstrom-Ion) * fluence(ions/cm²) / 1e-8 = knock-ons/cm³
            knockons_conc = avg_knockons * fluence / 1e-8
            
            knockons_concentrations.append(knockons_conc)
            
        return knockons_concentrations
    
    def calculate_dpa(self, vacancy_data, layer_indices, atoms_per_layer, fluence):
        """
        Calculate Displacements Per Atom (DPA) for each layer and atom.
        
        DPA = (ion fluence × vacancy) / Atomic density
            = (ions/cm²) × (vacancies/(ion·cm)) / (Atoms/cm³)
        
        Args:
            vacancy_data (numpy.ndarray): Array of vacancy values
            layer_indices (list): List of tuples (start_idx, end_idx) for each layer
            atoms_per_layer (list): List of number of atoms for each layer
            fluence (float): Fluence value in particles/cm²
            
        Returns:
            tuple: (dpa_values, atom_indices)
                - dpa_values: List of DPA values for each layer and atom
                - atom_indices: List of atom column indices for each layer
        """
        dpa_values = []
        atom_indices = []  # To track which atom (column) is used for each layer/atom
        
        for layer_idx, (start_idx, end_idx) in enumerate(layer_indices):
            layer_dpa = []
            layer_atom_indices = []
            num_atoms = atoms_per_layer[layer_idx]
            
            # Get atomic density for this layer
            atomic_density = self.layer_densities[layer_idx] if layer_idx < len(self.layer_densities) else 5.0e22
            
            # Calculate for each atom in the layer
            for atom_idx in range(num_atoms):
                # Column index in vacancy_data
                col_idx = sum(atoms_per_layer[:layer_idx]) + atom_idx
                layer_atom_indices.append(col_idx)
                
                if col_idx >= vacancy_data.shape[1]:
                    # If column index is out of bounds, add zero DPA
                    layer_dpa.append(0.0)
                    continue
                    
                # Extract vacancy data for this atom in this layer
                atom_vacancies = vacancy_data[start_idx:end_idx+1, col_idx]
                
                # Calculate average vacancy per Angstrom-Ion
                avg_vacancy = np.mean(atom_vacancies)
                
                # Convert to vacancies/(ion·cm)
                # 1 Angstrom = 1e-8 cm
                vacancies_per_ion_cm = avg_vacancy / 1e-8
                
                # Calculate DPA
                # DPA = (fluence × vacancies_per_ion_cm) / atomic_density
                dpa = (fluence * vacancies_per_ion_cm) / atomic_density
                
                layer_dpa.append(dpa)
                
            dpa_values.append(layer_dpa)
            atom_indices.append(layer_atom_indices)
            
        return dpa_values, atom_indices
    
    def calculate_average_defect_concentration(self, defect_concentrations, layer_boundaries):
        """
        Calculate average defect concentration for each layer and the whole structure.
        
        Args:
            defect_concentrations (list): List of defect concentrations for each layer and atom
            layer_boundaries (list): List of tuples (start_depth, end_depth) for each layer
            
        Returns:
            tuple: (avg_layer_concentrations, total_avg_concentration)
                - avg_layer_concentrations: List of average defect concentrations for each layer
                - total_avg_concentration: Average defect concentration for the whole structure
        """
        avg_layer_concentrations = []
        total_defects = 0.0
        total_volume = 0.0
        
        for layer_idx, layer_defects in enumerate(defect_concentrations):
            # Calculate average defect concentration for this layer
            if layer_defects:
                avg_layer_conc = np.mean(layer_defects)
            else:
                avg_layer_conc = 0.0
                
            avg_layer_concentrations.append(avg_layer_conc)
            
            # Calculate layer volume in cm³
            start_depth, end_depth = layer_boundaries[layer_idx]
            layer_thickness = (end_depth - start_depth) * 1e-4  # μm to cm
            layer_volume = layer_thickness * 1.0  # Assume 1 cm² area
            
            # Add to total defects and volume
            total_defects += avg_layer_conc * layer_volume
            total_volume += layer_volume
        
        # Calculate average defect concentration for the whole structure
        if total_volume > 0:
            total_avg_concentration = total_defects / total_volume
        else:
            total_avg_concentration = 0.0
            
        return avg_layer_concentrations, total_avg_concentration
    
    def calculate(self):
        try:
            # Get input values
            num_layers = self.num_layers.get()
            atoms_per_layer = [var.get() for var in self.atoms_per_layer[:num_layers]]
            layer_limits = [var.get() for var in self.layer_limits[:num_layers]]
            vacancy_file = self.vacancy_file.get()
            fluence = self.fluence.get()
            
            # Validate inputs
            if not os.path.isfile(vacancy_file):
                messagebox.showerror("Error", f"Vacancy file '{vacancy_file}' not found.")
                return
                
            if fluence <= 0:
                messagebox.showerror("Error", "Fluence must be positive.")
                return
                
            for i, limit in enumerate(layer_limits):
                if limit <= 0:
                    messagebox.showerror("Error", f"Layer {i+1} thickness must be positive.")
                    return
            
            # Update status
            self.status_var.set("Reading vacancy file...")
            self.root.update_idletasks()
            
            # Read vacancy file
            depths, knockons_data, vacancy_data = self.read_vacancy_file(vacancy_file)
            if depths is None or vacancy_data is None:
                return
                
            self.status_var.set(f"Found {len(depths)} depth points and {vacancy_data.shape[1]} vacancy columns")
            self.root.update_idletasks()
            
            # Find layer indices
            layer_indices, layer_boundaries = self.find_layer_indices(depths, layer_limits)
            
            # Calculate defect concentrations
            defect_concentrations, atom_indices = self.calculate_defect_concentration(
                vacancy_data, layer_indices, atoms_per_layer, fluence)
            
            # Calculate average defect concentrations
            avg_layer_concentrations, total_avg_concentration = self.calculate_average_defect_concentration(
                defect_concentrations, layer_boundaries)
            
            # Calculate knock-ons concentrations
            knockons_concentrations = self.calculate_knockons_concentration(
                knockons_data, layer_indices, fluence)
            
            # Calculate DPA values
            dpa_values, dpa_atom_indices = self.calculate_dpa(
                vacancy_data, layer_indices, atoms_per_layer, fluence)
            
            # Display results
            self.display_results(depths, vacancy_data, layer_indices, atoms_per_layer, 
                                layer_limits, fluence, defect_concentrations, atom_indices, 
                                layer_boundaries, avg_layer_concentrations, total_avg_concentration)
            
            # Display knock-ons results
            self.display_knockons_results(depths, knockons_data, layer_indices, 
                                         layer_limits, fluence, knockons_concentrations, layer_boundaries)
            
            # Display DPA results
            self.display_dpa_results(depths, vacancy_data, layer_indices, atoms_per_layer,
                                    layer_limits, fluence, dpa_values, dpa_atom_indices, layer_boundaries)
            
            # Create plots
            self.create_plots(depths, vacancy_data, layer_indices, atoms_per_layer, 
                             layer_limits, defect_concentrations, atom_indices, layer_boundaries,
                             avg_layer_concentrations)
            
            # Create knock-ons plot
            self.create_knockons_plot(depths, knockons_data, layer_indices, 
                                     layer_limits, knockons_concentrations, layer_boundaries)
            
            # Create DPA plot
            self.create_dpa_plot(depths, vacancy_data, layer_indices, atoms_per_layer,
                                layer_limits, dpa_values, dpa_atom_indices, layer_boundaries)
            
            # Switch to results tab
            self.notebook.select(1)  # Index 1 is the results tab
            
            self.status_var.set("Calculation completed successfully")
            
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred during calculation: {str(e)}")
            self.status_var.set(f"Error: {str(e)}")
    
    def display_results(self, depths, vacancy_data, layer_indices, atoms_per_layer, 
                       layer_limits, fluence, defect_concentrations, atom_indices, 
                       layer_boundaries, avg_layer_concentrations, total_avg_concentration):
        """Display calculation results in the results tab"""
        # Enable text widget for editing
        self.results_text.config(state=tk.NORMAL)
        self.results_text.delete(1.0, tk.END)
        
        # Add header
        self.results_text.insert(tk.END, "===== Defect Concentration Results =====\n\n")
        self.results_text.insert(tk.END, f"Fluence: {fluence:.2e} particles/cm²\n\n")
        
        # Add layer results
        for layer_idx, layer_defects in enumerate(defect_concentrations):
            self.results_text.insert(tk.END, f"Layer {layer_idx+1}:\n")
            
            # Show requested depth range
            requested_start = 0 if layer_idx == 0 else sum(layer_limits[:layer_idx])
            requested_end = sum(layer_limits[:layer_idx+1])
            self.results_text.insert(tk.END, f"  Requested depth range: {requested_start:.4f} - {requested_end:.4f} μm\n")
            
            # Show actual depth range from vacancy file
            actual_start, actual_end = layer_boundaries[layer_idx]
            self.results_text.insert(tk.END, f"  Actual depth range from file: {actual_start:.4f} - {actual_end:.4f} μm\n")
            
            # Show atomic density
            if layer_idx < len(self.layer_densities):
                self.results_text.insert(tk.END, f"  Atomic density: {self.layer_densities[layer_idx]:.4e} atoms/cm³\n")
            
            # Show depth points used
            start_idx, end_idx = layer_indices[layer_idx]
            num_points = end_idx - start_idx + 1
            self.results_text.insert(tk.END, f"  Number of depth points used: {num_points}\n")
            
            # Show defect concentrations for each atom
            for atom_idx, defect_conc in enumerate(layer_defects):
                # Get the actual atom name from the atom_names list using the column index
                col_idx = atom_indices[layer_idx][atom_idx]
                atom_name = self.atom_names[col_idx] if col_idx < len(self.atom_names) else f"Atom {atom_idx+1}"
                
                self.results_text.insert(tk.END, f"  {atom_name}: {defect_conc:.4e} defects/cm³\n")
            
            # Show average defect concentration for this layer
            self.results_text.insert(tk.END, f"  Average defect concentration: {avg_layer_concentrations[layer_idx]:.4e} defects/cm³\n")
            
            self.results_text.insert(tk.END, "\n")
        
        # Add average defect concentration for the whole structure
        self.results_text.insert(tk.END, "===== Overall Results =====\n\n")
        self.results_text.insert(tk.END, f"Average defect concentration for the entire structure: {total_avg_concentration:.4e} defects/cm³\n\n")
        
        # Add watermark
        self.results_text.insert(tk.END, "\n\nCreated by O.Z\n")
        
        # Disable text widget to prevent editing
        self.results_text.config(state=tk.DISABLED)
    
    def display_knockons_results(self, depths, knockons_data, layer_indices, 
                                layer_limits, fluence, knockons_concentrations, layer_boundaries):
        """Display knock-ons calculation results in the knock-ons tab"""
        # Enable text widget for editing
        self.knockons_text.config(state=tk.NORMAL)
        self.knockons_text.delete(1.0, tk.END)
        
        # Add header
        self.knockons_text.insert(tk.END, "===== Primary Knock-Ons Results =====\n\n")
        self.knockons_text.insert(tk.END, f"Fluence: {fluence:.2e} particles/cm²\n\n")
        
        # Add layer results
        total_knockons = 0
        for layer_idx, knockons_conc in enumerate(knockons_concentrations):
            self.knockons_text.insert(tk.END, f"Layer {layer_idx+1}:\n")
            
            # Show actual depth range from vacancy file
            actual_start, actual_end = layer_boundaries[layer_idx]
            self.knockons_text.insert(tk.END, f"  Depth range: {actual_start:.4f} - {actual_end:.4f} μm\n")
            
            # Show depth points used
            start_idx, end_idx = layer_indices[layer_idx]
            num_points = end_idx - start_idx + 1
            self.knockons_text.insert(tk.END, f"  Number of depth points used: {num_points}\n")
            
            # Show average knock-ons value
            layer_knockons = knockons_data[start_idx:end_idx+1]
            avg_knockons = np.mean(layer_knockons)
            self.knockons_text.insert(tk.END, f"  Average Knock-Ons: {avg_knockons:.4e} knock-ons/(Angstrom-Ion)\n")
            
            # Show knock-ons concentration
            self.knockons_text.insert(tk.END, f"  Knock-Ons Concentration: {knockons_conc:.4e} knock-ons/cm³\n")
            
            # Calculate layer volume in cm³ (thickness in cm)
            layer_thickness = (actual_end - actual_start) * 1e-4  # μm to cm
            # Assume 1 cm² area
            layer_volume = layer_thickness * 1.0  # cm³
            
            # Calculate total knock-ons in layer
            total_layer_knockons = knockons_conc * layer_volume
            self.knockons_text.insert(tk.END, f"  Total Knock-Ons in Layer: {total_layer_knockons:.4e} knock-ons\n")
            
            total_knockons += total_layer_knockons
            
            self.knockons_text.insert(tk.END, "\n")
        
        # Add total knock-ons across all layers
        self.knockons_text.insert(tk.END, f"Total Primary Knock-Ons across all layers: {total_knockons:.4e} knock-ons\n")
        
        # Add watermark
        self.knockons_text.insert(tk.END, "\n\nCreated by O.Z\n")
        
        # Disable text widget to prevent editing
        self.knockons_text.config(state=tk.DISABLED)
    
    def display_dpa_results(self, depths, vacancy_data, layer_indices, atoms_per_layer,
                           layer_limits, fluence, dpa_values, atom_indices, layer_boundaries):
        """Display DPA calculation results in the DPA tab"""
        # Enable text widget for editing
        self.dpa_text.config(state=tk.NORMAL)
        self.dpa_text.delete(1.0, tk.END)
        
        # Add header
        self.dpa_text.insert(tk.END, "===== Displacements Per Atom (DPA) Results =====\n\n")
        self.dpa_text.insert(tk.END, f"Fluence: {fluence:.2e} particles/cm²\n\n")
        self.dpa_text.insert(tk.END, "Formula: DPA = (ion fluence × vacancy) / Atomic density\n")
        self.dpa_text.insert(tk.END, "        = (ions/cm²) × (vacancies/(ion·cm)) / (Atoms/cm³)\n\n")
        
        # Add layer results
        for layer_idx, layer_dpa in enumerate(dpa_values):
            self.dpa_text.insert(tk.END, f"Layer {layer_idx+1}:\n")
            
            # Show actual depth range from vacancy file
            actual_start, actual_end = layer_boundaries[layer_idx]
            self.dpa_text.insert(tk.END, f"  Depth range: {actual_start:.4f} - {actual_end:.4f} μm\n")
            
            # Show atomic density
            atomic_density = self.layer_densities[layer_idx] if layer_idx < len(self.layer_densities) else 5.0e22
            self.dpa_text.insert(tk.END, f"  Atomic density: {atomic_density:.4e} atoms/cm³\n")
            
            # Show DPA values for each atom
            for atom_idx, dpa in enumerate(layer_dpa):
                # Get the actual atom name from the atom_names list using the column index
                col_idx = atom_indices[layer_idx][atom_idx]
                atom_name = self.atom_names[col_idx] if col_idx < len(self.atom_names) else f"Atom {atom_idx+1}"
                
                self.dpa_text.insert(tk.END, f"  {atom_name}: {dpa:.4e} DPA\n")
            
            # Calculate average DPA for the layer
            avg_dpa = np.mean(layer_dpa)
            self.dpa_text.insert(tk.END, f"  Average DPA: {avg_dpa:.4e}\n")
            
            self.dpa_text.insert(tk.END, "\n")
        
        # Add watermark
        self.dpa_text.insert(tk.END, "\n\nCreated by O.Z\n")
        
        # Disable text widget to prevent editing
        self.dpa_text.config(state=tk.DISABLED)
    
    def create_plots(self, depths, vacancy_data, layer_indices, atoms_per_layer, 
                    layer_limits, defect_concentrations, atom_indices, layer_boundaries,
                    avg_layer_concentrations):
        """Create plots in the plot tab"""
        # Clear existing plots
        for widget in self.plot_frame.winfo_children():
            widget.destroy()
        
        # Create figure with two subplots
        fig = plt.Figure(figsize=(10, 8), dpi=100)
        
        # Add watermark to figure
        fig.text(0.95, 0.05, "Created by O.Z", fontsize=8, color='gray', 
                ha='right', va='bottom', alpha=0.7, style='italic')
        
        # Plot 1: Vacancy vs Depth
        ax1 = fig.add_subplot(211)
        
        # Plot vacancy data for each atom in each layer
        for layer_idx, layer_atom_indices in enumerate(atom_indices):
            for i, col_idx in enumerate(layer_atom_indices):
                if col_idx < vacancy_data.shape[1]:
                    # Get the actual atom name
                    atom_name = self.atom_names[col_idx] if col_idx < len(self.atom_names) else f"Atom {i+1}"
                    ax1.semilogy(depths, vacancy_data[:, col_idx], 
                               label=f"Layer {layer_idx+1}, {atom_name}")
        
        # Add layer boundaries - use actual boundaries from file
        for i, (start_depth, end_depth) in enumerate(layer_boundaries):
            # Add vertical line at layer end
            ax1.axvline(x=end_depth, color='k', linestyle='--', alpha=0.5)
            # Add text label for layer
            mid_depth = (start_depth + end_depth) / 2
            ax1.text(mid_depth, ax1.get_ylim()[0] * 1.5, f"Layer {i+1}", 
                    horizontalalignment='center', verticalalignment='bottom',
                    bbox=dict(facecolor='white', alpha=0.7))
        
        ax1.set_xlabel('Depth (μm)')
        ax1.set_ylabel('Vacancy (Vacancies/Angstrom-Ion)')
        ax1.set_title('Vacancy vs Depth')
        ax1.legend()
        ax1.grid(True)
        
        # Plot 2: Defect Concentration by Layer and Atom
        ax2 = fig.add_subplot(212)
        
        # Prepare data for bar chart
        labels = []
        values = []
        colors = []
        
        color_map = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        
        for layer_idx, (layer_defects, layer_atom_indices) in enumerate(zip(defect_concentrations, atom_indices)):
            for atom_idx, (defect_conc, col_idx) in enumerate(zip(layer_defects, layer_atom_indices)):
                # Get the actual atom name
                atom_name = self.atom_names[col_idx] if col_idx < len(self.atom_names) else f"Atom {atom_idx+1}"
                labels.append(f"L{layer_idx+1} {atom_name}")
                values.append(defect_conc)
                colors.append(color_map[layer_idx % len(color_map)])
        
        # Add average values
        for layer_idx, avg_conc in enumerate(avg_layer_concentrations):
            labels.append(f"L{layer_idx+1} Avg")
            values.append(avg_conc)
            colors.append('black')
        
        ax2.bar(labels, values, color=colors)
        ax2.set_xlabel('Layer and Atom')
        ax2.set_ylabel('Defect Concentration (defects/cm³)')
        ax2.set_title('Defect Concentration by Layer and Atom')
        ax2.set_yscale('log')
        ax2.grid(True, axis='y')
        
        # Rotate x-axis labels for better readability
        plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
        
        # Adjust layout
        fig.tight_layout()
        
        # Add figure to frame
        canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Add toolbar
        from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
        toolbar_frame = ttk.Frame(self.plot_frame)
        toolbar_frame.pack(fill=tk.X)
        toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
        toolbar.update()
    
    def create_knockons_plot(self, depths, knockons_data, layer_indices, 
                            layer_limits, knockons_concentrations, layer_boundaries):
        """Create knock-ons plots in the knock-ons tab"""
        # Clear existing plots
        for widget in self.knockons_plot_frame.winfo_children():
            widget.destroy()
        
        # Create figure with two subplots
        fig = plt.Figure(figsize=(10, 8), dpi=100)
        
        # Add watermark to figure
        fig.text(0.95, 0.05, "Created by O.Z", fontsize=8, color='gray', 
                ha='right', va='bottom', alpha=0.7, style='italic')
        
        # Plot 1: Knock-Ons vs Depth
        ax1 = fig.add_subplot(211)
        
        # Plot knock-ons data
        ax1.semilogy(depths, knockons_data, label="Primary Knock-Ons", color='red')
        
        # Add layer boundaries - use actual boundaries from file
        for i, (start_depth, end_depth) in enumerate(layer_boundaries):
            # Add vertical line at layer end
            ax1.axvline(x=end_depth, color='k', linestyle='--', alpha=0.5)
            # Add text label for layer
            mid_depth = (start_depth + end_depth) / 2
            ax1.text(mid_depth, ax1.get_ylim()[0] * 1.5, f"Layer {i+1}", 
                    horizontalalignment='center', verticalalignment='bottom',
                    bbox=dict(facecolor='white', alpha=0.7))
        
        ax1.set_xlabel('Depth (μm)')
        ax1.set_ylabel('Knock-Ons (Knock-Ons/Angstrom-Ion)')
        ax1.set_title('Primary Knock-Ons vs Depth')
        ax1.legend()
        ax1.grid(True)
        
        # Plot 2: Knock-Ons Concentration by Layer
        ax2 = fig.add_subplot(212)
        
        # Prepare data for bar chart
        labels = [f"Layer {i+1}" for i in range(len(knockons_concentrations))]
        values = knockons_concentrations
        colors = ['red'] * len(knockons_concentrations)
        
        ax2.bar(labels, values, color=colors)
        ax2.set_xlabel('Layer')
        ax2.set_ylabel('Knock-Ons Concentration (knock-ons/cm³)')
        ax2.set_title('Primary Knock-Ons Concentration by Layer')
        ax2.set_yscale('log')
        ax2.grid(True, axis='y')
        
        # Adjust layout
        fig.tight_layout()
        
        # Add figure to frame
        canvas = FigureCanvasTkAgg(fig, master=self.knockons_plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Add toolbar
        from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
        toolbar_frame = ttk.Frame(self.knockons_plot_frame)
        toolbar_frame.pack(fill=tk.X)
        toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
        toolbar.update()
    
    def create_dpa_plot(self, depths, vacancy_data, layer_indices, atoms_per_layer,
                       layer_limits, dpa_values, atom_indices, layer_boundaries):
        """Create DPA plots in the DPA tab"""
        # Clear existing plots
        for widget in self.dpa_plot_frame.winfo_children():
            widget.destroy()
        
        # Create figure with two subplots
        fig = plt.Figure(figsize=(10, 8), dpi=100)
        
        # Add watermark to figure
        fig.text(0.95, 0.05, "Created by O.Z", fontsize=8, color='gray', 
                ha='right', va='bottom', alpha=0.7, style='italic')
        
        # Plot 1: Average DPA vs Depth
        ax1 = fig.add_subplot(211)
        
        # Calculate and plot average DPA for each depth point
        avg_dpa_by_depth = []
        for d_idx in range(len(depths)):
            # Find which layer this depth point belongs to
            layer_idx = None
            for l_idx, (start_idx, end_idx) in enumerate(layer_indices):
                if start_idx <= d_idx <= end_idx:
                    layer_idx = l_idx
                    break
            
            if layer_idx is not None:
                # Get atomic density for this layer
                atomic_density = self.layer_densities[layer_idx] if layer_idx < len(self.layer_densities) else 5.0e22
                
                # Calculate average vacancy across all atoms at this depth
                avg_vacancy = 0
                count = 0
                for col_idx in atom_indices[layer_idx]:
                    if col_idx < vacancy_data.shape[1]:
                        avg_vacancy += vacancy_data[d_idx, col_idx]
                        count += 1
                
                if count > 0:
                    avg_vacancy /= count
                    
                    # Convert to vacancies/(ion·cm)
                    vacancies_per_ion_cm = avg_vacancy / 1e-8
                    
                    # Calculate DPA
                    dpa = (self.fluence.get() * vacancies_per_ion_cm) / atomic_density
                    avg_dpa_by_depth.append(dpa)
                else:
                    avg_dpa_by_depth.append(0)
            else:
                avg_dpa_by_depth.append(0)
        
        # Plot average DPA vs depth
        ax1.semilogy(depths, avg_dpa_by_depth, label="Average DPA", color='green')
        
        # Add layer boundaries - use actual boundaries from file
        for i, (start_depth, end_depth) in enumerate(layer_boundaries):
            # Add vertical line at layer end
            ax1.axvline(x=end_depth, color='k', linestyle='--', alpha=0.5)
            # Add text label for layer
            mid_depth = (start_depth + end_depth) / 2
            ax1.text(mid_depth, ax1.get_ylim()[0] * 1.5, f"Layer {i+1}", 
                    horizontalalignment='center', verticalalignment='bottom',
                    bbox=dict(facecolor='white', alpha=0.7))
        
        ax1.set_xlabel('Depth (μm)')
        ax1.set_ylabel('DPA')
        ax1.set_title('Displacements Per Atom (DPA) vs Depth')
        ax1.legend()
        ax1.grid(True)
        
        # Plot 2: DPA by Layer and Atom
        ax2 = fig.add_subplot(212)
        
        # Prepare data for bar chart
        labels = []
        values = []
        colors = []
        
        color_map = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        
        for layer_idx, (layer_dpa, layer_atom_indices) in enumerate(zip(dpa_values, atom_indices)):
            for atom_idx, (dpa, col_idx) in enumerate(zip(layer_dpa, layer_atom_indices)):
                # Get the actual atom name
                atom_name = self.atom_names[col_idx] if col_idx < len(self.atom_names) else f"Atom {atom_idx+1}"
                labels.append(f"L{layer_idx+1} {atom_name}")
                values.append(dpa)
                colors.append(color_map[layer_idx % len(color_map)])
            
            # Add average DPA for this layer
            if layer_dpa:
                avg_dpa = np.mean(layer_dpa)
                labels.append(f"L{layer_idx+1} Avg")
                values.append(avg_dpa)
                colors.append('black')
        
        ax2.bar(labels, values, color=colors)
        ax2.set_xlabel('Layer and Atom')
        ax2.set_ylabel('DPA')
        ax2.set_title('Displacements Per Atom (DPA) by Layer and Atom')
        ax2.set_yscale('log')
        ax2.grid(True, axis='y')
        
        # Rotate x-axis labels for better readability
        plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
        
        # Adjust layout
        fig.tight_layout()
        
        # Add figure to frame
        canvas = FigureCanvasTkAgg(fig, master=self.dpa_plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Add toolbar
        from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
        toolbar_frame = ttk.Frame(self.dpa_plot_frame)
        toolbar_frame.pack(fill=tk.X)
        toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
        toolbar.update()

def main():
    root = tk.Tk()
    app = DefectCalculatorApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
