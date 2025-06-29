"""
Moment Arm Analysis Module

This module provides functions for analyzing and comparing muscle moment arms
between generic and personalized OpenSim models. It can be used to generate
plots for specific coordinates and muscles. To be used with 5_optimize_muscles.ipynb.

Author Ekaterina Stansfield
29 June 2025


Usage:
    from moment_arm_analysis import MomentArmAnalyzer

    analyzer = MomentArmAnalyzer(gen_model_path, pers_model_path, output_folder)
    
    Case 1: to plot all coordinates:
    analyzer.analyze_all_coordinates()

    Case 2: to plot a specific coordinate:
    analyzer.load_data(num_points=50)
    analyzer.plot_specific_coordinate('hip_flexion_r')

    Case 3: to plot specific muscles for specific coordinates:
    analyzer.load_data(num_points=50)
    analyzer.plot_specific_muscles(['hip_flexion_r'], ['glut_max1_r', 'rect_fem_r'])
"""

import opensim as osim
import numpy as np
import os
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from typing import List, Dict, Optional, Tuple


class MomentArmAnalyzer:
    """
    A class for analyzing muscle moment arms in OpenSim models.
    
    Attributes:
        gen_model_path (str): Path to generic model
        pers_model_path (str): Path to personalized model
        output_folder (str): Output directory for plots
        gen_data (dict): Generic model moment arm data
        pers_data (dict): Personalized model moment arm data
        coordinate_muscles_dict (dict): Mapping of coordinates to muscles
    """
    
    def __init__(self, gen_model_path: str, pers_model_path: str, output_folder: str = "./output"):
        """
        Initialize the MomentArmAnalyzer.
        
        Args:
            gen_model_path: Path to the generic OpenSim model
            pers_model_path: Path to the personalized OpenSim model
            output_folder: Directory to save output plots
        """
        self.gen_model_path = gen_model_path
        self.pers_model_path = pers_model_path
        self.output_folder = output_folder
        
        # Create output folder if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)
        
        # Initialize data containers
        self.gen_data = None
        self.pers_data = None
        self.coordinate_muscles_dict = None
        
        # Load model structure
        self._initialize_model_structure()
    
    def _initialize_model_structure(self):
        """Initialize the model structure and coordinate-muscle mappings."""
        # Load generic model for structure analysis
        model = osim.Model(self.gen_model_path)
        state = model.initSystem()
        self.coordinate_muscles_dict = self.coordinate_muscles(model)
        print(f"Initialized analysis for {len(self.coordinate_muscles_dict)} coordinates")
    
    @staticmethod
    def coordinate_bodies(model: osim.Model) -> Dict[str, List[str]]:
        """Create a dictionary mapping coordinates to their associated bodies."""
        coords = [coord.getName() for coord in model.getCoordinateSet()]
        coordinate_bodies = {}
        
        for coord in coords:
            side = '_r' if '_r' in coord else '_l' if '_l' in coord else ''
            
            if 'pelvis' in coord and '_t' not in coord:
                coordinate_bodies[coord] = ['pelvis', 'torso']
            elif 'hip_' in coord and side:
                coordinate_bodies[coord] = ['pelvis', f'femur{side}']
            elif 'knee_' in coord and 'beta' not in coord and side:
                coordinate_bodies[coord] = [f'femur{side}', f'tibia{side}']
            elif 'knee_' in coord and 'beta' in coord and side:
                coordinate_bodies[coord] = [f'femur{side}', f'tibia{side}', f'patella{side}']
            elif 'ankle' in coord and side:
                coordinate_bodies[coord] = [f'tibia{side}']
                    
        return coordinate_bodies

    @staticmethod
    def muscle_bodies(model: osim.Model) -> Dict[str, List[str]]:
        """Create a dictionary mapping muscles to their attachment bodies."""
        # Get muscle names (excluding last 3 forces which aren't muscles)
        muscle_names = [force.getName() for force in model.get_ForceSet()][:-3]
        
        muscle_bodies_dict = {}
        for name in muscle_names:
            muscle = model.getMuscles().get(name)
            path_points = muscle.getGeometryPath().getPathPointSet()
            
            bodies = []
            for i in range(path_points.getSize()):
                point = osim.PathPoint.safeDownCast(path_points.get(i))
                if point:
                    bodies.append(point.getBody().getName())
            
            muscle_bodies_dict[name] = bodies
        
        return muscle_bodies_dict

    @staticmethod
    def muscle_coordinate_mapping(muscle_bodies_dict: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """Map muscles to coordinates based on body attachments."""
        muscle_coords = {}
        
        for muscle, bodies in muscle_bodies_dict.items():
            coords = []
            side = '_r' if '_r' in muscle else '_l' if '_l' in muscle else ''
            
            # Hip coordinates for muscles attached to pelvis
            if 'pelvis' in bodies and side:
                coords.extend([f'hip_flexion{side}', f'hip_adduction{side}', f'hip_rotation{side}'])
            
            # Knee coordinates for muscles crossing knee joint
            if side:  # Only process if we have a side
                femur = f'femur{side}'
                tibia = f'tibia{side}'
                calcn = f'calcn{side}'
                
                # Muscles spanning femur to calcn or femur to tibia (but not calcn)
                if ((femur in bodies and calcn in bodies) or 
                    (tibia in bodies and calcn not in bodies)):
                    coords.append(f'knee_angle{side}')
            
            muscle_coords[muscle] = coords
        
        return muscle_coords

    @classmethod
    def coordinate_muscles(cls, model: osim.Model) -> Dict[str, List[str]]:
        """Create a dictionary mapping coordinates to associated muscles."""
        # Get coordinate-body and muscle-body mappings
        coord_bodies_dict = cls.coordinate_bodies(model)
        muscle_bodies_dict = cls.muscle_bodies(model)
        muscle_coords_dict = cls.muscle_coordinate_mapping(muscle_bodies_dict)
        
        # Initialize coordinate-muscle dictionary
        coord_muscles = {coord: [] for coord in coord_bodies_dict.keys()}
        
        # Populate with muscles for each coordinate
        for muscle, coords in muscle_coords_dict.items():
            for coord in coords:
                if coord in coord_muscles:
                    coord_muscles[coord].append(muscle)
        
        return coord_muscles

    @staticmethod
    def create_empty_data_dict(coordinates: Optional[List[str]] = None) -> Dict[str, List]:
        """Create an empty data dictionary with standard lower limb coordinates."""
        if coordinates is None:
            coordinates = [
                'hip_flexion_r', 'hip_adduction_r', 'hip_rotation_r', 'knee_angle_r', 'ankle_angle_r',
                'hip_flexion_l', 'hip_adduction_l', 'hip_rotation_l', 'knee_angle_l', 'ankle_angle_l'
            ]
        return {coord: [] for coord in coordinates}

    def populate_data(self, model_path: str, coordinate_muscles_dictionary: Dict[str, List[str]], 
                     data_dict: Optional[Dict] = None, num_points: int = 50) -> Dict:
        """
        Populate data dictionary with moment arm data for each coordinate-muscle pair.
        
        Args:
            model_path: Path to the OpenSim model file
            coordinate_muscles_dictionary: Dictionary mapping coordinates to lists of muscle names
            data_dict: Dictionary to populate. If None, creates new one
            num_points: Number of points to sample across coordinate range
            
        Returns:
            Updated data dictionary with DataFrames for each coordinate
        """
        if data_dict is None:
            data_dict = self.create_empty_data_dict(list(coordinate_muscles_dictionary.keys()))
        
        model = osim.Model(model_path)
        state = model.initSystem()
        
        # Store original coordinate values
        original_values = {}
        coordinates = {}
        
        # Get all coordinates at once
        for coord_name in data_dict.keys():
            try:
                coord = model.getCoordinateSet().get(coord_name)
                coordinates[coord_name] = coord
                original_values[coord_name] = coord.getValue(state)
            except:
                print(f"Warning: Coordinate '{coord_name}' not found in model")
        
        # Process each coordinate
        for coord_name, muscle_list in coordinate_muscles_dictionary.items():
            if coord_name not in coordinates:
                continue
                
            coordinate = coordinates[coord_name]
            coord_values = np.linspace(coordinate.getRangeMin(), coordinate.getRangeMax(), num_points)
            
            # Prepare data structure
            df_data = {'coordinate_values': coord_values}
            
            # Get all muscles for this coordinate
            muscles = {}
            for muscle_name in muscle_list:
                try:
                    muscle = osim.Muscle.safeDownCast(model.getForceSet().get(muscle_name))
                    if muscle:
                        muscles[muscle_name] = muscle
                except:
                    continue
            
            # Calculate moment arms for all muscles at each coordinate value
            for muscle_name, muscle in muscles.items():
                moment_arms = []
                for value in coord_values:
                    coordinate.setValue(state, value)
                    model.realizePosition(state)
                    moment_arm = muscle.computeMomentArm(state, coordinate)
                    moment_arms.append(moment_arm)
                df_data[muscle_name] = moment_arms
            
            # Create DataFrame
            df = pd.DataFrame(df_data)
            data_dict[coord_name].append(df)
            
            # Reset coordinate
            coordinate.setValue(state, original_values[coord_name])
        
        return data_dict

    def load_data(self, num_points: int = 50):
        """Load moment arm data for both generic and personalized models."""
        print("Loading generic model data...")
        self.gen_data = self.create_empty_data_dict()
        self.gen_data = self.populate_data(self.gen_model_path, self.coordinate_muscles_dict, 
                                          self.gen_data, num_points)
        
        print("Loading personalized model data...")
        self.pers_data = self.create_empty_data_dict()
        self.pers_data = self.populate_data(self.pers_model_path, self.coordinate_muscles_dict, 
                                           self.pers_data, num_points)
        
        print("Data loading complete.")

    def plot_coordinate_comparison(self, coord_name: str, ncols: int = 4, 
                                 specific_muscles: Optional[List[str]] = None,
                                 save_plot: bool = True, show_plot: bool = True) -> Optional[plt.Figure]:
        """
        Plot comparison of generic vs personalized moment arms for a given coordinate.
        All subplots will use the same y-axis range based on the maximum range across all muscles.
        
        Args:
            coord_name: Name of coordinate to plot
            ncols: Number of columns in subplot grid
            specific_muscles: List of specific muscles to plot. If None, plots all muscles
            save_plot: Whether to save the plot to file
            show_plot: Whether to display the plot
            
        Returns:
            matplotlib Figure object if show_plot=False, None otherwise
        """
        # Personal model name
        model_name = self.pers_model_path.split('/')[-1].replace('.osim', '')

        # Check if data is loaded
        if self.gen_data is None or self.pers_data is None:
            print("Data not loaded. Call load_data() first.")
            return None
        
        # Check if coordinate exists in both datasets
        if coord_name not in self.gen_data or coord_name not in self.pers_data:
            print(f"Warning: Coordinate '{coord_name}' not found in data")
            return None
        
        # Check if data exists for this coordinate
        if not self.gen_data[coord_name] or not self.pers_data[coord_name]:
            print(f"Warning: No data available for coordinate '{coord_name}'")
            return None
        
        # Get the data tables
        gen_table = pd.concat(self.gen_data[coord_name], ignore_index=True)
        pers_table = pd.concat(self.pers_data[coord_name], ignore_index=True)
        
        # Check if tables have the same structure
        if gen_table.shape != pers_table.shape:
            print(f"Warning: Data shape mismatch for '{coord_name}'. Generic: {gen_table.shape}, Personal: {pers_table.shape}")
            return None
        
        # Get muscle columns (exclude coordinate values column)
        muscle_columns = [col for col in gen_table.columns if col != 'coordinate_values']
        
        # Filter for specific muscles if requested
        if specific_muscles:
            muscle_columns = [muscle for muscle in muscle_columns if muscle in specific_muscles]
        
        if not muscle_columns:
            print(f"Warning: No muscle data found for coordinate '{coord_name}'")
            return None
        
        # Calculate global y-axis range across all muscles
        all_values = []
        for muscle in muscle_columns:
            all_values.extend(gen_table[muscle].values)
            all_values.extend(pers_table[muscle].values)
        
        # Remove any NaN or infinite values
        all_values = [val for val in all_values if np.isfinite(val)]
        
        if not all_values:
            print(f"Warning: No valid moment arm data found for coordinate '{coord_name}'")
            return None
        
        # Calculate range with a small margin
        y_min = np.min(all_values)
        y_max = np.max(all_values)
        y_range = y_max - y_min
        margin = 0.05 * y_range if y_range > 0 else 0.1 * abs(y_max) if y_max != 0 else 0.1
        
        global_y_min = y_min - margin
        global_y_max = y_max + margin
        
        print(f"Global y-axis range for {coord_name}: [{global_y_min:.4f}, {global_y_max:.4f}] meters")
        
        # Calculate subplot layout
        cols = ncols
        rows = int(np.ceil(len(muscle_columns) / cols))
        
        # Create figure and subplots
        fig, axs = plt.subplots(rows, cols, figsize=(cols*3, rows*3))
        fig.suptitle(f'Muscle Moment Arms - {coord_name.replace("_", " ").title()}', fontsize=18, y=1)
        fig.subplots_adjust(top=0.9, bottom=0.1, wspace=0.3, hspace=0.4)
        
        # Handle single subplot case
        if rows == 1 and cols == 1:
            axs = [axs]
        elif rows == 1 or cols == 1:
            axs = axs.flatten()
        else:
            axs = axs.flatten()
        
        # Get x-axis data (coordinate values) and convert to degrees if rotational
        x_raw = gen_table['coordinate_values']
        
        # Determine if this is a rotational coordinate (convert to degrees) or translational (keep in meters)
        is_rotational = any(keyword in coord_name.lower() for keyword in ['flexion', 'adduction', 'rotation', 'angle'])
        
        if is_rotational:
            x = np.degrees(x_raw)  # Convert from radians to degrees
            x_label = 'Coordinate Value (degrees)'
            x_unit = '°'
        else:
            x = x_raw  # Keep in meters for translational coordinates
            x_label = 'Coordinate Value (m)'
            x_unit = 'm'
        
        print(f"Coordinate '{coord_name}' detected as {'rotational' if is_rotational else 'translational'}")
        print(f"X-axis range: {x.min():.2f} to {x.max():.2f} {x_unit}")
        
        # Plot each muscle
        for i, muscle in enumerate(muscle_columns):
            if i < len(axs):
                ax = axs[i]
                
                # Plot generic data
                ax.plot(x, gen_table[muscle], color='gray', label='Generic', 
                       linestyle='--', linewidth=2, alpha=0.8)
                
                # Plot personalized data
                ax.plot(x, pers_table[muscle], color='red', label='Personalized', 
                       linestyle='-', linewidth=2)
                
                # Set consistent y-axis limits for all subplots
                ax.set_ylim(global_y_min, global_y_max)
                
                # Format subplot
                ax.set_title(muscle.replace('_', ' ').title(), fontsize=12, pad=10)
                ax.set_xlabel(x_label, fontsize=10)
                ax.set_ylabel('Moment Arm (m)', fontsize=10)
                ax.grid(True, alpha=0.3)
                ax.tick_params(labelsize=9)
                
                # Add horizontal line at y=0 for reference
                ax.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=0.8)
        
        # Hide unused subplots
        for i in range(len(muscle_columns), len(axs)):
            axs[i].set_visible(False)
        
        # Create custom legend
        custom_legend = [
            Line2D([0], [0], color="gray", lw=2, linestyle="--", label="Generic"),
            Line2D([0], [0], color="red", lw=2, linestyle="-", label="Personalized"),
        ]
        
        fig.legend(
            handles=custom_legend,
            loc='upper center', 
            bbox_to_anchor=(0.5, 0.97),
            ncol=2, 
            fontsize=12,
            frameon=True,
            fancybox=True,
            shadow=True
        )
        
        plt.tight_layout()
        
        if save_plot:
            filename = f'{model_name}_muscle_moment_arms_{coord_name}'
            if specific_muscles:
                filename += f'_selected_muscles'
            plt.savefig(os.path.join(self.output_folder, f'{filename}.png'), dpi=300, bbox_inches='tight')
        
        if show_plot:
            plt.show()
            return None
        else:
            return fig
        

    def plot_specific_muscles(self, coordinates: List[str], muscles: List[str], 
                            save_plot: bool = True, show_plot: bool = True) -> Optional[plt.Figure]:
        """
        Plot specific muscles across multiple coordinates.
        
        Args:
            coordinates: List of coordinates to include
            muscles: List of specific muscles to plot
            save_plot: Whether to save the plot
            show_plot: Whether to display the plot
            
        Returns:
            matplotlib Figure object if show_plot=False, None otherwise
        """
        for coord in coordinates:
            fig = self.plot_coordinate_comparison(coord, specific_muscles=muscles, 
                                                save_plot=save_plot, show_plot=show_plot)
            if not show_plot:
                return fig
        return None

    def analyze_all_coordinates(self, coordinates_to_plot: Optional[List[str]] = None, 
                              ncols: int = 4, num_points: int = 50):
        """
        Analyze and plot all specified coordinates.
        
        Args:
            coordinates_to_plot: List of coordinates to plot. If None, plots all available
            ncols: Number of columns in subplot grid
            num_points: Number of points for moment arm calculation
        """
        # Load data if not already loaded
        if self.gen_data is None or self.pers_data is None:
            self.load_data(num_points)
        
        if coordinates_to_plot is None:
            # Use coordinates that exist in both datasets
            coordinates_to_plot = list(set(self.gen_data.keys()) & set(self.pers_data.keys()))
            coordinates_to_plot.sort()
        
        # Filter coordinates to only include those that actually have data
        available_coordinates = []
        for coord in coordinates_to_plot:
            if (coord in self.gen_data and coord in self.pers_data and 
                self.gen_data[coord] and self.pers_data[coord]):
                available_coordinates.append(coord)
            else:
                print(f"Skipping {coord} - no data available")
        
        print(f"Available coordinates for plotting: {available_coordinates}")
        
        # Plot all available coordinates
        if available_coordinates:
            print(f"Plotting {len(available_coordinates)} coordinates...")
            for i, coord in enumerate(available_coordinates):
                print(f"Plotting {i+1}/{len(available_coordinates)}: {coord}")
                try:
                    self.plot_coordinate_comparison(coord, ncols=ncols)
                except Exception as e:
                    print(f"Error plotting {coord}: {e}")
                    continue
        else:
            print("No coordinates available for plotting. Check your data generation step.")

    def get_available_coordinates(self) -> List[str]:
        """Get list of available coordinates in the loaded data."""
        if self.coordinate_muscles_dict:
            return list(self.coordinate_muscles_dict.keys())
        return []

    def get_muscles_for_coordinate(self, coord_name: str) -> List[str]:
        """Get list of muscles associated with a specific coordinate."""
        if self.coordinate_muscles_dict and coord_name in self.coordinate_muscles_dict:
            return self.coordinate_muscles_dict[coord_name]
        return []

    def print_model_summary(self):
        """Print a summary of the model structure."""
        if self.coordinate_muscles_dict:
            print("\nModel Structure Summary:")
            print("=" * 50)
            for coord, muscles in self.coordinate_muscles_dict.items():
                print(f"{coord}: {len(muscles)} muscles")
                if muscles:
                    print(f"  Example muscles: {', '.join(muscles[:3])}")
                    if len(muscles) > 3:
                        print(f"  ... and {len(muscles) - 3} more")
                print()


def main():
    """Example usage of the MomentArmAnalyzer."""
    # Define model paths
    gen_model_path = "../final_results/generic_scaled/scaled_model.osim"
    pers_model_path = "../final_results/personalized/tps_skin_wrp_updated.osim"
    output_folder = "../final_results/personalized/"
    
    # Create analyzer
    analyzer = MomentArmAnalyzer(gen_model_path, pers_model_path, output_folder)
    
    # Print model summary
    analyzer.print_model_summary()
    
    # Example 1: Analyze all coordinates
    analyzer.analyze_all_coordinates()
    
    # Example 2: Plot specific coordinate
    # analyzer.load_data(num_points=50)
    # analyzer.plot_coordinate_comparison('hip_flexion_r')
    
    # Example 3: Plot specific muscles for specific coordinates
    # analyzer.load_data(num_points=50)
    # specific_muscles = ['glut_max1_r', 'rect_fem_r', 'semimem_r']
    # analyzer.plot_specific_muscles(['hip_flexion_r', 'knee_angle_r'], specific_muscles)
    
    # Example 4: Get available data
    # print("Available coordinates:", analyzer.get_available_coordinates())
    # print("Muscles for hip_flexion_r:", analyzer.get_muscles_for_coordinate('hip_flexion_r'))


if __name__ == "__main__":
    main()