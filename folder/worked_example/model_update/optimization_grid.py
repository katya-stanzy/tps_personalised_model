import numpy as np
import opensim as osim
import os
import logging
import time
from scipy.optimize import minimize


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s -  %(message)s'
)
logger = logging.getLogger('AdjustPersonal')

class AdjustPersonal:
    """
    Class to adjust muscle path points in a personalized OpenSim model to match
    moment arm profiles from a generic model using gradient-based optimization.
    """
    def __init__(self, gen_model_path, pers_model_path, error_threshold=0.003, shift_bounds = (-0.01, 0.01),
                learning_rate=0.001, max_iterations=50,
                convergence_threshold=0.001, regularization_weight=0.5, n_total=50):
            # Validate file existence
        if not os.path.exists(gen_model_path):
            raise FileNotFoundError(f"Generic model not found: {gen_model_path}")
        if not os.path.exists(pers_model_path):
            raise FileNotFoundError(f"Personalized model not found: {pers_model_path}")

        self.logger = logging.getLogger(f'AdjustPersonal')
        """
        Initialize with paths to generic and personalized models.
        
        Parameters:
            gen_model_path (str): Path to the generic OpenSim model
            pers_model_path (str): Path to the personalized OpenSim model
            error_threshold (float): Threshold for moment arm error (meters)
            learning_rate (float): Learning rate for gradient descent
            max_iterations (int): Maximum iterations for optimization
            convergence_threshold (float): Convergence threshold for optimization
            regularization_weight (float): Weight for regularization term
        """
        self.gen_model_path = gen_model_path
        self.pers_model_path = pers_model_path
        self.error_threshold = error_threshold
        self.shift_bounds = shift_bounds
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.regularization_weight = regularization_weight
        self.n_total = n_total

        # Load models
        self.logger.info(f"Loading generic model from {gen_model_path}")
        self.gen = osim.Model(gen_model_path)
        self.gen_state = self.gen.initSystem()
        
        self.logger.info(f"Loading personalized model from {pers_model_path}")
        self.pers = osim.Model(pers_model_path)
        self.pers_state = self.pers.initSystem()
        
        # Extract model information
        self.setup_model_info()
        
        # Configure optimization parameters
        self.setup_optimization_params()
        
        self.logger.info("AdjustPersonal initialized successfully")
    
    def setup_model_info(self):
        """Extract and cache model information."""
        # Get forces and coordinates
        self.mscl_names = [force.getName() for force in self.gen.get_ForceSet()]
        self.coords = [coord.getName() for coord in self.gen.getCoordinateSet()]
        
        # Map coordinates to bodies and muscles to bodies
        self.coordinate_bodies_dictionary = self._build_coordinate_bodies_map()
        self.muscle_bodies_dictionary = self._build_muscle_bodies_map()
    
    def setup_optimization_params(self):
        """Configure parameters for the optimization process."""      
        # Set step size for numerical gradient computation
        self.gradient_step = 0.001  # 1mm step for gradient computation
    
    def _build_coordinate_bodies_map(self):
        """
        Create a dictionary mapping coordinates to the bodies they affect.
        
        Returns:
            dict: Dictionary with coordinate names as keys and lists of body names as values
        """
        
        coordinate_bodies = {}
        
        for coord in self.coords:
            # Handle different coordinate types
            if 'pelvis' in coord and '_t' not in coord:
                coordinate_bodies[coord] = ['pelvis', 'torso']
            elif 'hip_' in coord:
                if '_r' in coord:
                    coordinate_bodies[coord] = ['pelvis', 'femur_r']
                elif '_l' in coord:
                    coordinate_bodies[coord] = ['pelvis', 'femur_l']
            elif 'knee_' in coord and 'beta' not in coord:
                if '_r' in coord:
                    coordinate_bodies[coord] = ['femur_r', 'tibia_r', 'patella_r']
                elif '_l' in coord:
                    coordinate_bodies[coord] = ['femur_l', 'tibia_l', 'patella_l']
            elif 'knee_' in coord and 'beta' in coord:
                if '_r' in coord:
                    coordinate_bodies[coord] = ['femur_r', 'tibia_r', 'patella_r']
                elif '_l' in coord:
                    coordinate_bodies[coord] = ['femur_l', 'tibia_l', 'patella_l']
            elif 'ankle' in coord:
                if '_r' in coord:
                    coordinate_bodies[coord] = ['tibia_r']
                elif '_l' in coord:
                    coordinate_bodies[coord] = ['tibia_l']
        
        return coordinate_bodies

    def _build_muscle_bodies_map(self):
        """
        Create a dictionary mapping muscles to the bodies they attach to.
        
        Returns:
            dict: Dictionary with muscle names as keys and lists of body names as values
        """
        # Get muscle names excluding the last 3 (which might be actuators)
        mscl_names = self.mscl_names[:-3]
        
        muscle_bodies = {}
        
        for muscle_name in mscl_names:
            muscle_bodies[muscle_name] = []
            muscle = self.gen.getMuscles().get(muscle_name)
            path = muscle.getGeometryPath()
            path_points = path.getPathPointSet()

            # Collect unique bodies this muscle attaches to
            bodies_seen = set()
            for i in range(path_points.getSize()):
                point = osim.PathPoint.safeDownCast(path_points.get(i))
                if not point:
                    continue
                
                body_name = point.getBody().getName()
                if body_name not in bodies_seen:
                    muscle_bodies[muscle_name].append(body_name)
                    bodies_seen.add(body_name)
        
        return muscle_bodies

    def process_muscles(self, muscle_list=None):
        """
        Process a list of muscles, adjusting their moment arms.
        
        Parameters:
            muscle_list (list, optional): List of muscle names to process.
                                        If None, process all muscles.
        """
        # If no muscle list is provided, use all muscles except the last 3
        if muscle_list is None:
            muscle_list = self.mscl_names[:-3]
        
        total_muscles = len(muscle_list)
        processed = 0
        
        for muscle_name in muscle_list:
            processed += 1
            self.logger.info(f" ------------ Processing muscle {processed}/{total_muscles}: {muscle_name} ----------------")
            
            # IMPORTANT: Reinitialize state at the start of each muscle
            self.pers_state = self.pers.initSystem()
            
            if muscle_name not in self.muscle_bodies_dictionary:
                self.logger.warning(f"Muscle {muscle_name} not found in muscle bodies dictionary. Skipping.")
                continue
            
            # Process muscle based on its attachments
            self._process_muscle_by_attachments(muscle_name)
            
            # IMPORTANT: Finalize model changes and reinitialize state after each muscle
            self.pers.finalizeFromProperties()
            self.pers_state = self.pers.initSystem()

    def _process_muscle_by_attachments(self, muscle_name):
        """
        Process a muscle based on its body attachments.
        
        Parameters:
            muscle_name (str): Name of the muscle to process
        """
        muscle_bodies = self.muscle_bodies_dictionary[muscle_name]
        
        # Hip muscles (attached to pelvis)
        if 'pelvis' in muscle_bodies:
            self.logger.info(f"Processing hip joint muscle: {muscle_name}")
            self._process_hip_muscle(muscle_name)

        # right patella
        if 'patella_r' in muscle_bodies:
            self.logger.info(f"Processing patella (right): {muscle_name}")
            self.process_one_muscle('knee_angle_r', muscle_name)#('knee_angle_r_beta', muscle_name) 
        
        # Right knee muscles
        if ('patella_r' not in muscle_bodies) and ('femur_r' in muscle_bodies) and ('calcn_r' in muscle_bodies):
            self.logger.info(f"Processing knee joint muscle (right): {muscle_name}")
            self.process_one_muscle('knee_angle_r', muscle_name)
        
        # Muscles crossing only right knee
        if ('patella_r' not in muscle_bodies) and ('tibia_r' in muscle_bodies) and ('calcn_r' not in muscle_bodies):
            self.logger.info(f"Processing knee-only muscle (right): {muscle_name}")
            self.process_one_muscle('knee_angle_r', muscle_name)

        # Right ankle muscles
        if 'tibia_r' in muscle_bodies and 'calcn_r' in muscle_bodies:
            self.logger.info(f"Processing ankle muscle (right): {muscle_name}")
            self.process_one_muscle('ankle_angle_r', muscle_name)
        
        # Left patella
        if 'patella_l' in muscle_bodies:
            self.logger.info(f"Processing patella (left): {muscle_name}")
            self.process_one_muscle('knee_angle_l', muscle_name) #('knee_angle_l_beta', muscle_name) 
    
       # Left knee muscles
        if ('patella_l' not in muscle_bodies) and ('femur_l' in muscle_bodies) and ('calcn_l' in muscle_bodies):
            self.logger.info(f"Processing knee joint muscle (left): {muscle_name}")
            self.process_one_muscle('knee_angle_l', muscle_name)           
        
        # Muscles crossing only left knee
        if ('patella_l' not in muscle_bodies) and ('tibia_l' in muscle_bodies) and ('calcn_l' not in muscle_bodies):
            self.logger.info(f"Processing knee-only muscle (left): {muscle_name}")
            self.process_one_muscle('knee_angle_l', muscle_name)
        
        # Left ankle muscles
        if 'tibia_l' in muscle_bodies and 'calcn_l' in muscle_bodies:
            self.logger.info(f"Processing ankle muscle (left): {muscle_name}")
            self.process_one_muscle('ankle_angle_l', muscle_name)
    
    def _process_hip_muscle(self, muscle_name):
        """
        Process a hip muscle that may affect multiple coordinates.
        
        Parameters:
            muscle_name (str): Name of the hip muscle
        """
        try:
            # Determine side (left or right)
            if muscle_name[-1] == 'l':
                three_coordinates_list = ['hip_rotation_l', 'hip_flexion_l', 'hip_adduction_l']
            elif muscle_name[-1] == 'r':
                three_coordinates_list = ['hip_rotation_r', 'hip_flexion_r', 'hip_adduction_r']
            else:
                self.logger.warning(f"Cannot determine side for muscle {muscle_name}, skipping")
                return
            
            self.logger.info(f"Processing hip muscle {muscle_name} affecting coordinates: {three_coordinates_list}")
            
            # Check if the muscle attaches to the pelvis
            if 'pelvis' in self.muscle_bodies_dictionary.get(muscle_name, []):
                self._process_multi_coordinate_muscle(three_coordinates_list, muscle_name)
            else:
                self.logger.info(f"Muscle {muscle_name} does not attach to pelvis, skipping")
        
        except Exception as e:
            self.logger.error(f"Error processing hip muscle {muscle_name}: {e}")

    def process_one_muscle(self, coordinate_name, muscle_name):
        """
        Process a single muscle for a single coordinate.
        
        Parameters:
            coordinate_name (str): Name of the coordinate to process
            muscle_name (str): Name of the muscle to process
        """
        try:
            # Initialize the model state
            self.pers_state = self.pers.initSystem()
            
            self.logger.info(f"Processing muscle {muscle_name} for coordinate {coordinate_name}")
            
            # Get model objects and calculate moment arms
            gen_muscle = osim.Muscle.safeDownCast(self.gen.getForceSet().get(muscle_name))
            gen_coordinate = self.gen.getCoordinateSet().get(coordinate_name)
            
            pers_muscle = osim.Muscle.safeDownCast(self.pers.getForceSet().get(muscle_name))
            pers_coordinate = self.pers.getCoordinateSet().get(coordinate_name)
            
            # Get coordinate range and calculate moment arms
            coord_values = self._get_coordinate_range(gen_coordinate)
            
            gen_moment_arms = self._calculate_moment_arms(
                self.gen, self.gen_state, gen_coordinate, gen_muscle, coord_values)
            
            pers_moment_arms = self._calculate_moment_arms(
                self.pers, self.pers_state, pers_coordinate, pers_muscle, coord_values)
            
            # Check if adjustments are needed
            diff_value, drop = self._compare_moment_arms(gen_moment_arms, pers_moment_arms)
            self.logger.info(f"Initial error: {diff_value:.6f}, drop: {drop}")
            
            # Determine if the muscle needs adjustment
            needs_adjustment = diff_value >= self.error_threshold or drop == 1
            
            if needs_adjustment:
                # Find attachment points that can be adjusted
                adjustable_points = self._find_adjustable_points(coordinate_name, muscle_name)
                
                if adjustable_points:
                    self.logger.info(f"Optimizing muscle path points for {muscle_name} on coordinate {coordinate_name}")
                    
                    # Perform optimization on the most promising point
                    for point in adjustable_points:  # Start with the first point
                        self._optimize_single_point(
                            point, coordinate_name, muscle_name, gen_moment_arms, coord_values)
                        
                else:
                    self.logger.info(f"No adjustable attachment points found for {muscle_name} on coordinate {coordinate_name}")
            else:
                self.logger.info(f"No adjustments needed for {muscle_name} on coordinate {coordinate_name}")
                
        except Exception as e:
            self.logger.error(f"Error processing muscle {muscle_name} for coordinate {coordinate_name}: {e}")
    
    def _find_adjustable_points(self, coordinate_name, muscle_name):
        """
        Find points that can be adjusted for this coordinate.
        
        Parameters:
            coordinate_name (str): Name of the coordinate
            muscle_name (str): Name of the muscle
            
        Returns:
            list: Indices of adjustable points
        """
        adjustable_points = []
        
        # Get the muscle's path points
        pers_muscle = self.pers.getMuscles().get(muscle_name)
        path = pers_muscle.getGeometryPath()
        path_points = path.getPathPointSet()
        
        # Find points attached to relevant bodies
        for i in range(path_points.getSize()):
            point = osim.PathPoint.safeDownCast(path_points.get(i))
            if not point:
                continue
            
            body_name = point.getBody().getName()
            if body_name in self.coordinate_bodies_dictionary.get(coordinate_name, []):
                adjustable_points.append(i)
        
        return adjustable_points
    
    def _process_multi_coordinate_muscle(self, coordinate_list, muscle_name):
        """
        Process one muscle that affects multiple coordinates.
        
        Parameters:
            coordinate_list (list): List of coordinate names
            muscle_name (str): Name of the muscle to process
        """
        self.logger.info(f"Processing muscle {muscle_name} for coordinates: {coordinate_list}")
        
        try:
            # Initialize model state
            self.pers_state = self.pers.initSystem()
            
            # Get generic muscle
            gen_muscle = osim.Muscle.safeDownCast(self.gen.getForceSet().get(muscle_name))
            pers_muscle = osim.Muscle.safeDownCast(self.pers.getForceSet().get(muscle_name))
            
            # Data structures to hold coordinate information
            gen_coords = []
            coord_ranges = []
            gen_moment_arms = []
            pers_moment_arms = []
            errors = []
            
            # Analyze each coordinate
            for coord_name in coordinate_list:
                # Get coordinate objects
                gen_coord = self.gen.getCoordinateSet().get(coord_name)
                pers_coord = self.pers.getCoordinateSet().get(coord_name)
                
                # Calculate coordinate range
                coord_range = self._get_coordinate_range(gen_coord)
                
                # Calculate moment arms
                gen_ma = self._calculate_moment_arms(
                    self.gen, self.gen_state, gen_coord, gen_muscle, coord_range)
                
                pers_ma = self._calculate_moment_arms(
                    self.pers, self.pers_state, pers_coord, pers_muscle, coord_range)
                
                # Compare moment arms
                diff_value, drop = self._compare_moment_arms(gen_ma, pers_ma)
                
                # Store results
                gen_coords.append(gen_coord)
                coord_ranges.append(coord_range)
                gen_moment_arms.append(gen_ma)
                pers_moment_arms.append(pers_ma)
                errors.append((diff_value, drop))
            
            # Extract error magnitudes and drop flags
            error_magnitudes = [abs(er) for er, drop in errors]
            drop_flags = [drop for er, drop in errors]
            
            self.logger.info(f"Original errors: {error_magnitudes}")
            self.logger.info(f"Original drops: {drop_flags}")
            
            # Determine if adjustment is needed
            needs_adjustment = (any(x >= self.error_threshold for x in error_magnitudes) or 
                            any(x == 1 for x in drop_flags))
            
            if needs_adjustment:
                self.logger.info("Adjustments needed, finding common adjustable points")
                
                # Find common adjustable points for all coordinates
                adjustable_points = self._find_common_adjustable_points(coordinate_list, muscle_name)
                
                if adjustable_points:
                    self.logger.info(f"Found {len(adjustable_points)} common adjustable points")
                    
                    # Store best results
                    best_error_sum = float('inf')
                    best_drop_count = sum(drop_flags)  # Start with all drops
                    best_point_index = None
                    
                    # Try each adjustable point and keep track of the best one
                    #pers_muscle = self.pers.getMuscles().get(muscle_name)

                    for point_index in adjustable_points:

                        self.logger.info(f"Attempting to optimize point {point_index}")
                        
                        # Create a deep copy of the current model state
                        model_copy = self.pers.clone()#osim.Model(self.pers_model_path)
                        model_state = model_copy.initSystem()
                        
                        # Optimize this point
                        try:
                            self._optimize_multi_point(
                                point_index, coordinate_list, muscle_name, 
                                gen_moment_arms, coord_ranges, error_magnitudes)
                            
                            # IMPORTANT: Reinitialize the state after optimization
                            self.pers_state = self.pers.initSystem()
                            
                            # Re-evaluate to see if this improved things
                            new_error_sum = 0
                            new_drop_count = 0
                            
                            for i, coord_name in enumerate(coordinate_list):
                                # Get updated moment arms
                                pers_coord = self.pers.getCoordinateSet().get(coord_name)
                                pers_muscle = osim.Muscle.safeDownCast(self.pers.getForceSet().get(muscle_name))
                                
                                pers_ma = self._calculate_moment_arms(
                                    self.pers, self.pers_state, pers_coord, pers_muscle, coord_ranges[i])
                                
                                # Calculate new error
                                new_diff, new_drop = self._compare_moment_arms(gen_moment_arms[i], pers_ma)
                                new_error_sum += abs(new_diff)
                                new_drop_count += new_drop
                            
                            self.logger.info(f"After optimizing point {point_index}: " 
                                            f"error_sum={new_error_sum}, drop_count={new_drop_count}")
                            
                            # Get displacement from original position
                            pers_muscle = self.pers.getMuscles().get(muscle_name)
                            path = pers_muscle.getGeometryPath()
                            path_points = path.getPathPointSet()
                            point = osim.PathPoint.safeDownCast(path_points.get(point_index))
                            new_loc = point.getLocation(self.pers_state)
                            new_location = np.array([new_loc.get(0), new_loc.get(1), new_loc.get(2)])
                            
                            # Get original location
                            model_muscle = model_copy.getMuscles().get(muscle_name)
                            model_path = model_muscle.getGeometryPath()
                            model_points = model_path.getPathPointSet()
                            model_point = osim.PathPoint.safeDownCast(model_points.get(point_index))
                            orig_loc = model_point.getLocation(model_state)
                            orig_location = np.array([orig_loc.get(0), orig_loc.get(1), orig_loc.get(2)])
                            
                            displacement = np.linalg.norm(new_location - orig_location)
                            
                            # Check if this is better than our best so far
                            # Prioritize drop reduction, but also consider proximity to original position
                            is_better = False
                            
                            # Major improvement: drop count reduced
                            if new_drop_count < best_drop_count:
                                is_better = True
                                self.logger.info(f"Point {point_index} reduces drops from {best_drop_count} to {new_drop_count}")
                            
                            # Same drop count but better error
                            elif new_drop_count == best_drop_count:
                                # If we've already found a point with the same drop count,
                                # prefer the one with better error or the one closer to original position
                                if best_point_index is not None:
                                    # If error is significantly better (>5%), accept it
                                    if new_error_sum < best_error_sum * 0.95:
                                        is_better = True
                                        self.logger.info(f"Point {point_index} has significantly better error")
                                    # If error is similar but point is closer to original, accept it
                                    elif new_error_sum < best_error_sum * 1.05 and displacement < 0.7 * (self.shift_bounds[1] - self.shift_bounds[0]):
                                        is_better = True
                                        self.logger.info(f"Point {point_index} has similar error but is closer to original position")
                                else:
                                    is_better = True
                            
                            if is_better:
                                best_error_sum = new_error_sum
                                best_drop_count = new_drop_count
                                best_point_index = point_index
                                self.logger.info(f"New best point: {point_index} (displacement: {displacement:.6f}m)")
                                # Keep this version of the model
                            else:
                                # Restore from copy if this point wasn't better
                                self.pers = model_copy
                                self.pers_state = model_state
                                self.logger.info(f"Point {point_index} was not better, reverting changes")
                        
                        except Exception as e:
                            self.logger.error(f"Error optimizing point {point_index}: {e}")
                            # Restore from copy on error
                            self.pers = model_copy
                            self.pers_state = model_state
                    
                    # Report final results
                    if best_point_index is not None:
                        # Make sure we've finalized from properties and reinitialized the state
                        self.pers.finalizeFromProperties()
                        self.pers_state = self.pers.initSystem()
                        
                        self.logger.info(f"Best adjustment was at point {best_point_index} with "
                                        f"error_sum={best_error_sum}, drop_count={best_drop_count}")
                        
                        # Compare to original
                        orig_error_sum = sum(error_magnitudes)
                        orig_drop_count = sum(drop_flags)
                        
                        if best_error_sum < orig_error_sum or best_drop_count < orig_drop_count:
                            self.logger.info("Successfully reduced error or eliminated drops")
                        else:
                            self.logger.warning("Could not improve on original error values")
                    else:
                        self.logger.warning("No improvement found across all adjustable points")
                else:
                    self.logger.info("No common adjustable points found")
            else:
                self.logger.info("No adjustments needed for this muscle")
                
        except Exception as e:
            self.logger.error(f"Error in _process_multi_coordinate_muscle for {muscle_name}: {e}")

    def _find_common_adjustable_points(self, coordinate_list, muscle_name):
        """
        Find points that can be adjusted for all coordinates.
        
        Parameters:
            coordinate_list (list): List of coordinate names
            muscle_name (str): Name of the muscle
            
        Returns:
            list: Indices of adjustable points
        """
        # Get the muscle's path points
        pers_muscle = self.pers.getMuscles().get(muscle_name)
        path = pers_muscle.getGeometryPath()
        path_points = path.getPathPointSet()
        
        # Find points attached to relevant bodies for all coordinates
        common_points = []
        
        for i in range(path_points.getSize()):
            point = osim.PathPoint.safeDownCast(path_points.get(i))
            if not point:
                continue
            
            body_name = point.getBody().getName()
            
            # Check if this point is relevant for all coordinates
            relevant_for_all = True
            for coord_name in coordinate_list:
                if body_name not in self.coordinate_bodies_dictionary.get(coord_name, []):
                    relevant_for_all = False
                    break
            
            if relevant_for_all:
                common_points.append(i)
        
        return common_points
    
    def _calculate_curve_shape_error(self, gen_moment_arms, pers_moment_arms):
        """
        Calculate the error in curve shape between generic and personalized moment arms.
        
        Parameters:
            gen_moment_arms (list): Generic model moment arms
            pers_moment_arms (list): Personalized model moment arms
            
        Returns:
            float: Error value representing the difference in curve shape
        """
        # Normalize both curves to focus on shape rather than absolute values
        if max(gen_moment_arms) - min(gen_moment_arms) == 0:
            gen_normalized = np.zeros_like(gen_moment_arms)
        else:
            gen_normalized = (np.array(gen_moment_arms) - min(gen_moment_arms)) / (max(gen_moment_arms) - min(gen_moment_arms))
            
        if max(pers_moment_arms) - min(pers_moment_arms) == 0:
            pers_normalized = np.zeros_like(pers_moment_arms)
        else:
            pers_normalized = (np.array(pers_moment_arms) - min(pers_moment_arms)) / (max(pers_moment_arms) - min(pers_moment_arms))
        
        # Calculate RMS difference between normalized curves
        diff_squared = np.sum((gen_normalized - pers_normalized) ** 2)
        rms_diff = np.sqrt(diff_squared / len(gen_moment_arms))
        
        return rms_diff #5.0 * rms_diff  # Weight factor to make this error component comparable to others

    def _get_optimization_bounds(self, muscle_name, point_index, orig_location, total_points):
        """
        Get optimization bounds for a specific muscle and point.
        
        Parameters:
            muscle_name (str): Name of the muscle
            point_index (int): Index of the point being optimized
            orig_location (np.array): Original location of the point
            total_points (int): Total number of points in the muscle path
            
        Returns:
            list: List of (min, max) bounds for each coordinate
        """
        # Default bounds
        default_bounds = [(orig_location[i] + self.shift_bounds[0]/2, 
                        orig_location[i] + self.shift_bounds[1]/2) 
                        for i in range(3)]
        
        # Special muscle-specific bounds (most restrictive first)
        if any(muscle in muscle_name for muscle in ['gasmed', 'gaslat']) and point_index == 0:
            return [(orig_location[i] + self.shift_bounds[0], 
                    orig_location[i] + self.shift_bounds[1]) 
                for i in range(3)]
        
        # Quadriceps and hamstring endpoint restrictions
        elif (any(muscle in muscle_name for muscle in ['recfem', 'vasint', 'vasmed', 'vaslat', 'bflh', 'bfsh']) 
            and (point_index == 0 or point_index == total_points - 1)):
            return [(orig_location[i] + self.shift_bounds[0] / 4, 
                    orig_location[i] + self.shift_bounds[1] / 4) 
                for i in range(3)]
        
        # General endpoint restrictions (for all other muscles)
        elif point_index == 0 or point_index == total_points - 1:
            return [(orig_location[i] + self.shift_bounds[0] / 2, 
                    orig_location[i] + self.shift_bounds[1] / 2) 
                for i in range(3)]
        
        # Default case
        else:
            return default_bounds

    def _optimize_single_point(self, point_index, coordinate_name, muscle_name, 
                                gen_moment_arms, coord_values):
        """
        Optimize a single point location using simplified optimization approach.
        
        Parameters:
            point_index (int): Index of the point to adjust
            coordinate_name (str): Name of the coordinate
            muscle_name (str): Name of the muscle
            gen_moment_arms (list): Generic model moment arms
            coord_values (list): Coordinate values
        """
        self.logger.info(f"***Optimizing point {point_index} for {muscle_name} on coordinate {coordinate_name}***")
        
        # IMPORTANT: Ensure state is initialized before we start
        self.pers_state = self.pers.initSystem()
        
        # Get the point's original location
        pers_muscle = self.pers.getMuscles().get(muscle_name)
        path = pers_muscle.getGeometryPath()
        path_points = path.getPathPointSet()
        point = osim.PathPoint.safeDownCast(path_points.get(point_index))
        
        location = point.getLocation(self.pers_state)
        orig_location = np.array([location.get(0), location.get(1), location.get(2)])
        
        # Get initial error and drop values
        pers_coord = self.pers.getCoordinateSet().get(coordinate_name)
        initial_moment_arms = self._calculate_moment_arms(
            self.pers, self.pers_state, pers_coord, pers_muscle, coord_values)
        initial_ma_diff, initial_drop = self._compare_moment_arms(gen_moment_arms, initial_moment_arms)
        self.logger.info(f"Initial m.a.error: {initial_ma_diff:.6f}, initial drop: {initial_drop}")
        
        # Define objective function
        def objective_function(params):
            """Objective function with weighted components."""
            # Create a new model to avoid state conflicts
            pers_copy = self.pers.clone()#osim.Model(self.pers_model_path)
            
            # Apply the point shift
            pers_muscle_copy = pers_copy.getMuscles().get(muscle_name)
            path_copy = pers_muscle_copy.getGeometryPath()
            path_points_copy = path_copy.getPathPointSet()
            point_copy = osim.PathPoint.safeDownCast(path_points_copy.get(point_index))
            
            # Set new point location
            point_copy.setLocation(osim.Vec3(params[0], params[1], params[2]))
            pers_copy.finalizeFromProperties()
            
            # Calculate moment arms
            state_copy = pers_copy.initSystem()
            coordinate_copy = pers_copy.getCoordinateSet().get(coordinate_name)
            muscle_copy = osim.Muscle.safeDownCast(pers_copy.getForceSet().get(muscle_name))
            
            moment_arms = self._calculate_moment_arms(
                pers_copy, state_copy, coordinate_copy, muscle_copy, coord_values)
            
            # Calculate error components
            error, drop = self._compare_moment_arms(gen_moment_arms, moment_arms)
            max_error = self._max_moment_arms_error(gen_moment_arms, moment_arms)
            
            # Weighted penalties for different error components
            error_penalty = abs(error)
            drop_penalty = 0 if drop == 0 else 10.0
            
            # Regularization term (penalize large movements)
            shift_magnitude = (np.sum((params - orig_location)**2))**0.5
            reg_term = shift_magnitude * self.regularization_weight
            
            # Add curve shape error
            curve_shape_error = self._calculate_curve_shape_error(gen_moment_arms, moment_arms)
            
            return error_penalty + max_error + drop_penalty + reg_term + curve_shape_error*0.05
        
        # Define bounds
        P = path_points.getSize()
        bounds = self._get_optimization_bounds(muscle_name, point_index, orig_location, P)
        
        # STEP 1: Simplified grid search with maximum 50 points
        self.logger.info("Starting simplified grid search")
        
        # We'll use a non-uniform grid with focus on the z-axis
        # Calculate dimensions to keep under n_total points
        n_total = self.n_total
        n_z = min(5, np.ceil(n_total ** 1/3).astype(int))  # More resolution in z
        #remaining = np.ceil((n_total // n_z)**0.5).astype(int)  # Roughly equal for x and y
        n_x = n_z#min(5, remaining)
        n_y = n_z#min(5, remaining)
        
        self.logger.info(f"Grid dimensions: {n_x}x{n_y}x{n_z} = {n_x*n_y*n_z} points")
        
        x_vals = np.linspace(bounds[0][0], bounds[0][1], n_x)
        y_vals = np.linspace(bounds[1][0], bounds[1][1], n_y)
        z_vals = np.linspace(bounds[2][0], bounds[2][1], n_z)
        
        best_point = orig_location.copy()
        orig_error = objective_function(orig_location)
        best_error = orig_error
        best_moment_arms = initial_moment_arms
        best_error_val, best_drop = initial_ma_diff, initial_drop
        self.logger.info(f"Original obj.funct.error {orig_error}")
        
        # Evaluate grid points
        total_points = n_x * n_y * n_z
        evaluated = 0
        
        for x in x_vals:
            for y in y_vals:
                for z in z_vals:
                    params = np.array([x, y, z])
                    error = objective_function(params)
                    
                    evaluated += 1
                    if evaluated % 5 == 0 or evaluated == total_points:
                        self.logger.info(f"Grid search progress: {evaluated}/{total_points}")
                    
                    if error < best_error:
                        best_error = error
                        best_point = params
                        
                        # Calculate moment arms and drop for this point
                        pers_copy = self.pers.clone() # osim.Model(self.pers_model_path)
                        pers_muscle_copy = pers_copy.getMuscles().get(muscle_name)
                        path_copy = pers_muscle_copy.getGeometryPath()
                        path_points_copy = path_copy.getPathPointSet()
                        point_copy = osim.PathPoint.safeDownCast(path_points_copy.get(point_index))
                        point_copy.setLocation(osim.Vec3(params[0], params[1], params[2]))
                        pers_copy.finalizeFromProperties()
                        state_copy = pers_copy.initSystem()
                        coordinate_copy = pers_copy.getCoordinateSet().get(coordinate_name)
                        muscle_copy = osim.Muscle.safeDownCast(pers_copy.getForceSet().get(muscle_name))
                        moment_arms = self._calculate_moment_arms(
                            pers_copy, state_copy, coordinate_copy, muscle_copy, coord_values)
                        best_ma_diff_val, best_drop = self._compare_moment_arms(gen_moment_arms, moment_arms)
                        best_moment_arms = moment_arms

                        self.logger.info(f"New best point: {params}, obj.funct.error: {error}, m.a.error {best_ma_diff_val}, drop: {best_drop}")

        self.logger.info(f"Grid search complete. Best m.a.error: {best_ma_diff_val}, best drop: {best_drop}")
        
        # STEP 2: One local optimization from best grid point
        self.logger.info("Starting local optimization from best grid point")
        
        result = minimize(
            objective_function,
            best_point,
            method='SLSQP',#,'trust-constr' # 'trust-constr'
            bounds=bounds,
            options={
                'maxiter': 30,  # Limit iterations for faster completion
                'disp': True
            }
        )
        
        # Check if optimization found a better solution
        if result.success:
            new_location = result.x
            
            # Get error and drop for optimized location
            pers_copy = self.pers.clone() # osim.Model(self.pers_model_path)
            pers_muscle_copy = pers_copy.getMuscles().get(muscle_name)
            path_copy = pers_muscle_copy.getGeometryPath()
            path_points_copy = path_copy.getPathPointSet()
            point_copy = osim.PathPoint.safeDownCast(path_points_copy.get(point_index))
            point_copy.setLocation(osim.Vec3(new_location[0], new_location[1], new_location[2]))
            pers_copy.finalizeFromProperties()
            state_copy = pers_copy.initSystem()
            coordinate_copy = pers_copy.getCoordinateSet().get(coordinate_name)
            muscle_copy = osim.Muscle.safeDownCast(pers_copy.getForceSet().get(muscle_name))
            
            opt_moment_arms = self._calculate_moment_arms(
                pers_copy, state_copy, coordinate_copy, muscle_copy, coord_values)
            opt_error, opt_drop = self._compare_moment_arms(gen_moment_arms, opt_moment_arms)
            
            # Calculate displacement
            displacement = np.linalg.norm(new_location - orig_location)

            self.logger.info(f"Optimization result: optim.func.val {result.fun}, m.a.error={opt_error:.6f}, drop={opt_drop}, displacement={displacement:.6f}m")

        # Decide whether to accept the optimized solution
            accept_solution = False
            
            # Gradient Case 1: Significant error improvement
            if opt_error < initial_ma_diff * 0.9 and opt_drop == 0:
                accept_solution = True
                self.logger.info("Solution accepted: Significant ma improvement (>10%)")
            
            # Gradient Case 2: Drop elimination with reasonable displacement
            if initial_drop == 1 and opt_drop == 0 and result.fun <= orig_error:
                accept_solution = True
                self.logger.info("Solution accepted: Drop eliminated with reasonable displacement")

            # Gradient Case 3: Some error improvement and some drop improvement
            elif result.fun < orig_error and opt_drop == 0 and displacement <= (self.shift_bounds[1] - self.shift_bounds[0]) / 2:
                accept_solution = True
                self.logger.info("Solution accepted: Smaller error and zero drop")
            
            if accept_solution:
                point.setLocation(osim.Vec3(new_location[0], new_location[1], new_location[2]))
                self.pers.finalizeFromProperties()
                
                # IMPORTANT: Reinitialize state after making changes
                self.pers_state = self.pers.initSystem()
                
                self.logger.info(f"Applied new location: {new_location}. Old location: {orig_location}")
                return True
            else:
                self.logger.info("Optimization solution rejected, checking if grid search found better solution")
                
        # Check if the grid search solution is better than the initial
                grid_displacement = np.linalg.norm(best_point - orig_location)
                accept_grid = False
                
                # Grid Case 1: Similar criteria for grid search solution
                if best_ma_diff_val < initial_ma_diff * 0.5:
                    accept_grid = True
                    self.logger.info("Grid solution accepted: Significant ma improvement (>50%)")

                # Grid Case 2: Drop elimination with reasonable displacement
                elif initial_drop == 1 and best_drop == 0 and grid_displacement <= (self.shift_bounds[1] - self.shift_bounds[0]) / 2:
                    accept_grid = True
                    self.logger.info("Grid solution accepted: Drop eliminated with reasonable displacement")

                # Grid Case 3: Some error improvement and some drop improvement
                elif best_ma_diff_val < initial_ma_diff and best_drop == 0:
                    accept_grid = True
                    self.logger.info("Grid solution accepted: Smaller error and zero drop")
                
                if accept_grid:
                    point.setLocation(osim.Vec3(best_point[0], best_point[1], best_point[2]))
                    self.pers.finalizeFromProperties()
                    
                    # IMPORTANT: Reinitialize state after making changes
                    self.pers_state = self.pers.initSystem()
                    
                    self.logger.info(f"Applied grid search solution: {best_point}")
                    return True
                else:
                    self.logger.info("No acceptable solution found, leaving point unchanged")
                    return False
        else:
            self.logger.warning(f"Optimization failed: {result.message}")
            
            # Fall back to grid search result if it's better
            grid_displacement = np.linalg.norm(best_point - orig_location)
            accept_grid = False
            
            # Grid Case 1: Similar criteria for grid search solution
            if best_ma_diff_val < initial_ma_diff * 0.5:
                accept_grid = True
                self.logger.info("Grid solution accepted: Significant ma improvement (>50%)")

            # Grid Case 2: Drop elimination with reasonable displacement
            if initial_drop == 1 and best_drop == 0 and grid_displacement <= (self.shift_bounds[1] - self.shift_bounds[0]) / 2:
                accept_grid = True
                self.logger.info("Grid solution accepted: Drop eliminated with reasonable displacement")

            # Grid Case 3: Some error improvement and some drop improvement
            elif best_ma_diff_val < initial_ma_diff and best_drop == 0:
                accept_grid = True
                self.logger.info("Grid solution accepted: Smaller error and zero drop")
            
            if accept_grid:
                point.setLocation(osim.Vec3(best_point[0], best_point[1], best_point[2]))
                self.pers.finalizeFromProperties()
                # IMPORTANT: Reinitialize state after making changes
                self.pers_state = self.pers.initSystem()
                
                self.logger.info(f"Applied grid search solution: {best_point}")
                return True
            else:
                self.logger.info("No acceptable solution found, leaving point unchanged")
                return False

    def _get_multi_coord_optimization_bounds(self, muscle_name, point_index, orig_location, total_points):
        """
        Get optimization bounds for multi-coordinate optimization.
        
        Parameters:
            muscle_name (str): Name of the muscle
            point_index (int): Index of the point being optimized
            orig_location (np.array): Original location of the point
            total_points (int): Total number of points in the muscle path
            
        Returns:
            list: List of (min, max) bounds for each coordinate
        """
        # Special cases for specific muscles
        if any(muscle in muscle_name for muscle in ['psoas']):
            return [(orig_location[i] + self.shift_bounds[0] / 2, 
                    orig_location[i] + self.shift_bounds[1] / 2) 
                for i in range(3)]
        
        # Increased range for gluteus maximus middle points
        elif (any(muscle in muscle_name for muscle in ['glmax']) 
            and point_index in [1, 2]):
            return [(orig_location[i] + self.shift_bounds[0] * 2, 
                    orig_location[i] + self.shift_bounds[1] * 2) 
                for i in range(3)]
        
        # Endpoint restrictions
        elif point_index == 0 or point_index == total_points - 1:
            return [(orig_location[i] + self.shift_bounds[0] / 2, 
                    orig_location[i] + self.shift_bounds[1] / 2) 
                for i in range(3)]
        
        # Default case
        else:
            return [(orig_location[i] + self.shift_bounds[0], 
                    orig_location[i] + self.shift_bounds[1]) 
                for i in range(3)]

    def _optimize_multi_point(self, point_index, coordinate_list, muscle_name,
                            gen_moment_arms_list, coord_values_list, orig_errors):
        """
        Optimize a point that affects multiple coordinates with simplified approach.
        
        Parameters:
            point_index (int): Index of the point to adjust
            coordinate_list (list): List of coordinate names
            muscle_name (str): Name of the muscle
            gen_moment_arms_list (list): List of generic moment arms for each coordinate
            coord_values_list (list): List of coordinate values for each coordinate
            orig_errors (list): List of original error values
        """
        self.logger.info(f"***Optimizing point {point_index} for {muscle_name} on coordinates {coordinate_list}***")
        
        # IMPORTANT: Ensure state is initialized before we start
        self.pers_state = self.pers.initSystem()
        
        # Get the point's original location
        pers_muscle = self.pers.getMuscles().get(muscle_name)
        path = pers_muscle.getGeometryPath()
        path_points = path.getPathPointSet()
        point = osim.PathPoint.safeDownCast(path_points.get(point_index))
        
        location = point.getLocation(self.pers_state)
        orig_location = np.array([location.get(0), location.get(1), location.get(2)])
        
        # Calculate initial errors and drops for each coordinate
        initial_moment_arms = []
        initial_errors = []
        initial_drops = []
        
        for i, coord_name in enumerate(coordinate_list):
            pers_coord = self.pers.getCoordinateSet().get(coord_name)
            moment_arms = self._calculate_moment_arms(
                self.pers, self.pers_state, pers_coord, pers_muscle, coord_values_list[i])
            error, drop = self._compare_moment_arms(gen_moment_arms_list[i], moment_arms)
            
            initial_moment_arms.append(moment_arms)
            initial_errors.append(error)
            initial_drops.append(drop)
        
        initial_error_sum = sum(abs(e) for e in initial_errors)
        initial_drop_count = sum(initial_drops)
        
        self.logger.info(f"Initial errors: {initial_errors}")
        self.logger.info(f"Initial drops: {initial_drops}")
        self.logger.info(f"Initial error sum: {initial_error_sum}, drop count: {initial_drop_count}")
        
        # Define the objective function for optimization
        def objective_function(params):
            """Enhanced objective function for multiple coordinates."""
            # Create a new model to avoid state conflicts
            pers_copy = self.pers.clone() #osim.Model(self.pers_model_path)
            
            # Apply the point shift
            pers_muscle_copy = pers_copy.getMuscles().get(muscle_name)
            path_copy = pers_muscle_copy.getGeometryPath()
            path_points_copy = path_copy.getPathPointSet()
            point_copy = osim.PathPoint.safeDownCast(path_points_copy.get(point_index))
            
            # Set new point location
            point_copy.setLocation(osim.Vec3(params[0], params[1], params[2]))
            pers_copy.finalizeFromProperties()
            
            # Calculate moment arms for each coordinate
            state_copy = pers_copy.initSystem()
            total_error_penalty = 0
            total_drop_penalty = 0
            total_curve_error = 0
            
            for i, coord_name in enumerate(coordinate_list):
                coordinate_copy = pers_copy.getCoordinateSet().get(coord_name)
                muscle_copy = osim.Muscle.safeDownCast(pers_copy.getForceSet().get(muscle_name))
                
                moment_arms = self._calculate_moment_arms(
                    pers_copy, state_copy, coordinate_copy, muscle_copy, coord_values_list[i])
                
                # Calculate error components
                error, drop = self._compare_moment_arms(gen_moment_arms_list[i], moment_arms)
                max_error = self._max_moment_arms_error(gen_moment_arms_list[i], moment_arms)
                
                # Add to total error penalty
                total_error_penalty += abs(error)
                
                # Add to drop penalty
                total_drop_penalty += (0.0 if drop == 0 else 10.0)
                
                # Add curve shape error
                curve_error = self._calculate_curve_shape_error(gen_moment_arms_list[i], moment_arms)
                total_curve_error += curve_error
            
            # Regularization (penalizes larger shifts)
            shift_magnitude = (np.sum((params - orig_location)**2))**0.5
            reg_term = self.regularization_weight * shift_magnitude
            
            return total_error_penalty + max_error + total_drop_penalty + reg_term + total_curve_error*0.1
        
        # Function to evaluate actual errors and drops for a given point
        def evaluate_point(params):
            # Create a model to evaluate
            pers_copy = self.pers.clone() #osim.Model(self.pers_model_path)
            pers_muscle_copy = pers_copy.getMuscles().get(muscle_name)
            path_copy = pers_muscle_copy.getGeometryPath()
            path_points_copy = path_copy.getPathPointSet()
            point_copy = osim.PathPoint.safeDownCast(path_points_copy.get(point_index))
            point_copy.setLocation(osim.Vec3(params[0], params[1], params[2]))
            pers_copy.finalizeFromProperties()
            state_copy = pers_copy.initSystem()
            
            errors = []
            drops = []
            
            for i, coord_name in enumerate(coordinate_list):
                coordinate_copy = pers_copy.getCoordinateSet().get(coord_name)
                muscle_copy = osim.Muscle.safeDownCast(pers_copy.getForceSet().get(muscle_name))
                
                moment_arms = self._calculate_moment_arms(
                    pers_copy, state_copy, coordinate_copy, muscle_copy, coord_values_list[i])
                
                error, drop = self._compare_moment_arms(gen_moment_arms_list[i], moment_arms)
                errors.append(error)
                drops.append(drop)
            
            return errors, drops
        
        # Define bounds
        P = path_points.getSize()
        bounds = self._get_multi_coord_optimization_bounds(muscle_name, point_index, orig_location, P)

        # STEP 1: Simplified grid search 
        self.logger.info("Starting simplified grid search")
        
        # Calculate dimensions to keep under n_total points
        n_total = self.n_total
        n_z = min(5, np.ceil(n_total ** 1/3).astype(int))  # More resolution in z
        remaining = np.ceil((n_total // n_z)**0.5).astype(int)  # Roughly equal for x and y
        n_x = min(5, remaining)
        n_y = min(5, remaining)
        
        self.logger.info(f"Grid dimensions: {n_x}x{n_y}x{n_z} = {n_x*n_y*n_z} points")
        
        x_vals = np.linspace(bounds[0][0], bounds[0][1], n_x)
        y_vals = np.linspace(bounds[1][0], bounds[1][1], n_y)
        z_vals = np.linspace(bounds[2][0], bounds[2][1], n_z)
        
        # Start with original location as best
        best_point = orig_location.copy()
        orig_obj_error = objective_function(orig_location)
        best_obj_error = orig_obj_error
        self.logger.info(f"Original objective function: {best_obj_error}")

        best_errors = initial_errors.copy()
        best_drops = initial_drops.copy()
        best_error_sum = initial_error_sum
        best_drop_count = initial_drop_count
        
        # Grid search
        total_points = n_x * n_y * n_z
        for idx, (x, y, z) in enumerate([(x, y, z) for x in x_vals for y in y_vals for z in z_vals]):
            params = np.array([x, y, z])
            obj_error = objective_function(params)
            
            if idx % 5 == 0 or idx == total_points - 1:
                self.logger.info(f"Grid search progress: {idx+1}/{total_points}")
            
            if obj_error < best_obj_error:
                # Evaluate actual errors and drops
                errors, drops = evaluate_point(params)
                error_sum = sum(abs(e) for e in errors)
                drop_count = sum(drops)
                
                self.logger.info(f"Found better point: {params}, objective function: {obj_error}, error sum: {error_sum}, drops: {drop_count}")
                
                best_obj_error = obj_error
                best_point = params
                best_errors = errors.copy()
                best_drops = drops.copy()
                best_error_sum = error_sum
                best_drop_count = drop_count
        
        self.logger.info(f"Grid search complete. Best error sum: {best_error_sum}, best drop count: {best_drop_count}")
        
        # STEP 2: Local optimization from best grid point
        self.logger.info("Starting local optimization from best grid point")
        
        result = minimize(
            objective_function,
            best_point,
            method='SLSQP',  # Using just one optimization method
            bounds=bounds,
            options={
                'maxiter': 30,  # Limit iterations for speed
                'disp': True
            }
        )
        
        # Evaluate optimization result
        if result.success:
            new_location = result.x
            opt_errors, opt_drops = evaluate_point(new_location)
            opt_error_sum = sum(abs(e) for e in opt_errors)
            opt_drop_count = sum(opt_drops)
            
            # Calculate displacement
            displacement = np.linalg.norm(new_location - orig_location)
            
            self.logger.info(f"Optimization result: error sum={opt_error_sum}, drop count={opt_drop_count}, displacement={displacement:.6f}m")
            
            # Decide whether to accept the optimized solution
            apply_opt = False
            
            # Case 1: Significant error improvement
            if initial_drop_count == opt_drop_count == 0 and best_obj_error < orig_obj_error:
                apply_opt = True
                self.logger.info("Optimization accepted: improved objective error")
            
            # Case 2: Drop reduction with reasonable displacement
            elif opt_drop_count < initial_drop_count and displacement <= abs(self.shift_bounds[1] - self.shift_bounds[0]) / 4:
                apply_opt = True
                self.logger.info("Optimization accepted: Drop count reduced with reasonable displacement")
            
            # Case 3: Some error improvement and no increase in drops
            elif opt_error_sum < initial_error_sum*0.7 and opt_drop_count <= initial_drop_count:
                apply_opt = True
                self.logger.info("Optimization accepted: Some error improvement with no increase in drops")

            # Case 4: No drops
            elif initial_drop_count> 0 and opt_drop_count == 0 and displacement <= abs(self.shift_bounds[1] - self.shift_bounds[0]) / 4: 
                apply_opt = True
                self.logger.info("Optimization accepted: small displacement")

            # Case 5: glmax
            elif ('glmax' in muscle_name) and (point_index in [1,2]) and initial_drop_count> 0 and opt_drop_count == 0 and best_obj_error < orig_obj_error:
                apply_opt = True
                self.logger.info("Optimization accepted: glmax drop eliminated")
            
            if apply_opt:
                point.setLocation(osim.Vec3(new_location[0], new_location[1], new_location[2]))
                self.pers.finalizeFromProperties()
                
                # IMPORTANT: Reinitialize state after making changes
                self.pers_state = self.pers.initSystem()
                
                self.logger.info(f"Applied optimization solution: {new_location}. Old location: {orig_location}")
                return True
            else:
                self.logger.info("Optimization solution rejected, checking grid search solution")
        else:
            self.logger.warning(f"Optimization failed: {result.message}")
            self.logger.info("Checking if grid search found an acceptable solution")
        
        # If optimization wasn't applied, check if grid search solution is acceptable
        grid_displacement = np.linalg.norm(best_point - orig_location)
        apply_grid = False
        
        # Similar criteria for grid search
        if initial_drop_count == best_drop_count == 0 and best_obj_error < orig_obj_error:
            apply_grid = True
            self.logger.info("Grid solution accepted: improved objective error")
        
        elif best_drop_count < initial_drop_count and grid_displacement <= abs(self.shift_bounds[1] - self.shift_bounds[0]) / 4:
            apply_grid = True
            self.logger.info("Grid solution accepted: Drop count reduced with reasonable displacement")
        
        elif best_error_sum < initial_error_sum * 0.7 and best_drop_count <= initial_drop_count:
            apply_grid = True
            self.logger.info("Grid solution accepted: Some error improvement with no increase in drops")

        elif initial_drop_count > 0 and best_drop_count == 0 and best_error_sum < initial_error_sum  and grid_displacement <= abs(self.shift_bounds[1] - self.shift_bounds[0]) / 2:
            apply_grid = True
            self.logger.info("Grid solution accepted: Drop eliminated, small displacement")
        # Case 5: glmax
        elif ('glmax' in muscle_name) and (point_index in [1,2]) and initial_drop_count> 0 and best_drop_count == 0 and best_obj_error < orig_obj_error:
            apply_grid = True
            self.logger.info("Optimization accepted: glmax drop eliminated")
        
        if apply_grid:
            point.setLocation(osim.Vec3(best_point[0], best_point[1], best_point[2]))
            self.pers.finalizeFromProperties()
            
            # IMPORTANT: Reinitialize state after making changes
            self.pers_state = self.pers.initSystem()
            
            self.logger.info(f"Applied grid search solution: {best_point}")
            return True
        else:
            self.logger.info("No acceptable solution found, leaving point unchanged")
            return False
    

    @staticmethod
    def _get_coordinate_range(coordinate):
        """
        Get a range of values spanning the coordinate's range of motion.
        
        Parameters:
            coordinate (osim.Coordinate): OpenSim coordinate object
            
        Returns:
            numpy.ndarray: Array of coordinate values spanning the range of motion
        """
        min_coord_value = coordinate.getRangeMin()
        max_coord_value = coordinate.getRangeMax()
        return np.linspace(min_coord_value, max_coord_value, 50)
    
    @staticmethod
    def _calculate_moment_arms(model, state, coordinate, muscle, coord_values):
        """
        Calculate moment arms for a muscle across a range of coordinate values.
        
        Parameters:
            model (osim.Model): OpenSim model
            state (osim.State): Model state
            coordinate (osim.Coordinate): Coordinate to vary
            muscle (osim.Muscle): Muscle to calculate moment arms for
            coord_values (numpy.ndarray): Coordinate values to evaluate
            
        Returns:
            list: Moment arm values for each coordinate value
        """
        moment_arms = []
        
        for value in coord_values:
            # Set the coordinate value
            coordinate.setValue(state, value)
            
            # Realize position to update the model
            model.realizePosition(state)
            
            # Compute moment arm
            moment_arm = muscle.computeMomentArm(state, coordinate)
            moment_arms.append(moment_arm)
            
        return moment_arms
    
    @staticmethod
    def _compare_moment_arms(generic_moment_arms, personal_moment_arms):
        """
        Compare moment arms between generic and personalized models.
        
        Parameters:
            generic_moment_arms (list): Moment arms from generic model
            personal_moment_arms (list): Moment arms from personalized model
            
        Returns:
            tuple: (error_value, drop_flag) - The error value and whether there's a drop
        """
        # Initialize error values
        #error = 0
        current_error_val_sum = 0
        
        # Ensure arrays are of equal length
        if len(generic_moment_arms) != len(personal_moment_arms):
            raise ValueError('Moment arms are of unequal length')
        
        # Find maximum positive error (when personal < generic)
        for i in range(len(generic_moment_arms)):
            error_val = abs(generic_moment_arms[i]-personal_moment_arms[i])
            # if error_val > 0.005:  # Error threshold of 5mm
            #     error = 1
            #if error_val > current_error_val_max:
            #if current_error_val_max > error_val:
            current_error_val_sum += error_val
        
        # Check for sudden drops in moment arm profile
        n = len(personal_moment_arms)
        #ma_average_speed = np.abs(np.mean(np.array(personal_moment_arms)))/n
        drop = 0
        
        for i in range(n-1):
            dx = abs(personal_moment_arms[i] - personal_moment_arms[i+1])
            if i < 2 and i+4 < n:
                comp1 = abs(personal_moment_arms[i+1] - personal_moment_arms[i+3])*0.5
                comp2 = abs(personal_moment_arms[i+2] - personal_moment_arms[i+4])*0.5
                if dx > (comp1 + comp2)*2:
                    drop += 1
            elif 2 <= i < n-3:
                comp1 = abs(personal_moment_arms[i-2] - personal_moment_arms[i])*0.5
                comp2 = abs(personal_moment_arms[i+1] - personal_moment_arms[i+3])*0.5
                if dx > (comp1 + comp2)*2:
                    drop += 1
            elif n - 3 <= i:
                comp1 = abs(personal_moment_arms[i-2] - personal_moment_arms[i])*0.5
                comp2 = abs(personal_moment_arms[i-3] - personal_moment_arms[i-1])*0.5
                if dx > (comp1 + comp2)*2:
                    drop += 1                
                # Flag if change is larger than average moment arm
                # if dx >= ma_average_speed*5:
                #     drop = 1
                #     break
        
        return  current_error_val_sum/n, drop #error *
    
    @staticmethod
    def _max_moment_arms_error(generic_moment_arms, personal_moment_arms):
         # Initialize error values
        #error = 0
        current_error_val = 0
        
        # Ensure arrays are of equal length
        if len(generic_moment_arms) != len(personal_moment_arms):
            raise ValueError('Moment arms are of unequal length')
        
        # Find maximum positive error (when personal < generic)
        for i in range(len(generic_moment_arms)):
            error_val = abs(generic_moment_arms[i]) - abs(personal_moment_arms[i])
            if error_val > current_error_val:
                current_error_val = error_val

        return current_error_val


    def save_model(self, output_path=None):
        """
        Save the adjusted personalized model.
        
        Parameters:
            output_path (str, optional): Path to save the model to
                                        If None, append '_adjusted' to original path
        """
        if output_path is None:
            # Create default output path by adding '_adjusted' before extension
            base, ext = os.path.splitext(self.pers_model_path)
            output_path = f"{base}_adjusted{ext}"
        
        self.logger.info(f"Saving adjusted model to {output_path}")
        self.pers.printToXML(output_path)
        self.logger.info("Model saved successfully")


def main():
    """Main function to run the adjustment process."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Adjust muscle moment arms in personalized OpenSim model')
    parser.add_argument('--generic', required=True, help='Path to generic OpenSim model')
    parser.add_argument('--personalized', required=True, help='Path to personalized OpenSim model')
    parser.add_argument('--output', help='Path to save adjusted model (default: adds _adjusted suffix)')
    parser.add_argument('--muscles', nargs='+', help='List of muscles to process (default: all)')
    parser.add_argument('--log', default='INFO', help='Logging level')
    parser.add_argument('--workers', type=int, default=8, help='Maximum number of parallel workers')
    parser.add_argument('--threshold', type=float, default=0.005, help='Error threshold in meters')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for gradient descent')
    parser.add_argument('--iterations', type=int, default=50, help='Maximum iterations for optimization')
    parser.add_argument('--n_total', type=int, default=50, help='Total number of points for grid search')
    parser.add_argument('--shift_bounds', nargs=2, type=float, default=[-0.01, 0.01], help='Bounds for point shifts')
    parser.add_argument('--regularization_weight', type=float, default=0.1, help='Weight for regularization term')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    parser.add_argument('--quiet', action='store_true', help='Disable logging')
    parser.add_argument('--no_log', action='store_true', help='Disable logging to file')
    parser.add_argument('--log_file', default='adjust_personal.log', help='Log file name')
    parser.add_argument('--log_dir', default='.', help='Directory for log file')
    parser.add_argument('--log_format', default='%(asctime)s - %(message)s', help='Log format')

    
    args = parser.parse_args()
    
    # Set logging level
    numeric_level = getattr(logging, args.log.upper(), None)
    if not isinstance(numeric_level, int):
        print(f"Invalid log level: {args.log}")
        numeric_level = logging.INFO
    logging.getLogger().setLevel(numeric_level)
    
    # Create adjuster
    adjuster = AdjustPersonal(
        args.generic, 
        args.personalized,
        error_threshold=args.threshold,
        learning_rate=args.learning_rate,
        max_iterations=args.iterations
    )
    
    # Process muscles
    adjuster.process_muscles(args.muscles)
    
    # Save adjusted model
    adjuster.save_model(args.output)
    
    logger.info("Processing complete")


if __name__ == "__main__":
    main()