import os
import time
import pickle
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
import gymnasium as gym
from gymnasium import spaces
from gymnasium.core import Wrapper
from datetime import datetime
import warnings
import threading
import json
import shutil
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.monitor import Monitor

# Import the custom environment
try:
    from lander_with_traj import (
        LunarLanderEnv, CENTER_X, CENTER_Y, GRAVITY_CONST, TIME_STEP,
        DEFAULT_ALTITUDE_TARGET_NN_PX, EARTH_RADIUS_KM_MATLAB,
        convert_state_to_km_for_nn, convert_vel_kms_to_px_s,
        predict_nn_trajectory_segment, generate_trajectory_from_nn_prediction
    )
except ImportError as e:
    print(f"Error importing LunarLanderEnv: {e}")
    print("Please ensure 'lander_with_traj.py' is in the current directory or PYTHONPATH.")
    exit(1)

# -----------------------------------------------------------------------------
# Configuration class for better organization
# -----------------------------------------------------------------------------

class Config:
    """Configuration container for the training script"""
    
    def __init__(self, args=None):
        """Initialize configuration with command line arguments or defaults"""
        # Generate a unique run ID based on timestamp
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Base directories
        self.base_dir = Path("lunar_lander_rl")
        
        # Assets paths
        self.assets_dir = self.base_dir / "assets"
        self.keras_model_path = self.assets_dir / "optimal_trajectory_predictor_2d_csv.keras"
        self.x_scaler_path = self.assets_dir / "x_scaler_nn.pkl"
        self.y_scaler_path = self.assets_dir / "y_scaler_nn.pkl"
        
        # Run-specific directories
        self.runs_dir = self.base_dir / "runs"
        self.current_run_dir = self.runs_dir / f"run_{self.run_id}"
        self.logs_dir = self.current_run_dir / "logs"
        self.models_dir = self.current_run_dir / "models"
        self.eval_dir = self.current_run_dir / "eval"
        self.tensorboard_dir = self.logs_dir / "tensorboard"
        
        self.load_model_path = None
        self.load_vecnorm_path = None

        # Create all required directories
        for dir_path in [self.assets_dir, self.logs_dir, self.models_dir, 
                         self.eval_dir, self.tensorboard_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Training hyperparameters (defaults)
        self.model_name = f"ppo_lunar_lander_{self.run_id}"
        self.total_timesteps = 1_000_000
        self.learning_rate = 5e-5
        self.n_steps = 2048
        self.batch_size = 64
        self.n_epochs = 10
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.clip_range = 0.2
        self.ent_coef = 0.001
        self.vf_coef = 0.5
        self.max_grad_norm = 0.5
        self.n_envs = 7
        
        # Evaluation parameters
        self.eval_freq = max(self.n_steps * 10, 1)
        self.n_eval_episodes = 5
        self.checkpoint_freq = max(self.n_steps * 5, 1)
        self.vis_episodes = 2
        
        # Environment parameters
        self.pixels_per_km_fallback = 1.0
        
        # Override with command line arguments if provided
        if args:
            for key, value in vars(args).items():
                if value is not None:  # Only override if explicitly provided
                    setattr(self, key, value)
    
    def save_config(self):
        """Save the configuration to a JSON file"""
        config_path = self.current_run_dir / "config.json"
        
        # Convert to a dictionary, handling Path objects
        config_dict = {k: str(v) if isinstance(v, Path) else v 
                      for k, v in vars(self).items()}
        
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=4)
        
        print(f"Configuration saved to {config_path}")
        return config_path
    
    @property
    def best_model_path(self):
        """Path to the best model during training"""
        return self.models_dir / "best_model" / "model.zip"
    
    @property 
    def best_vecnorm_path(self):
        """Path to the VecNormalize stats for the best model"""
        return self.models_dir / "best_model" / "vecnormalize.pkl"
    
    @property
    def final_model_path(self):
        """Path to the final model after training"""
        return self.models_dir / "final_model.zip"
    
    @property
    def final_vecnorm_path(self):
        """Path to the VecNormalize stats for the final model"""
        return self.models_dir / "final_vecnormalize.pkl"

# -----------------------------------------------------------------------------
# Asset Management
# -----------------------------------------------------------------------------

def load_prediction_assets(config):
    """Load the Keras model and scalers needed for trajectory prediction"""
    keras_model_path = config.keras_model_path
    x_scaler_path = config.x_scaler_path
    y_scaler_path = config.y_scaler_path
    
    if not os.path.exists(keras_model_path):
        print(f"ERROR: Keras model not found at {keras_model_path}")
        return None, None, None
    if not os.path.exists(x_scaler_path) or not os.path.exists(y_scaler_path):
        print(f"ERROR: Scaler files not found ({x_scaler_path}, {y_scaler_path})")
        return None, None, None

    try:
        keras_model = tf.keras.models.load_model(keras_model_path)
        with open(x_scaler_path, "rb") as f:
            x_scaler = pickle.load(f)
        with open(y_scaler_path, "rb") as f:
            y_scaler = pickle.load(f)
        print("Keras model and scalers loaded successfully.")
        return keras_model, x_scaler, y_scaler
    except Exception as e:
        print(f"Error loading Keras model or scalers: {e}")
        return None, None, None

def calculate_pixels_per_km():
    """Calculate the scaling factor between pixels and kilometers"""
    temp_env = LunarLanderEnv(render_mode=None)
    _, _ = temp_env.reset()
    
    pixels_per_km = 1.0  # Default fallback value
    
    if hasattr(temp_env, 'actual_moon_radius') and temp_env.actual_moon_radius > 0:
        pixels_per_km = temp_env.actual_moon_radius / EARTH_RADIUS_KM_MATLAB
        print(f"Dynamically calculated PIXELS_PER_KM: {pixels_per_km:.3f}")
    else:
        print(f"Warning: Could not get actual_moon_radius from environment. Using fallback value: {pixels_per_km}")
    
    temp_env.close()
    return pixels_per_km

# -----------------------------------------------------------------------------
# Environment Wrappers
# -----------------------------------------------------------------------------

class TrajectoryGuidanceWrapper(Wrapper):
    """Wrapper to enable neural network trajectory guidance for the lunar lander"""
    
    def __init__(self, env, keras_model, x_scaler, y_scaler, pixels_per_km):
        super().__init__(env)
        self.keras_model = keras_model
        self.x_scaler = x_scaler
        self.y_scaler = y_scaler
        self.pixels_per_km = pixels_per_km
        self.env_unwrapped = self.env.unwrapped

        if not all([self.keras_model, self.x_scaler, self.y_scaler]):
            print("WARNING: TrajectoryGuidanceWrapper initialized without all prediction assets. Guidance may fail.")

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        if self.keras_model and self.x_scaler and self.y_scaler and self.env_unwrapped.ship_pos is not None:
            try:
                # Generate a random landing target on the moon's surface
                moon_radius_px = self.env_unwrapped.actual_moon_radius
                max_angle_offset = np.pi  # Full semicircle for potential landing spots
                
                # Random angle offset from vertical
                random_angle_offset_from_vertical = np.random.uniform(-max_angle_offset, max_angle_offset)
                
                # Calculate surface landing coordinates
                target_x_on_surface = CENTER_X + moon_radius_px * np.sin(random_angle_offset_from_vertical)
                target_y_on_surface = CENTER_Y + moon_radius_px * np.cos(random_angle_offset_from_vertical)
                target_surface_pos_px = np.array([target_x_on_surface, target_y_on_surface])
                
                # Calculate hover point above landing site
                vec_center_to_target_surface = target_surface_pos_px - np.array([CENTER_X, CENTER_Y])
                
                # Normalize to get unit vector direction
                if moon_radius_px < 1e-9:  # Avoid division by zero
                    unit_radial_to_target_surface = np.array([
                        np.sin(random_angle_offset_from_vertical), 
                        np.cos(random_angle_offset_from_vertical)
                    ])
                else:
                    unit_radial_to_target_surface = vec_center_to_target_surface / moon_radius_px
                
                # Calculate hover position at specified altitude
                target_hover_radius_from_center = moon_radius_px + DEFAULT_ALTITUDE_TARGET_NN_PX
                target_hover_pos_px = np.array([CENTER_X, CENTER_Y]) + unit_radial_to_target_surface * target_hover_radius_from_center
                
                # Convert positions and velocities for NN trajectory prediction
                current_pos_px = self.env_unwrapped.ship_pos.copy()
                current_vel_px_s = self.env_unwrapped.ship_vel.copy()

                current_pos_km_nn, current_vel_kms_nn = convert_state_to_km_for_nn(
                    current_pos_px, current_vel_px_s, CENTER_X, CENTER_Y, self.pixels_per_km
                )
                target_hover_pos_km_nn, _ = convert_state_to_km_for_nn(
                    target_hover_pos_px, np.zeros(2), CENTER_X, CENTER_Y, self.pixels_per_km
                )

                # Predict optimal trajectory
                pred_vi_vx_kms, pred_vi_vy_kms, pred_tof_s = predict_nn_trajectory_segment(
                    current_pos_km_nn, current_vel_kms_nn, target_hover_pos_km_nn,
                    self.keras_model, self.x_scaler, self.y_scaler
                )

                if pred_tof_s <= 1e-2:
                    return self.env_unwrapped._get_obs(), info

                # Convert predicted initial velocity back to pixels/second
                nn_initial_vel_px_s = convert_vel_kms_to_px_s(
                    np.array([pred_vi_vx_kms, pred_vi_vy_kms]), self.pixels_per_km
                )

                # Generate trajectory waypoints
                num_trajectory_points = 50
                reference_trajectory_waypoints = generate_trajectory_from_nn_prediction(
                    current_pos_px=current_pos_px,
                    nn_predicted_initial_vel_px_s=nn_initial_vel_px_s,
                    nn_predicted_tof_s=pred_tof_s,
                    pixels_per_km_env=self.pixels_per_km,
                    screen_center_x_env=CENTER_X,
                    screen_center_y_env=CENTER_Y,
                    moon_radius_px_env=self.env_unwrapped.actual_moon_radius,
                    gravity_const_env=GRAVITY_CONST,
                    time_step=TIME_STEP,
                    num_points=num_trajectory_points
                )

                # Activate trajectory guidance if we have valid waypoints
                if reference_trajectory_waypoints and len(reference_trajectory_waypoints) >= 2:
                    self.env_unwrapped.activate_nn_trajectory_guidance(reference_trajectory_waypoints)
                    
            except Exception as e:
                print(f"Error during trajectory generation/activation in wrapper: {e}")
                import traceback
                traceback.print_exc()
        
        return self.env_unwrapped._get_obs(), info

    def step(self, action):
        return self.env.step(action)

# -----------------------------------------------------------------------------
# Visualization Functionality
# -----------------------------------------------------------------------------

def visualize_model(model_path, vec_normalize_path, keras_model_vis, x_scaler_vis, y_scaler_vis,
                   pixels_per_km_vis, num_episodes=3, seed_offset=1000, verbose=True):
    """Visualize a trained model's performance"""
    if verbose:
        print(f"\n--- Visualizing model from: {model_path} ---")
        print(f"--- Using VecNormalize stats from: {vec_normalize_path} ---")

    # Check if model file exists
    if not os.path.isfile(model_path):
        if verbose:
            print(f"ERROR: Model file not found at {model_path}")
        # Try adding .zip extension if missing
        if not model_path.endswith(".zip"):
            model_path_with_ext = f"{model_path}.zip"
            if os.path.isfile(model_path_with_ext):
                model_path = model_path_with_ext
                if verbose:
                    print(f"Found model file with .zip extension: {model_path}")
            else:
                if verbose:
                    print(f"Could not find model file with .zip extension either")
                return
        else:
            return

    # Check if VecNormalize stats file exists
    if not os.path.exists(vec_normalize_path):
        if verbose:
            print(f"ERROR: VecNormalize stats file not found: {vec_normalize_path}")
        return

    # Function to create environment for the model
    def make_vis_env_for_loading(rank, seed=0):
        def _init():
            env = LunarLanderEnv(render_mode=None, random_seed=seed + rank)
            env = Monitor(env)
            env = TrajectoryGuidanceWrapper(env, keras_model_vis, x_scaler_vis, y_scaler_vis, pixels_per_km_vis)
            return env
        return _init

    try:
        # Load the model and create vectorized environment
        loaded_model_vis = PPO.load(model_path, device="auto")
        vis_load_vec_env = DummyVecEnv([make_vis_env_for_loading(0, seed=seed_offset)])
        
        # Load VecNormalize stats
        loaded_vec_normalize_vis = VecNormalize.load(vec_normalize_path, vis_load_vec_env)
        loaded_vec_normalize_vis.training = False
        loaded_vec_normalize_vis.norm_reward = False

        # Create render environment
        render_env_raw_vis = LunarLanderEnv(render_mode='human')
        render_env_wrapped_vis = TrajectoryGuidanceWrapper(
            render_env_raw_vis, keras_model_vis, x_scaler_vis, y_scaler_vis, pixels_per_km_vis
        )

        # Run visualization episodes
        episode_results = []
        for episode in range(num_episodes):
            obs_dict, info = render_env_wrapped_vis.reset()
            terminated = False
            truncated = False
            total_reward_eval = 0
            step_count = 0
            
            if verbose:
                print(f"\n  Visualizing episode {episode + 1}/{num_episodes}")

            while not (terminated or truncated):
                # Normalize observation and predict action
                obs_to_predict = loaded_vec_normalize_vis.normalize_obs(np.array([obs_dict]))
                action, _ = loaded_model_vis.predict(obs_to_predict, deterministic=True)

                # Take step in environment
                obs_dict, reward, terminated, truncated, info = render_env_wrapped_vis.step(action[0])
                total_reward_eval += reward
                step_count += 1
                render_env_raw_vis.render()
                time.sleep(1/60)  # Control visualization speed

            # Collect episode results
            outcome_message = getattr(render_env_raw_vis, 'episode_outcome_message', "Unknown")
            episode_results.append({
                'episode': episode + 1,
                'steps': step_count,
                'reward': total_reward_eval,
                'outcome': outcome_message
            })
            
            if verbose:
                print(f"  Episode {episode + 1} finished after {step_count} steps.")
                print(f"    Outcome: {outcome_message}")
                print(f"    Total Reward: {total_reward_eval:.2f}")
        
        # Cleanup environments
        render_env_raw_vis.close()
        vis_load_vec_env.close()
        
        return episode_results

    except Exception as e:
        if verbose:
            print(f"Error during visualization: {e}")
            import traceback
            traceback.print_exc()
        return None
    finally:
        if verbose:
            print(f"--- Visualization finished ---")

# -----------------------------------------------------------------------------
# Custom Callbacks
# -----------------------------------------------------------------------------

class CustomEvalCallback(EvalCallback):
    """Enhanced evaluation callback with optional visualization"""
    
    def __init__(self,
                 eval_env,
                 keras_model_vis,
                 x_scaler_vis,
                 y_scaler_vis,
                 pixels_per_km_vis,
                 config,
                 n_visual_episodes=1,
                 visualize_on_new_best=True,
                 **kwargs):
        
        # Create directory for best model if it doesn't exist
        if 'best_model_save_path' in kwargs and kwargs['best_model_save_path'] is not None:
            os.makedirs(kwargs['best_model_save_path'], exist_ok=True)
        
        super().__init__(eval_env, **kwargs)
        self.keras_model_vis = keras_model_vis
        self.x_scaler_vis = x_scaler_vis
        self.y_scaler_vis = y_scaler_vis
        self.pixels_per_km_vis = pixels_per_km_vis
        self.n_visual_episodes = n_visual_episodes
        self.visualize_on_new_best = visualize_on_new_best
        self.last_best_reward = -float('inf')
        self.config = config
        self.custom_logger = get_logger(config.logs_dir / "eval_callback.log") # NOME CAMBIATO

        self.custom_logger.info(f"CustomEvalCallback initialized with visualize_on_new_best={self.visualize_on_new_best}") # NOME CAMBIATO

    def _on_step(self) -> bool:
        """Process evaluation step and optionally visualize new best models"""
        # Call parent method for standard evaluation
        result = super()._on_step()
        
        # Check if we have a new best model
        if (self.visualize_on_new_best and 
            hasattr(self, 'best_mean_reward') and 
            self.best_mean_reward is not None and
            self.best_mean_reward > self.last_best_reward and 
            self.best_model_save_path):
            
            self.custom_logger.info(f"New best model! Mean reward: {self.best_mean_reward:.2f} vs previous: {self.last_best_reward:.2f}")
            
            # Update last best reward immediately
            self.last_best_reward = self.best_mean_reward
            
            # Save VecNormalize stats for the evaluation environment
            if self.eval_env is not None and hasattr(self.eval_env, "save"):
                vec_normalize_save_path = os.path.join(os.path.dirname(self.best_model_save_path), 
                                                      "vecnormalize.pkl")
                self.custom_logger.info(f"Saving VecNormalize stats to: {vec_normalize_save_path}")
                try:
                    self.eval_env.save(vec_normalize_save_path)
                    self.custom_logger.info("VecNormalize stats saved successfully")
                except Exception as e:
                    self.custom_logger.error(f"Error saving VecNormalize stats: {e}")
            
            # Start a timer to visualize the best model after a short delay
            # This allows files to be fully written to disk
            if self.visualize_on_new_best:
                threading.Timer(1.0, self._visualize_best_model).start()
        
        return result
    
    def _visualize_best_model(self):
        """Visualize the best model using the current settings"""
        if not self.best_model_save_path:
            self.custom_logger.warning("No path to save best model specified, skipping visualization")
            return
        
        # Ensure model path has the .zip extension
        model_path = os.path.join(self.best_model_save_path, "model.zip")
        vec_normalize_path = os.path.join(self.best_model_save_path, "vecnormalize.pkl")
        
        # Check if both files exist
        model_exists = os.path.exists(model_path)
        stats_exist = os.path.exists(vec_normalize_path)
        
        if model_exists and stats_exist:
            self.custom_logger.info(f"Visualizing best model from {model_path}")
            try:
                visualize_model(
                    model_path=model_path,
                    vec_normalize_path=vec_normalize_path,
                    keras_model_vis=self.keras_model_vis,
                    x_scaler_vis=self.x_scaler_vis,
                    y_scaler_vis=self.y_scaler_vis,
                    pixels_per_km_vis=self.pixels_per_km_vis,
                    num_episodes=self.n_visual_episodes,
                    seed_offset=self.num_timesteps + 1000
                )
                self.custom_logger.info("Visualization completed successfully")
            except Exception as e:
                self.custom_logger.error(f"Error during visualization: {e}")
                import traceback
                traceback.print_exc()
        else:
            self.custom_logger.warning(f"Cannot visualize: Model exists: {model_exists}, VecNormalize stats exist: {stats_exist}")

# -----------------------------------------------------------------------------
# Logging Utilities
# -----------------------------------------------------------------------------

def get_logger(log_file=None, name="lunar_lander_rl"):
    """Set up logging to both console and file"""
    import logging
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Remove existing handlers if any
    if logger.handlers:
        for handler in logger.handlers:
            logger.removeHandler(handler)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Create formatter and add it to the handler
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(console_handler)
    
    # Add file handler if log file is specified
    if log_file:
        # Ensure parent directory exists
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

# -----------------------------------------------------------------------------
# Environment Creation
# -----------------------------------------------------------------------------

def make_env(rank, keras_model, x_scaler, y_scaler, pixels_per_km, seed=0):
    """Create a single environment instance with proper wrapping"""
    def _init():
        env = LunarLanderEnv(render_mode=None, random_seed=seed + rank)
        env = Monitor(env)
        env = TrajectoryGuidanceWrapper(env, keras_model, x_scaler, y_scaler, pixels_per_km)
        return env
    return _init

def create_training_env(config, keras_model, x_scaler, y_scaler, pixels_per_km):
    """Create and configure the vectorized training environment"""
    logger = get_logger(config.logs_dir / "env_creation.log")
    logger.info(f"Creating vectorized environment with {config.n_envs} envs")
    
    # Create vectorized environment based on number of environments
    if config.n_envs > 1:
        vec_env = SubprocVecEnv(
            [make_env(i, keras_model, x_scaler, y_scaler, pixels_per_km) for i in range(config.n_envs)],
            start_method="spawn"
        )
    else:
        vec_env = DummyVecEnv([make_env(0, keras_model, x_scaler, y_scaler, pixels_per_km)])
    
    # Apply VecNormalize wrapper
    normalized_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.)
    logger.info("Training environment created and wrapped with VecNormalize")
    
    return normalized_env

def create_eval_env(config, keras_model, x_scaler, y_scaler, pixels_per_km, eval_seed_offset=100):
    """Create and configure the evaluation environment"""
    logger = get_logger(config.logs_dir / "env_creation.log")
    logger.info("Creating evaluation environment")
    
    # Always use a single environment for evaluation
    eval_env_list = [make_env(0, keras_model, x_scaler, y_scaler, pixels_per_km, seed=eval_seed_offset)]
    eval_vec_env = DummyVecEnv(eval_env_list)
    
    # Apply VecNormalize wrapper in eval mode (no training, no reward normalization)
    eval_env = VecNormalize(
        eval_vec_env, 
        norm_obs=True, 
        norm_reward=False, 
        clip_obs=10., 
        training=False
    )
    
    logger.info("Evaluation environment created")
    return eval_env

# -----------------------------------------------------------------------------
# Training Process
# -----------------------------------------------------------------------------

def setup_callbacks(config, eval_env, keras_model, x_scaler, y_scaler, pixels_per_km):
    """Set up the training callbacks"""
    logger = get_logger(config.logs_dir / "callback_setup.log") # Potresti voler usare main_logger qui
    
    # Checkpoint callback to save models periodically
    checkpoint_callback = CheckpointCallback(
        save_freq=config.checkpoint_freq,
        save_path=str(config.models_dir / "checkpoints"), # Assicura sia stringa
        name_prefix=config.model_name,
        save_replay_buffer=False,
        save_vecnormalize=True,
        verbose=1
    )
    logger.info(f"Checkpoint callback configured: save frequency = {config.checkpoint_freq} steps")
    
    # Custom evaluation callback with visualization
    best_model_dir = str(config.models_dir / "best_model") # Assicura sia stringa
    eval_logs_dir = str(config.eval_dir) # Assicura sia stringa
    
    eval_callback = CustomEvalCallback(
        eval_env, 
        keras_model_vis=keras_model,
        x_scaler_vis=x_scaler,
        y_scaler_vis=y_scaler,
        pixels_per_km_vis=pixels_per_km,
        config=config,
        n_visual_episodes=config.vis_episodes,
        visualize_on_new_best=True,
        # Standard EvalCallback parameters:
        best_model_save_path=best_model_dir,
        log_path=eval_logs_dir,
        eval_freq=config.eval_freq,
        n_eval_episodes=config.n_eval_episodes,
        deterministic=True,
        render=False, 
        verbose=1
    )
    logger.info(f"Evaluation callback configured: eval frequency = {config.eval_freq} steps")
    
    return CallbackList([checkpoint_callback, eval_callback])

import os
import pathlib
from typing import Optional, Dict, Any, Union
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.utils import get_schedule_fn, get_device
import logging
import torch

class TrainingManager:
    """
    Manages the training workflow, especially for continuing training from saved models.
    Handles model loading, hyperparameter adjustments, and training continuation.
    """
    
    def __init__(self, config, training_env, main_logger):
        """
        Initialize the training manager.
        
        Args:
            config: Configuration object with training parameters
            training_env: Vectorized and normalized training environment
            main_logger: Logger instance for outputting status
        """
        self.config = config
        self.training_env = training_env
        self.logger = main_logger
        self.model = None
        self.callbacks = None
        
    def set_callbacks(self, callbacks):
        """Set the callbacks for training"""
        self.callbacks = callbacks
        
    def _resolve_model_path(self, model_path: Union[str, pathlib.Path]) -> str:
        """
        Resolve the model path, checking for existence and adding .zip extension if needed.
        
        Args:
            model_path: Path to the model file
            
        Returns:
            Resolved model path as string
        """
        if not os.path.exists(model_path):
            # Try adding .zip extension if missing
            if not str(model_path).endswith(".zip"):
                model_path_with_ext = f"{model_path}.zip"
                if os.path.exists(model_path_with_ext):
                    self.logger.info(f"Found model file with .zip extension: {model_path_with_ext}")
                    return model_path_with_ext
                else:
                    self.logger.warning(f"Could not find model file at {model_path} or with .zip extension")
                    return None
            else:
                self.logger.warning(f"Model file not found at {model_path}")
                return None
        return str(model_path)

    def _update_model_hyperparameters(self, model: PPO) -> PPO:
        """
        Update the model's hyperparameters based on the current configuration.
        
        Args:
            model: The loaded PPO model
            
        Returns:
            Updated PPO model
        """
        self.logger.info("Updating hyperparameters for continued training")
        original_lr = model.learning_rate
        
        # Basic hyperparameters
        model.learning_rate = self.config.learning_rate
        model.n_steps = self.config.n_steps
        model.batch_size = self.config.batch_size
        model.n_epochs = self.config.n_epochs
        model.gamma = self.config.gamma
        model.gae_lambda = self.config.gae_lambda
        
        # Handle clip_range which could be a schedule
        if hasattr(self.config, 'clip_range'):
            # Check if clip_range is a constant or callable in the model
            if callable(model.clip_range):
                self.logger.info("Converting constant clip_range value to schedule function")
            
            # Always convert to a schedule function for consistency
            model.clip_range = get_schedule_fn(self.config.clip_range)
            
            # Also handle clip_range_vf if present and not None
            if hasattr(model, 'clip_range_vf') and model.clip_range_vf is not None:
                if hasattr(self.config, 'clip_range_vf'):
                    model.clip_range_vf = get_schedule_fn(self.config.clip_range_vf)
                else:
                    # Use the same value as clip_range if clip_range_vf not specified
                    model.clip_range_vf = get_schedule_fn(self.config.clip_range)
        
        # Other coefficients
        model.ent_coef = self.config.ent_coef
        model.vf_coef = self.config.vf_coef
        model.max_grad_norm = self.config.max_grad_norm
        
        # Update TensorBoard log directory
        model.tensorboard_log = str(self.config.tensorboard_dir)
        
        # Set verbosity for console output
        model.verbose = 1
        
        self.logger.info(f"Hyperparameters updated. Learning rate: {original_lr} -> {model.learning_rate}")
        return model
        
    def _create_new_model(self) -> PPO:
        """
        Create a new PPO model with the current configuration.
        
        Returns:
            New PPO model instance
        """
        self.logger.info("Creating new PPO model with current configuration")
        
        # Ensure device is properly selected
        device = get_device("auto")
        self.logger.info(f"Using device: {device}")
        
        # Create the model with specified hyperparameters
        model = PPO(
            "MlpPolicy",
            self.training_env,
            learning_rate=self.config.learning_rate,
            n_steps=self.config.n_steps,
            batch_size=self.config.batch_size,
            n_epochs=self.config.n_epochs,
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda,
            clip_range=self.config.clip_range,
            ent_coef=self.config.ent_coef,
            vf_coef=self.config.vf_coef,
            max_grad_norm=self.config.max_grad_norm,
            verbose=1,
            tensorboard_log=str(self.config.tensorboard_dir),
            device=device
        )
        
        # Log model info
        self.logger.info(f"New PPO model created with policy: {model.policy}")
        return model
    
    def load_or_create_model(self) -> bool:
        """
        Load an existing model for continued training or create a new one.
        
        Returns:
            Boolean indicating success
        """
        # Check if we need to load an existing model
        if self.config.load_model_path:
            model_path = self._resolve_model_path(self.config.load_model_path)
            
            if model_path:
                try:
                    self.logger.info(f"Loading PPO model from: {model_path}")
                    
                    # Check if we're using CUDA and free memory before loading
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    # Load the model with the current environment (already VecNormalized)
                    self.model = PPO.load(
                        model_path,
                        env=self.training_env,
                        device="auto",
                        # If using a custom policy, add custom_objects here
                    )
                    
                    # Update hyperparameters for continued training
                    self.model = self._update_model_hyperparameters(self.model)
                    
                    self.logger.info(f"Model loaded successfully: {type(self.model).__name__}")
                    self.logger.info(f"Model device: {self.model.device}")
                    self.logger.info(f"Model parameters: Learning rate={self.model.learning_rate}, " +
                                     f"n_steps={self.model.n_steps}, batch_size={self.model.batch_size}")
                    return True
                    
                except Exception as e:
                    self.logger.error(f"Failed to load model: {e}", exc_info=True)
                    return False
            else:
                self.logger.warning("Model path not found, creating a new model instead")
        
        # Create a new model
        try:
            self.model = self._create_new_model()
            return True
        except Exception as e:
            self.logger.error(f"Failed to create new model: {e}", exc_info=True)
            return False
    
    def train(self) -> Optional[PPO]:
        """
        Train or continue training the model.
        
        Returns:
            Trained PPO model on success, None on failure
        """
        if not self.model:
            self.logger.error("No model available for training. Call load_or_create_model first.")
            return None
            
        if not self.callbacks:
            self.logger.warning("No callbacks set for training. Performance monitoring will be limited.")
        
        try:
            # Determine whether to reset timestep counting
            reset_timesteps = not self.config.load_model_path
            
            # Log training start
            self.logger.info(f"Starting {'new' if reset_timesteps else 'continued'} training " +
                           f"for {self.config.total_timesteps} timesteps")
            
            # Begin training
            self.model.learn(
                total_timesteps=self.config.total_timesteps,
                callback=self.callbacks,
                tb_log_name=self.config.model_name,
                reset_num_timesteps=reset_timesteps
            )
            
            self.logger.info("Training completed successfully")
            return self.model
            
        except Exception as e:
            self.logger.error(f"Training failed: {e}", exc_info=True)
            return None
    
    def save_final_model(self) -> Dict[str, str]:
        """
        Save the final model and VecNormalize stats.
        
        Returns:
            Dictionary with paths to saved files
        """
        if not self.model:
            self.logger.error("No model to save")
            return {}
            
        results = {}
        
        # Save the model
        final_model_path = str(self.config.final_model_path)
        self.model.save(final_model_path)
        self.logger.info(f"Final model saved to {final_model_path}")
        results['model'] = final_model_path
        
        # Save VecNormalize stats
        if hasattr(self.training_env, "save"):
            final_vecnorm_path = str(self.config.final_vecnorm_path)
            self.training_env.save(final_vecnorm_path)
            self.logger.info(f"Final VecNormalize stats saved to {final_vecnorm_path}")
            results['vecnorm'] = final_vecnorm_path
        
        return results


# This function replaces the original train_model function
def train_model_improved(config, training_env, callbacks, main_logger):
    """
    Train or continue training the RL model with improved robustness.
    
    Args:
        config: Configuration object with training parameters
        training_env: Vectorized and normalized training environment
        callbacks: Training callbacks for monitoring and checkpointing
        main_logger: Logger for tracking training process
        
    Returns:
        Trained model on success, None on failure
    """
    # Create training manager
    trainer = TrainingManager(config, training_env, main_logger)
    
    # Set callbacks
    trainer.set_callbacks(callbacks)
    
    # Load or create model
    if not trainer.load_or_create_model():
        main_logger.error("Failed to initialize model. Aborting training.")
        return None
    
    # Train model
    model = trainer.train()
    
    # Save model if training was successful
    if model:
        saved_paths = trainer.save_final_model()
        if saved_paths:
            main_logger.info(f"Training complete. Model saved to: {saved_paths.get('model', 'unknown')}")
        else:
            main_logger.warning("Training complete but model could not be saved")
    
    return model


# Additional improvements for loading VecNormalize stats
def load_vecnormalize_stats(vec_normalize_path, vec_env, logger):
    """
    Load VecNormalize statistics with error handling.
    
    Args:
        vec_normalize_path: Path to the VecNormalize pickle file
        vec_env: Vector environment to apply normalization to
        logger: Logger for messages
        
    Returns:
        VecNormalize environment on success, None on failure
    """
    if not os.path.exists(vec_normalize_path):
        logger.error(f"VecNormalize stats file not found: {vec_normalize_path}")
        return None
        
    try:
        # Load VecNormalize with the prepared vector environment
        vec_normalize = VecNormalize.load(vec_normalize_path, vec_env)
        logger.info(f"VecNormalize stats loaded successfully from {vec_normalize_path}")
        
        # Log some statistics to verify proper loading
        if hasattr(vec_normalize, "obs_rms") and vec_normalize.obs_rms is not None:
            mean = vec_normalize.obs_rms.mean
            var = vec_normalize.obs_rms.var
            logger.info(f"Observation normalization loaded - mean shape: {mean.shape}, var shape: {var.shape}")
            
        if hasattr(vec_normalize, "ret_rms") and vec_normalize.ret_rms is not None:
            logger.info(f"Return normalization loaded - mean: {vec_normalize.ret_rms.mean}, " +
                       f"var: {vec_normalize.ret_rms.var}")
            
        return vec_normalize
        
    except Exception as e:
        logger.error(f"Failed to load VecNormalize stats: {e}", exc_info=True)
        return None


# Improvements for main function to use these new components
def create_environments_improved(config, keras_model, x_scaler, y_scaler, pixels_per_km, main_logger):
    """
    Create or load training and evaluation environments with better error handling.
    
    Args:
        config: Configuration with environment settings
        keras_model, x_scaler, y_scaler, pixels_per_km: Assets for trajectory guidance
        main_logger: Logger for messages
        
    Returns:
        Tuple of (training_env, eval_env) or (None, None) on failure
    """
    from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
    
    # Import make_env function from your main script
    # For this example, we assume make_env is defined and imported
    
    # Function is used in your main script, refers to the make_env function defined in the original code
    
    # First, create the base vector environments (without normalization)
    if config.load_model_path and config.load_vecnorm_path and os.path.exists(config.load_vecnorm_path):
        main_logger.info("Creating environments for loading saved normalization stats")
        
        # Create the base environment for training
        if config.n_envs > 1:
            base_train_env = SubprocVecEnv(
                [make_env(i, keras_model, x_scaler, y_scaler, pixels_per_km, seed=i*10) 
                 for i in range(config.n_envs)],
                start_method="spawn"
            )
        else:
            base_train_env = DummyVecEnv(
                [make_env(0, keras_model, x_scaler, y_scaler, pixels_per_km, seed=0)]
            )
            
        # Create the base environment for evaluation
        base_eval_env = DummyVecEnv(
            [make_env(0, keras_model, x_scaler, y_scaler, pixels_per_km, seed=config.n_envs + 100)]
        )
        
        # Load normalization statistics
        training_env = load_vecnormalize_stats(config.load_vecnorm_path, base_train_env, main_logger)
        if training_env is None:
            base_train_env.close()
            base_eval_env.close()
            return None, None
            
        # Set training mode for the training environment
        training_env.training = True
        training_env.norm_reward = True
        
        # Load normalization statistics for evaluation environment
        eval_env = load_vecnormalize_stats(config.load_vecnorm_path, base_eval_env, main_logger)
        if eval_env is None:
            training_env.close()
            base_eval_env.close()
            return None, None
            
        # Set evaluation mode for the evaluation environment
        eval_env.training = False
        eval_env.norm_reward = False
        
    else:
        # Create new environments with fresh normalization
        main_logger.info("Creating new environments with fresh normalization")
        
        if config.load_model_path and not os.path.exists(config.load_vecnorm_path):
            main_logger.warning(f"VecNormalize stats file not found: {config.load_vecnorm_path}")
            main_logger.warning("Creating new environments with fresh normalization instead")
        
        # Create new environments using the existing functions
        training_env = create_training_env(config, keras_model, x_scaler, y_scaler, pixels_per_km)
        eval_env = create_eval_env(config, keras_model, x_scaler, y_scaler, pixels_per_km)
    
    return training_env, eval_env


# To use this in the main function, replace the relevant parts with calls to these functions
# For example, in the main function you would use:
#
# training_env, eval_env = create_environments_improved(config, keras_model, x_scaler, y_scaler, pixels_per_km, main_logger)
# if training_env is None or eval_env is None:
#     main_logger.error("Failed to create environments. Aborting.")
#     return
#
# callbacks = setup_callbacks(config, eval_env, keras_model, x_scaler, y_scaler, pixels_per_km)
# model = train_model_improved(config, training_env, callbacks, main_logger)
# -----------------------------------------------------------------------------
# Main Function
# -----------------------------------------------------------------------------

def main():
    """Main function to handle the training process"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train a PPO agent for Lunar Lander with trajectory guidance")

    # Environment settings
    parser.add_argument("--n_envs", type=int, default=None, help="Number of parallel environments (default in Config)")

    # Training hyperparameters
    parser.add_argument("--total_timesteps", type=int, default=None, help="Total timesteps for training (default in Config)")
    parser.add_argument("--learning_rate", type=float, default=None, help="Learning rate (default in Config)")
    parser.add_argument("--n_steps", type=int, default=None, help="Number of steps per update (default in Config)")
    parser.add_argument("--batch_size", type=int, default=None, help="Minibatch size (default in Config)")
    parser.add_argument("--n_epochs", type=int, default=None, help="Number of epochs for optimization (default in Config)")
    parser.add_argument("--gamma", type=float, default=None, help="Discount factor (default in Config)")
    parser.add_argument("--gae_lambda", type=float, default=None, help="GAE lambda parameter (default in Config)")
    parser.add_argument("--clip_range", type=float, default=None, help="Clipping parameter for PPO (default in Config)")
    parser.add_argument("--ent_coef", type=float, default=None, help="Entropy coefficient (default in Config)")
    parser.add_argument("--vf_coef", type=float, default=None, help="Value function coefficient (default in Config)")
    parser.add_argument("--max_grad_norm", type=float, default=None, help="Maximum norm for gradient clipping (default in Config)")

    # Evaluation parameters
    parser.add_argument("--eval_freq", type=int, default=None, help="Evaluation frequency in timesteps (default in Config)")
    parser.add_argument("--n_eval_episodes", type=int, default=None, help="Number of episodes for evaluation (default in Config)")
    parser.add_argument("--checkpoint_freq", type=int, default=None, help="Checkpoint frequency in timesteps (default in Config)")
    parser.add_argument("--vis_episodes", type=int, default=None, help="Number of episodes for visualization (default in Config)")

    # Arguments for continuing training
    parser.add_argument("--load_model_path", type=str, default=None, help="Path to the .zip model file to load and continue training.")
    parser.add_argument("--load_vecnorm_path", type=str, default=None, help="Path to the .pkl VecNormalize statistics file to load.")

    args = parser.parse_args()

    # Initialize configuration
    config = Config(args)

    # Setup logging
    main_logger = get_logger(config.logs_dir / "main.log", name=f"rl_train_{config.run_id}")
    main_logger.info(f"Starting Lunar Lander RL training run: {config.run_id}")
    main_logger.info(f"Full configuration path: {config.save_config()}")
    main_logger.info(f"TensorBoard log directory: {config.tensorboard_dir}")


    # Set up environment deterministically for reproducible results
    np.random.seed(42) # Consider making this configurable
    tf.random.set_seed(42) # Consider making this configurable
    # torch.manual_seed(42) # If using PyTorch directly for other parts

    # Filter TensorFlow warnings
    warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow')
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress extensive TF INFO messages

    # Load prediction assets (Keras model, scalers for trajectory guidance)
    main_logger.info("Loading trajectory prediction assets (Keras model, scalers)...")
    keras_model, x_scaler, y_scaler = load_prediction_assets(config)
    if not all([keras_model, x_scaler, y_scaler]):
        main_logger.error("Failed to load critical prediction assets. Aborting.")
        return

    # Calculate pixels per km conversion
    pixels_per_km = calculate_pixels_per_km()
    if pixels_per_km <= 0: # Should ideally not happen if env initializes correctly
        pixels_per_km = config.pixels_per_km_fallback
        main_logger.warning(f"Using fallback pixels_per_km value: {pixels_per_km}")

    # Create/Load training and evaluation environments using the new improved function
    main_logger.info("Creating/Loading training and evaluation environments...")
    training_env, eval_env = create_environments_improved(
        config, keras_model, x_scaler, y_scaler, pixels_per_km, main_logger
    )

    if training_env is None or eval_env is None:
        main_logger.error("Failed to create or load environments. Aborting.")
        if training_env: training_env.close() # Attempt to close if one was partially created
        if eval_env: eval_env.close()
        return
    main_logger.info("Training and evaluation environments are ready.")

    # Setup callbacks
    main_logger.info("Setting up callbacks...")
    try:
        callbacks = setup_callbacks(config, eval_env, keras_model, x_scaler, y_scaler, pixels_per_km)
        main_logger.info("Callbacks configured successfully.")
    except Exception as e:
        main_logger.error(f"Error setting up callbacks: {e}", exc_info=True)
        training_env.close()
        eval_env.close()
        return

    # Train model using the new improved function
    main_logger.info("Starting model training process...")
    model = train_model_improved(config, training_env, callbacks, main_logger)

    if model is None:
        main_logger.error("Training process failed or was aborted. See logs for details.")
        # Environments should be closed by train_model_improved or its callers on failure,
        # but ensure cleanup here as a fallback.
        if training_env: training_env.close()
        if eval_env: eval_env.close()
        return
    main_logger.info("Model training process finished.")

    # Visualize final model (paths from config properties are Path objects, convert to str)
    main_logger.info("Visualizing final model...")
    try:
        final_model_path_str = str(config.final_model_path)
        final_vecnorm_path_str = str(config.final_vecnorm_path)

        if os.path.exists(final_model_path_str) and os.path.exists(final_vecnorm_path_str):
            visualize_model(
                model_path=final_model_path_str,
                vec_normalize_path=final_vecnorm_path_str,
                keras_model_vis=keras_model,
                x_scaler_vis=x_scaler,
                y_scaler_vis=y_scaler,
                pixels_per_km_vis=pixels_per_km,
                num_episodes=config.vis_episodes * 2, # Visualize more for final model
                verbose=True
            )
        else:
            main_logger.warning(f"Could not visualize final model. Model or VecNormalize file missing.")
            main_logger.warning(f"Checked model path: {final_model_path_str} (Exists: {os.path.exists(final_model_path_str)})")
            main_logger.warning(f"Checked vecnorm path: {final_vecnorm_path_str} (Exists: {os.path.exists(final_vecnorm_path_str)})")

    except Exception as e:
        main_logger.error(f"Error visualizing final model: {e}", exc_info=True)

    # Cleanup
    main_logger.info("Closing environments...")
    if training_env: training_env.close()
    if eval_env: eval_env.close()

    main_logger.info(f"--- Training run {config.run_id} completed. ---")
    main_logger.info(f"Output directory: {config.current_run_dir}")
    if os.path.exists(config.best_model_path):
         main_logger.info(f"Best model saved during this run: {config.best_model_path}")
    if os.path.exists(config.final_model_path):
         main_logger.info(f"Final model saved at the end of this run: {config.final_model_path}")

if __name__ == "__main__":
    main()