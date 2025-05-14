import os
import time
import pickle
import argparse
import numpy as np
import pandas as pd
import warnings
import threading
import json
import shutil
from datetime import datetime
from pathlib import Path
import logging
import traceback

# Configure TensorFlow before importing it to prevent illegal hardware instructions
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce logging
os.environ['PYTHONWARNINGS'] = 'ignore::UserWarning'  # Suppress warnings
# Disable hardware optimizations that can cause crashes
os.environ['TF_DISABLE_MKL'] = '1'
os.environ['TF_DISABLE_SEGMENT_REDUCTION'] = '1'
# Prevent TensorFlow from using AVX/AVX2/FMA instructions
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Now import TensorFlow with safer configuration
import tensorflow as tf

# Configure TensorFlow further to avoid problematic CPU instructions
def configure_tensorflow():
    # Limit TensorFlow to single thread operation which often helps with stability
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)
    
    # Disable GPU if available (since we're having CPU issues)
    try:
        tf.config.set_visible_devices([], 'GPU')
        physical_devices = tf.config.list_physical_devices('GPU')
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
    except:
        pass
    
    # More options to make TensorFlow safer but slower
    try:
        tf.config.experimental.enable_op_determinism()
    except:
        pass

# Call this function immediately
configure_tensorflow()

# Continue with the rest of your imports
import gymnasium as gym
from gymnasium import spaces
from gymnasium.core import Wrapper
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import get_schedule_fn, get_device

# Import the custom environment
try:
    from lander_with_traj import (
        LunarLanderEnv, CENTER_X, CENTER_Y, GRAVITY_CONST, TIME_STEP,
        DEFAULT_ALTITUDE_TARGET_NN_PX, EARTH_RADIUS_KM_MATLAB,
        convert_state_to_km_for_nn, convert_vel_kms_to_px_s,
        predict_nn_trajectory_segment, generate_trajectory_from_nn_prediction,
        PIXELS_PER_KM  # Added if defined in lander_with_traj.py, otherwise calculated
    )
except ImportError as e:
    print(f"Error importing LunarLanderEnv components: {e}")
    print("Please ensure 'lander_with_traj.py' is in the current directory or PYTHONPATH,")
    print("and all necessary constants/functions are importable.")
    exit(1)

# Try importing custom policy
try:
    from custom_policy import CustomLunarLanderPolicy
    CUSTOM_POLICY_AVAILABLE = True
except ImportError:
    CUSTOM_POLICY_AVAILABLE = False
    print("Custom policy not available, will use default MlpPolicy")

# Import della nuova classe
try:
    from reward_modifier import LunarLanderRewardWrapper
    REWARD_WRAPPER_AVAILABLE = True
except ImportError:
    REWARD_WRAPPER_AVAILABLE = False
    print("LunarLanderRewardWrapper not available, will use default rewards")

# -----------------------------------------------------------------------------
# Configuration class
# -----------------------------------------------------------------------------
class Config:
    def __init__(self, args=None):
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.base_dir = Path("lunar_lander_rl_curriculum")  # Nome base per run di curriculum
        self.model_name_base = "ppo_lunar_lander_curriculum"  # Verrà aggiunto lo stage
        self.model_name = self.model_name_base  # Oppure una variante più specifica
        self.assets_dir = self.base_dir / "assets"
        self.keras_model_path = self.assets_dir / "optimal_trajectory_predictor_2d_csv.keras"
        self.x_scaler_path = self.assets_dir / "x_scaler_nn.pkl"
        self.y_scaler_path = self.assets_dir / "y_scaler_nn.pkl"

        self.runs_dir = self.base_dir / "runs"
        self.current_run_dir = self.runs_dir / f"run_{self.run_id}"
        self.logs_dir = self.current_run_dir / "logs"
        self.models_dir = self.current_run_dir / "models"  # Cartella base per i modelli di questo run
        self.eval_dir = self.current_run_dir / "eval"   # Cartella base per i log di eval di questo run
        self.tensorboard_dir = self.logs_dir / "tensorboard"

        self.load_model_path = None  # Per caricare un modello per il primo stage del curriculum
        self.load_vecnorm_path = None  # Per caricare VecNormalize per il primo stage

        # Create required directories
        for dir_path in [self.assets_dir, self.logs_dir, self.models_dir, self.eval_dir, self.tensorboard_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Curriculum parameters 
        self.curriculum_stages = [1, 2, 3]
        # Ottimizzazione dei timesteps in base ai grafici forniti
        self.timesteps_per_stage = [400_000, 600_000, 1_000_000]  # Ridotto da [1M, 1M, 2M]
        
        # Reward wrapper settings
        self.use_reward_wrapper = False  # Default to False
        self.fuel_efficiency_weights = [0.2, 0.4, 0.6]  # Incremental weights per stage
        self.time_penalty_factors = [0.1, 0.2, 0.3]     # Incremental time penalty factors
        
        # Training hyperparameters ottimizzati per evitare oscillazioni
        self.learning_rate = 2.5e-4    # Ridotto per stabilizzare l'apprendimento
        self.n_steps = 1024            # Già ottimizzato
        self.batch_size = 64           # Già ottimizzato
        self.n_epochs = 5              # Ridotto per evitare overfitting
        self.gamma = 0.995             # Già ottimizzato
        self.gae_lambda = 0.9          # Ridotto per dare meno importanza ai reward lontani
        self.clip_range = 0.15         # Già ottimizzato
        self.ent_coef = 0.02           # Aumentato per favorire più esplorazione
        self.vf_coef = 1.0             # Aumentato significativamente per migliorare la stima del valore
        self.max_grad_norm = 0.4       # Ridotto per limitare cambiamenti troppo bruschi
        self.n_envs = 8                # Già ottimizzato
        
        # Implementa learning rate schedule
        self.use_lr_schedule = True   # Nuova opzione per abilitare schedule
        self.lr_schedule_type = "linear"  # linear, exponential, cosine, constant
        self.lr_end_factor = 0.1      # LR finale = LR iniziale * lr_end_factor

        self.current_training_stage_in_config = self.curriculum_stages[0] if self.curriculum_stages else 1

        # Evaluation parameters
        self.eval_freq_factor = 10  # Eval ogni N * n_steps
        self.n_eval_episodes = 5
        self.checkpoint_freq_factor = 5  # Checkpoint ogni N * n_steps
        self.vis_episodes = 1  # Episodi da visualizzare al new best
        self.visualize_best_model = False  # Default to false, can be enabled via CLI

        self.pixels_per_km_fallback = 1.0

        if args:
            for key, value in vars(args).items():
                if value is not None:
                    if key == "total_timesteps":
                        print(f"WARNING: '--total_timesteps' CLI arg received. It will be applied to the FIRST stage of the curriculum if 'timesteps_per_stage' is not also specified via CLI.")
                        # Se viene fornito solo total_timesteps, lo usiamo per il primo stage
                        if not hasattr(args, 'timesteps_per_stage_cli') or not args.timesteps_per_stage_cli:
                           self.timesteps_per_stage[0] = value
                    elif key == "timesteps_per_stage_cli":  # Nuovo arg per CLI
                        try:
                            self.timesteps_per_stage = [int(ts.strip()) for ts in value.split(',')]
                            if len(self.timesteps_per_stage) != len(self.curriculum_stages):
                                print(f"ERROR: Number of --timesteps_per_stage_cli values ({len(self.timesteps_per_stage)}) must match number of curriculum_stages ({len(self.curriculum_stages)}).")
                                exit(1)
                        except ValueError:
                            print("ERROR: --timesteps_per_stage_cli must be a comma-separated list of integers.")
                            exit(1)
                    else:
                         setattr(self, key, value)
        
        # Derived eval/checkpoint frequencies
        self.eval_freq = max(self.n_steps * self.eval_freq_factor // self.n_envs, 1)
        self.checkpoint_freq = max(self.n_steps * self.checkpoint_freq_factor // self.n_envs, 1)

    def save_config(self):
        config_path = self.current_run_dir / "config.json"
        config_dict = {k: str(v) if isinstance(v, Path) else v for k, v in vars(self).items()}
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=4)
        print(f"Configuration saved to {config_path}")
        return config_path

    def get_model_name_for_stage(self, stage_num):
        return f"{self.model_name_base}_stage{stage_num}_{self.run_id}"

    def get_stage_specific_model_dir(self, stage_num):
        path = self.models_dir / f"stage_{stage_num}"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def get_stage_specific_eval_dir(self, stage_num):
        path = self.eval_dir / f"stage_{stage_num}"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def overall_final_model_dir(self):  # Directory per il modello finale dopo tutti gli stage
        path = self.models_dir / "final_curriculum_model"
        path.mkdir(parents=True, exist_ok=True)
        return path


# -----------------------------------------------------------------------------
# Asset Management & Env Creation
# -----------------------------------------------------------------------------
def get_logger(log_file=None, name="lunar_lander_rl"):
    """Set up logging to both console and file"""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Remove existing handlers to avoid duplicate logs
    if logger.hasHandlers():
        logger.handlers.clear()

    # Formatter for logs
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (if log_file is specified)
    if log_file:
        log_file_path = Path(log_file)  # Convert to Path object if not already
        # Ensure parent directory exists
        log_file_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(str(log_file_path))  # logging.FileHandler wants a string
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def load_prediction_assets(config):
    """Load neural network model and scalers for trajectory prediction"""
    keras_model_path = config.keras_model_path
    x_scaler_path = config.x_scaler_path
    y_scaler_path = config.y_scaler_path
    
    logger = get_logger(config.logs_dir / "prediction_assets.log")
    
    if not os.path.exists(keras_model_path):
        logger.error(f"ERROR: Keras model not found at {keras_model_path}")
        return None, None, None
    if not os.path.exists(x_scaler_path) or not os.path.exists(y_scaler_path):
        logger.error(f"ERROR: Scaler files not found ({x_scaler_path}, {y_scaler_path})")
        return None, None, None
    
    try:
        keras_model = tf.keras.models.load_model(keras_model_path)
        with open(x_scaler_path, "rb") as f:
            x_scaler = pickle.load(f)
        with open(y_scaler_path, "rb") as f:
            y_scaler = pickle.load(f)
        logger.info("Keras model and scalers loaded successfully.")
        return keras_model, x_scaler, y_scaler
    except tf.errors.OpError as tf_err:
        logger.error(f"TensorFlow error loading model: {tf_err}")
        return None, None, None
    except Exception as e:
        logger.error(f"Error loading Keras model or scalers: {e}")
        return None, None, None


def calculate_pixels_per_km(config):
    """Calculate pixels per km conversion ratio for the lunar lander environment"""
    logger = get_logger(config.logs_dir / "pixels_calc.log")
    try:
        # Don't need render_mode for this calculation
        temp_env_params = {"render_mode": None, "training_stage": 1}
        temp_env = LunarLanderEnv(**temp_env_params)
        
        pixels_per_km = config.pixels_per_km_fallback
        if hasattr(temp_env, 'actual_moon_radius') and temp_env.actual_moon_radius > 0:
            pixels_per_km = temp_env.actual_moon_radius / EARTH_RADIUS_KM_MATLAB
            logger.info(f"Dynamically calculated PIXELS_PER_KM: {pixels_per_km:.3f}")
        else:
            logger.warning(f"Could not get actual_moon_radius from temp_env. Using fallback: {pixels_per_km}")
        
        temp_env.close()  # Close the temporary environment
        return pixels_per_km
    except Exception as e:
        logger.error(f"Error in calculate_pixels_per_km: {e}. Using fallback.")
        return config.pixels_per_km_fallback


class TrajectoryGuidanceWrapper(Wrapper):
    """Wrapper that adds trajectory guidance capability to the environment"""
    def __init__(self, env, keras_model, x_scaler, y_scaler, pixels_per_km):
        super().__init__(env)
        self.keras_model = keras_model
        self.x_scaler = x_scaler
        self.y_scaler = y_scaler
        self.pixels_per_km = pixels_per_km
        self.env_unwrapped = self.env.unwrapped
        
        # Check if we have all needed prediction assets
        if not all([self.keras_model, self.x_scaler, self.y_scaler]):
            print("WARNING: TrajectoryGuidanceWrapper initialized without all prediction assets.")

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        
        # Only generate trajectory for stages > 1 and when the ship position is available
        if (hasattr(self.env_unwrapped, 'training_stage') and self.env_unwrapped.training_stage > 1 and
            self.keras_model and self.x_scaler and self.y_scaler and
            hasattr(self.env_unwrapped, 'ship_pos') and self.env_unwrapped.ship_pos is not None):
            try:
                # Generate a random landing target on the surface
                moon_radius_px = self.env_unwrapped.actual_moon_radius
                # Random angle for landing point (almost entire circumference)
                random_surface_angle = self.env_unwrapped.np_random.uniform(-np.pi*0.8, np.pi*0.8)
                
                target_x_on_surface = CENTER_X + moon_radius_px * np.cos(random_surface_angle)
                target_y_on_surface = CENTER_Y + moon_radius_px * np.sin(random_surface_angle)
                target_surface_pos_px = np.array([target_x_on_surface, target_y_on_surface])

                vec_center_to_target_surface = target_surface_pos_px - np.array([CENTER_X, CENTER_Y])
                norm_vec = np.linalg.norm(vec_center_to_target_surface)
                unit_radial_to_target_surface = vec_center_to_target_surface / (norm_vec + 1e-9)
                
                target_hover_radius_from_center = moon_radius_px + DEFAULT_ALTITUDE_TARGET_NN_PX
                target_hover_pos_px = np.array([CENTER_X, CENTER_Y]) + unit_radial_to_target_surface * target_hover_radius_from_center
                
                # Generate trajectory
                current_pos_px_w = self.env_unwrapped.ship_pos.copy()
                current_vel_px_s_w = self.env_unwrapped.ship_vel.copy()

                current_pos_km_nn_w, current_vel_kms_nn_w = convert_state_to_km_for_nn(
                    current_pos_px_w, current_vel_px_s_w, CENTER_X, CENTER_Y, self.pixels_per_km
                )
                target_hover_pos_km_nn_w, _ = convert_state_to_km_for_nn(
                    target_hover_pos_px, np.zeros(2), CENTER_X, CENTER_Y, self.pixels_per_km
                )
                pred_vi_vx_kms_w, pred_vi_vy_kms_w, pred_tof_s_w = predict_nn_trajectory_segment(
                    current_pos_km_nn_w, current_vel_kms_nn_w, target_hover_pos_km_nn_w,
                    self.keras_model, self.x_scaler, self.y_scaler
                )

                if pred_tof_s_w > 1e-2:
                    nn_initial_vel_px_s_w = convert_vel_kms_to_px_s(
                        np.array([pred_vi_vx_kms_w, pred_vi_vy_kms_w]), self.pixels_per_km
                    )
                    num_traj_points_w = 50
                    ref_traj_w = generate_trajectory_from_nn_prediction(
                        current_pos_px=current_pos_px_w,
                        nn_predicted_initial_vel_px_s=nn_initial_vel_px_s_w,
                        nn_predicted_tof_s=pred_tof_s_w,
                        pixels_per_km_env=self.pixels_per_km, 
                        screen_center_x_env=CENTER_X, 
                        screen_center_y_env=CENTER_Y,
                        moon_radius_px_env=self.env_unwrapped.actual_moon_radius, 
                        gravity_const_env=GRAVITY_CONST,
                        time_step=TIME_STEP, 
                        num_points=num_traj_points_w
                    )
                    if ref_traj_w and len(ref_traj_w) >= 2:
                        self.env_unwrapped.activate_nn_trajectory_guidance(ref_traj_w)
            except Exception as e:
                print(f"Error during trajectory generation in TrajectoryGuidanceWrapper for stage {self.env_unwrapped.training_stage}: {e}")
                # import traceback; traceback.print_exc()  # Uncomment for more detailed debugging
        
        # Get the updated observation which might have changed after trajectory activation
        final_obs = self.env_unwrapped._get_obs()
        return final_obs, info

    def step(self, action):
        return self.env.step(action)


def make_env(rank, keras_model_rl_agent_unused, x_scaler_rl_agent_unused, y_scaler_rl_agent_unused,
             pixels_per_km_for_env, current_stage_num,
             trajectory_predictor_model_path_for_env,
             trajectory_x_scaler_path_for_env,
             trajectory_y_scaler_path_for_env,
             seed=0, env_config_params=None):
    """Helper function to create a single environment instance"""
    def _init():
        # Filter out any parameters not accepted by LunarLanderEnv
        valid_env_kwargs = {
            "render_mode": None,
            "random_seed": seed + rank,
            "training_stage": current_stage_num,
            "trajectory_predictor_model_path": trajectory_predictor_model_path_for_env,
            "trajectory_x_scaler_path": trajectory_x_scaler_path_for_env,
            "trajectory_y_scaler_path": trajectory_y_scaler_path_for_env,
        }
        
        # Conserva i parametri specifici del wrapper separatamente
        wrapper_params = {}
        if env_config_params:
            for key, value in env_config_params.items():
                # Filtra i parametri per il reward wrapper
                if key in ['use_reward_wrapper', 'fuel_efficiency_weight', 'time_penalty_factor']:
                    wrapper_params[key] = value
                else:
                    # Aggiungi solo i parametri validi per LunarLanderEnv
                    valid_env_kwargs[key] = value
            
        # Don't pass pixels_per_km as it's not an expected parameter
        env = LunarLanderEnv(**valid_env_kwargs)
        env = Monitor(env)
        
        # Applica il RewardWrapper prima di TrajectoryGuidanceWrapper
        if REWARD_WRAPPER_AVAILABLE and wrapper_params.get('use_reward_wrapper', False):
            # Usa i parametri dal wrapper_params
            fuel_weight = wrapper_params.get('fuel_efficiency_weight', 0.4)
            time_factor = wrapper_params.get('time_penalty_factor', 0.01)
            
            env = LunarLanderRewardWrapper(
                env, 
                fuel_efficiency_weight=fuel_weight,
                time_penalty_factor=time_factor
            )
        
        # Apply trajectory guidance wrapper
        env = TrajectoryGuidanceWrapper(
            env, 
            keras_model_rl_agent_unused, 
            x_scaler_rl_agent_unused, 
            y_scaler_rl_agent_unused, 
            pixels_per_km_for_env
        )
        return env
    return _init


def setup_vecnormalize(base_env, stats_path=None, training=True, logger=None):
    """Set up a VecNormalize wrapper with optional loading of statistics"""
    if stats_path and os.path.exists(stats_path):
        try:
            env = VecNormalize.load(stats_path, base_env)
            env.training = training
            env.norm_reward = False
            if logger:
                logger.info(f"VecNormalize stats loaded from {stats_path}")
            return env
        except Exception as e:
            if logger:
                logger.error(f"Failed to load VecNormalize stats: {e}. Creating new VecNormalize.")
    
    # Create new VecNormalize wrapper
    env = VecNormalize(base_env, norm_obs=True, norm_reward=False)
    env.training = training
    if logger:
        logger.info("Created new VecNormalize wrapper")
    return env


def create_environments_improved(config,
                                keras_model_for_wrapper, x_scaler_for_wrapper, y_scaler_for_wrapper,
                                pixels_per_km_val, main_logger,
                                current_stage_num, path_to_load_vecnorm_stats_from=None,
                                env_config_params_from_main=None):
    """Create training and evaluation environments with proper error handling"""
    main_logger.info(f"Creating vectorized environments for Stage {current_stage_num} with {config.n_envs} parallel envs.")
    env_creation_params = env_config_params_from_main if env_config_params_from_main else {}

    # Aggiungi parametri per il reward wrapper
    stage_idx = min(current_stage_num - 1, 2)  # 0-based index, max 2 (stage 3)
    env_config_params = env_config_params_from_main if env_config_params_from_main else {}
    
    # Aggiungi parametri del reward wrapper agli env_params se non presenti
    if hasattr(config, 'use_reward_wrapper') and config.use_reward_wrapper and REWARD_WRAPPER_AVAILABLE:
        env_config_params['use_reward_wrapper'] = True
        env_config_params['fuel_efficiency_weight'] = config.fuel_efficiency_weights[stage_idx]
        env_config_params['time_penalty_factor'] = config.time_penalty_factors[stage_idx]
        main_logger.info(f"Using reward wrapper for Stage {current_stage_num} with fuel weight={config.fuel_efficiency_weights[stage_idx]}, time factor={config.time_penalty_factors[stage_idx]}")

    # Create the environment factory function
    env_factory_partial = lambda i, s: make_env(
        rank=i,
        keras_model_rl_agent_unused=keras_model_for_wrapper,
        x_scaler_rl_agent_unused=x_scaler_for_wrapper,
        y_scaler_rl_agent_unused=y_scaler_for_wrapper,
        pixels_per_km_for_env=pixels_per_km_val,
        current_stage_num=current_stage_num,
        trajectory_predictor_model_path_for_env=str(config.keras_model_path),
        trajectory_x_scaler_path_for_env=str(config.x_scaler_path),
        trajectory_y_scaler_path_for_env=str(config.y_scaler_path),
        seed=s,
        env_config_params=env_config_params
    )

    # Create training environment
    try:
        if config.n_envs > 1:
            try:
                base_train_env = SubprocVecEnv(
                    [env_factory_partial(i, i*20 + current_stage_num*100)
                     for i in range(config.n_envs)],
                    start_method="spawn"
                )
            except Exception as e:
                main_logger.error(f"Failed to create SubprocVecEnv: {e}. Falling back to DummyVecEnv.")
                base_train_env = DummyVecEnv([env_factory_partial(0, current_stage_num*100)])
        else:
            base_train_env = DummyVecEnv([env_factory_partial(0, current_stage_num*100)])
        
        # Apply VecNormalize to training environment
        train_env = setup_vecnormalize(
            base_train_env, 
            stats_path=path_to_load_vecnorm_stats_from,
            training=True, 
            logger=main_logger
        )
        
        # Create evaluation environment (always single env)
        eval_base_env = DummyVecEnv([env_factory_partial(0, 999 + current_stage_num*100)])
        eval_env = setup_vecnormalize(
            eval_base_env, 
            stats_path=path_to_load_vecnorm_stats_from,
            training=False, 
            logger=main_logger
        )
        
        return train_env, eval_env
    
    except Exception as e:
        main_logger.error(f"Failed to create environments: {e}")
        traceback.print_exc()
        return None, None  # Return None, None on failure


def visualize_model(model_path, vec_normalize_path, keras_model_vis, x_scaler_vis, y_scaler_vis,
                   pixels_per_km_vis, num_episodes=3, seed_offset=1000, verbose=True):
    """Visualize a trained model's performance with proper resource management"""
    if verbose:
        print(f"\n--- Visualizing model from: {model_path} ---")
        print(f"--- Using VecNormalize stats from: {vec_normalize_path} ---")

    # Check if files exist
    if not os.path.isfile(model_path):
        if verbose:
            print(f"ERROR: Model file not found at {model_path}")
        # Try with .zip extension
        if not model_path.endswith(".zip"):
            model_path_with_ext = f"{model_path}.zip"
            if os.path.isfile(model_path_with_ext):
                model_path = model_path_with_ext
                if verbose:
                    print(f"Found model file with .zip extension: {model_path}")
            else:
                if verbose:
                    print("Could not find model file with .zip extension either")
                return None
        else:
            return None

    if not os.path.exists(vec_normalize_path):
        if verbose:
            print(f"ERROR: VecNormalize stats file not found: {vec_normalize_path}")
        return None

    # Variables to track resources for cleanup
    vis_load_vec_env = None
    render_env_raw_vis = None
    render_env_wrapped_vis = None
    
    try:
        # Load the model
        loaded_model_vis = PPO.load(model_path, device="auto")
        
        # Create vectorized environment for normalization
        def make_vis_env_for_loading(rank, seed=0):
            def _init():
                env = LunarLanderEnv(render_mode=None, random_seed=seed + rank)
                env = Monitor(env)
                env = TrajectoryGuidanceWrapper(
                    env, keras_model_vis, x_scaler_vis, y_scaler_vis, pixels_per_km_vis
                )
                return env
            return _init
        
        # Create and load normalized environment
        vis_load_vec_env = DummyVecEnv([make_vis_env_for_loading(0, seed=seed_offset)])
        loaded_vec_normalize_vis = VecNormalize.load(vec_normalize_path, vis_load_vec_env)
        loaded_vec_normalize_vis.training = False
        loaded_vec_normalize_vis.observation_space = vis_load_vec_env.observation_space
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
                
                # Render with error handling
                try:
                    render_env_raw_vis.render()
                    time.sleep(1/60)  # Control visualization speed
                except Exception as render_error:
                    if verbose:
                        print(f"Warning: Render failed: {render_error}")
            
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
        
        return episode_results
        
    except Exception as e:
        if verbose:
            print(f"Error during visualization: {e}")
            traceback.print_exc()
        return None
    finally:
        # Clean up resources to prevent leaks
        if verbose:
            print("Cleaning up visualization resources...")
        
        # Close environments in reverse order of creation
        if render_env_wrapped_vis is not None:
            try:
                render_env_wrapped_vis.close()
            except:
                pass
        
        if render_env_raw_vis is not None:
            try:
                render_env_raw_vis.close()
            except:
                pass
        
        if vis_load_vec_env is not None:
            try:
                vis_load_vec_env.close()
            except:
                pass
            
        # Force garbage collection
        import gc
        gc.collect()
        
        if verbose:
            print("--- Visualization finished ---")


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
                 visualize_on_new_best=False,
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
        self.custom_logger = get_logger(config.logs_dir / "eval_callback.log")
        self.visualization_in_progress = False
        self.visualization_timer = None
        self.custom_logger.info(f"CustomEvalCallback initialized with visualize_on_new_best={self.visualize_on_new_best}")

    def _on_step(self) -> bool:
        """Process evaluation step and optionally visualize new best models"""
        # Store information about the best model before calling the parent method
        best_model_exists_before = hasattr(self, 'best_model_path') and self.best_model_path is not None
        previous_best_model_path = self.best_model_path if best_model_exists_before else None
        previous_mean_reward = getattr(self, 'best_mean_reward', None)
        
        # Call parent method for standard evaluation
        result = super()._on_step()
        
        # Determina se un nuovo best model è stato salvato controllando sia il path che il reward
        current_best_exists = hasattr(self, 'best_model_path') and self.best_model_path is not None
        current_mean_reward = getattr(self, 'best_mean_reward', None)
        
        # Un nuovo best model è stato salvato se:
        # 1. Prima non esisteva ed ora esiste, oppure
        # 2. Il reward è migliorato, oppure
        # 3. Il percorso è cambiato
        new_best_model_saved = (
            (not best_model_exists_before and current_best_exists) or
            (current_mean_reward is not None and previous_mean_reward is not None and 
             current_mean_reward > previous_mean_reward) or
            (previous_best_model_path != self.best_model_path and current_best_exists)
        )
        
        # Aggiungiamo log dettagliati per il debug
        if new_best_model_saved:
            self.custom_logger.info(f"New best model detected! Previous reward: {previous_mean_reward}, Current reward: {current_mean_reward}")
            self.custom_logger.info(f"Previous path: {previous_best_model_path}, Current path: {self.best_model_path}")
        
        # Se un nuovo best model è stato salvato, salviamo SEMPRE le statistiche VecNormalize
        if new_best_model_saved and self.best_model_save_path:
            self.custom_logger.info("New best model detected. Saving VecNormalize stats...")
            vec_normalize_save_path = os.path.join(self.best_model_save_path, "vecnormalize.pkl")
            if self.eval_env is not None and hasattr(self.eval_env, "save"):
                try:
                    self.eval_env.save(vec_normalize_save_path)
                    self.custom_logger.info(f"VecNormalize stats successfully saved to {vec_normalize_save_path}")
                    
                    # Verifica che il file sia stato effettivamente creato
                    if os.path.exists(vec_normalize_save_path):
                        self.custom_logger.info(f"Verified: vecnormalize.pkl file exists at {vec_normalize_save_path}")
                    else:
                        self.custom_logger.error(f"File not found after save: {vec_normalize_save_path}")
                except Exception as e:
                    self.custom_logger.error(f"Error saving VecNormalize stats: {e}")
                    traceback.print_exc()
            else:
                self.custom_logger.error("Cannot save VecNormalize stats: eval_env is None or has no save method")
        
        # Rest of the visualization logic
        # Check if we have a new best model and should visualize
        # This is separate from saving VecNormalize stats
        if (self.visualize_on_new_best and 
            hasattr(self, 'best_mean_reward') and 
            self.best_mean_reward is not None and
            self.best_mean_reward > self.last_best_reward and 
            self.best_model_save_path and
            not self.visualization_in_progress):
            
            self.custom_logger.info(f"New best model! Mean reward: {self.best_mean_reward:.2f} vs previous: {self.last_best_reward:.2f}")
            
            # Update last best reward
            self.last_best_reward = self.best_mean_reward
            
            # Save VecNormalize stats for the evaluation environment
            if self.eval_env is not None and hasattr(self.eval_env, "save"):
                vec_normalize_save_path = os.path.join(self.best_model_save_path, "vecnormalize.pkl")
                self.custom_logger.info(f"Saving VecNormalize stats to: {vec_normalize_save_path}")
                try:
                    self.eval_env.save(vec_normalize_save_path)
                    self.custom_logger.info("VecNormalize stats saved successfully")
                except Exception as e:
                    self.custom_logger.error(f"Error saving VecNormalize stats: {e}")

            # Cancel any existing visualization timer
            if self.visualization_timer:
                self.visualization_timer.cancel()
                
            # Schedule visualization with a delay to ensure files are saved
            self.custom_logger.info("Scheduling visualization for the new best model")
            self.visualization_timer = threading.Timer(3.0, self._visualize_best_model)
            self.visualization_timer.daemon = True
            self.visualization_timer.start()

        return result

    def _visualize_best_model(self):
        """Visualize the best model using the current settings"""
        if self.visualization_in_progress:
            self.custom_logger.warning("Visualization already in progress, skipping")
            return
        
        # Set flag to prevent concurrent visualizations
        self.visualization_in_progress = True
        
        try:
            if not self.best_model_save_path:
                self.custom_logger.warning("No path specified for best model, skipping visualization")
                return

            # Get file paths
            model_path = os.path.join(self.best_model_save_path, "BEST_model.zip")
            vec_normalize_path = os.path.join(self.best_model_save_path, "vecnormalize.pkl")

            # Check if both files exist and are not empty
            model_exists = os.path.exists(model_path) and os.path.getsize(model_path) > 0
            stats_exist = os.path.exists(vec_normalize_path) and os.path.getsize(vec_normalize_path) > 0
            
            if model_exists and stats_exist:
                self.custom_logger.info(f"Visualizing best model from {model_path}")
                
                # Use the direct paths instead of making temporary copies
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
                    traceback.print_exc()
            else:
                self.custom_logger.warning(
                    f"Cannot visualize: Model exists: {model_exists}, VecNormalize stats exist: {stats_exist}"
                )
        finally:
            # Always clear flag when done, regardless of errors
            self.visualization_in_progress = False


def setup_callbacks(config, eval_env, keras_model, x_scaler, y_scaler, pixels_per_km, current_stage_num):
    """Set up training callbacks with proper directory structures"""
    logger = get_logger(config.logs_dir / f"callback_setup_stage_{current_stage_num}.log")

    # Configure checkpoint callback
    checkpoint_save_path = config.get_stage_specific_model_dir(current_stage_num) / "checkpoints"
    checkpoint_callback = CheckpointCallback(
        save_freq=config.checkpoint_freq,
        save_path=str(checkpoint_save_path),
        name_prefix=f"{config.get_model_name_for_stage(current_stage_num)}_ckpt",
        save_replay_buffer=False,
        save_vecnormalize=True,
        verbose=1
    )
    logger.info(f"Checkpoint callback for Stage {current_stage_num} configured: save freq {config.checkpoint_freq}")

    # Configure evaluation callback
    best_model_dir_stage = config.get_stage_specific_model_dir(current_stage_num) / "best_model"
    eval_logs_dir_stage = config.get_stage_specific_eval_dir(current_stage_num)

    eval_callback = CustomEvalCallback(
        eval_env,
        keras_model_vis=keras_model,
        x_scaler_vis=x_scaler,
        y_scaler_vis=y_scaler,
        pixels_per_km_vis=pixels_per_km,
        config=config,
        n_visual_episodes=config.vis_episodes,
        visualize_on_new_best=config.visualize_best_model,  # Use config setting
        best_model_save_path=str(best_model_dir_stage),
        log_path=str(eval_logs_dir_stage),
        eval_freq=config.eval_freq,
        n_eval_episodes=config.n_eval_episodes,
        deterministic=True,
        render=False,
        verbose=1
    )
    logger.info(f"Evaluation callback for Stage {current_stage_num} configured: eval freq {config.eval_freq}")
    
    # Return combined callbacks
    return CallbackList([checkpoint_callback, eval_callback])


# -----------------------------------------------------------------------------
# Learning Rate Scheduling
# -----------------------------------------------------------------------------
def create_learning_rate_schedule(initial_lr, schedule_type="linear", total_steps=1000000, end_factor=0.1):
    """
    Creates a learning rate schedule function.
    
    Args:
        initial_lr: Initial learning rate
        schedule_type: Type of scheduling ("linear", "exponential", "cosine", "constant")
        total_steps: Total number of steps for the schedule
        end_factor: Factor to multiply initial_lr by to get the final learning rate
        
    Returns:
        A function that takes progress (0-1) and returns the learning rate
    """
    final_lr = initial_lr * end_factor
    
    def linear_schedule(progress):
        return initial_lr + progress * (final_lr - initial_lr)
        
    def exponential_schedule(progress):
        return initial_lr * (end_factor ** progress)
        
    def cosine_schedule(progress):
        cosine_decay = 0.5 * (1 + np.cos(np.pi * progress))
        return final_lr + cosine_decay * (initial_lr - final_lr)
        
    def constant_schedule(_):
        return initial_lr
        
    schedulers = {
        "linear": linear_schedule,
        "exponential": exponential_schedule,
        "cosine": cosine_schedule,
        "constant": constant_schedule
    }
    
    return schedulers.get(schedule_type, linear_schedule)

# -----------------------------------------------------------------------------
# Training Manager
# -----------------------------------------------------------------------------
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

    def _resolve_model_path(self, model_path):
        """
        Resolve the model path, checking for existence and adding .zip extension if needed.
        """
        # Convert to string if Path object
        model_path = str(model_path)
        
        if not os.path.exists(model_path):
            # Try adding .zip extension if missing
            if not model_path.endswith(".zip"):
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
        return model_path

    def _update_model_hyperparameters(self, model):
        """
        Update the model's hyperparameters based on the current configuration.
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

    def _create_new_model(self):
        """
        Create a new PPO model with the current configuration.
        """
        self.logger.info("Creating new PPO model with current configuration")
        
        # Ensure device is properly selected
        device = get_device("auto")
        self.logger.info(f"Using device: {device}")

        # Use the custom policy if available
        policy_class = "MlpPolicy"
        if CUSTOM_POLICY_AVAILABLE:
            policy_class = CustomLunarLanderPolicy
            self.logger.info("Using custom LunarLander policy architecture")

        # Configure learning rate scheduling
        learning_rate = self.config.learning_rate
        if hasattr(self.config, "use_lr_schedule") and self.config.use_lr_schedule:
            # Prendi total_timesteps dalla somma dei timesteps per stage se disponibili
            total_steps = sum(self.config.timesteps_per_stage) if hasattr(self.config, "timesteps_per_stage") else 1000000
            learning_rate = create_learning_rate_schedule(
                initial_lr=self.config.learning_rate,
                schedule_type=self.config.lr_schedule_type,
                total_steps=total_steps,
                end_factor=self.config.lr_end_factor
            )
            self.logger.info(f"Using {self.config.lr_schedule_type} learning rate schedule from {self.config.learning_rate} to {self.config.learning_rate * self.config.lr_end_factor}")
        
        # Create the model with specified hyperparameters
        model = PPO(
            policy_class,
            self.training_env,
            learning_rate=learning_rate,
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

    def load_or_create_model(self):
        """
        Load an existing model for continued training or create a new one.
        """
        # Check if we need to load an existing model
        if self.config.load_model_path:
            model_path = self._resolve_model_path(self.config.load_model_path)
            if model_path:
                try:
                    self.logger.info(f"Loading PPO model from: {model_path}")
                    
                    # Free memory if using CUDA
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    # Load the model with the current environment
                    self.model = PPO.load(
                        model_path,
                        env=self.training_env,
                        device="auto",
                    )
                    
                    # Update hyperparameters for continued training
                    self.model = self._update_model_hyperparameters(self.model)
                    
                    self.logger.info(f"Model loaded successfully: {type(self.model).__name__}")
                    self.logger.info(f"Model device: {self.model.device}")
                    self.logger.info(
                        f"Model parameters: Learning rate={self.model.learning_rate}, " +
                        f"n_steps={self.model.n_steps}, batch_size={self.model.batch_size}"
                    )
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

    def train(self, custom_total_timesteps=None, custom_tb_log_name=None, custom_reset_num_timesteps=None):
        """
        Train or continue training the model.
        """
        if not self.model:
            self.logger.error("No model available for training. Call load_or_create_model first.")
            return None

        if not self.callbacks:
            self.logger.warning("No callbacks set for training. Performance monitoring will be limited.")

        try:
            # Use custom values or defaults from config
            total_timesteps = custom_total_timesteps if custom_total_timesteps is not None else self.config.total_timesteps
            tb_log_name = custom_tb_log_name if custom_tb_log_name is not None else self.config.model_name_base
            reset_timesteps = custom_reset_num_timesteps if custom_reset_num_timesteps is not None else not self.config.load_model_path

            # Log training start
            self.logger.info(
                f"Starting {'new' if reset_timesteps else 'continued'} training " +
                f"for {total_timesteps} timesteps"
            )

            # Begin training
            self.model.learn(
                total_timesteps=total_timesteps,
                callback=self.callbacks,
                tb_log_name=tb_log_name,
                reset_num_timesteps=reset_timesteps
            )

            self.logger.info("Training completed successfully")
            return self.model

        except Exception as e:
            self.logger.error(f"Training failed: {e}", exc_info=True)
            return None

    def save_final_model(self, model_path, vecnorm_path):
        """
        Save the final model and VecNormalize stats.
        """
        if not self.model:
            self.logger.error("No model to save")
            return {}

        results = {}
        
        # Save the model
        model_path_str = str(model_path)  # Convert Path to string if needed
        self.model.save(model_path_str)
        self.logger.info(f"Final model saved to {model_path_str}")
        results['model'] = model_path_str

        # Save VecNormalize stats
        vecnorm_path_str = str(vecnorm_path)  # Convert Path to string if needed
        if hasattr(self.training_env, "save"):
            self.training_env.save(vecnorm_path_str)
            self.logger.info(f"Final VecNormalize stats saved to {vecnorm_path_str}")
            results['vecnorm'] = vecnorm_path_str
            
        return results


# -----------------------------------------------------------------------------
# Main Function
# -----------------------------------------------------------------------------
def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train PPO for Lunar Lander with Curriculum")
    parser.add_argument("--training_stages", type=str, help="Comma-separated list of stages to run (e.g., '1,2,3')")
    parser.add_argument("--timesteps_per_stage_cli", type=str, help="Comma-separated list of timesteps for each stage")
    parser.add_argument("--visualize_best_model", action="store_true", help="Enable visualization of best model during training")
    parser.add_argument("--load_model_path", type=str, help="Path to model to load for starting the curriculum")
    parser.add_argument("--load_vecnorm_path", type=str, help="Path to VecNormalize stats to load")
    args = parser.parse_args()
    
    # Initialize configuration
    config = Config(args)

    # Set up logging
    main_logger = get_logger(config.logs_dir / "main_curriculum.log", name=f"rl_train_curr_{config.run_id}")
    main_logger.info(f"Starting CURRICULUM training run: {config.run_id}")
    config_path = config.save_config()
    main_logger.info(f"Full configuration saved to: {config_path}")
    main_logger.info(f"TensorBoard log directory: {config.tensorboard_dir}")

    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    torch.manual_seed(42)
    warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow')
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    # Load trajectory prediction assets
    main_logger.info("Loading trajectory prediction assets (Keras model, scalers)...")
    keras_model, x_scaler, y_scaler = load_prediction_assets(config)
    if not all([keras_model, x_scaler, y_scaler]):
        main_logger.error("Failed to load critical prediction assets. Aborting.")
        return

    # Calculate pixels per km conversion ratio
    pixels_per_km = calculate_pixels_per_km(config)
        
    # --- CURRICULUM LOOP ---
    model = None
    # These are for the initial loading of the very first stage, if continuing an interrupted run
    current_run_initial_model_path = config.load_model_path 
    current_run_initial_vecnorm_path = config.load_vecnorm_path
    # Path of the model to load (only for the first stage of this script run, if specified)
    initial_model_load_path_for_script = config.load_model_path

    for i, stage_num_from_curriculum in enumerate(config.curriculum_stages):
        main_logger.info(f"========== STARTING CURRICULUM STAGE {stage_num_from_curriculum} ==========")
        config.current_training_stage_in_config = stage_num_from_curriculum
        
        # Environment-specific parameters for this stage
        env_creation_specific_params = {
            # Add stage-specific parameters here
        }
        
        # Create training and evaluation environments
        training_env_stage, eval_env_stage = create_environments_improved(
            config,
            keras_model, x_scaler, y_scaler, pixels_per_km, main_logger,
            stage_num_from_curriculum,
            path_to_load_vecnorm_stats_from=current_run_initial_vecnorm_path,
            env_config_params_from_main=env_creation_specific_params
        )

        if training_env_stage is None or eval_env_stage is None:
            main_logger.error(f"Failed to create/load environments for Stage {stage_num_from_curriculum}. Aborting curriculum.")
            return

        # Set up callbacks
        callbacks_stage = setup_callbacks(
            config, eval_env_stage, keras_model, x_scaler, y_scaler, pixels_per_km, stage_num_from_curriculum
        )
        
        # Initialize training manager
        trainer_stage = TrainingManager(config, training_env_stage, main_logger)
        trainer_stage.set_callbacks(callbacks_stage)

        # Load or create model 
        if model is None:  # Only for the first stage processed by this script
            # Temporarily override load_model_path in config just for this load_or_create_model call
            original_config_load_model_path = config.load_model_path
            config.load_model_path = initial_model_load_path_for_script
            
            if not trainer_stage.load_or_create_model():
                main_logger.error(f"Failed to initialize model for Stage {stage_num_from_curriculum}. Aborting.")
                if training_env_stage:
                    training_env_stage.close()
                if eval_env_stage:
                    eval_env_stage.close()
                return
                
            model = trainer_stage.model
            config.load_model_path = original_config_load_model_path  # Restore original
        else:  # Subsequent stages within same script run
            main_logger.info(f"Continuing with model from previous stage for Stage {stage_num_from_curriculum}.")
            model.set_env(training_env_stage)  # Use the new environment with this stage's VecNormalize
            model.tensorboard_log = str(config.tensorboard_dir)
            trainer_stage.model = model

        # Get timesteps for this stage
        timesteps_this_stage = config.timesteps_per_stage[i]
        main_logger.info(f"Training Stage {stage_num_from_curriculum} for {timesteps_this_stage} timesteps...")
        
        # Determine if we should reset the timestep counter
        reset_timesteps_for_sb3_learn = True
        if i > 0:  # Not the first stage in this run
            reset_timesteps_for_sb3_learn = False
        elif initial_model_load_path_for_script:  # First stage but loading a model
            reset_timesteps_for_sb3_learn = False

        # Run training for this stage
        model = trainer_stage.train(
            custom_total_timesteps=timesteps_this_stage,
            custom_tb_log_name=config.get_model_name_for_stage(stage_num_from_curriculum),
            custom_reset_num_timesteps=reset_timesteps_for_sb3_learn
        )

        if model is None:
            main_logger.error(f"Training for Stage {stage_num_from_curriculum} failed. Aborting curriculum.")
            if training_env_stage:
                training_env_stage.close()
            if eval_env_stage:
                eval_env_stage.close()
            return

        # Save model for this stage
        stage_model_dir = config.get_stage_specific_model_dir(stage_num_from_curriculum)
        final_stage_model_path = stage_model_dir / f"final_model_stage{stage_num_from_curriculum}.zip"
        model.save(str(final_stage_model_path))
        
        # Save VecNormalize stats for this stage and update path for next stage
        current_stage_vecnorm_save_path = stage_model_dir / f"vecnormalize_stage{stage_num_from_curriculum}.pkl"
        if hasattr(training_env_stage, "save"):
            training_env_stage.save(str(current_stage_vecnorm_save_path))
            main_logger.info(f"VecNormalize stats for Stage {stage_num_from_curriculum} saved to {current_stage_vecnorm_save_path}")
            current_run_initial_vecnorm_path = str(current_stage_vecnorm_save_path)  # Update for next loop
        else:
            main_logger.warning(f"Training environment for stage {stage_num_from_curriculum} does not have a save method.")
            current_run_initial_vecnorm_path = None

        main_logger.info(f"--- Stage {stage_num_from_curriculum} completed. Model saved to {final_stage_model_path} ---")
        
        # Clean up environments from this stage
        if training_env_stage:
            training_env_stage.close()
        if eval_env_stage:
                eval_env_stage.close()
        
        # No longer need the initial model path after first stage
        initial_model_load_path_for_script = None

    main_logger.info("=== CURRICULUM TRAINING COMPLETED SUCCESSFULLY ===")

    # Save the final model from the curriculum
    if model:
        main_logger.info("Saving final model from last curriculum stage...")
        final_model_overall_path = config.overall_final_model_dir / "final_model_curriculum.zip"
        model.save(str(final_model_overall_path))

        # Copy the last VecNormalize stats to the final folder
        final_vecnorm_path = config.overall_final_model_dir / "final_vecnormalize_curriculum.pkl"
        if current_run_initial_vecnorm_path and os.path.exists(current_run_initial_vecnorm_path):
            shutil.copy(current_run_initial_vecnorm_path, final_vecnorm_path)
            main_logger.info(f"Overall final VecNormalize stats (from last stage) copied to {final_vecnorm_path}")
        else:
            # If the last VecNormalize stats weren't saved, try saving from current model's environment
            if hasattr(model.get_env(), "save"):
                try:
                    model.get_env().save(str(final_vecnorm_path))
                    main_logger.info(f"Overall final VecNormalize stats saved to {final_vecnorm_path}")
                except Exception as e:
                    main_logger.error(f"Could not save final vecnormalize for the overall model: {e}")

        main_logger.info(f"Overall final model saved to {final_model_overall_path}")

        # Visualize the final model
        main_logger.info("Visualizing final model from curriculum...")
        model_to_vis_path_str = str(final_model_overall_path)
        vecnorm_to_vis_path_str = str(final_vecnorm_path)

        if os.path.exists(model_to_vis_path_str) and os.path.exists(vecnorm_to_vis_path_str):
            # Force gc before visualization
            import gc
            gc.collect()
            
            # Instead of calling the method directly, which might leak resources
            # Use a separate process for visualization
            if config.visualize_best_model:
                try:
                    import multiprocessing as mp
                    
                    # Define a wrapper function for multiprocessing
                    def run_visualization():
                        # Set process-specific settings
                        np.random.seed(42)
                        tf.random.set_seed(42)
                        
                        # Run visualization
                        visualize_model(
                            model_path=model_to_vis_path_str,
                            vec_normalize_path=vecnorm_to_vis_path_str,
                            keras_model_vis=keras_model, 
                            x_scaler_vis=x_scaler, 
                            y_scaler_vis=y_scaler,
                            pixels_per_km_vis=pixels_per_km,
                            num_episodes=config.vis_episodes, 
                            verbose=True
                        )
                    
                    # Run in a separate process to avoid resource leaks
                    ctx = mp.get_context('spawn')
                    p = ctx.Process(target=run_visualization)
                    p.start()
                    p.join(timeout=60)  # Wait up to 60 seconds
                    
                    # If process didn't finish, terminate it
                    if p.is_alive():
                        main_logger.warning("Visualization timed out after 60 seconds, terminating.")
                        p.terminate()
                        p.join(1)
                        
                except Exception as vis_err:
                    main_logger.error(f"Failed to run visualization in separate process: {vis_err}")
        else:
            main_logger.warning(
                f"Could not visualize: final model ({model_to_vis_path_str} exists: {os.path.exists(model_to_vis_path_str)}) or " +
                f"VecNormalize ({vecnorm_to_vis_path_str} exists: {os.path.exists(vecnorm_to_vis_path_str)}) not found."
            )
            
    main_logger.info(f"--- Training run {config.run_id} completed. Output in: {config.current_run_dir} ---")
    
    # Final cleanup
    import gc
    gc.collect()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Unhandled exception in main: {e}")
        traceback.print_exc()
    finally:
        # Make sure all resources are released
        import gc
        gc.collect()
