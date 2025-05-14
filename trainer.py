import os
import time
import pickle
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
import gymnasium as gym
from gymnasium import spaces # Non più usato direttamente qui, ma LunarLanderEnv lo usa
from gymnasium.core import Wrapper
from datetime import datetime
import warnings
import threading # Usato in CustomEvalCallback
import json
import shutil # Non usato direttamente, ma utile per la gestione dei file
from pathlib import Path
import logging
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import get_schedule_fn, get_device # Usato in TrainingManager
import torch # Usato in TrainingManager

# Import the custom environment
try:
    from lander_with_traj import (
        LunarLanderEnv, CENTER_X, CENTER_Y, GRAVITY_CONST, TIME_STEP,
        DEFAULT_ALTITUDE_TARGET_NN_PX, EARTH_RADIUS_KM_MATLAB,
        convert_state_to_km_for_nn, convert_vel_kms_to_px_s,
        predict_nn_trajectory_segment, generate_trajectory_from_nn_prediction,
        PIXELS_PER_KM # Aggiunto se definito in lander_with_traj.py, altrimenti calcolato
    )
except ImportError as e:
    print(f"Error importing LunarLanderEnv components: {e}")
    print("Please ensure 'lander_with_traj.py' is in the current directory or PYTHONPATH,")
    print("and all necessary constants/functions are importable.")
    exit(1)

# -----------------------------------------------------------------------------
# Configuration class
# -----------------------------------------------------------------------------
class Config:
    def __init__(self, args=None):
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.base_dir = Path("lunar_lander_rl_curriculum") # Nome base per run di curriculum
        # Nella funzione __init__ della classe Config
        self.model_name_base = "ppo_lunar_lander_curriculum" # Verrà aggiunto lo stage
        self.model_name = self.model_name_base  # Oppure una variante più specifica
        self.assets_dir = self.base_dir / "assets"
        self.keras_model_path = self.assets_dir / "optimal_trajectory_predictor_2d_csv.keras"
        self.x_scaler_path = self.assets_dir / "x_scaler_nn.pkl"
        self.y_scaler_path = self.assets_dir / "y_scaler_nn.pkl"

        self.runs_dir = self.base_dir / "runs"
        self.current_run_dir = self.runs_dir / f"run_{self.run_id}"
        self.logs_dir = self.current_run_dir / "logs"
        self.models_dir = self.current_run_dir / "models" # Cartella base per i modelli di questo run
        self.eval_dir = self.current_run_dir / "eval"   # Cartella base per i log di eval di questo run
        self.tensorboard_dir = self.logs_dir / "tensorboard"

        self.load_model_path = None # Per caricare un modello per il primo stage del curriculum
        self.load_vecnorm_path = None # Per caricare VecNormalize per il primo stage

        for dir_path in [self.assets_dir, self.logs_dir, self.models_dir, self.eval_dir, self.tensorboard_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        
        
        # Curriculum parameters
        self.curriculum_stages = [1, 2, 3]
        self.timesteps_per_stage = [1_000_000, 1_000_000, 2_000_000] # Esempio
        self.current_training_stage_in_config = self.curriculum_stages[0] if self.curriculum_stages else 1

        # Training hyperparameters (defaults for PPO)
        self.learning_rate = 5e-5 
        self.n_steps = 2048       
        self.batch_size = 64      
        self.n_epochs = 10        
        self.gamma = 0.99         
        self.gae_lambda = 0.95    
        self.clip_range = 0.2     
        self.ent_coef = 0.0005    
        self.vf_coef = 0.5        
        self.max_grad_norm = 0.5  
        self.n_envs = 7 # Default per SubprocVecEnv

        # Evaluation parameters
        self.eval_freq_factor = 10 # Eval ogni N * n_steps
        self.n_eval_episodes = 5
        self.checkpoint_freq_factor = 5 # Checkpoint ogni N * n_steps
        self.vis_episodes = 1 # Episodi da visualizzare al new best

        self.pixels_per_km_fallback = 1.0

        if args:
            for key, value in vars(args).items():
                if value is not None:
                    if key == "total_timesteps":
                        print(f"WARNING: '--total_timesteps' CLI arg received. It will be applied to the FIRST stage of the curriculum if 'timesteps_per_stage' is not also specified via CLI.")
                        # Se viene fornito solo total_timesteps, lo usiamo per il primo stage.
                        # Se viene fornito anche timesteps_per_stage via CLI, quello avrà la precedenza.
                        if not hasattr(args, 'timesteps_per_stage_cli') or not args.timesteps_per_stage_cli:
                           self.timesteps_per_stage[0] = value
                    elif key == "timesteps_per_stage_cli": # Nuovo arg per CLI
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
    def overall_final_model_dir(self): # Directory per il modello finale dopo tutti gli stage
        path = self.models_dir / "final_curriculum_model"
        path.mkdir(parents=True, exist_ok=True)
        return path


# -----------------------------------------------------------------------------
# Asset Management & Env Creation (Simili a prima, ma make_env ora prende lo stage)
# -----------------------------------------------------------------------------
def load_prediction_assets(config):
    # ... (come nel tuo script, ma usa config.keras_model_path etc.)
    keras_model_path = config.keras_model_path
    x_scaler_path = config.x_scaler_path
    y_scaler_path = config.y_scaler_path
    if not os.path.exists(keras_model_path): #ecc.
        print(f"ERROR: Keras model not found at {keras_model_path}")
        return None, None, None
    if not os.path.exists(x_scaler_path) or not os.path.exists(y_scaler_path):
        print(f"ERROR: Scaler files not found ({x_scaler_path}, {y_scaler_path})")
        return None, None, None
    try:
        keras_model = tf.keras.models.load_model(keras_model_path)
        with open(x_scaler_path, "rb") as f: x_scaler = pickle.load(f)
        with open(y_scaler_path, "rb") as f: y_scaler = pickle.load(f)
        print("Keras model and scalers loaded successfully.")
        return keras_model, x_scaler, y_scaler
    except Exception as e:
        print(f"Error loading Keras model or scalers: {e}")
        return None, None, None


def calculate_pixels_per_km(config): # Passa config per fallback
    # Usa LunarLanderEnv per calcolare, instanziandola con lo stage di default (1)
    # o uno stage che non influenzi actual_moon_radius
    try:
        # Non serve render_mode per questo calcolo
        temp_env_params = {"render_mode": None, "training_stage": 1}
        temp_env = LunarLanderEnv(**temp_env_params)
        # Non è necessario resettare se actual_moon_radius è impostato in __init__
        # Se _load_sprites_and_update_dims() è chiamato in __init__ è sufficiente
        # _, _ = temp_env.reset() # Potrebbe non essere necessario se actual_moon_radius è già settato
        
        pixels_per_km = config.pixels_per_km_fallback 
        if hasattr(temp_env, 'actual_moon_radius') and temp_env.actual_moon_radius > 0:
            pixels_per_km = temp_env.actual_moon_radius / EARTH_RADIUS_KM_MATLAB
            print(f"Dynamically calculated PIXELS_PER_KM: {pixels_per_km:.3f}")
        else:
            print(f"Warning: Could not get actual_moon_radius from temp_env. Using fallback: {pixels_per_km}")
        temp_env.close() # Chiudi l'ambiente temporaneo
        return pixels_per_km
    except Exception as e:
        print(f"Error in calculate_pixels_per_km: {e}. Using fallback.")
        return config.pixels_per_km_fallback


class TrajectoryGuidanceWrapper(Wrapper):
    def __init__(self, env, keras_model, x_scaler, y_scaler, pixels_per_km):
        super().__init__(env)
        self.keras_model = keras_model
        self.x_scaler = x_scaler
        self.y_scaler = y_scaler
        self.pixels_per_km = pixels_per_km
        self.env_unwrapped = self.env.unwrapped # Accedi all'ambiente base per training_stage
        if not all([self.keras_model, self.x_scaler, self.y_scaler]):
            print("WARNING: TrajectoryGuidanceWrapper initialized without all prediction assets.")

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs) # Chiama LunarLanderEnv.reset()
        
        # La logica di generazione traiettoria del wrapper si attiva solo per stage > 1,
        # perché LunarLanderEnv.reset() gestisce la traiettoria semplice per Stage 1.
        # E solo se una traiettoria non è GIA' attiva da LunarLanderEnv.reset()
        # (anche se per stage > 1, LunarLanderEnv.reset() non attiva traiettorie di default)
        if hasattr(self.env_unwrapped, 'training_stage') and self.env_unwrapped.training_stage > 1:
            if self.keras_model and self.x_scaler and self.y_scaler and \
               hasattr(self.env_unwrapped, 'ship_pos') and self.env_unwrapped.ship_pos is not None:
                try:
                    # Genera un target di atterraggio casuale sulla superficie
                    moon_radius_px = self.env_unwrapped.actual_moon_radius
                    # Angolo casuale per il punto di atterraggio (es. emisfero inferiore)
                    # random_surface_angle = self.env_unwrapped.np_random.uniform(np.pi * 0.1, np.pi * 0.9) # Per emisfero inferiore
                    random_surface_angle = self.env_unwrapped.np_random.uniform(-np.pi*0.8, np.pi*0.8) # Quasi tutta la circonferenza
                    
                    target_x_on_surface = CENTER_X + moon_radius_px * np.cos(random_surface_angle)
                    target_y_on_surface = CENTER_Y + moon_radius_px * np.sin(random_surface_angle)
                    target_surface_pos_px = np.array([target_x_on_surface, target_y_on_surface])

                    vec_center_to_target_surface = target_surface_pos_px - np.array([CENTER_X, CENTER_Y])
                    norm_vec = np.linalg.norm(vec_center_to_target_surface)
                    unit_radial_to_target_surface = vec_center_to_target_surface / (norm_vec + 1e-9)
                    
                    target_hover_radius_from_center = moon_radius_px + DEFAULT_ALTITUDE_TARGET_NN_PX
                    target_hover_pos_px = np.array([CENTER_X, CENTER_Y]) + unit_radial_to_target_surface * target_hover_radius_from_center
                    
                    # --- Logica di generazione traiettoria (come nel tuo tasto 'N') ---
                    # (Questa è una versione semplificata, dovresti integrare la tua logica uno/due archi se necessario)
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
                        num_traj_points_w = 50 # O parametrizzalo
                        ref_traj_w = generate_trajectory_from_nn_prediction(
                            current_pos_px=current_pos_px_w, # Parte dalla pos attuale del lander
                            nn_predicted_initial_vel_px_s=nn_initial_vel_px_s_w,
                            nn_predicted_tof_s=pred_tof_s_w,
                            # ... (altri parametri per generate_trajectory_from_nn_prediction)
                            pixels_per_km_env=self.pixels_per_km, screen_center_x_env=CENTER_X, screen_center_y_env=CENTER_Y,
                            moon_radius_px_env=self.env_unwrapped.actual_moon_radius, gravity_const_env=GRAVITY_CONST,
                            time_step=TIME_STEP, num_points=num_traj_points_w
                        )
                        if ref_traj_w and len(ref_traj_w) >= 2:
                            self.env_unwrapped.activate_nn_trajectory_guidance(ref_traj_w)
                            # Anche per PID se lo usi per debug
                            # self.env_unwrapped.set_trajectory(ref_traj_w) 
                    # Se la predizione fallisce, l'ambiente partirà senza guida NN attiva per questo episodio

                except Exception as e:
                    print(f"Error during trajectory generation/activation in TrajectoryGuidanceWrapper for stage {self.env_unwrapped.training_stage}: {e}")
                    # import traceback; traceback.print_exc() # Per debug approfondito
        
        # L'osservazione potrebbe essere cambiata se activate_nn_trajectory_guidance è stata chiamata
        # e modifica lo stato osservabile (es. target relativi). È più sicuro riottenerla.
        final_obs = self.env_unwrapped._get_obs()
        # Info è quella del reset dell'ambiente base, che è OK.
        return final_obs, info

    def step(self, action):
        return self.env.step(action)

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
                vec_normalize_save_path = os.path.join(self.best_model_save_path, "vecnormalize.pkl")
        
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
        model_path = os.path.join(self.best_model_save_path, "BEST_model.zip")
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

def make_env(rank, keras_model_rl_agent_unused, x_scaler_rl_agent_unused, y_scaler_rl_agent_unused,
             pixels_per_km_for_env, current_stage_num,
             trajectory_predictor_model_path_for_env, # Spostati prima
             trajectory_x_scaler_path_for_env,      # Spostati prima
             trajectory_y_scaler_path_for_env,      # Spostati prima
             seed=0, env_config_params=None):  
    """Helper function to create a single environment instance"""
    def _init():
        env_kwargs = {
            "render_mode": None,
            "random_seed": seed + rank,
            "training_stage": current_stage_num,
            "trajectory_predictor_model_path": trajectory_predictor_model_path_for_env,
            "trajectory_x_scaler_path": trajectory_x_scaler_path_for_env,
            "trajectory_y_scaler_path": trajectory_y_scaler_path_for_env,
            "pixels_per_km": pixels_per_km_for_env, # <-- Questo argomento non è atteso da __init__
        }
        if env_config_params: # Permette di passare altri parametri da Config se necessario
            env_kwargs.update(env_config_params)

        env = LunarLanderEnv(**env_kwargs)
        env = Monitor(env)
        # Il TrajectoryGuidanceWrapper usa il *suo* keras_model, x_scaler, y_scaler
        # che sono quelli passati come argomenti principali (keras_model_rl_agent_unused etc.
        # il cui nome ho modificato per chiarezza, ma nel tuo codice sono solo keras_model, x_scaler, y_scaler)
        # Questi sono per la *guida* della traiettoria (Stage 2-3), non per la *generazione fissa* in Stage 1 dentro l'env.
        env = TrajectoryGuidanceWrapper(env, keras_model_rl_agent_unused, x_scaler_rl_agent_unused, y_scaler_rl_agent_unused, pixels_per_km_for_env)
        return env
    return _init

def create_environments_improved(config, # config ora contiene i path giusti
                                 keras_model_for_wrapper, x_scaler_for_wrapper, y_scaler_for_wrapper, # Per TrajectoryGuidanceWrapper
                                 pixels_per_km_val, main_logger,
                                 current_stage_num, path_to_load_vecnorm_stats_from=None,
                                 env_config_params_from_main=None):
    main_logger.info(f"Creating vectorized environments for Stage {current_stage_num} with {config.n_envs} parallel envs.")
    env_creation_params = env_config_params_from_main if env_config_params_from_main else {}

    # I percorsi per il predittore di traiettoria interno a LunarLanderEnv (per Stage 1)
    # vengono da config.keras_model_path, config.x_scaler_path, config.y_scaler_path
    # che sono già definiti nella tua classe Config come i path degli asset di predizione.
    
    env_factory_partial = lambda i, s: make_env( # 's' è il secondo parametro posizionale della lambda
        rank=i,
        keras_model_rl_agent_unused=keras_model_for_wrapper,
        x_scaler_rl_agent_unused=x_scaler_for_wrapper,
        y_scaler_rl_agent_unused=y_scaler_for_wrapper,
        pixels_per_km_for_env=pixels_per_km_val,
        current_stage_num=current_stage_num,
        trajectory_predictor_model_path_for_env=str(config.keras_model_path),
        trajectory_x_scaler_path_for_env=str(config.x_scaler_path),
        trajectory_y_scaler_path_for_env=str(config.y_scaler_path),
        seed=s, # Il parametro 's' della lambda viene passato come 'seed' a make_env
        env_config_params=env_creation_params
    )

    if config.n_envs > 1:
        base_train_env = SubprocVecEnv(
            [env_factory_partial(i, i*20 + current_stage_num*100) # Chiamata corretta: secondo argomento posizionale
             for i in range(config.n_envs)],
            start_method="spawn"
        )
    else:
        base_train_env = DummyVecEnv(
            [env_factory_partial(0, current_stage_num*100)] # Chiamata corretta
        )

    base_eval_env = DummyVecEnv(
        [env_factory_partial(0, 10000 + current_stage_num*100)] # Chiamata corretta
    )


    # Gestione VecNormalize
    if path_to_load_vecnorm_stats_from and os.path.exists(path_to_load_vecnorm_stats_from):
        main_logger.info(f"Loading VecNormalize stats from {path_to_load_vecnorm_stats_from} for Stage {current_stage_num}")
        try:
            training_env = VecNormalize.load(str(path_to_load_vecnorm_stats_from), base_train_env)
            training_env.training = True  # Assicurati che sia in modalità training
            training_env.norm_reward = False # O basato sulla config dello stage se vuoi cambiarlo

            eval_env = VecNormalize.load(str(path_to_load_vecnorm_stats_from), base_eval_env)
            eval_env.training = False
            eval_env.norm_reward = False
            main_logger.info(f"Successfully loaded VecNormalize stats for Stage {current_stage_num}.")
        except Exception as e:
            main_logger.error(f"Error loading VecNormalize from {path_to_load_vecnorm_stats_from}: {e}. Creating fresh VecNormalize.")
            # Fallback a creare uno nuovo se il caricamento fallisce
            training_env = VecNormalize(base_train_env, norm_obs=True, norm_reward=False, clip_obs=10.)
            eval_env = VecNormalize(base_eval_env, norm_obs=True, norm_reward=False, clip_obs=10., training=False)
            if hasattr(training_env, 'obs_rms'): eval_env.obs_rms = training_env.obs_rms
            if hasattr(training_env, 'ret_rms'): eval_env.ret_rms = training_env.ret_rms

    else:
        if path_to_load_vecnorm_stats_from:
            main_logger.warning(f"VecNormalize stats path specified but not found: {path_to_load_vecnorm_stats_from}. Creating fresh VecNormalize for Stage {current_stage_num}.")
        else:
            main_logger.info(f"No VecNormalize stats path provided for loading. Creating fresh VecNormalize for Stage {current_stage_num}.")

        training_env = VecNormalize(base_train_env, norm_obs=True, norm_reward=False, clip_obs=10.)
        eval_env = VecNormalize(base_eval_env, norm_obs=True, norm_reward=False, clip_obs=10., training=False)
        # Copia le statistiche iniziali (saranno vuote se appena creato, il che è corretto)
        if hasattr(training_env, 'obs_rms'): eval_env.obs_rms = training_env.obs_rms
        if hasattr(training_env, 'ret_rms'): eval_env.ret_rms = training_env.ret_rms

    main_logger.info(f"Environments for Stage {current_stage_num} created and wrapped.")
    return training_env, eval_env

# -----------------------------------------------------------------------------
# Callbacks & Training Manager (invariati rispetto alla tua ultima versione, ma assicurati che i percorsi siano gestiti)
# -----------------------------------------------------------------------------
# class TrainingManager: ... (come l'hai definita)
# class CustomEvalCallback(EvalCallback): ... (come l'hai definita, ma i path di salvataggio sono ora relativi allo stage)
def setup_callbacks(config, eval_env, keras_model, x_scaler, y_scaler, pixels_per_km, current_stage_num):
    logger = get_logger(config.logs_dir / f"callback_setup_stage_{current_stage_num}.log")
    
    checkpoint_save_path = config.get_stage_specific_model_dir(current_stage_num) / "checkpoints"
    checkpoint_callback = CheckpointCallback(
        save_freq=config.checkpoint_freq,
        save_path=str(checkpoint_save_path),
        name_prefix=f"{config.get_model_name_for_stage(current_stage_num)}_ckpt",
        save_replay_buffer=False, # Generalmente False per PPO
        save_vecnormalize=True, # Salva le stats di VecNormalize
        verbose=1
    )
    logger.info(f"Checkpoint callback for Stage {current_stage_num} configured: save freq {config.checkpoint_freq}")

    best_model_dir_stage = config.get_stage_specific_model_dir(current_stage_num) / "best_model"
    eval_logs_dir_stage = config.get_stage_specific_eval_dir(current_stage_num)
    
    eval_callback = CustomEvalCallback(
        eval_env, 
        keras_model_vis=keras_model, x_scaler_vis=x_scaler, y_scaler_vis=y_scaler, 
        pixels_per_km_vis=pixels_per_km,
        config=config, # Passa l'intera config
        n_visual_episodes=config.vis_episodes,
        visualize_on_new_best=False, # O configuralo
        best_model_save_path=str(best_model_dir_stage), 
        log_path=str(eval_logs_dir_stage),
        eval_freq=config.eval_freq,
        n_eval_episodes=config.n_eval_episodes,
        deterministic=True,
        render=False, 
        verbose=1
    )
    logger.info(f"Evaluation callback for Stage {current_stage_num} configured: eval freq {config.eval_freq}")
    return CallbackList([checkpoint_callback, eval_callback])

# TrainingManager and other utilities (get_logger) dovrebbero essere ok come le hai definite.
def get_logger(log_file=None, name="lunar_lander_rl"):
    """Set up logging to both console and file"""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Rimuovi handler esistenti per evitare log duplicati se la funzione viene chiamata più volte
    if logger.hasHandlers():
        logger.handlers.clear()

    # Formatter per i log
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (se log_file è specificato)
    if log_file:
        log_file_path = Path(log_file) # Converte in oggetto Path se non lo è già
        # Assicura che la directory genitore esista
        log_file_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(str(log_file_path)) # logging.FileHandler vuole una stringa
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger

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
    
    def train(self, custom_total_timesteps=None, custom_tb_log_name=None, custom_reset_num_timesteps=None) -> Optional[PPO]:
        """
        Train or continue training the model.
        
        Args:
            custom_total_timesteps: Override total_timesteps from config
            custom_tb_log_name: Override model_name for TensorBoard logging
            custom_reset_num_timesteps: Override reset_num_timesteps decision
        
        Returns:
            Trained PPO model on success, None on failure
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
            self.logger.info(f"Starting {'new' if reset_timesteps else 'continued'} training " +
                        f"for {total_timesteps} timesteps")
            
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

    def save_final_model(self, model_path, vecnorm_path) -> Dict[str, str]:
        """
        Save the final model and VecNormalize stats.
        
        Args:
            model_path: Path to save the model
            vecnorm_path: Path to save the VecNormalize stats
            
        Returns:
            Dictionary with paths to saved files
        """
        if not self.model:
            self.logger.error("No model to save")
            return {}
            
        results = {}
        
        # Save the model
        self.model.save(model_path)
        self.logger.info(f"Final model saved to {model_path}")
        results['model'] = str(model_path)
        
        # Save VecNormalize stats
        if hasattr(self.training_env, "save"):
            self.training_env.save(vecnorm_path)
            self.logger.info(f"Final VecNormalize stats saved to {vecnorm_path}")
            results['vecnorm'] = str(vecnorm_path)
        
        return results



# -----------------------------------------------------------------------------
# Main Function
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Train PPO for Lunar Lander with Curriculum")
    # ... (i tuoi argomenti CLI come prima) ...
    parser.add_argument("--training_stages", type=str, help="Comma-separated list of stages to run (e.g., '1,2,3')")
    parser.add_argument("--timesteps_per_stage_cli", type=str, help="Comma-separated list of timesteps for each stage")
    # ...
    args = parser.parse_args()
    config = Config(args)
    
    main_logger = get_logger(config.logs_dir / "main_curriculum.log", name=f"rl_train_curr_{config.run_id}")
    main_logger.info(f"Starting CURRICULUM training run: {config.run_id}")
    config_path = config.save_config()
    main_logger.info(f"Full configuration saved to: {config_path}")
    main_logger.info(f"TensorBoard log directory: {config.tensorboard_dir}")

    # Seed (come prima)
    np.random.seed(42); tf.random.set_seed(42); torch.manual_seed(42)
    warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow')
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    main_logger.info("Loading trajectory prediction assets (Keras model, scalers)...")
    keras_model, x_scaler, y_scaler = load_prediction_assets(config)
    if not all([keras_model, x_scaler, y_scaler]):
        main_logger.error("Failed to load critical prediction assets. Aborting."); return

    pixels_per_km = calculate_pixels_per_km(config)

    # --- CURRICULUM LOOP ---
    model = None 
    # Questi sono per il caricamento *iniziale* del primissimo stage del curriculum, se si continua un run interrotto.
    # Per gli stage successivi al primo *all'interno dello stesso run dello script*, il modello viene passato.
    current_run_initial_model_path = config.load_model_path 
    current_run_initial_vecnorm_path = config.load_vecnorm_path
    vecnorm_stats_path_to_load_for_current_stage = config.load_vecnorm_path

    # Path del modello da caricare (solo per il primo stage di questo script run, se specificato)
    initial_model_load_path_for_script = config.load_model_path

    for i, stage_num_from_curriculum in enumerate(config.curriculum_stages):
        main_logger.info(f"========== STARTING CURRICULUM STAGE {stage_num_from_curriculum} ==========")
        config.current_training_stage_in_config = stage_num_from_curriculum

        env_creation_specific_params = {
            # "wind_power": config.wind_power_stage_map.get(stage_num_from_curriculum, 0.0)
        }
        training_env_stage, eval_env_stage = create_environments_improved(
            config, # Passa la config globale, non una copia temporanea
            keras_model, x_scaler, y_scaler, pixels_per_km, main_logger,
            stage_num_from_curriculum,
            path_to_load_vecnorm_stats_from=vecnorm_stats_path_to_load_for_current_stage, # NUOVO ARGOMENTO
            env_config_params_from_main=env_creation_specific_params
        )
        if training_env_stage is None or eval_env_stage is None:
            main_logger.error(f"Failed to create/load environments for Stage {stage_num_from_curriculum}. Aborting curriculum.")
            return

        callbacks_stage = setup_callbacks(config, eval_env_stage, keras_model, x_scaler, y_scaler, pixels_per_km, stage_num_from_curriculum)

        trainer_stage = TrainingManager(config, training_env_stage, main_logger)
        trainer_stage.set_callbacks(callbacks_stage)

        if model is None: # Solo per il primissimo stage processato da questo script
            # Sovrascrivi temporaneamente load_model_path in config solo per questa chiamata a load_or_create_model
            # per il primissimo stage del run dello script
            original_config_load_model_path = config.load_model_path
            config.load_model_path = initial_model_load_path_for_script

            if not trainer_stage.load_or_create_model():
                main_logger.error(f"Failed to initialize model for Stage {stage_num_from_curriculum}. Aborting.")
                training_env_stage.close(); eval_env_stage.close(); return
            model = trainer_stage.model
            config.load_model_path = original_config_load_model_path # Ripristina
        else: # Stage successivi all'interno dello stesso script run
            main_logger.info(f"Continuing with model from previous stage for Stage {stage_num_from_curriculum}.")
            model.set_env(training_env_stage) # Cruciale: il modello userà il nuovo env (con VecNormalize aggiornato/caricato)
            model.tensorboard_log = str(config.tensorboard_dir)
            # Qui potresti voler reimpostare/adattare il learning rate schedule se necessario
            # Esempio: model.learning_rate = get_schedule_fn(nuovo_lr_per_stage)
            # model._setup_lr_schedule() # Per far sì che PPO ricalcoli lo scheduler
            trainer_stage.model = model

        timesteps_this_stage = config.timesteps_per_stage[i]
        main_logger.info(f"Training Stage {stage_num_from_curriculum} for {timesteps_this_stage} timesteps...")

        reset_timesteps_for_sb3_learn = True
        if i > 0: # Non è il primo stage del curriculum IN QUESTO SCRIPT RUN
            reset_timesteps_for_sb3_learn = False
        elif initial_model_load_path_for_script: # È il primo stage MA stiamo caricando un modello
            reset_timesteps_for_sb3_learn = False
        # Altrimenti (primo stage e nessun modello caricato), reset_timesteps_for_sb3_learn rimane True

        model = trainer_stage.train(
            custom_total_timesteps=timesteps_this_stage,
            custom_tb_log_name=config.get_model_name_for_stage(stage_num_from_curriculum),
            custom_reset_num_timesteps=reset_timesteps_for_sb3_learn
        )

        if model is None:
            main_logger.error(f"Training for Stage {stage_num_from_curriculum} failed. Aborting curriculum.")
            training_env_stage.close(); eval_env_stage.close(); return

        stage_model_dir = config.get_stage_specific_model_dir(stage_num_from_curriculum)
        final_stage_model_path = stage_model_dir / f"final_model_stage{stage_num_from_curriculum}.zip"
        model.save(final_stage_model_path)

        # Salva VecNormalize PER QUESTO STAGE e aggiorna il path per il PROSSIMO stage
        current_stage_vecnorm_save_path = stage_model_dir / f"vecnormalize_stage{stage_num_from_curriculum}.pkl"
        if hasattr(training_env_stage, "save"):
            training_env_stage.save(str(current_stage_vecnorm_save_path)) # Salva come stringa
            main_logger.info(f"VecNormalize stats for Stage {stage_num_from_curriculum} saved to {current_stage_vecnorm_save_path}")
            vecnorm_stats_path_to_load_for_current_stage = str(current_stage_vecnorm_save_path) # Aggiorna per il prossimo loop
        else:
            main_logger.warning(f"Training environment for stage {stage_num_from_curriculum} does not have a save method. Cannot save VecNormalize stats.")
            # Se non possiamo salvare, per il prossimo stage non ci sarà nulla da caricare (a meno che non sia fornito da CLI)
            # e quindi ne verrà creato uno nuovo (il che potrebbe o meno essere ciò che si desidera).
            # Considera di impostarlo a None per forzare la creazione di uno nuovo per il prossimo stage se questo accade.
            vecnorm_stats_path_to_load_for_current_stage = None


        main_logger.info(f"--- Stage {stage_num_from_curriculum} completed. Model saved to {final_stage_model_path} ---")

        training_env_stage.close()
        eval_env_stage.close()
        # Per il prossimo stage del loop, il 'model' corrente verrà riutilizzato e gli verrà dato
        # un nuovo env (con VecNormalize inizializzato da vecnorm_stats_path_to_load_for_current_stage).
        # initial_model_load_path_for_script non è più rilevante dopo il primo stage processato.
        initial_model_load_path_for_script = None # Assicura che non venga riutilizzato per errore

    main_logger.info("=== CURRICULUM TRAINING COMPLETED SUCCESSFULLY ===")

    if model:
        main_logger.info("Saving final model from last curriculum stage...")
        final_model_overall_path = config.overall_final_model_dir / "final_model_curriculum.zip"
        model.save(final_model_overall_path)

        # L'ultimo VecNormalize salvato (vecnorm_stats_path_to_load_for_current_stage)
        # è quello dell'ultimo ambiente di training. Copialo nella cartella finale.
        if vecnorm_stats_path_to_load_for_current_stage and os.path.exists(vecnorm_stats_path_to_load_for_current_stage):
            shutil.copy(vecnorm_stats_path_to_load_for_current_stage, config.overall_final_model_dir / "final_vecnormalize_curriculum.pkl")
            main_logger.info(f"Overall final VecNormalize stats (from last stage) copied to {config.overall_final_model_dir}")
        else:
             # Se l'ultimo training_env era disponibile e aveva un metodo save (ma non è stato salvato per qualche motivo sopra), potresti provare a salvarlo qui.
             # Tuttavia, la logica sopra dovrebbe averlo già gestito.
            if hasattr(model.get_env(), "save"):
                try:
                    model.get_env().save(config.overall_final_model_dir / "final_vecnormalize_curriculum.pkl") #type: ignore
                    main_logger.info(f"Overall final VecNormalize stats saved to {config.overall_final_model_dir / 'final_vecnormalize_curriculum.pkl'}")
                except Exception as e:
                    main_logger.error(f"Could not save final vecnormalize for the overall model: {e}")


        main_logger.info(f"Overall final model saved to {final_model_overall_path}")

        main_logger.info("Visualizing final model from curriculum...")
        # Il modello da visualizzare è l'ultimo (`model`)
        # Le stats di VecNormalize da usare sono quelle dell'ultimo stage (`vecnorm_stats_path_to_load_for_current_stage`)
        model_to_vis_path_str = str(final_model_overall_path)
        vecnorm_to_vis_path_str = str(config.overall_final_model_dir / "final_vecnormalize_curriculum.pkl")

        if os.path.exists(model_to_vis_path_str) and os.path.exists(vecnorm_to_vis_path_str):
            visualize_model(model_to_vis_path_str, vecnorm_to_vis_path_str,
                            keras_model, x_scaler, y_scaler, pixels_per_km,
                            num_episodes=config.vis_episodes, verbose=True) # Usa config.vis_episodes
        else:
            main_logger.warning(f"Could not visualize: final model ({model_to_vis_path_str} exists: {os.path.exists(model_to_vis_path_str)}) or " +
                                f"VecNormalize ({vecnorm_to_vis_path_str} exists: {os.path.exists(vecnorm_to_vis_path_str)}) not found.")

    main_logger.info(f"--- Training run {config.run_id} completed. Output in: {config.current_run_dir} ---")


if __name__ == "__main__":
    # Devi avere 'shutil' importato per la copia del file vecnormalize finale
    # import shutil # (assicurati che sia in cima al file)
    main()