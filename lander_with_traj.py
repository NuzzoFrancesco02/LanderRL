import pygame
import time
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import os
import pickle
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import pandas as pd  # <<< ASSICURATI CHE QUESTA RIGA SIA QUI ALL'INIZIO DEL FILE
import tensorflow as tf # <<< ANCHE QUESTA SE USI TF GLOBALMENTE

WEIGHT_VERTICAL_CONTROL_ORIENTATION = 2 # Esempio, da tarare!
VERTICAL_CONTROL_DEADZONE_PX = 5.0      # Non applicare se molto vicino verticalmente
MAX_RADIAL_SPEED_FOR_BONUS_CONSIDERATION = 0.05 # Considera solo se la velocità radiale non è eccessiva


MIN_EPISODE_SCORE_FOR_TRUNCATION = -100000.0 
# Bonus per l'avvicinamento al marker (shaping reward)
WEIGHT_APPROACHING_MARKER_BONUS_PER_PIXEL = 0.5  # Esempio: 0.5 punti per ogni pixel di avvicinamento
                                               # Sintonizzalo attentamente!

# Bonus per la precisione dell'atterraggio vicino al marker (terminal reward)
WEIGHT_PRECISION_LANDING_BONUS_MAX = 5000.0     # Bonus massimo se atterra esattamente sul marker
MAX_DIST_FOR_PRECISION_BONUS_PX = 10.0          # Distanza massima dal marker per ricevere parte di questo bonus (in pixel)
MIN_DIST_FOR_MAX_PRECISION_BONUS_PX = 2.0 
PROJECTION_UPDATE_RATE = 1
WEIGHT_SPATIAL_PROXIMITY_TO_REF_PATH = 0.05  # Bonus base per la vicinanza alla forma della traiettoria NN

# Fattore di decadimento per la ricompensa di prossimità (un valore più piccolo significa decadimento più rapido)
PROXIMITY_DECAY_FACTOR = 0.01 # Es: exp(-0.01 * distance)
NN_PATH_CORRIDOR_RADIUS = 20
WEIGHT_VELOCITY_ALIGNMENT_WITH_REF_PATH = 0.005 # Bonus per allineare la velocità con la direzione locale della traiettoria NN
WEIGHT_DEVIATE_NN_PATH_PENALTY = -0.005   # Penalità per pixel per cui si sfora NN_PATH_CORRIDOR_RADIUS (deve essere negativo)
WEIGHT_PROGRESS_ALONG_PATH_SHAPE = 1
# Soglia per considerare la traiettoria NN completata spazialmente (per il bonus una tantum)
REF_TRAJ_END_REACH_THRESHOLD_PX = 300.0

# --- Costanti Fisiche e di Gioco ---
SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 800
CENTER_X = SCREEN_WIDTH // 2
CENTER_Y = SCREEN_HEIGHT // 2
WEIGHT_TIME_PENALTY = 0.005  # Da 0.08 a 0.05            # AUMENTATO DRASTICAMENTE (da 0.005)
MAX_RADIAL_SPEED_FINAL_LANDING = -1
# --- NUOVI PESI RICOMPENSA PER AGENTE RL CHE SEGUE TRAIETTORIA NN ---
WEIGHT_NN_WAYPOINT_DISTANCE_REDUCTION = 5*0 # Bonus per ridurre la distanza al waypoint NN
WEIGHT_NN_WAYPOINT_REACHED = 20.0 * 0     # Bonus elevato per aver raggiunto un waypoint NN
WEIGHT_NN_TRAJECTORY_COMPLETED = 1000.0    # Bonus per aver completato l'intera traiettoria NN
WEIGHT_DEVIATION_FROM_NN_PATH = -0.05     # Penalità (per pixel) se troppo lontano dal segmento corrente della traiettoria NN
MAX_ALLOWED_DEVIATION_NN_PATH = 30.0     # Distanza in pixel oltre la quale si applica una forte penalità
WEIGHT_FINAL_LANDING_SPEED_REDUCTION = 150.0
WEIGHT_FINAL_LANDING_ANGLE_ALIGNMENT = 200.0
WEIGHT_EXCESSIVE_FINAL_DESCENT_SPEED_PENALTY = 5.0 

WEIGHT_FUEL_EMPTY_PENALTY = 1000.0                # AUMENTATO SIGNIFICATIVAMENTE
WEIGHT_FUEL_CONSUMPTION_PENALTY_PER_UNIT = 0.05 # NUOVO: Penalità per unità di carburante consumato

# Scala per normalizzare le coordinate relative al waypoint
WAYPOINT_COORD_SCALE_XY = SCREEN_WIDTH # o SCREEN_HEIGHT, scegli una scala ragionevole
WAYPOINT_VEL_SCALE = 15.0 # Scala per normalizzare le velocità relative al waypoint


WEIGHT_TRAJECTORY_SIMILARITY = 2.0 # (Dal tuo snippet)
MAX_EXPECTED_DEVIATION = 50.0   # (Dal tuo snippet)
WEIGHT_EXCESSIVE_DEVIATION = 1.0    # (Dal tuo snippet, da applicare come penalità)
MAX_ALLOWED_DEVIATION = 30.0      # (Dal tuo snippet)
REF_TRAJ_END_REACH_THRESHOLD_PX = 60.0 # Per bonus completamento spaziale

# --- FATTORI DI SCALA SPRITE ---
SHIP_SCALE_FACTOR = 0.2
MOON_SCALE_FACTOR = 1
FLAME_SCALE_FACTOR = 0.08
BACKGROUND_SCALE_FACTOR = 0.8



GRAVITY_CONST = 10000
TIME_STEP = 0.1
MAIN_THRUST_FORCE = 100.0 # Potrebbe essere necessario regolarlo per hover fine
ROTATION_THRUST_TORQUE = 350

DRY_MASS = 50.0
INITIAL_FUEL_MASS = 60.0
FUEL_CONSUMPTION_MAIN = 3
FUEL_CONSUMPTION_ROTATION = 1

SHIP_CENTER_TO_REAR_DIST = 25
SHIP_SIDE_THRUSTER_X_OFFSET = -10
SHIP_SIDE_THRUSTER_Y_OFFSET = 25

DEFAULT_MOON_RADIUS = 171.5
DEFAULT_SHIP_W = 89
DEFAULT_SHIP_H = 87

LANDING_SPEED_LIMIT = 1.5
LANDING_ANGLE_LIMIT = np.radians(8)
TARGET_ANGLE_LANDING = 0.0

MAX_STEPS_PER_EPISODE = 4000

DESCENT_PHASE_ALTITUDE = 200  # Altitudine per fase di discesa
APPROACH_PHASE_ALTITUDE = 110  # Altitudine per fase di avvicinamento
NEAR_SURFACE_ALTITUDE_THRESHOLD = 80
LANDING_PHASE_ALTITUDE = NEAR_SURFACE_ALTITUDE_THRESHOLD  # Riutilizza o modifica

LANDING_AID_LINE_LENGTH = 40
LANDING_AID_ARC_RADIUS = 50
FUEL_BAR_WIDTH = 100
FUEL_BAR_HEIGHT = 12

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
GRAY = (128, 128, 128)
YELLOW = (255, 255, 0)
ORANGE = (255, 165, 0)
PURPLE = (128, 0, 128)
CYAN = (0, 255, 255)
DARK_GRAY = (50, 50, 50)
LIGHT_GREEN = (144, 238, 144)
DARK_RED = (139, 0, 0)


WEIGHT_ROTATION_SPEED_PENALTY = 5           # Leggermente ridotto per permettere manovre
WEIGHT_HIGH_SPEED_NEAR_SURFACE_PENALTY = 0.05   # AUMENTATO DRASTICAMENTE (da 0.005)
WEIGHT_CRASH_PENALTY = 150.0                    # MANTENUTO (forte deterrente)
WEIGHT_OOB_PENALTY = 15.0                       # Aumentato leggermente
WEIGHT_TIMEOUT_PENALTY = 15.0                   # Aumentato leggermente

WEIGHT_ROTATING_TOWARDS_RETROGRADE_BONUS = 100 # Da sintonizzare! (Potrebbe servire un valore diverso da 0.1)
# Tolleranza angolare: se l'errore è minore di questo, consideriamo allineati
ALIGNMENT_ERROR_THRESHOLD_RAD = np.radians(5.0) # Esempio: 5 gradi
MIN_ALTITUDE_FOR_RETROGRADE_BONUS = 0
                                         # Set higher than NEAR_SURFACE_ALTITUDE_THRESHOLD.

WEIGHT_LOW_SPEED_STASIS_PENALTY = 0.5 # Penalità per velocità troppo bassa
LOW_SPEED_STASIS_THRESHOLD = 1.5     # Soglia di velocità totale per considerare stasi
LOW_SPEED_STASIS_DURATION = 100       # Numero di step consecutivi a bassa velocità per penalizzare
LOW_SPEED_STASIS_MIN_ALT = 50         # Altitudine minima sopra la quale applicare questa penalità (non penalizzare se si sta cercando di atterrare)


RADIAL_SPEED_DESCENT_LIMIT=-0.2
WEIGHT_EXCESSIVE_DESCENT_SPEED_PENALTY = 2
WEIGHT_RADIAL_DIRECTION = 0.05 # Valore da sintonizzare!


# Bonus (valori positivi o da aggiungere)
WEIGHT_APPROACH_SURFACE_BONUS = 15     # Aumentato leggermente per superare la nuova time penalty
WEIGHT_ALIGN_ANGLE_NEAR_SURFACE_BONUS = 100.0
WEIGHT_REDUCE_SPEED_NEAR_SURFACE_BONUS = 90.0  # Da 60.0 a 90.0    # Aumentato leggermente
WEIGHT_SUCCESSFUL_SURFACE_LANDING_BONUS = 20000.0  # Aumentato per renderlo ancora l'obiettivo primario
STATIC_ALTITUDE_BONUS_NUMERATOR = 100
WEIGHT_STATIC_ALTITUDE_BONUS_SCALER = 100 # Molto piccolo
OPTIMAL_DESCENT_SPEED = -0.8  # Velocità di discesa ottimale
WEIGHT_OPTIMAL_DESCENT_BONUS = 3.0  # Peso del bonus
ASSET_FOLDER = "assets"
SHIP_IMAGE_FILE = os.path.join(ASSET_FOLDER, "spaceship.png")
MOON_IMAGE_FILE = os.path.join(ASSET_FOLDER, "moon.png")
BACKGROUND_IMAGE_FILE = os.path.join(ASSET_FOLDER, "starfield.png")
THRUSTER_FLAME_FILE = os.path.join(ASSET_FOLDER, "flame.png")

assets_folder_exists = os.path.isdir(ASSET_FOLDER)
if not assets_folder_exists:
    print(f"ATTENZIONE: Cartella '{ASSET_FOLDER}' non trovata. La grafica userà i fallback.")

AP_ON = False
AP_UPDATE_RATE = 1
AP_KP_ANGLE = 0.6
AP_KD_ANGLE = -1.0
AP_KP_TANGENTIAL_VEL = 0.4
AP_KP_RADIAL_VEL = -0.8
AP_KI_RADIAL_VEL = -0.2
AP_RADIAL_VEL_TARGET_FAR = -0.0
AP_RADIAL_VEL_TARGET_NEAR = -0.0
AP_RADIAL_NEAR_THRESHOLD_ALT = 50
AP_INTEGRAL_MAX = 20.0
AP_THRUST_ACTIVATION_THRESHOLD = -0.05
AP_ROTATION_OUTPUT_THRESHOLD = 0.01

# ... (dopo AP_ROTATION_OUTPUT_THRESHOLD) ...
TRAJECTORY_AP_ON = False # Attivabile dall'utente
TRAJECTORY_UPDATE_RATE = 1 # Ogni quanti step ricalcolare comandi PID

# Esempio di guadagni PID (DA TARARE ACCURATAMENTE!)
PID_POS_KP, PID_POS_KI, PID_POS_KD = 0.1, 0.05*0, 0.3*0  # Per posizione -> velocità desiderata
PID_VEL_KP, PID_VEL_KI, PID_VEL_KD = 0.1, 0.005*0, 0.3  # Per velocità -> forza desiderata
PID_ANGLE_KP, PID_ANGLE_KI, PID_ANGLE_KD = 300, 5, 1600 # Per orientamento -> coppia desiderata

PID_POS_KP, PID_POS_KI, PID_POS_KD = 0.08, 0.02, 0.15

# Per velocità -> forza desiderata (aggiunto Ki per eliminare errori residui)
PID_VEL_KP, PID_VEL_KI, PID_VEL_KD = 0.08, 0.008, 0.25

# Per orientamento -> coppia desiderata (ridotto Kd, aumentato Ki leggermente)
PID_ANGLE_KP, PID_ANGLE_KI, PID_ANGLE_KD = 280, 8, 1200

def normalize_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

class PIDController:
    """
    Controllore PID migliorato per sistemi con attuatori on/off.
    Include funzionalità anti-windup e filtri per il termine derivativo.
    """
    def __init__(self, Kp=1.0, Ki=0.0, Kd=0.0, output_limit=float('inf'), int_limit=float('inf')):
        # Parametri di guadagno
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        
        # Limiti
        self.output_limit = output_limit
        self.int_limit = int_limit
        
        # Variabili di stato
        self.prev_error = 0.0
        self.prev_d = 0.0
        self.i = 0.0
        self.p = 0.0
        self.d = 0.0
        self.last_output = 0.0
        
        # Parametri del filtro
        self.d_filter_alpha = 0.2  # Peso del nuovo valore nel filtro (0-1)
        
    def update_setpoint_based_error(self, error, process_var, dt):
        """
        Aggiorna il controllore PID basato sull'errore di setpoint.
        
        Args:
            error: Differenza tra setpoint e valore attuale
            process_var: Valore attuale della variabile controllata
            dt: Delta tempo dall'ultimo aggiornamento
            
        Returns:
            Output del controllore PID
        """
        # Evita divisione per zero
        if dt <= 0:
            dt = 0.01
            
        # Calcolo termine proporzionale
        self.p = error * self.Kp
        
        # Calcolo termine derivativo con filtro passa-basso
        raw_d = self.Kd * (error - self.prev_error) / dt
        self.d = self.prev_d * (1 - self.d_filter_alpha) + raw_d * self.d_filter_alpha
        self.prev_d = self.d
        
        # Calcolo termine integrale con anti-windup
        if abs(self.last_output) < self.output_limit * 0.95:
            # Accumula integrale solo se l'output non è in saturazione
            self.i += error * dt * self.Ki
            # Limita l'accumulo integrale
            self.i = max(-self.int_limit, min(self.int_limit, self.i))
        
        # Salva errore per il prossimo ciclo
        self.prev_error = error
        
        # Calcolo output finale
        output = self.p + self.i + self.d
        
        # Limita l'output se necessario
        self.last_output = max(-self.output_limit, min(self.output_limit, output))
        
        return self.last_output
    
    def reset(self):
        """Resetta lo stato interno del controllore"""
        self.prev_error = 0.0
        self.prev_d = 0.0
        self.i = 0.0
        self.p = 0.0
        self.d = 0.0
        self.last_output = 0.0
        
    def set_params(self, Kp=None, Ki=None, Kd=None):
        """Aggiorna i parametri del controllore"""
        if Kp is not None:
            self.Kp = Kp
        if Ki is not None:
            self.Ki = Ki
        if Kd is not None:
            self.Kd = Kd
            
    def get_state(self):
        """Restituisce lo stato attuale del PID per debugging"""
        return {
            'P': self.p,
            'I': self.i,
            'D': self.d,
            'output': self.last_output,
            'Kp': self.Kp,
            'Ki': self.Ki,
            'Kd': self.Kd
        }


# --- COSTANTI PER LA NUOVA RICOMPENSA DI TRACCIAMENTO FORMA TRAIETTORIA NN ---



COS_15_DEGREES = np.cos(np.radians(5))  # Valore soglia per il coseno (circa 0.9659)
WEIGHT_PRECISE_ALIGNMENT_BONUS = 2   # Esempio: Bonus per allineamento stretto
WEIGHT_OPPOSITE_DIRECTION_PENALTY_FACTOR =4 # Esempio: Fattore per la pe

class LunarLanderEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 120}

    def __init__(self,
                 render_mode=None,
                 visualize_hitboxes=False,
                 wind_power: float = 0.0,
                 target_render_fps: int = None,
                 turbulence_power: float = 0.0,
                 initial_random_angle: bool = True,
                 random_seed = None,
                 enable_wind: bool = False,
                 landing_pad_width: float = 0.4
                ):
        super().__init__()
        self.previous_dist_to_marker = float('inf')
        self.target_render_fps = target_render_fps if target_render_fps is not None else self.metadata["render_fps"]
        self.nn_guidance_time_elapsed = 0.0 # <<< RIGA DA AGGIUNGERE
        self.nn_trajectory_completed_flag_for_bonus = False # <<< RIGA DA AGGIUNGERE (se usi la logica di reward spaziale)
        self.last_closest_ref_segment_idx = -1 # Per tracciare il progresso spaziale
        self.nn_path_total_segments = 0        # Numero di segmenti nella traiettoria di riferimento

        self.current_trajectory = [] # Cronologia (pos, vel, angle, action) del lander nell'episodio

        self.consecutive_low_speed_steps = 0
        self.render_mode = render_mode
        self.visualize_hitboxes = visualize_hitboxes
        self.episode_info = {} # Aggiungi questa riga

        # --- STORE NEW PARAMETERS ---
        self.wind_power = wind_power
        self.turbulence_power = turbulence_power
        self.apply_initial_random_angle = initial_random_angle # Renamed to avoid clash if parent has it
        self.enable_wind = enable_wind
        self.landing_pad_width = landing_pad_width
        # Note: random_seed will be used in the first reset or to seed self.np_random
        self._initial_seed = random_seed
        # --- END STORE NEW PARAMETERS ---

        self.screen = None
        self.clock = None
        self.is_pygame_initialized = False
        self.font = None
        self.large_font = None

        self.is_recording_hud = False
        self.episode_outcome_message = ""

        self.ship_sprite_original = None
        self.moon_sprite = None
        self.background_sprite = None
        self.flame_sprite_original = None
        self.assets_loaded = False

        self.actual_ship_w = DEFAULT_SHIP_W
        self.actual_ship_h = DEFAULT_SHIP_H
        self.actual_moon_radius = DEFAULT_MOON_RADIUS

        if assets_folder_exists:
            self._lazy_init_pygame()
            if self.is_pygame_initialized:
                 self._load_sprites_and_update_dims()
            else:
                print("WARN: Pygame non inizializzato, uso dimensioni di default anche se assets esistono.")
                self._use_default_dims()
        else:
            self._use_default_dims()

        self.action_space = spaces.MultiDiscrete([2, 3]) # Main engine (OFF/ON), Rotation (OFF/CCW/CW)

        # Observation space:
        # x_pos_norm, y_pos_norm, x_vel_norm, y_vel_norm,
        # ship_angle_norm (relative to its own "up"), angular_vel_norm,
        # altitude_norm, fuel_percentage
        obs_low_list = [
            -SCREEN_WIDTH, -SCREEN_HEIGHT, -10, -10,
            -np.pi, -5*np.pi,
            0, 0.0,
            # NUOVE OSSERVAZIONI PER RL (valori raw prima della normalizzazione in _get_obs)
            -SCREEN_WIDTH, -SCREEN_HEIGHT, # rel_target_x, rel_target_y
            -WAYPOINT_VEL_SCALE*2, -WAYPOINT_VEL_SCALE*2, # rel_target_vx, rel_target_vy
            0.0, # progress_on_nn_traj
            0.0  # is_nn_traj_active_flag
        ]
        obs_high_list = [
            SCREEN_WIDTH, SCREEN_HEIGHT, 10, 10,
            np.pi, 5*np.pi,
            SCREEN_HEIGHT, 1.0,
            # NUOVE OSSERVAZIONI PER RL
            SCREEN_WIDTH, SCREEN_HEIGHT,
            WAYPOINT_VEL_SCALE*2, WAYPOINT_VEL_SCALE*2,
            1.0,
            1.0
        ]
        self.observation_space = spaces.Box(low=np.array(obs_low_list, dtype=np.float32),
                                            high=np.array(obs_high_list, dtype=np.float32),
                                            shape=(len(obs_low_list),), dtype=np.float32) # Shape ora è 14

        
        self.is_nn_trajectory_guidance_active = False # Flag per indicare se l'agente RL deve seguire la traiettoria NN
        self.nn_trajectory_points = None # Traiettoria generata dalla NN da seguire
        self.current_nn_waypoint_index = 0
        self.previous_distance_to_nn_waypoint = float('inf')
        self.nn_trajectory_total_waypoints = 0


        self.ship_pos = None
        self.ship_vel = None
        self.ship_angle = None # Relative angle of the ship
        self.ship_angular_vel = None
        self.current_fuel = None
        self.current_mass = None
        self.fuel_empty = False
        self.steps_taken = 0

        self.autopilot_on = AP_ON
        self.ap_integral_radial_error = 0.0
        self.ap_last_radial_error = 0.0

        self.current_episode_score = 0.0
        self.landing_target_circumference_angle = None  # Angle in radians
        self.landing_target_display_pos = None 

        self.preview_target_display_pos = None          # NUOVO: Posizione per anteprima
        self.show_target_selected_message = False       # NUOVO: Flag per messaggio feedback
        self.target_selected_message_start_time = 0     # NUOVO: Timer per messaggio feedback
        self.is_in_targeting_mode = False # NUOVO: Stato per modalità mira
        # Seed the environment if a seed was passed to __init__
        # This is a fallback if the agent/wrapper doesn't call env.seed() or env.reset(seed=...)
        self.autopilot_on = AP_ON
        self.trajectory_autopilot_on = TRAJECTORY_AP_ON # Nuovo flag
        self.trajectory_points = None
        self.current_trajectory_target_idx = 0
        self.trajectory_time_elapsed = 0.0

        # Limiti per output PID (es. MAIN_THRUST_FORCE, ROTATION_THRUST_TORQUE)
        # Questi PID controlleranno la *forza* o *coppia* desiderata.
        # Un PID per la velocità X e uno per la Y, che generano la forza desiderata
        self.pid_vel_x = PIDController(Kp=PID_VEL_KP, Ki=PID_VEL_KI, Kd=PID_VEL_KD,
                                       output_limit=MAIN_THRUST_FORCE,
                                       int_limit=MAIN_THRUST_FORCE/2)
        self.pid_vel_y = PIDController(Kp=PID_VEL_KP, Ki=PID_VEL_KI, Kd=PID_VEL_KD,
                                       output_limit=MAIN_THRUST_FORCE,
                                       int_limit=MAIN_THRUST_FORCE/2)

        # PID per l'angolo, che genera la coppia desiderata
        self.pid_angle = PIDController(Kp=PID_ANGLE_KP, Ki=PID_ANGLE_KI, Kd=PID_ANGLE_KD,
                                       output_limit=ROTATION_THRUST_TORQUE*1.5, # Un po' di margine
                                       int_limit=ROTATION_THRUST_TORQUE)
        self.current_trajectory = [] # <<< AGGIUNGI QUESTA RIGA: Inizializza come lista vuota

        self.ap_trajectory_step_counter = 0 # Per TRAJECTORY_UPDATE_RATE



        if self._initial_seed is not None:
            self.seed(self._initial_seed)
    def set_trajectory(self, trajectory_points_list):
        """
        Imposta la traiettoria da seguire.
        Usata sia dall'autopilota PID ('P') sia come input per l'agente RL (quando 'N' viene premuto).
        """
        if trajectory_points_list and isinstance(trajectory_points_list, list) and len(trajectory_points_list) > 0:
            self.trajectory_points = sorted(trajectory_points_list, key=lambda p: p['t'])
            # Se l'agente RL deve usare questa traiettoria, impostare i suoi parametri specifici
            # Questo avviene esternamente o quando l'utente preme 'N' e attiva la logica RL
            # Qui resettiamo solo l'indice e il tempo per l'autopilota PID.
            self.current_trajectory_target_idx = 0 # Per PID
            self.trajectory_time_elapsed = 0.0     # Per PID
            # Resetta i PID quando una nuova traiettoria è impostata (per autopilota PID)
            self.pid_vel_x.reset()
            self.pid_vel_y.reset()
            self.pid_angle.reset()
            print(f"INFO (set_trajectory): Traiettoria di {len(self.trajectory_points)} punti caricata e PID resettati.")
        else:
            self.trajectory_points = None
            self.current_trajectory_target_idx = 0
            print("INFO (set_trajectory): Traiettoria rimossa o non valida.")
    def _find_closest_point_on_polyline(self, point, polyline_points_list_of_arrays):
        """
        Trova il punto più vicino su una polilinea (lista di np.array) a un dato punto.
        Restituisce: (closest_point_on_path, min_distance, best_segment_index, tangent_vector_at_closest_point)
        Restituisce (None, float('inf'), -1, None) se la polilinea è invalida.
        """
        if not polyline_points_list_of_arrays or len(polyline_points_list_of_arrays) == 0:
            return None, float('inf'), -1, None
        
        # Se la polilinea ha un solo punto
        if len(polyline_points_list_of_arrays) == 1:
            closest_pt = np.array(polyline_points_list_of_arrays[0])
            dist = np.linalg.norm(point - closest_pt)
            return closest_pt, dist, 0, None # Nessuna tangente definibile

        min_dist_sq = float('inf')
        closest_point_overall = None
        best_segment_idx = -1
        tangent_at_closest = None

        for i in range(len(polyline_points_list_of_arrays) - 1):
            p1 = np.array(polyline_points_list_of_arrays[i])
            p2 = np.array(polyline_points_list_of_arrays[i+1])
            
            line_vec = p2 - p1
            point_vec = point - p1
            segment_len_sq = np.dot(line_vec, line_vec)

            current_closest_on_segment = None
            if segment_len_sq < 1e-12: # Il segmento è praticamente un punto
                current_closest_on_segment = p1
                # t per il calcolo della tangente non è ben definito, usiamo la direzione del prossimo segmento se possibile
            else:
                t = np.dot(point_vec, line_vec) / segment_len_sq
                if t < 0.0:
                    current_closest_on_segment = p1
                elif t > 1.0:
                    current_closest_on_segment = p2
                else:
                    current_closest_on_segment = p1 + t * line_vec
            
            dist_sq = np.sum((point - current_closest_on_segment)**2)

            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
                closest_point_overall = current_closest_on_segment
                best_segment_idx = i
                # Calcola la tangente del segmento che contiene il punto più vicino
                current_segment_norm = np.linalg.norm(line_vec) # Calcola la norma
                if current_segment_norm > 1e-9:
                    tangent_at_closest = line_vec / current_segment_norm # Normalizza correttamente
                else: # Segmento degenere, prova a guardare il prossimo se esiste
                    if i + 2 < len(polyline_points_list_of_arrays):
                        p2_val = polyline_points_list_of_arrays[i+1] # Assicurati che p2 sia corretto qui
                        next_line_vec = polyline_points_list_of_arrays[i+2] - p2_val # Usa p2_val
                        
                        next_segment_norm = np.linalg.norm(next_line_vec) # Calcola la norma
                        if next_segment_norm > 1e-9:
                            tangent_at_closest = next_line_vec / next_segment_norm # Normalizza correttamente
                        else: 
                            tangent_at_closest = None
                    else: 
                        tangent_at_closest = None
        
        if closest_point_overall is None : # Dovrebbe essere impostato se polyline_points non è vuota
            return None, float('inf'), -1, None
            
        return closest_point_overall, np.sqrt(min_dist_sq), best_segment_idx, tangent_at_closest
    def activate_nn_trajectory_guidance(self, nn_trajectory_points_list):
        if nn_trajectory_points_list and isinstance(nn_trajectory_points_list, list) and len(nn_trajectory_points_list) >= 2:
            self.nn_trajectory_points = sorted(nn_trajectory_points_list, key=lambda p: p['t'])
            self.is_nn_trajectory_guidance_active = True
            
            self.nn_guidance_time_elapsed = 0.0
            self.nn_trajectory_completed_flag_for_bonus = False
            self.last_closest_ref_segment_idx = -1 # Inizializza per il tracciamento del progresso
            self.nn_path_total_segments = len(self.nn_trajectory_points) - 1

            self.current_trajectory = [] # Pulisci la cronologia del lander per la nuova guida

            # Per _get_obs (se ancora usa info basate su waypoint)
            self.current_nn_waypoint_index = 0 
            self.nn_trajectory_total_waypoints = len(self.nn_trajectory_points)
            if self.ship_pos is not None:
                first_wp_pos = np.array(self.nn_trajectory_points[0]['pos'])
                self.previous_distance_to_nn_waypoint = np.linalg.norm(self.ship_pos - first_wp_pos)
            else:
                self.previous_distance_to_nn_waypoint = float('inf')
            
            self.autopilot_on = False 
            self.trajectory_autopilot_on = False
                    
            
        else:
            self.is_nn_trajectory_guidance_active = False
            self.nn_trajectory_points = None
            self.current_nn_waypoint_index = 0
            self.nn_trajectory_total_waypoints = 0
            self.nn_guidance_time_elapsed = 0.0
            self.nn_trajectory_completed_flag_for_bonus = False
            print("INFO (activate_nn_trajectory_guidance): Guida RL su traiettoria NN DISATTIVATA (traiettoria non valida).")


    def _get_current_trajectory_target(self):
        if not self.trajectory_points:
            return None

        # Avanza l'indice se il tempo corrente ha superato il tempo del target attuale
        # o se siamo molto vicini al punto target (per evitare di rimanere bloccati)
        while (self.current_trajectory_target_idx < len(self.trajectory_points) - 1 and
               self.trajectory_time_elapsed >= self.trajectory_points[self.current_trajectory_target_idx]['t']):
            # Potremmo aggiungere una logica per verificare se siamo vicini al punto
            # current_target = self.trajectory_points[self.current_trajectory_target_idx]
            # distance_to_target = np.linalg.norm(self.ship_pos - np.array(current_target['pos']))
            # if distance_to_target < 5.0: # Esempio: 5 pixel
            self.current_trajectory_target_idx += 1

        return self.trajectory_points[self.current_trajectory_target_idx]
    
    
    def set_parameters(self, wind_power: float = None, turbulence_power: float = None, enable_wind: bool = None, initial_random_angle: bool = None):
            """
            Permette di modificare dinamicamente alcuni parametri dell'ambiente.
            """
            if wind_power is not None:
                self.wind_power = wind_power
            if turbulence_power is not None:
                self.turbulence_power = turbulence_power
            if enable_wind is not None:
                self.enable_wind = enable_wind
            if initial_random_angle is not None:
                # Assumendo che l'attributo che controlla questo sia self.apply_initial_random_angle
                self.apply_initial_random_angle = initial_random_angle
            
            # Stampa di debug semplificata per evitare errori con self.verbose
            # Questa stampa avverrà ogni volta che i parametri vengono cambiati tramite questo metodo.
            current_angle_setting = 'N/A'
            if hasattr(self, 'apply_initial_random_angle'):
                current_angle_setting = self.apply_initial_random_angle

            print(f"DEBUG: LunarLanderEnv parametri aggiornati via set_parameters. Nuovi valori: "
                  f"wind_power={self.wind_power}, "
                  f"turbulence_power={self.turbulence_power}, "
                  f"enable_wind={self.enable_wind}, "
                  f"initial_random_angle={current_angle_setting}")
    
    def reset_episode_info(self):
        """
        Resetta le informazioni dell'episodio accumulate.
        Chiamato dal LunarLanderStatsCallback.
        """
        self.episode_info = {}

    def seed(self, seed=None):
        """Seeds the environment's random number generator."""
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        self._initial_seed = seed # Store the seed actually used
        # print(f"INFO: LunarLanderEnv seeded with {seed}") # For debugging
        return [seed]

    def _use_default_dims(self):
        self.actual_ship_w = DEFAULT_SHIP_W
        self.actual_ship_h = DEFAULT_SHIP_H
        self.actual_moon_radius = DEFAULT_MOON_RADIUS
        print(f"INFO: Usate dimensioni di default: Ship({self.actual_ship_w}x{self.actual_ship_h}), MoonRadius({self.actual_moon_radius:.1f})")

    def _lazy_init_pygame(self):
        if self.is_pygame_initialized:
            return
        print("INFO: Inizializzazione Pygame...")
        try:
            pygame.init()
            pygame.font.init()
            if pygame.display.get_surface() is None and assets_folder_exists:
                pygame.display.set_mode((1, 1), pygame.NOFRAME)
            if self.render_mode == "human":
                self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
                pygame.display.set_caption("Lunar Lander AI")
            elif self.render_mode == "rgb_array":
                self.screen = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont(None, 24)
            self.large_font = pygame.font.SysFont(None, 72)
            self.is_pygame_initialized = True
            print("INFO: Pygame inizializzato.")
        except pygame.error as e:
            print(f"ERRORE Pygame: Inizializzazione Pygame fallita: {e}")
            print("Potrebbe essere necessario un display server (es. Xvfb) per ambienti headless se pygame.init() fallisce.")
            self.is_pygame_initialized = False; self.font = None; self.large_font = None; self.screen = None
        except Exception as e:
            print(f"ERRORE generico: Inizializzazione Pygame fallita: {e}")
            self.is_pygame_initialized = False; self.font = None; self.large_font = None; self.screen = None

    def _load_sprites_and_update_dims(self):
        if not self.is_pygame_initialized:
            print("WARN: Impossibile caricare sprite: Pygame non inizializzato (chiamare _lazy_init_pygame prima).")
            self._use_default_dims(); return
        print("INFO: Caricamento e scaling sprite...")
        self.assets_loaded = False; loaded_count = 0; essential_loaded = True
        def load_and_scale(filepath, scale_factor):
            try:
                img = pygame.image.load(filepath).convert_alpha()
                if scale_factor != 1.0:
                    orig_w, orig_h = img.get_size()
                    new_w = max(1, int(orig_w * scale_factor)); new_h = max(1, int(orig_h * scale_factor))
                    scaled_img = pygame.transform.scale(img, (new_w, new_h)); return scaled_img
                else: return img
            except pygame.error as e: print(f" -> ERRORE caricamento {os.path.basename(filepath)}: {e}"); return None
            except Exception as e: print(f" -> ERRORE generico caricamento {os.path.basename(filepath)}: {e}"); return None
        self.ship_sprite_original = load_and_scale(SHIP_IMAGE_FILE, SHIP_SCALE_FACTOR)
        self.moon_sprite = load_and_scale(MOON_IMAGE_FILE, MOON_SCALE_FACTOR)
        self.background_sprite = load_and_scale(BACKGROUND_IMAGE_FILE, 1.0)
        self.flame_sprite_original = load_and_scale(THRUSTER_FLAME_FILE, FLAME_SCALE_FACTOR)
        if self.ship_sprite_original: self.actual_ship_w=self.ship_sprite_original.get_width(); self.actual_ship_h=self.ship_sprite_original.get_height()-10; loaded_count+=1
        else: essential_loaded = False; self.actual_ship_w = DEFAULT_SHIP_W; self.actual_ship_h = DEFAULT_SHIP_H
        if self.moon_sprite: self.actual_moon_radius = self.moon_sprite.get_width()/2; loaded_count+=1
        else: self.actual_moon_radius = DEFAULT_MOON_RADIUS
        if self.background_sprite: loaded_count += 1
        if self.flame_sprite_original: loaded_count += 1
        self.assets_loaded = essential_loaded
        if not essential_loaded : print("WARN: Caricamento Sprite Essenziali FALLITO! Uso dimensioni di default."); self._use_default_dims()
        print(f"INFO: Caricamento/Scaling sprite completato. {loaded_count} asset caricati.")
        print(f"INFO: Dimensioni fisiche in uso: Ship({self.actual_ship_w}x{self.actual_ship_h}), MoonRadius({self.actual_moon_radius:.1f})")

    def _get_local_vertical_angle_and_altitude(self):
        if self.ship_pos is None: return 0.0, SCREEN_HEIGHT
        vector_center_to_ship = self.ship_pos - np.array([CENTER_X, CENTER_Y])
        dist_center_moon = np.linalg.norm(vector_center_to_ship)
        altitude = dist_center_moon - self.actual_moon_radius
        if dist_center_moon < 1e-6: local_vertical_angle = 0.0
        else: local_vertical_angle = np.arctan2(vector_center_to_ship[1], vector_center_to_ship[0])
        return local_vertical_angle, altitude

    def _get_obs(self):
        if self.ship_pos is None: return np.zeros(self.observation_space.shape, dtype=np.float32)
        _, altitude = self._get_local_vertical_angle_and_altitude()
        normalized_relative_angle = normalize_angle(self.ship_angle) / np.pi # Assumendo ship_angle sia l'angolo relativo del lander
        fuel_percentage = (self.current_fuel / INITIAL_FUEL_MASS) if INITIAL_FUEL_MASS > 0 else 0.0

        base_obs_list = [
            self.ship_pos[0]/SCREEN_WIDTH, self.ship_pos[1]/SCREEN_HEIGHT,
            self.ship_vel[0]/10.0, self.ship_vel[1]/10.0,
            normalized_relative_angle, self.ship_angular_vel/(np.pi),
            max(0, altitude)/(SCREEN_HEIGHT/2), fuel_percentage
        ]

        # --- AGGIUNTA OSSERVAZIONI PER GUIDA RL SU TRAIETTORIA NN ---
        rel_target_x_norm, rel_target_y_norm = 0.0, 0.0
        rel_target_vx_norm, rel_target_vy_norm = 0.0, 0.0
        progress_on_nn_traj_norm = 0.0
        is_nn_traj_active_flag = 0.0

        if self.is_nn_trajectory_guidance_active and self.nn_trajectory_points and \
           self.current_nn_waypoint_index < len(self.nn_trajectory_points):
            
            is_nn_traj_active_flag = 1.0
            target_wp = self.nn_trajectory_points[self.current_nn_waypoint_index]
            target_pos_px = np.array(target_wp['pos'])
            target_vel_px_s = np.array(target_wp['vel'])

            rel_target_x_norm = (target_pos_px[0] - self.ship_pos[0]) / WAYPOINT_COORD_SCALE_XY
            rel_target_y_norm = (target_pos_px[1] - self.ship_pos[1]) / WAYPOINT_COORD_SCALE_XY
            
            rel_target_vx_norm = (target_vel_px_s[0] - self.ship_vel[0]) / WAYPOINT_VEL_SCALE
            rel_target_vy_norm = (target_vel_px_s[1] - self.ship_vel[1]) / WAYPOINT_VEL_SCALE

            if self.nn_trajectory_total_waypoints > 0:
                progress_on_nn_traj_norm = min(1.0, float(self.current_nn_waypoint_index) / self.nn_trajectory_total_waypoints)
        elif self.is_nn_trajectory_guidance_active and self.nn_trajectory_points and \
             self.current_nn_waypoint_index >= len(self.nn_trajectory_points):
            # Traiettoria NN completata, l'agente è in fase di atterraggio finale
            is_nn_traj_active_flag = 1.0 # Manteniamo attivo per indicare che la fase è post-traiettoria
            progress_on_nn_traj_norm = 1.0
            # rel_target_x/y/vx/vy rimangono 0 (nessun waypoint specifico da seguire)
            # L'agente ora deve usare altitudine, velocità, angolo per atterrare.

        rl_obs_list = [
            rel_target_x_norm, rel_target_y_norm,
            rel_target_vx_norm, rel_target_vy_norm,
            progress_on_nn_traj_norm,
            is_nn_traj_active_flag
        ]
        # --- FINE AGGIUNTA OSSERVAZIONI RL ---

        observation = np.array(base_obs_list + rl_obs_list, dtype=np.float32)

        if not np.all(np.isfinite(observation)):
            print(f"WARN: Osservazione non finita! Stato: p={self.ship_pos}, v={self.ship_vel}, rel_a={self.ship_angle}, av={self.ship_angular_vel}, fuel={self.current_fuel}")
            print(f"RL Obs: rel_target_x={rel_target_x_norm}, rel_target_y={rel_target_y_norm}, prog={progress_on_nn_traj_norm}")
            observation = np.nan_to_num(observation, nan=0.0, posinf=self.observation_space.high, neginf=self.observation_space.low)
            observation = np.clip(observation, self.observation_space.low, self.observation_space.high)
        return observation
    
    
    
    def _get_polar_velocities(self):
        if self.ship_pos is None or self.ship_vel is None: return 0.0, 0.0
        vector_center_to_ship = self.ship_pos - np.array([CENTER_X, CENTER_Y])
        distance_to_center = np.linalg.norm(vector_center_to_ship)
        if distance_to_center < 1e-6: return 0.0, 0.0
        radial_unit_vector = vector_center_to_ship / distance_to_center
        tangential_unit_vector = np.array([-radial_unit_vector[1], radial_unit_vector[0]])
        radial_speed = np.dot(self.ship_vel, radial_unit_vector)
        tangential_speed = np.dot(self.ship_vel, tangential_unit_vector)
        return radial_speed, tangential_speed

    def _get_info(self):
        if self.ship_pos is None: return {}
        local_vertical_angle, altitude = self._get_local_vertical_angle_and_altitude()
        radial_speed, tangential_speed = self._get_polar_velocities()
        absolute_ship_angle_world = normalize_angle(local_vertical_angle + self.ship_angle)

        current_fuel_percentage = (self.current_fuel / INITIAL_FUEL_MASS) * 100 if INITIAL_FUEL_MASS > 0 else 0

        info_dict = {
            "altitude_surface": altitude,
            "speed_total": np.linalg.norm(self.ship_vel),
            "speed_x": self.ship_vel[0], "speed_y": self.ship_vel[1],
            "speed_radial": radial_speed, "speed_tangential": tangential_speed,
            "ship_angle_relative_deg": np.degrees(normalize_angle(self.ship_angle)),
            "ship_angle_absolute_deg": np.degrees(absolute_ship_angle_world),
            "local_vertical_angle_deg": np.degrees(local_vertical_angle),
            "angular_velocity_dps": np.degrees(self.ship_angular_vel),
            "current_fuel": self.current_fuel,
            "fuel_percentage": current_fuel_percentage, # Modificato per riutilizzo
            "fuel_remaining_percent": current_fuel_percentage, # Aggiunto per EnhancedRenderCallback
            "fuel_empty": self.fuel_empty,
            "autopilot_on": self.autopilot_on,
            "steps": self.steps_taken,
        }
        return info_dict
    
    def reset(self, seed=None, options=None):
        # If a seed is passed to reset, it takes precedence. Otherwise, use the one from __init__ if available.
        if seed is None and self._initial_seed is not None:
            seed = self._initial_seed
            # Avoid re-seeding with the same initial seed on every reset unless explicitly passed.
            # self.np_random might have already been seeded by an explicit call to env.seed() by the wrapper.
            # If self.np_random is None, this is the first reset, seed it.
        # if self.np_random is None or seed is not None: # Only re-seed if new seed or not seeded yet
        super().reset(seed=seed) # This will seed self.np_random via gym.utils.seeding
        # --- RESET VARIABILI GUIDA RL SU TRAIETTORIA NN ---
        self.current_trajectory = [] # <<< AGGIUNGI QUESTA RIGA: Inizializza come lista vuota
        # --- RESET VARIABILI GUIDA RL SU TRAIETTORIA NN ---
        self.is_nn_trajectory_guidance_active = False 
        self.nn_trajectory_points = None
        self.current_nn_waypoint_index = 0       # Per _get_obs e vecchio sistema di waypoint (se riattivato)
        self.previous_distance_to_nn_waypoint = float('inf') # Per _get_obs
        self.nn_trajectory_total_waypoints = 0   # Per _get_obs

        self.nn_guidance_time_elapsed = 0.0      # Per la nuova logica di tracciamento temporale/spaziale
        self.nn_trajectory_completed_flag_for_bonus = False # Per bonus di completamento spaziale
        self.last_closest_ref_segment_idx = -1   # Per tracciare progresso sulla forma
        self.nn_path_total_segments = 0
        # --- FINE RESET VARIABILI RL ---
        self.current_trajectory_target_idx = 0
        self.trajectory_time_elapsed = 0.0
        self.pid_vel_x.reset()
        self.pid_vel_y.reset()
        self.pid_angle.reset()
        self.ap_trajectory_step_counter = 0
        self.consecutive_low_speed_steps = 0
        self.nn_guidance_time_elapsed = 0.0 # <<< RIGA DA AGGIUNGERE/CONFERMARE
        self.nn_trajectory_completed_flag_for_bonus = False # <<< RIGA DA AGGIUNGERE/CONFERMARE
        
        START_ALTITUDE_PIXELS = self.np_random.uniform(low=150.0, high=250.0)
        current_moon_radius_pixels = max(1.0, self.actual_moon_radius)
        orbit_radius_pixels = current_moon_radius_pixels + START_ALTITUDE_PIXELS
        self.trajectory_autopilot_on = TRAJECTORY_AP_ON # Resetta allo stato di default
        if self.trajectory_points: # Se una traiettoria era caricata per il PID
            self.current_trajectory_target_idx = 0
            self.trajectory_time_elapsed = 0.0

        # Initial orbital position (always randomized for now)
        start_pos_angle_rad = self.np_random.uniform(0, 2 * np.pi)
        # 1. SCEGLI UN'ALTITUDINE DI PARTENZA FISSA (in pixels sopra la superficie)
        # START_ALTITUDE_PIXELS = self.np_random.uniform(low=100.0, high=200.0) # RIGA ORIGINALE CASUALE
        #START_ALTITUDE_PIXELS = 100.0  # ESEMPIO: Altitudine fissa di 150 pixels

        current_moon_radius_pixels = max(1.0, self.actual_moon_radius)
        orbit_radius_pixels = current_moon_radius_pixels + START_ALTITUDE_PIXELS

        # 2. SCEGLI UN ANGOLO DI PARTENZA FISSO (in radianti)
        # start_pos_angle_rad = self.np_random.uniform(0, 2 * np.pi) # RIGA ORIGINALE CASUALE
        start_pos_angle_rad = np.pi # ESEMPIO: 0 radianti (a destra del centro della luna, "ore 3")
        # Altri esempi per start_pos_angle_rad:
        # start_pos_angle_rad = np.pi / 2  # Sotto il centro ("ore 6")
        # start_pos_angle_rad = np.pi      # A sinistra del centro ("ore 9")
        # start_pos_angle_rad = 3 * np.pi / 2 # Sopra il centro ("ore 12")
        # start_pos_angle_rad = np.deg2rad(45) # Angolo specifico in gradi, es. 45°

        # --- FINE MODIFICHE PER POSIZIONE DI PARTENZA ---
        self.ship_pos = np.array([CENTER_X + orbit_radius_pixels * np.cos(start_pos_angle_rad),
                                CENTER_Y + orbit_radius_pixels * np.sin(start_pos_angle_rad)], dtype=np.float32)

        # Dopo che self.ship_pos è stato inizializzato:
        if self.landing_target_display_pos is not None and self.ship_pos is not None:
            self.previous_dist_to_marker = np.linalg.norm(self.ship_pos - np.array(self.landing_target_display_pos))
        else:
            self.previous_dist_to_marker = float('inf')

        # Assicurati che le chiavi per final_info siano presenti se le usi per logging
        self.episode_info.update({
            'reward_comp_approaching_marker': 0.0,
            'reward_comp_precision_landing': 0.0,
            'final_dist_to_marker_on_land': -1.0
        })
        try:
            orbit_speed_magnitude = np.sqrt(GRAVITY_CONST / max(orbit_radius_pixels, 1.0))
        except ValueError: # Should not happen with max(..., 1.0)
            orbit_speed_magnitude = 1.0

        vel_dir_angle = start_pos_angle_rad - np.pi / 2 # Tangential velocity for orbit
        self.ship_vel = np.array([orbit_speed_magnitude * np.cos(vel_dir_angle),
                                  orbit_speed_magnitude * np.sin(vel_dir_angle)], dtype=np.float32)

        ADD_VELOCITY_PERTURBATION = True # Keep this for variability
        if ADD_VELOCITY_PERTURBATION:
            vel_perturb_angle = self.np_random.uniform(0, 2*np.pi)
            vel_perturb_magn = self.np_random.uniform(-0.5,0.5) # Small perturbation
            self.ship_vel += np.array([vel_perturb_magn*np.cos(vel_perturb_angle), vel_perturb_magn*np.sin(vel_perturb_angle)])

        # --- USE self.apply_initial_random_angle ---
        if self.apply_initial_random_angle:
            self.ship_angle = self.np_random.uniform(-np.pi/3, np.pi/3) # Wider random angle if true
            self.ship_angular_vel = self.np_random.uniform(-0.1, 0.1)   # Slightly more initial spin
        else:
            self.ship_angle = self.np_random.uniform(-np.pi/12, np.pi/12) # Narrower random angle (closer to upright)
            self.ship_angular_vel = self.np_random.uniform(-0.02, 0.02)  # Less initial spin
        # --- END USE ---

        self.current_fuel = INITIAL_FUEL_MASS
        self.current_mass = DRY_MASS + self.current_fuel
        self.fuel_empty = False
        self.steps_taken = 0
        self.autopilot_on = AP_ON # Reset autopilot state based on constant
        self.ap_integral_radial_error = 0.0
        self.is_recording_hud = False
        self.episode_outcome_message = ""
        self.current_episode_score = 0.0
        self.episode_info = {} # Aggiungi/Modifica questa riga
        self.show_target_selected_message = False # Reset feedback
        self.is_in_targeting_mode = False # NUOVO: Assicura che la modalità sia disattivata

        if self.render_mode == "human" or self.render_mode == "rgb_array":
            self._lazy_init_pygame() # Ensure pygame is ready for rendering

        observation = self._get_obs()
        info_data = self._get_info()

        if self.render_mode == "human":
            self._render_frame()
         # Inizializzazione per il bonus di consistenza della discesa
        self.prev_speed_radial_direction = 0 # 0: non definito, -1: discesa, 1: salita
        self.descent_consistency_counter = 0

        # Esempio: se volessi definire un landing_target_pos specifico per l'episodio
        # self.landing_target_pos = np.array([CENTER_X, CENTER_Y - self.actual_moon_radius - 2]) # Esempio: poco sopra il "polo nord" lunare

        if self.render_mode == "human" or self.render_mode == "rgb_array":
            self._lazy_init_pygame() # Ensure pygame is ready for rendering

        observation = self._get_obs()
        info_data = self._get_info()

        if self.render_mode == "human":
            self._render_frame()
        return observation, info_data

    def _get_trajectory_autopilot_action(self):
        """
        Funzione principale di controllo orbitale migliorata.
        Con parametri PID preimpostati per sistemi con thruster on/off.
        """
        if not self.trajectory_points or self.ship_pos is None or self.ship_vel is None:
            return [0, 0]  # Nessuna azione se non c'è traiettoria o stato

        current_target_waypoint = self._get_current_trajectory_target()
        if not current_target_waypoint:
            print("WARN: Nessun waypoint valido dalla traiettoria.")
            return [0, 0]

        target_pos = np.array(current_target_waypoint['pos'], dtype=np.float32)
        target_vel = np.array(current_target_waypoint['vel'], dtype=np.float32)

        # Calcolo distanza dal target 
        distance_to_target = np.linalg.norm(target_pos - self.ship_pos)
        
        # --- 1. Calcolo della correzione di posizione ---
        # Guadagno variabile basato sulla distanza per migliore convergenza
        pos_correction_gain = min(0.01, 0.05 / (1 + distance_to_target * 0.001))
        error_pos = target_pos - self.ship_pos
        pos_correction_vel = error_pos * pos_correction_gain
        
        # Combiniamo errore di posizione e velocità
        error_vel_x = (target_vel[0] + pos_correction_vel[0]) - self.ship_vel[0]
        error_vel_y = (target_vel[1] + pos_correction_vel[1]) - self.ship_vel[1]

        # --- 2. Calcolo forza di spinta necessaria ---
        # Output PID per la forza desiderata nelle direzioni X e Y
        desired_accel_by_pid_x = self.pid_vel_x.update_setpoint_based_error(error_vel_x, self.ship_vel[0], TIME_STEP)
        desired_accel_by_pid_y = self.pid_vel_y.update_setpoint_based_error(error_vel_y, self.ship_vel[1], TIME_STEP)

        desired_thrust_force_world_x = self.current_mass * desired_accel_by_pid_x
        desired_thrust_force_world_y = self.current_mass * desired_accel_by_pid_y

        # Compensazione gravità
        direction_to_center_grav = np.array([CENTER_X, CENTER_Y]) - self.ship_pos
        distance_sq_grav = max(np.dot(direction_to_center_grav, direction_to_center_grav), 1.0)
        dist_to_center_grav = np.sqrt(distance_sq_grav)
        gravity_magnitude_ap = GRAVITY_CONST / distance_sq_grav
        
        # Calcolo del vettore forza gravitazionale
        if dist_to_center_grav > 1e-9:
            current_gravity_force_world = direction_to_center_grav / dist_to_center_grav * gravity_magnitude_ap
        else:
            current_gravity_force_world = np.zeros(2)

        # La spinta richiesta è la forza calcolata dai PID + compensazione gravità
        required_thrust_vector_world = np.array([desired_thrust_force_world_x, desired_thrust_force_world_y]) + current_gravity_force_world*0

        # --- 3. Calcolo orientamento target ---
        required_thrust_magnitude = np.linalg.norm(required_thrust_vector_world)
        
        # Calcolo orientamento attuale (spostato qui sopra per risolvere il problema dell'accesso alla variabile)
        local_vertical_angle_ap, _ = self._get_local_vertical_angle_and_altitude()
        self.current_absolute_ship_orientation_world = normalize_angle(local_vertical_angle_ap + self.ship_angle)
        
        # Strategia di orientamento basata sulla situazione
        #if required_thrust_magnitude > 0.15 * MAIN_THRUST_FORCE and required_thrust_magnitude < MAIN_THRUST_FORCE * 1.2:
        if required_thrust_magnitude > 0.15 * MAIN_THRUST_FORCE:
            # Per spinte significative, orienta nella direzione della spinta
            target_ship_orientation_world = np.arctan2(required_thrust_vector_world[1], required_thrust_vector_world[0])
        else:
            # Per spinte minime, valuta la strategia migliore in base alla situazione orbitale
            # MODIFICA: Per spinte minime, mantieni l'orientamento attuale a meno che non sia necessario ruotare
            if required_thrust_magnitude < 0.05 * MAIN_THRUST_FORCE:  # Soglia ancora più bassa
                # Mantieni l'orientamento attuale per evitare rotazioni non necessarie
                target_ship_orientation_world = self.current_absolute_ship_orientation_world
            else:
                # Quando la spinta è modesta, mantieni l'allineamento con la velocità per efficienza orbitale
                vel_direction = np.arctan2(self.ship_vel[1], self.ship_vel[0])
                target_ship_orientation_world = vel_direction
        
        # --- 4. PID per l'orientamento ---
        # Calcolo errore orientamento
             
        angle_error = normalize_angle(target_ship_orientation_world - self.current_absolute_ship_orientation_world)
        if abs(angle_error) < 0.02:  # ~1 grado
            angle_error = 0
        # Output PID per la coppia desiderata
        desired_torque = self.pid_angle.update_setpoint_based_error(angle_error, self.ship_angle, TIME_STEP)
        
        # --- 5. Conversione in azioni discrete ---
        thrust_cmd = 0
        rotation_cmd = 0
        
        # Strategia migliorata per l'attivazione della spinta principale
        alignment_threshold = np.radians(10.0)  # Soglia di allineamento in radianti (10 gradi)
        
        # Attiva la spinta quando:
        # - L'orientamento è sufficientemente allineato 
        # - La spinta richiesta è significativa ma non eccessiva
        if abs(angle_error) < alignment_threshold:
            # Attiva solo se la spinta richiesta è in un range ragionevole
            #if required_thrust_magnitude > 0.15 * MAIN_THRUST_FORCE and required_thrust_magnitude < MAIN_THRUST_FORCE * 1.2:
            if required_thrust_magnitude > 0.15 * MAIN_THRUST_FORCE:
                thrust_cmd = 1
        
        # MODIFICA: Controllo di rotazione migliorato per evitare rotazioni eccessive
        # Modifica nella parte del controllo di rotazione
        # Aumenta significativamente la deadband quando la spinta è bassa
        if required_thrust_magnitude < 0.15 * MAIN_THRUST_FORCE:
            # Deadband molto più ampia quando il thrust non è attivo
            rotation_deadband = ROTATION_THRUST_TORQUE * 0.3  # 6 volte più grande
        else:
            # Deadband normale quando potrebbe essere necessario allinearsi
            rotation_deadband = ROTATION_THRUST_TORQUE * 0.05
        
        # Controllo predittivo migliorato
        angular_velocity = getattr(self, 'ship_angular_vel', 0)  # Valore di default se non esiste
        predicted_angle_error = angle_error + angular_velocity * 0.5  # predizione per 0.5 secondi
        
        # MODIFICA: Aumenta la soglia e aggiungi controllo isteresi per angoli piccoli
        if abs(predicted_angle_error) < np.radians(5.0):  # Aumentata da 2.0 a 5.0 gradi
            # Se l'errore previsto è piccolo, lascia che la nave si stabilizzi da sola
            rotation_cmd = 0
        # Aggiungi controllo di isteresi - usa una soglia più grande per FERMARE la rotazione
        elif abs(angle_error) < np.radians(3.0) and abs(angular_velocity) < np.radians(1.0):
            # Se l'errore è piccolo e la velocità angolare è bassa, interrompi la rotazione
            rotation_cmd = 0
        elif desired_torque > rotation_deadband:
            rotation_cmd = 1  # CCW
        elif desired_torque < -rotation_deadband:
            rotation_cmd = 2  # CW
        else:
            rotation_cmd = 0
            
        # Se il fuel è vuoto, nessuna azione possibile
        if self.fuel_empty:
            thrust_cmd = 0
            rotation_cmd = 0
            
        # Debug informazioni PID se abilitato
        if getattr(self, '_enable_pid_debug', False):
            self.debug_pid_performance(error_vel_x, error_vel_y, angle_error)
        
        return [thrust_cmd, rotation_cmd]
    
    def _find_closest_vertex_and_next_on_ref(self, current_pos, ref_traj_points):
        if not ref_traj_points:
            return None, -1, float('inf'), None

        ref_positions = np.array([wp['pos'] for wp in ref_traj_points])
        distances_sq = np.sum((ref_positions - current_pos)**2, axis=1)
        closest_vertex_idx = np.argmin(distances_sq)
        min_dist = np.sqrt(distances_sq[closest_vertex_idx])
        
        closest_vertex_pos = ref_positions[closest_vertex_idx]
        
        next_vertex_pos = None
        if closest_vertex_idx < len(ref_positions) - 1:
            next_vertex_pos = ref_positions[closest_vertex_idx + 1]
        
        return closest_vertex_pos, closest_vertex_idx, min_dist, next_vertex_pos

    # All'interno della classe LunarLanderEnv
    def _resample_trajectory(self, trajectory_points, num_points):
        """
        Ricampiona una traiettoria per ottenere un numero specifico di punti equidistanti.
        
        Args:
            trajectory_points: Lista di array numpy rappresentanti i punti della traiettoria
            num_points: Numero di punti desiderati nella traiettoria ricampionata
        
        Returns:
            Una lista di array numpy con i punti ricampionati
        """
        if len(trajectory_points) <= 1:
            return trajectory_points
        
        # Calcola la lunghezza totale della traiettoria
        total_length = 0.0
        segment_lengths = []
        
        for i in range(len(trajectory_points) - 1):
            length = np.linalg.norm(trajectory_points[i + 1] - trajectory_points[i])
            total_length += length
            segment_lengths.append(length)
        
        if total_length < 1e-6:  # Se la traiettoria è praticamente un punto
            return [trajectory_points[0]] * num_points
        
        # Calcola le posizioni normalizzate per l'interpolazione
        normalized_positions = [0.0]
        cumulative_length = 0.0
        
        for length in segment_lengths:
            cumulative_length += length
            normalized_positions.append(cumulative_length / total_length)
        
        # Crea i punti ricampionati
        resampled_points = []
        
        for i in range(num_points):
            t = i / (num_points - 1) if num_points > 1 else 0.0
            
            # Trova il segmento corrispondente
            segment_idx = 0
            while segment_idx < len(normalized_positions) - 1 and normalized_positions[segment_idx + 1] < t:
                segment_idx += 1
            
            if segment_idx >= len(normalized_positions) - 1:
                resampled_points.append(trajectory_points[-1])
                continue
            
            # Interpolazione lineare nel segmento
            p0 = normalized_positions[segment_idx]
            p1 = normalized_positions[segment_idx + 1]
            
            if abs(p1 - p0) < 1e-6:  # Evita divisione per zero
                alpha = 0.0
            else:
                alpha = (t - p0) / (p1 - p0)
            
            point = trajectory_points[segment_idx] * (1.0 - alpha) + trajectory_points[segment_idx + 1] * alpha
            resampled_points.append(point)
        
        return resampled_points
    
    def _predict_lander_ballistic_segment(self, 
                                        initial_pos_arr, 
                                        initial_vel_arr, 
                                        target_spatial_length_px: float, 
                                        num_prediction_points: int = 15, # Punti per il segmento ricampionato
                                        max_simulation_steps: int = 50): # Limite per la simulazione
        """
        Prevede il segmento di traiettoria balistica a breve termine del lander.
        Cerca di generare un percorso di circa target_spatial_length_px, 
        campionato in num_prediction_points.
        Restituisce una lista di posizioni np.array([x,y]).
        """
        if target_spatial_length_px <= 0:
            return [initial_pos_arr.copy()] * num_prediction_points if num_prediction_points > 0 else []

        raw_sim_points = [initial_pos_arr.copy()] # Inizia con la posizione attuale

        current_pred_pos = initial_pos_arr.copy()
        current_pred_vel = initial_vel_arr.copy()

        path_length_covered_so_far = 0.0

        for _ in range(max_simulation_steps):
            if path_length_covered_so_far >= target_spatial_length_px and len(raw_sim_points) > 1:
                break 

            # Calcolo gravità (semplificato, come in altre parti del tuo codice)
            direction_to_center = np.array([CENTER_X, CENTER_Y]) - current_pred_pos
            distance_sq = max(np.dot(direction_to_center, direction_to_center), 1.0)
            dist = np.sqrt(distance_sq)

            gravity_accel = np.zeros(2)
            if dist > 1e-9:
                gravity_force_magnitude = GRAVITY_CONST / distance_sq # Assumendo sia un'accelerazione
                # Potresti omettere landing_assist_factor qui per una predizione puramente balistica
                gravity_accel = (direction_to_center / dist) * gravity_force_magnitude

            current_pred_vel += gravity_accel * TIME_STEP 
            next_pos = current_pred_pos + current_pred_vel * TIME_STEP

            segment_len = np.linalg.norm(next_pos - current_pred_pos)
            if segment_len < 1e-3 and path_length_covered_so_far > target_spatial_length_px * 0.5:
                # Evita loop se la velocità è quasi zero dopo aver percorso un po'
                break

            path_length_covered_so_far += segment_len
            raw_sim_points.append(next_pos.copy())
            current_pred_pos = next_pos

            # Controllo opzionale: impatto con la luna durante la predizione
            # altitude_pred = np.linalg.norm(current_pred_pos - np.array([CENTER_X, CENTER_Y])) - self.actual_moon_radius
            # if altitude_pred <= 0:
            #     break

        if len(raw_sim_points) < 2:
            return [initial_pos_arr.copy()] * num_prediction_points if num_prediction_points > 0 else []

        # Ricampiona i punti grezzi simulati per ottenere il numero desiderato di punti
        # per il segmento di lookahead del lander.
        resampled_path = self._resample_trajectory(raw_sim_points, num_prediction_points)

        return resampled_path
    
    def step(self, action):
        if self.ship_pos is None:
            obs = self._get_obs() 
            reward = -WEIGHT_CRASH_PENALTY * 2
            self.current_episode_score += reward
            return obs, reward, True, False, {"error": "Called step before reset"}

        if self.is_nn_trajectory_guidance_active:
            self.nn_guidance_time_elapsed += TIME_STEP
        else: 
            self.trajectory_time_elapsed += TIME_STEP

        actual_action = action 

        if self.trajectory_autopilot_on and not self.is_nn_trajectory_guidance_active:
            self.ap_trajectory_step_counter +=1
            if self.ap_trajectory_step_counter >= TRAJECTORY_UPDATE_RATE:
                self.ap_trajectory_step_counter = 0
                self._cached_trajectory_action = self._get_trajectory_autopilot_action() 
            if hasattr(self, '_cached_trajectory_action') and self._cached_trajectory_action is not None:
                actual_action = self._cached_trajectory_action
            else: 
                actual_action = self._get_trajectory_autopilot_action()
                self._cached_trajectory_action = actual_action
        
        thrust_cmd = actual_action[0]
        rotation_cmd = actual_action[1]
        prev_info_data = self._get_info()
        # --- Inizio Blocco Fisica (COPIA LA TUA FISICA COMPLETA QUI) ---
        # ... (Vento, Gravità, Propulsori, Consumo Carburante, Damping, Integrazione Posizione/Angolo) ...
        # Questa parte è cruciale e deve essere la tua logica di fisica testata.
        # Per esempio:
        fuel_consumed_this_step = 0.0 # DEVE ESSERE CALCOLATO QUI DENTRO LA FISICA
        if self.enable_wind and not self.fuel_empty:
            base_wind_x = self.wind_power * (self.np_random.choice([-1,1]) if self.turbulence_power > 0 else 1)
            gust_x = (self.np_random.uniform() - 0.5) * 2 * self.turbulence_power
            gust_y = (self.np_random.uniform() - 0.5) * 2 * self.turbulence_power
            actual_wind_force_x = base_wind_x + gust_x
            actual_wind_force_x += self.wind_power * np.sign(CENTER_X - self.ship_pos[0]) * 0.5
            actual_wind_force_y = gust_y
            if self.current_mass > 0:
                wind_accel_x = (actual_wind_force_x / self.current_mass); wind_accel_y = (actual_wind_force_y / self.current_mass)
                self.ship_vel[0] += wind_accel_x * TIME_STEP; self.ship_vel[1] += wind_accel_y * TIME_STEP
        direction_to_center = np.array([CENTER_X, CENTER_Y]) - self.ship_pos
        distance_sq = max(np.dot(direction_to_center, direction_to_center), 1.0); dist_to_center = np.sqrt(distance_sq)
        gravity_force_magnitude = GRAVITY_CONST / distance_sq
        altitude_for_gravity_assist = prev_info_data["altitude_surface"]
        if altitude_for_gravity_assist < 50:
            gravity_force_magnitude *= max(0.7, 1.0 - (50 - altitude_for_gravity_assist) / 50 * 0.3)
        gravity_accel_vector = np.zeros(2)
        if dist_to_center > 1e-9: gravity_accel_vector = (direction_to_center / dist_to_center) * gravity_force_magnitude
        self.ship_vel += gravity_accel_vector * TIME_STEP
        thrust_force_on_ship = np.zeros(2, dtype=np.float32); torque_on_ship_val = 0.0
        effective_thrust_cmd = thrust_cmd; effective_rotation_cmd = rotation_cmd
        if self.fuel_empty: effective_thrust_cmd = 0; effective_rotation_cmd = 0
        current_local_vertical_angle_phys, _ = self._get_local_vertical_angle_and_altitude() # rinominato per evitare clash
        absolute_ship_orientation_world_phys = normalize_angle(current_local_vertical_angle_phys + self.ship_angle)
        if effective_thrust_cmd == 1:
            thrust_dir_world = np.array([np.cos(absolute_ship_orientation_world_phys), np.sin(absolute_ship_orientation_world_phys)])
            thrust_force_on_ship = thrust_dir_world * MAIN_THRUST_FORCE
            fuel_consumed_this_step += FUEL_CONSUMPTION_MAIN * TIME_STEP
        if effective_rotation_cmd == 1: torque_on_ship_val = ROTATION_THRUST_TORQUE; fuel_consumed_this_step += FUEL_CONSUMPTION_ROTATION * TIME_STEP
        elif effective_rotation_cmd == 2: torque_on_ship_val = -ROTATION_THRUST_TORQUE; fuel_consumed_this_step += FUEL_CONSUMPTION_ROTATION * TIME_STEP
        self.current_fuel -= fuel_consumed_this_step
        if self.current_fuel <= 0: self.current_fuel = 0; self.fuel_empty = True
        self.current_mass = DRY_MASS + self.current_fuel
        if self.current_mass > 0:
            linear_accel_from_thrust = thrust_force_on_ship / self.current_mass
            inertia_approx = self.current_mass * ((self.actual_ship_w**2 + self.actual_ship_h**2) / 1200.0); inertia_approx = max(inertia_approx, 1e-6)
            angular_accel_from_torque = torque_on_ship_val / inertia_approx
            self.ship_vel += linear_accel_from_thrust * TIME_STEP; self.ship_angular_vel += angular_accel_from_torque * TIME_STEP
        is_thrust_intended_phys = actual_action[0] == 1; is_thrust_active_for_damping_phys = is_thrust_intended_phys and not self.fuel_empty
        is_near_surface_damping_phys = prev_info_data["altitude_surface"] < NEAR_SURFACE_ALTITUDE_THRESHOLD
        base_damping_factor_phys = 0.98 if is_thrust_active_for_damping_phys else 0.90
        surface_damping_factor_phys = 0.95 if is_near_surface_damping_phys else base_damping_factor_phys
        self.ship_angular_vel *= surface_damping_factor_phys
        if actual_action[1] == 0:
            if abs(self.ship_angular_vel) > 0.01: self.ship_angular_vel *= (0.92 - min(0.1, abs(self.ship_angular_vel) * 0.05))
            if not is_thrust_active_for_damping_phys and abs(self.ship_angular_vel) < 0.1:
                angle_error_stabilization_phys = normalize_angle(0.0 - self.ship_angle)
                if abs(angle_error_stabilization_phys) > 0.05: self.ship_angular_vel += min(0.0005, abs(angle_error_stabilization_phys) * 0.002) * np.sign(angle_error_stabilization_phys)
        self.ship_angle += self.ship_angular_vel * TIME_STEP; self.ship_angle = normalize_angle(self.ship_angle)
        self.ship_pos += self.ship_vel * TIME_STEP; self.steps_taken += 1
        if actual_action[1] == 0 and abs(self.ship_angular_vel) > 0.01: # REPLACEMENT 2
            if not is_thrust_active_for_damping_phys:
                angle_error_stabilization2_phys = normalize_angle(0.0 - self.ship_angle)
                if (angle_error_stabilization2_phys * self.ship_angular_vel > 0) or abs(angle_error_stabilization2_phys) < 0.1: self.ship_angular_vel *= 0.85
        # --- Fine Blocco Fisica ---

        if hasattr(self, 'current_trajectory'):
            self.current_trajectory.append(
                (self.ship_pos.copy(), self.ship_vel.copy(), self.ship_angle, actual_action.copy())
            )

        current_info_data = self._get_info()
        current_altitude = current_info_data["altitude_surface"]
        current_speed_radial = current_info_data["speed_radial"]
        current_ship_angle_relative_to_local_vertical = current_info_data["ship_angle_relative_deg"] * (np.pi / 180.0)

        reward = 0.0
        terminated = False; truncated = False
        landed_successfully = False; crashed = False
        
        final_info = self._get_info() 
        final_info['nn_guidance_active_shape'] = False # Default, sovrascritto sotto se attiva
        final_info['nn_time_elapsed_guidance'] = self.nn_guidance_time_elapsed
        final_info['nn_dist_to_ref_shape'] = -1.0
        final_info['reward_comp_spatial_proximity'] = 0.0
        final_info['reward_comp_velocity_alignment'] = 0.0
        final_info['reward_comp_nn_traj_completed_spatial'] = 0.0
        # Aggiunte chiavi per le nuove metriche di similarità con traiettoria proiettata
        final_info['trajectory_similarity'] = 0.0 
        final_info['max_trajectory_deviation'] = 0.0
        final_info['avg_dist_paths'] = -1.0
        final_info['trajectory_reward_comp'] = 0.0

        # 1. PENALITÀ GENERALI
        reward -= WEIGHT_TIME_PENALTY 
        if fuel_consumed_this_step > 0:
            reward -= WEIGHT_FUEL_CONSUMPTION_PENALTY_PER_UNIT * fuel_consumed_this_step
        reward -= WEIGHT_ROTATION_SPEED_PENALTY * (abs(self.ship_angular_vel))

        # Questo è solo un esempio, assicurati che final_info sia gestito coerentemente
        if 'reward_comp_approaching_marker' not in final_info:
            final_info['reward_comp_approaching_marker'] = 0.0
        if 'reward_comp_precision_landing' not in final_info: # Anche se è un terminal reward, inizializzalo per il log
            final_info['reward_comp_precision_landing'] = 0.0
        if 'final_dist_to_marker_on_land' not in final_info:
            final_info['final_dist_to_marker_on_land'] = -1.0


        # --- BONUS PER AVVICINAMENTO AL MARKER (SHAPING REWARD) ---
        approaching_marker_bonus_this_step = 0.5
        if self.landing_target_display_pos is not None and self.ship_pos is not None:
            current_dist_to_marker = np.linalg.norm(self.ship_pos - np.array(self.landing_target_display_pos))
            
            if self.previous_dist_to_marker is not None and self.previous_dist_to_marker != float('inf'):
                distance_reduction = self.previous_dist_to_marker - current_dist_to_marker
                
                # Ricompensa solo se c'è una riduzione effettiva della distanza
                if distance_reduction > 0: # Si è avvicinato
                    approaching_marker_bonus_this_step = WEIGHT_APPROACHING_MARKER_BONUS_PER_PIXEL * distance_reduction
                    reward += approaching_marker_bonus_this_step
            
            self.previous_dist_to_marker = current_dist_to_marker # Aggiorna per il prossimo step
        
        final_info['reward_comp_approaching_marker'] = approaching_marker_bonus_this_step
        # --- FINE BONUS AVVICINAMENTO MARKER ---

        # --- INIZIO NUOVA SEZIONE: PREMIO PER MATCH TRA TRAIETTORIA PROIETTATA E TARGET ---
        # Calcola la traiettoria proiettata - simile a quanto fatto in _render_frame
        WEIGHT_TRAJECTORY_MATCH = 1  # Peso per premiare la vicinanza alla traiettoria target
        WEIGHT_TRAJECTORY_DEVIATION_PENALTY = 0.25  # Peso per penalizzare le deviazioni massime
        MAX_PROJECTION_STEPS = 50  # Numero ragionevole di step di proiezione per il calcolo
        
        is_final_landing_phase = (not self.is_nn_trajectory_guidance_active or \
                                 self.nn_trajectory_completed_flag_for_bonus) and \
                                 current_altitude < NEAR_SURFACE_ALTITUDE_THRESHOLD * 1.2

        if (self.steps_taken -1 ) % PROJECTION_UPDATE_RATE == 0:
            if hasattr(self, 'trajectory_points') and self.trajectory_points and len(self.trajectory_points) > 1:
                # Estrai i punti della traiettoria target
                target_trajectory_points = [np.array(p['pos']) for p in self.trajectory_points]
                
                # Calcola la traiettoria proiettata
                pred_pos = self.ship_pos.copy()
                pred_vel = self.ship_vel.copy()
                projected_trajectory = [pred_pos.copy()]
                
                # Proietta la traiettoria in avanti usando la fisica
                prediction_time_step = TIME_STEP * 5  # Usa un time step più grande per la proiezione
                for _ in range(MAX_PROJECTION_STEPS):
                    direction_to_center = np.array([CENTER_X, CENTER_Y]) - pred_pos
                    distance_sq = max(np.dot(direction_to_center, direction_to_center), 1.0)
                    dist = np.sqrt(distance_sq)
                    
                    try:
                        gravity_force_magnitude = GRAVITY_CONST / distance_sq
                        gravity_accel = (direction_to_center / dist) * gravity_force_magnitude
                    except (ValueError, ZeroDivisionError):
                        gravity_accel = np.zeros(2)
                    
                    pred_vel += gravity_accel * prediction_time_step
                    pred_pos += pred_vel * prediction_time_step
                    projected_trajectory.append(pred_pos.copy())
                    
                    # Interrompi se colpisce la superficie o esce dallo schermo
                    pred_dist_center = np.linalg.norm(pred_pos - np.array([CENTER_X, CENTER_Y]))
                    if pred_dist_center <= self.actual_moon_radius:
                        break
                    if not (-SCREEN_WIDTH < pred_pos[0] < 2*SCREEN_WIDTH and -SCREEN_HEIGHT < pred_pos[1] < 2*SCREEN_HEIGHT):
                        break
                
                # Calcola la similarità tra le due traiettorie
                max_deviation = 0.0
                total_distance = 0.0
                valid_comparisons = 0
                
                # Per ogni punto della traiettoria proiettata, trova il punto più vicino sulla traiettoria target
                for proj_point in projected_trajectory:
                    min_dist = float('inf')
                    for target_point in target_trajectory_points:
                        dist = np.linalg.norm(proj_point - target_point)
                        min_dist = min(min_dist, dist)
                    
                    if min_dist < float('inf'):
                        max_deviation = max(max_deviation, min_dist)
                        total_distance += min_dist
                        valid_comparisons += 1
                
                # Calcola la distanza media tra i punti
                avg_distance = total_distance / valid_comparisons if valid_comparisons > 0 else float('inf')
                
                # Calcola ricompense basate sulle metriche di similarità
                trajectory_similarity_score = np.exp(-0.1 * avg_distance)  # Più vicino è, maggiore è il punteggio
                trajectory_similarity_reward = WEIGHT_TRAJECTORY_MATCH * trajectory_similarity_score
                
                # Penalità per deviazioni massime
                max_deviation_penalty = -WEIGHT_TRAJECTORY_DEVIATION_PENALTY * (1.0 - np.exp(-0.01 * max_deviation))
                
                # Combina ricompense e penalità
                trajectory_reward_component = trajectory_similarity_reward + max_deviation_penalty
                reward += trajectory_reward_component
                
                # Aggiorna le informazioni finali con le metriche
                final_info['trajectory_similarity'] = trajectory_similarity_score
                final_info['max_trajectory_deviation'] = max_deviation
                final_info['avg_dist_paths'] = avg_distance
                final_info['trajectory_reward_comp'] = trajectory_reward_component
            # --- FINE NUOVA SEZIONE ---

            
            # 2. RICOMPENSA PER SEGUIRE LA FORMA DELLA TRAIETTORIA NN
            if self.is_nn_trajectory_guidance_active and self.nn_trajectory_points and \
            len(self.nn_trajectory_points) >= 2:
                
                final_info['nn_guidance_active_shape'] = True # Indica che questa logica è attiva
                ref_path_positions_np = [np.array(wp['pos']) for wp in self.nn_trajectory_points]
                
                closest_pt_on_ref, dist_to_ref, seg_idx_ref, tangent_at_closest = \
                    self._find_closest_point_on_polyline(self.ship_pos, ref_path_positions_np)
                
                final_info['nn_dist_to_ref_shape'] = dist_to_ref

                if closest_pt_on_ref is not None:
                    # A. Ricompensa di Prossimità
                    proximity_bonus = WEIGHT_SPATIAL_PROXIMITY_TO_REF_PATH * np.exp(-PROXIMITY_DECAY_FACTOR * dist_to_ref)
                    reward += proximity_bonus
                    final_info['reward_comp_spatial_proximity'] = proximity_bonus

                    if dist_to_ref > NN_PATH_CORRIDOR_RADIUS:
                        deviation_penalty = WEIGHT_DEVIATE_NN_PATH_PENALTY * (dist_to_ref - NN_PATH_CORRIDOR_RADIUS)
                        reward += deviation_penalty 
                    
                    # B. Ricompensa di Allineamento Velocità
                    if tangent_at_closest is not None and WEIGHT_VELOCITY_ALIGNMENT_WITH_REF_PATH > 1e-9 :
                        norm_ship_vel = np.linalg.norm(self.ship_vel)
                        if norm_ship_vel > 0.1: # Solo se il lander si sta muovendo significativamente
                            unit_ship_vel_dir = self.ship_vel / norm_ship_vel
                            cos_angle_vel_path = np.dot(unit_ship_vel_dir, tangent_at_closest)
                            directional_reward_component = 0.0

                            # Condizione per applicare questo bonus/penalità (opzionale, ma consigliata)
                            # Solo se il lander è ragionevolmente vicino al percorso
                            if dist_to_ref < NN_PATH_CORRIDOR_RADIUS * 1.5 and not is_final_landing_phase: # O usa NN_PATH_CORRIDOR_RADIUS se vuoi essere più stringente

                                # 1. BONUS per allineamento stretto (angolo < 15 gradi)
                                if cos_angle_vel_path > COS_15_DEGREES:
                                    # Il bonus può essere fisso o proporzionale a quanto è buono l'allineamento
                                    # Esempio di bonus proporzionale:
                                    # Scaliamo cos_angle_vel_path (che è tra COS_15_DEGREES e 1.0) a un fattore tra 0 e 1
                                    bonus_factor_precise = (cos_angle_vel_path - COS_15_DEGREES) / (1.0 - COS_15_DEGREES)
                                    directional_reward_component = WEIGHT_PRECISE_ALIGNMENT_BONUS * bonus_factor_precise
                                    final_info['reward_comp_precise_align_bonus'] = directional_reward_component
                                
                                # 2. PENALITÀ per direzioni opposte (angolo > 90 gradi)
                                elif cos_angle_vel_path < COS_15_DEGREES: # Se non è nel range del bonus, controlla se è opposto
                                    # cos_angle_vel_path è tra -1 (perfettamente opposto) e 0 (perpendicolare).
                                    # Moltiplicando per WEIGHT_OPPOSITE_DIRECTION_PENALTY_FACTOR (positivo),
                                    # otteniamo una penalità negativa che è massima a -WEIGHT_OPPOSITE_DIRECTION_PENALTY_FACTOR.
                                    if cos_angle_vel_path <= 0: 
                                        directional_reward_component = cos_angle_vel_path * WEIGHT_OPPOSITE_DIRECTION_PENALTY_FACTOR
                                    else:
                                        directional_reward_component = -cos_angle_vel_path * WEIGHT_OPPOSITE_DIRECTION_PENALTY_FACTOR
                                    final_info['reward_comp_opposite_dir_penalty'] = directional_reward_component
                                
                                # Nessun bonus/penalità da questo blocco se l'angolo è tra 15° e 90°
                                # (cioè 0 <= cos_angle_vel_path <= COS_15_DEGREES)
                                else:
                                    final_info['reward_comp_precise_align_bonus'] = 0.0
                                    final_info['reward_comp_opposite_dir_penalty'] = 0.0


                            reward += directional_reward_component
                    
                    # C. Ricompensa Progresso Spaziale (basato sull'indice del segmento più vicino)
                    if seg_idx_ref != -1 and hasattr(self, 'last_closest_ref_segment_idx'): # Assicura che l'attributo esista
                        if seg_idx_ref > self.last_closest_ref_segment_idx :
                            # Calcola quanti segmenti sono stati superati per un bonus proporzionale
                            # Questo previene bonus enormi se salta molti segmenti (improbabile con controllo fine)
                            num_segments_advanced = seg_idx_ref - self.last_closest_ref_segment_idx
                            progress_bonus = WEIGHT_PROGRESS_ALONG_PATH_SHAPE * num_segments_advanced
                            reward += progress_bonus
                            # final_info['reward_comp_progress_along_path'] = progress_bonus # Aggiungi se vuoi vederlo
                        self.last_closest_ref_segment_idx = seg_idx_ref

         
                    # --- NUOVO: BONUS/PENALITÀ PER ORIENTAMENTO VERTICALE CORRETTO RISPETTO ALLA TRAIETTORIA ---
                    vertical_orientation_reward = 0.0
                    if self.is_nn_trajectory_guidance_active and self.nn_trajectory_points and \
                    len(self.nn_trajectory_points) >= 2 and closest_pt_on_ref is not None:
                        
                        # Altitudine del lander (già calcolata come current_altitude)
                        # Altitudine del punto più vicino sulla traiettoria di riferimento
                        vector_center_to_ref_pt = closest_pt_on_ref - np.array([CENTER_X, CENTER_Y])
                        altitude_of_ref_pt = np.linalg.norm(vector_center_to_ref_pt) - self.actual_moon_radius
                        
                        # Errore di altitudine: positivo se il lander è più alto della traiettoria, negativo se più basso
                        altitude_error_to_ref = current_altitude - altitude_of_ref_pt

                        # Angolo attuale del lander rispetto alla verticale locale (ship_angle è già questo)
                        # ship_angle: 0 è motori verso il basso (spinta verso l'alto), pi è motori verso l'alto (spinta verso il basso)
                        # Più precisamente, ship_angle è l'angolo rispetto alla verticale locale, 
                        # quindi 0 = lander dritto, motori sotto.
                        # Per spingere verso l'alto (frenare discesa), ship_angle deve essere vicino a 0.
                        # Per spingere verso il basso (accelerare discesa), ship_angle deve essere vicino a +/- pi.

                        # Vogliamo incentivare un orientamento specifico a seconda dell'errore di altitudine e della velocità radiale.

                        # CASO 1: Lander è SOPRA la traiettoria di riferimento (altitude_error_to_ref > DEADZONE)
                        # e/o sta SCENDENDO TROPPO VELOCEMENTE (current_speed_radial è molto negativo).
                        # Deve frenare la discesa o addirittura risalire -> orientare i motori verso il basso (ship_angle vicino a 0).
                        if altitude_error_to_ref > VERTICAL_CONTROL_DEADZONE_PX or \
                        (current_speed_radial < -MAX_RADIAL_SPEED_FOR_BONUS_CONSIDERATION and altitude_error_to_ref > -NN_PATH_CORRIDOR_RADIUS): # Scende veloce, anche se poco sotto
                            # Premiamo se ship_angle è vicino a 0 (motori sotto, per spingere lander verso l'alto o frenare discesa)
                            # Usiamo una gaussiana o exp per premiare l'allineamento a 0
                            angle_alignment_score_brake = np.exp(-5.0 * (normalize_angle(self.ship_angle - 0.0)**2) ) # Più vicino a 0 è, più alto il punteggio
                            vertical_orientation_reward += WEIGHT_VERTICAL_CONTROL_ORIENTATION * angle_alignment_score_brake
                            final_info['reward_comp_vert_orient_brake'] = WEIGHT_VERTICAL_CONTROL_ORIENTATION * angle_alignment_score_brake

                        # CASO 2: Lander è SOTTO la traiettoria di riferimento (altitude_error_to_ref < -DEADZONE)
                        # e/o sta SALENDO TROPPO VELOCEMENTE (current_speed_radial è molto positivo) o deve accelerare la discesa.
                        # Deve accelerare la discesa o smettere di salire -> potrebbe ridurre spinta o orientare motori verso l'alto (ship_angle vicino a +/- pi).
                        # Questa parte è più délicata, perché spingere verso il basso è raramente ottimale.
                        # Forse è meglio penalizzare l'orientamento sbagliato nel CASO 1 piuttosto che premiare attivamente una spinta verso il basso.
                        # Tuttavia, se deve SCENDERE per raggiungere una traiettoria più bassa e sta salendo o scendendo troppo lentamente:
                        elif altitude_error_to_ref < -VERTICAL_CONTROL_DEADZONE_PX and \
                            (current_speed_radial > MAX_RADIAL_SPEED_FOR_BONUS_CONSIDERATION / 2 or current_speed_radial > 0): # Sale, o scende troppo lentamente
                            # Penalizziamo se ship_angle è vicino a 0 (sta cercando di frenare la discesa/salire quando dovrebbe scendere)
                            angle_misalignment_penalty_accel = np.exp(-5.0 * (normalize_angle(self.ship_angle - 0.0)**2) )
                            vertical_orientation_reward -= WEIGHT_VERTICAL_CONTROL_ORIENTATION * angle_misalignment_penalty_accel * 0.5 # Penalità più piccola
                            final_info['reward_comp_vert_orient_accel_penalty'] = -WEIGHT_VERTICAL_CONTROL_ORIENTATION * angle_misalignment_penalty_accel * 0.5

                        reward += vertical_orientation_reward
                    # --- FINE NUOVO BONUS/PENALITÀ ORIENTAMENTO VERTICALE ---
                # D. Bonus una tantum per completamento spaziale
                if ref_path_positions_np: # Verifica che la lista non sia vuota
                    dist_to_end_of_ref_shape = np.linalg.norm(self.ship_pos - ref_path_positions_np[-1])
                    if dist_to_end_of_ref_shape < REF_TRAJ_END_REACH_THRESHOLD_PX and \
                    not self.nn_trajectory_completed_flag_for_bonus:
                        completion_bonus = WEIGHT_NN_TRAJECTORY_COMPLETED 
                        reward += completion_bonus
                        self.nn_trajectory_completed_flag_for_bonus = True
                        final_info['reward_comp_nn_traj_completed_spatial'] = completion_bonus
                        print("INFO (step): Lander ha raggiunto la fine spaziale della traiettoria NN.")
            
        # 3. RICOMPENSE/PENALITÀ PER LA FASE FINALE DI ATTERRAGGIO
        
        if is_final_landing_phase:
            if current_speed_radial < 0: 
                if abs(current_speed_radial) < abs(MAX_RADIAL_SPEED_FINAL_LANDING): 
                     reward += WEIGHT_FINAL_LANDING_SPEED_REDUCTION * (1.0 - (abs(current_speed_radial) / abs(MAX_RADIAL_SPEED_FINAL_LANDING))) * 0.5
                else: 
                    reward -= WEIGHT_EXCESSIVE_FINAL_DESCENT_SPEED_PENALTY * (abs(current_speed_radial) - abs(MAX_RADIAL_SPEED_FINAL_LANDING))
            tangential_speed_reduction_bonus = (LANDING_SPEED_LIMIT * 0.5 - abs(current_info_data["speed_tangential"])) / (LANDING_SPEED_LIMIT * 0.5 + 1e-9)
            if tangential_speed_reduction_bonus > 0:
                reward += WEIGHT_FINAL_LANDING_SPEED_REDUCTION * tangential_speed_reduction_bonus * 0.3
            angle_error_final_land = abs(current_ship_angle_relative_to_local_vertical - TARGET_ANGLE_LANDING)
            angle_alignment_bonus = np.exp(-5.0 * angle_error_final_land) 
            reward += WEIGHT_FINAL_LANDING_ANGLE_ALIGNMENT * angle_alignment_bonus

        # 4. CONDIZIONI DI TERMINAZIONE EPISODIO
        contact_threshold_altitude = (self.actual_ship_h / 2.0)*1
        if current_altitude <= contact_threshold_altitude:
            terminated = True 
            speed_total_contact = current_info_data["speed_total"]
            radial_speed_contact = current_info_data["speed_radial"] 
            tangential_speed_contact = abs(current_info_data["speed_tangential"])
            angle_contact_rad = current_ship_angle_relative_to_local_vertical 
            speed_ok = speed_total_contact < LANDING_SPEED_LIMIT
            radial_speed_ok = abs(radial_speed_contact) < LANDING_SPEED_LIMIT * 0.8 and radial_speed_contact < -0.01 
            tangential_speed_ok = tangential_speed_contact < LANDING_SPEED_LIMIT * 0.5
            angle_ok = abs(angle_contact_rad - TARGET_ANGLE_LANDING) < LANDING_ANGLE_LIMIT
            if speed_ok and radial_speed_ok and tangential_speed_ok and angle_ok:
                landed_successfully = True
                reward += WEIGHT_SUCCESSFUL_SURFACE_LANDING_BONUS # Bonus base per atterraggio riuscito

                # --- BONUS PER PRECISIONE ATTERRAGGIO VICINO AL MARKER ---
                precision_landing_bonus_this_ep = 0.0
                actual_final_dist_to_marker = -1.0 # Per logging

                if self.landing_target_display_pos is not None and self.ship_pos is not None:
                    final_dist = np.linalg.norm(self.ship_pos - np.array(self.landing_target_display_pos))
                    actual_final_dist_to_marker = final_dist # Salva per il log

                    if final_dist <= MIN_DIST_FOR_MAX_PRECISION_BONUS_PX:
                        # Atterraggio molto preciso, assegna il bonus massimo
                        precision_landing_bonus_this_ep = WEIGHT_PRECISION_LANDING_BONUS_MAX
                    elif final_dist < MAX_DIST_FOR_PRECISION_BONUS_PX:
                        # Atterraggio abbastanza vicino, assegna un bonus che scala linearmente
                        # Il fattore va da quasi 1 (vicino a MIN_DIST) a 0 (a MAX_DIST)
                        # Scaliamo in modo che a MIN_DIST_FOR_MAX_PRECISION_BONUS_PX sia 1, e a MAX_DIST_FOR_PRECISION_BONUS_PX sia 0
                        # Evitiamo la divisione per zero se MAX_DIST è uguale a MIN_DIST
                        range_dist = MAX_DIST_FOR_PRECISION_BONUS_PX - MIN_DIST_FOR_MAX_PRECISION_BONUS_PX
                        if range_dist > 1e-6 : # Evita divisione per zero o numeri molto piccoli
                            precision_factor = 1.0 - ( (final_dist - MIN_DIST_FOR_MAX_PRECISION_BONUS_PX) / range_dist )
                            precision_factor = max(0, min(1, precision_factor)) # Assicura che sia tra 0 e 1
                            precision_landing_bonus_this_ep = WEIGHT_PRECISION_LANDING_BONUS_MAX * precision_factor
                        else: # Se MIN_DIST e MAX_DIST sono uguali, e siamo entro MIN_DIST, diamo il massimo
                             precision_landing_bonus_this_ep = WEIGHT_PRECISION_LANDING_BONUS_MAX


                    reward += precision_landing_bonus_this_ep
                
                final_info['reward_comp_precision_landing'] = precision_landing_bonus_this_ep
                final_info['final_dist_to_marker_on_land'] = actual_final_dist_to_marker
                # --- FINE BONUS PRECISIONE ATTERRAGGIO ---
            
            else: # Non atterrato con successo (crash)
                crashed = True
                reward -= WEIGHT_CRASH_PENALTY
                final_info['reward_comp_precision_landing'] = 0.0 # Nessun bonus precisione se crash
                if self.landing_target_display_pos is not None and self.ship_pos is not None:
                     final_info['final_dist_to_marker_on_land'] = np.linalg.norm(self.ship_pos - np.array(self.landing_target_display_pos))
                else:
                     final_info['final_dist_to_marker_on_land'] = -1.0
        
        oob_margin = 50
        is_oob = not (-oob_margin < self.ship_pos[0] < SCREEN_WIDTH + oob_margin and \
                      -oob_margin < self.ship_pos[1] < SCREEN_HEIGHT + oob_margin)
        if is_oob and not terminated: 
            truncated = True; reward -= WEIGHT_OOB_PENALTY
        
        if self.steps_taken >= MAX_STEPS_PER_EPISODE and not terminated and not truncated:
            truncated = True; reward -= WEIGHT_TIMEOUT_PENALTY
        
        if self.fuel_empty and not terminated and not truncated:
            truncated = True; reward -= WEIGHT_FUEL_EMPTY_PENALTY
            if current_altitude > NEAR_SURFACE_ALTITUDE_THRESHOLD * 2.0:
                reward -= WEIGHT_FUEL_EMPTY_PENALTY 
        
        if (terminated or truncated) and not self.fuel_empty:
             fuel_efficiency_bonus = (self.current_fuel / INITIAL_FUEL_MASS) * 50 
             reward += fuel_efficiency_bonus

        # --- Aggiornamento Score e Info Finali (finale) ---
        self.current_episode_score += reward
        
        # Aggiorna final_info con risultati finali
        final_info['landed_successfully'] = landed_successfully
        final_info['crashed'] = crashed
        final_info['fuel_consumed_step'] = fuel_consumed_this_step
        final_info['out_of_bounds'] = is_oob and truncated 
        final_info['timeout'] = (self.steps_taken >= MAX_STEPS_PER_EPISODE) and truncated
        final_info['fuel_ended_episode'] = self.fuel_empty and truncated
        final_info['reward_step'] = reward

        if terminated or truncated:
            self.episode_info = final_info.copy() 
            # ... (costruzione outcome_message) ...
            self.episode_outcome_message = final_info.get("outcome_message", "N/A") # Aggiorna per HUD
            self.show_target_selected_message = False 
            self.is_in_targeting_mode = False
        # --- NUOVO: CONTROLLO PER SCORE TROPPO NEGATIVO ---
        # Applica questo controllo solo se l'episodio non è già terminato o troncato per altre ragioni
        if not terminated and not truncated:
            if self.current_episode_score < MIN_EPISODE_SCORE_FOR_TRUNCATION:
                print(f"INFO (step): Episodio TRONCATO per score ({self.current_episode_score:.2f}) < soglia ({MIN_EPISODE_SCORE_FOR_TRUNCATION}).")
                truncated = True  # Usiamo truncated per interruzioni artificiali dell'episodio
                
                # Opzionale: puoi aggiungere una piccola penalità specifica per questo evento,
                # ma lo score già molto negativo è un forte segnale.
                # specific_low_score_penalty = -50.0 
                # reward += specific_low_score_penalty
                # self.current_episode_score += specific_low_score_penalty # Aggiorna anche lo score se aggiungi penalità

                # Aggiorna il messaggio di outcome per il rendering e il logging
                # (Questa parte potrebbe aver bisogno di essere integrata meglio con la tua gestione esistente di episode_outcome_message)
                current_outcome_base = "TRUNCATED"
                if self.episode_outcome_message and "TRUNCATED" not in self.episode_outcome_message.upper():
                     self.episode_outcome_message = f"{current_outcome_base} [SCORE LOW] & {self.episode_outcome_message}"
                else:
                     self.episode_outcome_message = f"{current_outcome_base} [SCORE LOW ({self.current_episode_score:.0f})]"
                
                # Aggiungi a final_info per il logging da parte di SB3
                final_info['low_score_truncation'] = True
        
        # Assicurati che la chiave esista in final_info anche se non si verifica la truncazione
        if 'low_score_truncation' not in final_info:
            final_info['low_score_truncation'] = False
        # --- FINE NUOVO CONTROLLO ---

        # Blocco finale per aggiornare episode_info e outcome_message (potrebbe necessitare di revisione)
        if (terminated or truncated):
            # Questo blocco viene eseguito quando l'episodio finisce per qualsiasi motivo.
            # Assicurati che self.episode_info venga popolato una sola volta alla fine.
            # La logica esatta qui dipende da come vuoi che episode_outcome_message sia finalizzato.
            # Se hai già un blocco simile, integra questo.
            if not self.episode_info: # Riempi solo se non già riempito (prima terminazione)
                self.episode_info = final_info.copy()
                # Assicurati che l'outcome message in self.episode_info sia quello finale
                self.episode_info["outcome_message"] = self.episode_outcome_message 
            
            # Se il rendering è attivo e il messaggio non è stato ancora impostato dall'evento specifico
            if not self.episode_outcome_message: # Fallback se nessun messaggio specifico è stato impostato
                 if landed_successfully: self.episode_outcome_message = "LANDED SUCCESSFULLY"
                 elif crashed: self.episode_outcome_message = "CRASHED"
                 elif final_info.get('timeout', False): self.episode_outcome_message = "TRUNCATED [TIMEOUT]"
                 elif final_info.get('out_of_bounds', False): self.episode_outcome_message = "TRUNCATED [OOB]"
                 elif final_info.get('fuel_ended_episode', False): self.episode_outcome_message = "TRUNCATED [NO FUEL]"
                 elif final_info.get('low_score_truncation', False) : pass # Già gestito sopra
                 else: self.episode_outcome_message = "EPISODE ENDED"
        # --- Rendering ---
        if self.render_mode == "human":
            if not hasattr(self, 'is_pygame_initialized') or not self.is_pygame_initialized: 
                self._lazy_init_pygame()
            if hasattr(self, 'is_pygame_initialized') and self.is_pygame_initialized: 
                self._render_frame(action=actual_action, current_step_info=final_info)

        # --- Osservazione Finale e Controllo Validità ---
        observation = self._get_obs()
        if not np.all(np.isfinite(observation)):
            print(f"ERRORE: Osservazione finale non finita! {observation}. Stato: p={self.ship_pos} v={self.ship_vel} angle={self.ship_angle} fuel={self.current_fuel}. Terminazione forzata.")
            observation = np.nan_to_num(observation, nan=0.0, posinf=self.observation_space.high, neginf=self.observation_space.low)
            observation = np.clip(observation, self.observation_space.low, self.observation_space.high)
            if not terminated: 
                error_penalty = WEIGHT_CRASH_PENALTY * 1.5 
                reward -= error_penalty 
                self.current_episode_score -= error_penalty 
                terminated = True; crashed = True; 
                final_info['crashed'] = True 
                final_info['error_internal'] = "Invalid observation generated"
                if 'outcome_message' in final_info: final_info['outcome_message'] += " [OBS_ERR]"
                else: final_info['outcome_message'] = "CRASHED [OBS_ERR]"
                self.episode_outcome_message = final_info['outcome_message']

        return observation, reward, terminated, truncated, final_info
    
    
    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame() # _render_frame dovrebbe restituire l'array NumPy
        elif self.render_mode == "human":
            self._render_frame() # _render_frame gestisce il display Pygame
            return None # Le human-rendering di solito restituiscono None
        # Non fare nulla se render_mode è None o non supportato

    def _render_frame(self, action=None, current_step_info=None): 
        if not self.is_pygame_initialized or self.screen is None:
            # Potrebbe essere necessario gestire questo caso se _lazy_init_pygame non è stato chiamato
            # o se l'inizializzazione è fallita.
            # Se render_mode è specificato, _lazy_init_pygame dovrebbe essere chiamato in __init__.
            if self.render_mode in ["human", "rgb_array"]: # Prova a inizializzare se non fatto
                 self._lazy_init_pygame()
                 if not self.is_pygame_initialized: # Se ancora non inizializzato
                     # Potresti sollevare un errore o restituire qualcosa che indichi il fallimento
                     # print("WARN: Pygame non inizializzato, impossibile renderizzare.")
                     if self.render_mode == "rgb_array": return np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH, 3), dtype=np.uint8) # Fallback
                     return
            else: # Se nessun render_mode specificato, non fare nulla
                return
        if action is None: action = [0, 0]

        thrust_cmd = action[0]
        rotation_cmd = action[1]

        if self.assets_loaded and self.background_sprite: self.screen.blit(self.background_sprite, (0, 0))
        else: self.screen.fill(BLACK)

        if self.assets_loaded and self.moon_sprite:
            moon_rect = self.moon_sprite.get_rect(center=(CENTER_X, CENTER_Y))
            self.screen.blit(self.moon_sprite, moon_rect)
        else: pygame.draw.circle(self.screen, GRAY, (CENTER_X, CENTER_Y), int(self.actual_moon_radius))
        if self.visualize_hitboxes: pygame.draw.circle(self.screen, CYAN, (CENTER_X, CENTER_Y), int(self.actual_moon_radius), 1)

        # --- Disegna la traiettoria NN ATTIVA per l'AGENTE RL ---
        if self.is_nn_trajectory_guidance_active and self.nn_trajectory_points and len(self.nn_trajectory_points) > 1:
            path_points_to_draw_nn = [tuple(map(int, p['pos'])) for p in self.nn_trajectory_points]
            if len(path_points_to_draw_nn) > 1:
                try:
                    pygame.draw.lines(self.screen, CYAN, False, path_points_to_draw_nn, 1) # Ciano per traiettoria RL
                except TypeError: pass # Evita crash se i punti non sono validi

            # Evidenzia il waypoint NN corrente per l'RL
            if self.current_nn_waypoint_index < len(self.nn_trajectory_points):
                current_target_wp_nn_pos = self.nn_trajectory_points[self.current_nn_waypoint_index]['pos']
                try:
                    target_pos_draw_nn = tuple(map(int, current_target_wp_nn_pos))
                    pygame.draw.circle(self.screen, (255,165,0), target_pos_draw_nn, 7, 2) # Arancione per target RL
                    pygame.draw.circle(self.screen, (255,255,0), target_pos_draw_nn, 5, 0) # Giallo interno
                except (TypeError, ValueError): pass
        # --- FINE Disegno Traiettoria NN per RL ---

        # --- Disegna la traiettoria predefinita da seguire (se self.trajectory_points è impostato) ---
        if self.trajectory_points and len(self.trajectory_points) > 1:
            path_points_to_draw = []
            for point_dict in self.trajectory_points: # Ora self.trajectory_points è impostato da 'L'
                try:
                    path_points_to_draw.append(tuple(map(int, point_dict['pos'])))
                except (TypeError, ValueError) as e:
                    # print(f"Attenzione: punto della traiettoria non valido per il disegno: {point_dict['pos']} - {e}")
                    continue 
            if len(path_points_to_draw) > 1:
                trajectory_line_color = (80, 80, 80) # Grigio scuro
                try:
                    pygame.draw.lines(self.screen, trajectory_line_color, False, path_points_to_draw, 1)
                except TypeError as e:
                    # print(f"Errore Pygame nel disegnare la traiettoria: {e}")
                    pass

            # Evidenzia il waypoint target corrente della traiettoria PID *solo se il PID è ATTIVO*
            if self.trajectory_autopilot_on: # Questo flag è ancora cruciale per il target del PID
                current_target_waypoint_details = self._get_current_trajectory_target()
                if current_target_waypoint_details:
                    try:
                        target_pos_draw = tuple(map(int, current_target_waypoint_details['pos']))
                        pygame.draw.circle(self.screen, ORANGE, target_pos_draw, 6, 2)
                        pygame.draw.circle(self.screen, YELLOW, target_pos_draw, 4, 0)
                    except (TypeError, ValueError):
                        pass
        # --- FINE Disegno Traiettoria ---
        # NEW: Draw marked target on moon
        # --- MODIFIED: Drawing Target Markers with NEW Colors ---
        # Draw FIXED target marker (BLUE) - Era Giallo
        if self.landing_target_display_pos and self.font:
            marker_x, marker_y = int(self.landing_target_display_pos[0]), int(self.landing_target_display_pos[1])
            # Usa BLU per il bersaglio fissato
            pygame.draw.circle(self.screen, BLUE, (marker_x, marker_y), 8, 2)
            pygame.draw.line(self.screen, BLUE, (marker_x - 12, marker_y), (marker_x + 12, marker_y), 2)
            pygame.draw.line(self.screen, BLUE, (marker_x, marker_y - 12), (marker_x, marker_y + 12), 2)

        # Draw PREVIEW target marker (YELLOW) ONLY if in targeting mode - Era Ciano
        if self.is_in_targeting_mode and self.preview_target_display_pos and self.font:
             # Controllo per non sovrapporre anteprima e fisso (opzionale, ma utile)
             is_previewing_at_fixed_target = False
             if self.landing_target_display_pos and self.preview_target_display_pos:
                  dist_sq = (self.preview_target_display_pos[0] - self.landing_target_display_pos[0])**2 + \
                            (self.preview_target_display_pos[1] - self.landing_target_display_pos[1])**2
                  if dist_sq < 1: is_previewing_at_fixed_target = True

             # Disegna l'anteprima GIALLA solo se non coincide col fisso (o sempre, a scelta)
             if not is_previewing_at_fixed_target:
                  marker_x, marker_y = int(self.preview_target_display_pos[0]), int(self.preview_target_display_pos[1])
                  # Usa GIALLO per l'anteprima
                  pygame.draw.circle(self.screen, YELLOW, (marker_x, marker_y), 7, 1) # Leggermente diverso per distinguerlo
                  pygame.draw.line(self.screen, YELLOW, (marker_x - 10, marker_y), (marker_x + 10, marker_y), 1)
                  pygame.draw.line(self.screen, YELLOW, (marker_x, marker_y - 10), (marker_x, marker_y + 10), 1)

        if self.ship_pos is not None and self.ship_vel is not None:
            num_prediction_steps=500; prediction_time_step=TIME_STEP*5; prediction_color=YELLOW
            pred_pos=self.ship_pos.copy(); pred_vel=self.ship_vel.copy(); trajectory_points=[(int(pred_pos[0]),int(pred_pos[1]))]
            for _ in range(num_prediction_steps):
                direction_to_center=np.array([CENTER_X,CENTER_Y])-pred_pos; distance_sq=max(np.dot(direction_to_center,direction_to_center),1.0); dist=np.sqrt(distance_sq)
                try: gravity_force_magnitude=GRAVITY_CONST/distance_sq; gravity_accel=direction_to_center/dist*gravity_force_magnitude
                except (ValueError,ZeroDivisionError): gravity_accel=np.zeros(2)
                pred_vel+=gravity_accel*prediction_time_step; pred_pos+=pred_vel*prediction_time_step; trajectory_points.append((int(pred_pos[0]),int(pred_pos[1])))
                pred_dist_center=np.linalg.norm(pred_pos-np.array([CENTER_X,CENTER_Y]))
                if pred_dist_center<=self.actual_moon_radius: break
                if not (-SCREEN_WIDTH<pred_pos[0]<2*SCREEN_WIDTH and -SCREEN_HEIGHT<pred_pos[1]<2*SCREEN_HEIGHT): break
            if len(trajectory_points)>1:
                try: pygame.draw.lines(self.screen,prediction_color,False,trajectory_points,1)
                except TypeError: pass

        if self.ship_pos is None: return

        ship_center_x, ship_center_y = int(self.ship_pos[0]), int(self.ship_pos[1])
        local_vertical_angle, current_altitude_render = self._get_local_vertical_angle_and_altitude()
        absolute_ship_orientation_world = normalize_angle(local_vertical_angle + self.ship_angle)

        is_near_surface_for_aid = current_altitude_render < NEAR_SURFACE_ALTITUDE_THRESHOLD * 1.2
        if is_near_surface_for_aid and self.font :
            start_arc_angle = normalize_angle(local_vertical_angle - LANDING_ANGLE_LIMIT)
            end_arc_angle = normalize_angle(local_vertical_angle + LANDING_ANGLE_LIMIT)
            if end_arc_angle < start_arc_angle: end_arc_angle += 2 * np.pi
            arc_rect = pygame.Rect(ship_center_x - LANDING_AID_ARC_RADIUS,
                                   ship_center_y - LANDING_AID_ARC_RADIUS,
                                   2 * LANDING_AID_ARC_RADIUS,
                                   2 * LANDING_AID_ARC_RADIUS)
            try: pygame.draw.arc(self.screen, LIGHT_GREEN, arc_rect, -end_arc_angle, -start_arc_angle, 2)
            except TypeError: pass
            nose_line_end_x = ship_center_x + LANDING_AID_LINE_LENGTH * np.cos(absolute_ship_orientation_world)
            nose_line_end_y = ship_center_y + LANDING_AID_LINE_LENGTH * np.sin(absolute_ship_orientation_world)
            is_angle_within_limit = abs(normalize_angle(self.ship_angle - TARGET_ANGLE_LANDING)) < LANDING_ANGLE_LIMIT
            nose_line_color = GREEN if is_angle_within_limit else RED
            pygame.draw.line(self.screen, nose_line_color, (ship_center_x, ship_center_y), (nose_line_end_x, nose_line_end_y), 2)

        if self.assets_loaded and self.ship_sprite_original:
            render_angle_degrees = -np.degrees(absolute_ship_orientation_world)
            rotated_ship = pygame.transform.rotate(self.ship_sprite_original, render_angle_degrees)
            rotated_rect = rotated_ship.get_rect(center=(ship_center_x, ship_center_y))
            self.screen.blit(rotated_ship, rotated_rect)
        else:
            nose=(12*(self.actual_ship_w/DEFAULT_SHIP_W),0);left_tail=(-6*(self.actual_ship_w/DEFAULT_SHIP_W),-6*(self.actual_ship_h/DEFAULT_SHIP_H));right_tail=(-6*(self.actual_ship_w/DEFAULT_SHIP_W),6*(self.actual_ship_h/DEFAULT_SHIP_H))
            base_points=[nose,left_tail,right_tail];cos_a,sin_a=np.cos(absolute_ship_orientation_world),np.sin(absolute_ship_orientation_world);rotated_points=[]
            for bx,by in base_points:rx=bx*cos_a-by*sin_a;ry=bx*sin_a+by*cos_a;px=ship_center_x+rx;py=ship_center_y+ry;rotated_points.append((int(px),int(py)))
            pygame.draw.polygon(self.screen,BLUE,rotated_points,1)

        if self.visualize_hitboxes:
            w,h=self.actual_ship_w,self.actual_ship_h;corners=np.array([[-w/2,-h/2],[w/2,-h/2],[w/2,h/2],[-w/2,h/2]])
            cos_a=np.cos(absolute_ship_orientation_world);sin_a=np.sin(absolute_ship_orientation_world);rotation_matrix=np.array([[cos_a,-sin_a],[sin_a,cos_a]])
            rotated_corners=(rotation_matrix@corners.T).T;translated_corners=rotated_corners+self.ship_pos
            pygame.draw.polygon(self.screen,RED,translated_corners.astype(int),1)

        if self.assets_loaded and self.flame_sprite_original:
            cos_abs,sin_abs=np.cos(absolute_ship_orientation_world),np.sin(absolute_ship_orientation_world)
            if thrust_cmd==1 and not self.fuel_empty:
                flame_attach_angle_rad=normalize_angle(absolute_ship_orientation_world+np.pi); render_flame_degrees=-np.degrees(flame_attach_angle_rad)
                rotated_flame=pygame.transform.rotate(self.flame_sprite_original,render_flame_degrees)
                rear_nozzle_offset_local=np.array([-SHIP_CENTER_TO_REAR_DIST,0])
                world_offset=np.array([rear_nozzle_offset_local[0]*cos_abs-rear_nozzle_offset_local[1]*sin_abs, rear_nozzle_offset_local[0]*sin_abs+rear_nozzle_offset_local[1]*cos_abs])
                nozzle_pos_world=self.ship_pos+world_offset; flame_rect=rotated_flame.get_rect(center=(int(nozzle_pos_world[0]),int(nozzle_pos_world[1]))); self.screen.blit(rotated_flame,flame_rect)
            if rotation_cmd==1 and not self.fuel_empty:
                flame_attach_angle_rad=normalize_angle(absolute_ship_orientation_world+np.pi/2); render_flame_degrees=-np.degrees(flame_attach_angle_rad)
                rotated_flame=pygame.transform.rotate(self.flame_sprite_original,render_flame_degrees)
                nozzle_local=np.array([SHIP_SIDE_THRUSTER_X_OFFSET,SHIP_SIDE_THRUSTER_Y_OFFSET])
                world_offset=np.array([nozzle_local[0]*cos_abs-nozzle_local[1]*sin_abs, nozzle_local[0]*sin_abs+nozzle_local[1]*cos_abs])
                nozzle_pos_world=self.ship_pos+world_offset; flame_rect=rotated_flame.get_rect(center=(int(nozzle_pos_world[0]),int(nozzle_pos_world[1]))); self.screen.blit(rotated_flame,flame_rect)
            if rotation_cmd==2 and not self.fuel_empty:
                flame_attach_angle_rad=normalize_angle(absolute_ship_orientation_world-np.pi/2); render_flame_degrees=-np.degrees(flame_attach_angle_rad)
                rotated_flame=pygame.transform.rotate(self.flame_sprite_original,render_flame_degrees)
                nozzle_local=np.array([SHIP_SIDE_THRUSTER_X_OFFSET,-SHIP_SIDE_THRUSTER_Y_OFFSET])
                world_offset=np.array([nozzle_local[0]*cos_abs-nozzle_local[1]*sin_abs, nozzle_local[0]*sin_abs+nozzle_local[1]*cos_abs])
                nozzle_pos_world=self.ship_pos+world_offset; flame_rect=rotated_flame.get_rect(center=(int(nozzle_pos_world[0]),int(nozzle_pos_world[1]))); self.screen.blit(rotated_flame,flame_rect)
        elif not (self.assets_loaded and self.flame_sprite_original): pass

        if self.font and self.ship_pos is not None:
            info_data_render = self._get_info()
            y_pos = 10

            # NUOVO: Visualizza lo score corrente
            score_surface = self.font.render(f"Score: {self.current_episode_score:.2f}", True, WHITE)
            self.screen.blit(score_surface, (10, y_pos))
            y_pos += 20
            # --- Fine NUOVO ---

            text_lines = [
                f"Alt: {info_data_render['altitude_surface']:.1f} m",
                f"RadSpeed: {info_data_render['speed_radial']:.2f} m/s",
                f"TanSpeed: {info_data_render['speed_tangential']:.2f} m/s",
                f"RelAngle: {info_data_render['ship_angle_relative_deg']:.1f} deg",
                f"AngVel: {info_data_render['angular_velocity_dps']:.1f} dps",
            ]
            for line in text_lines:
                text_surf = self.font.render(line, True, WHITE)
                self.screen.blit(text_surf, (10, y_pos)); y_pos += 20

            fuel_text_y = y_pos
            fuel_perc_val = info_data_render['fuel_percentage']
            fuel_status_text = f"Fuel: {fuel_perc_val:.1f}%" + (" EMPTY" if info_data_render['fuel_empty'] else "")
            text_surf = self.font.render(fuel_status_text, True, WHITE)
            self.screen.blit(text_surf, (10, fuel_text_y))

            fuel_bar_x = text_surf.get_width() + 20
            fuel_bar_bg_rect = pygame.Rect(fuel_bar_x, fuel_text_y + 2, FUEL_BAR_WIDTH, FUEL_BAR_HEIGHT)
            pygame.draw.rect(self.screen, DARK_GRAY, fuel_bar_bg_rect)
            current_fuel_width = int(FUEL_BAR_WIDTH * (fuel_perc_val / 100.0))
            fuel_bar_color = GREEN
            if fuel_perc_val <= 20: fuel_bar_color = RED
            elif fuel_perc_val <= 50: fuel_bar_color = YELLOW
            if current_fuel_width > 0:
                fuel_bar_fill_rect = pygame.Rect(fuel_bar_x, fuel_text_y + 2, current_fuel_width, FUEL_BAR_HEIGHT)
                pygame.draw.rect(self.screen, fuel_bar_color, fuel_bar_fill_rect)
                y_pos = fuel_text_y + 20
                if current_step_info: # Verifica che current_step_info sia passato
                    # Visualizza la ricompensa dello step corrente
                    step_reward_val = current_step_info.get('reward_step', 0.0)
                    reward_color = GREEN if step_reward_val > 0 else (RED if step_reward_val < 0 else WHITE)
                    reward_surf = self.font.render(f"Step Reward: {step_reward_val:.3f}", True, reward_color)
                    self.screen.blit(reward_surf, (10, y_pos)); y_pos += 20 # Aggiorna y_pos

                    # Info specifiche per la guida NN basata sulla forma
                    if current_step_info.get('nn_guidance_active_shape', False):
                        nn_hud_color = ORANGE 
                        
                        time_guid = current_step_info.get('nn_time_elapsed_guidance', 0.0)
                        dist_shape = current_step_info.get('nn_dist_to_ref_shape', -1.0)
                        r_prox = current_step_info.get('reward_comp_spatial_proximity', 0)
                        r_align = current_step_info.get('reward_comp_velocity_alignment', 0) # Sarà 0 se il peso è 0
                        r_compl_spatial = current_step_info.get('reward_comp_nn_traj_completed_spatial',0)
                        # r_prog = current_step_info.get('reward_comp_progress_along_path', 0) # Se aggiungi questa chiave

                        nn_shape_info_lines = [
                            f"NN Shape Guide: ON (Time: {time_guid:.1f}s)",
                            f"  Dist to Ref Shape: {dist_shape:.1f} px"
                        ]
                        if abs(r_prox) > 1e-3: nn_shape_info_lines.append(f"  R_Prox: {r_prox:.2f}")
                        if abs(r_align) > 1e-3: nn_shape_info_lines.append(f"  R_Align: {r_align:.2f}")
                        # if abs(r_prog) > 1e-3: nn_shape_info_lines.append(f"  R_Prog: {r_prog:.2f}")
                        if abs(r_compl_spatial) > 1e-3: nn_shape_info_lines.append(f"  R_ComplShape: {r_compl_spatial:.2f}")

                        for line in nn_shape_info_lines:
                            text_surf = self.font.render(line, True, nn_hud_color)
                            self.screen.blit(text_surf, (10, y_pos)); y_pos += 18 
                        y_pos += 5 
                    else: # Guida NN non attiva
                        text_surf = self.font.render("NN Shape Guide: OFF", True, GRAY)
                        self.screen.blit(text_surf, (10, y_pos)); y_pos += 20
                
                else: # Nessuna guida NN
                    text_surf = self.font.render("NN Guide: OFF", True, GRAY)
                    self.screen.blit(text_surf, (10, y_pos)); y_pos += 20
            other_lines = [
                f"Steps: {info_data_render['steps']}",
                f"Autopilot PPO (A): {'ON' if self.autopilot_on else 'OFF'}", # Rinominato per chiarezza
                f"Autopilot PID (P): {'ON' if self.trajectory_autopilot_on else 'OFF'}", # Rinominato
                f"Hitboxes (H): {'ON' if self.visualize_hitboxes else 'OFF'}",
                f"REC (R): {'ON' if self.is_recording_hud else 'OFF'}",
                f"Man. Target (C): {'SET' if self.landing_target_circumference_angle is not None else 'None'}",
                f"Aim Mode (M): {'ON' if self.is_in_targeting_mode else 'OFF'}"
            ]
            # Disegna info RL
            rl_hud_color = CYAN if self.is_nn_trajectory_guidance_active else GRAY
            #rl_status_text = f"RL Guidance NN: {'ON' if self.is_nn_trajectory_guidance_active else 'OFF'}"
            #if self.is_nn_trajectory_guidance_active and self.nn_trajectory_points:
            #    rl_status_text += f" (WP {self.current_nn_waypoint_index}/{self.nn_trajectory_total_waypoints})"
            #   other_lines.append(rl_status_text)
            if self.show_target_selected_message and self.large_font:
                current_time = time.time()
                # Show message for ~2 seconds
                if current_time - self.target_selected_message_start_time < 2.0:
                    feedback_surf = self.large_font.render("TARGET SELECTED", True, GREEN)
                    feedback_rect = feedback_surf.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2))
                    # Optional: add a semi-transparent background box
                    bg_rect = feedback_rect.inflate(20, 10)
                    bg_surf = pygame.Surface(bg_rect.size, pygame.SRCALPHA)
                    bg_surf.fill((0, 0, 0, 180)) # Black semi-transparent background
                    self.screen.blit(bg_surf, bg_rect.topleft)
                    self.screen.blit(feedback_surf, feedback_rect)
                else:
                    # Time expired, stop showing
                    self.show_target_selected_message = False
            
            for line in other_lines:
                text_surf = self.font.render(line, True, WHITE)
                self.screen.blit(text_surf, (10, y_pos)); y_pos += 20

            action_indicator_y = SCREEN_HEIGHT - 30
            action_spacing = 80
            thrust_active = thrust_cmd == 1 and not info_data_render['fuel_empty']
            thrust_color = GREEN if thrust_active else GRAY
            thrust_text_surf = self.font.render("THRUST [^]", True, thrust_color)
            self.screen.blit(thrust_text_surf, (10, action_indicator_y))
            rot_ccw_active = rotation_cmd == 1 and not info_data_render['fuel_empty']
            rot_ccw_color = GREEN if rot_ccw_active else GRAY
            rot_ccw_text_surf = self.font.render("ROT [<]", True, rot_ccw_color)
            self.screen.blit(rot_ccw_text_surf, (10 + action_spacing, action_indicator_y))
            rot_cw_active = rotation_cmd == 2 and not info_data_render['fuel_empty']
            rot_cw_color = GREEN if rot_cw_active else GRAY
            rot_cw_text_surf = self.font.render("ROT [>]", True, rot_cw_color)
            self.screen.blit(rot_cw_text_surf, (10 + 2 * action_spacing, action_indicator_y))

            if self.autopilot_on:
                ap_indicator_rect = pygame.Rect(SCREEN_WIDTH - 30, 10, 20, 20)
                pygame.draw.rect(self.screen, PURPLE, ap_indicator_rect)

            if self.episode_outcome_message and self.large_font:
                msg_color = WHITE
                if "LANDED" in self.episode_outcome_message.upper(): msg_color = GREEN
                
                elif "CRASHED" in self.episode_outcome_message.upper(): msg_color = RED
                elif "TRUNCATED" in self.episode_outcome_message.upper() or \
                     "TIMEOUT" in self.episode_outcome_message.upper() or \
                     "BOUNDS" in self.episode_outcome_message.upper() or \
                     "FUEL EMPTY" in self.episode_outcome_message.upper(): msg_color = YELLOW # Modificato per includere fuel empty
                outcome_surf = self.large_font.render(self.episode_outcome_message, True, msg_color)
                outcome_rect = outcome_surf.get_rect(center=(SCREEN_WIDTH//2, SCREEN_HEIGHT//3))
    
                self.screen.blit(outcome_surf, outcome_rect)
                self.screen.blit(outcome_surf, outcome_rect)


        if self.episode_outcome_message and self.large_font:
                msg_color = WHITE # Default
                msg_upper = self.episode_outcome_message.upper()
                if "LANDED" in msg_upper: msg_color = GREEN
                elif "CRASHED" in msg_upper or "ERROR" in msg_upper: msg_color = RED
                elif "TRUNCATED" in msg_upper or "TIMEOUT" in msg_upper or \
                     "BOUNDS" in msg_upper or "FUEL" in msg_upper : msg_color = YELLOW # Simplified "FUEL" check
                
                outcome_surf = self.large_font.render(self.episode_outcome_message, True, msg_color)
                outcome_rect = outcome_surf.get_rect(center=(SCREEN_WIDTH//2, SCREEN_HEIGHT//3))
                self.screen.blit(outcome_surf, outcome_rect)
        

        if self.render_mode == "human":
            if self.is_pygame_initialized: # Aggiunto controllo
                pygame.event.pump()
                pygame.display.flip()
                if self.clock: self.clock.tick(self.target_render_fps)
        elif self.render_mode == "rgb_array":
            if self.is_pygame_initialized: # Aggiunto controllo
                frame = pygame.surfarray.pixels3d(self.screen)
                return np.transpose(frame, axes=(1, 0, 2))
            else: # Fallback se pygame non è inizializzato per rgb_array
                return np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH, 3), dtype=np.uint8)
        return None # Per coerenza, se nessun caso matcha

    def close(self):
        if self.is_pygame_initialized:
            print("INFO: Chiusura Pygame...")
            pygame.font.quit()
            pygame.display.quit()
            pygame.quit()
            self.is_pygame_initialized = False
            self.screen=None; self.clock=None; self.font=None; self.large_font=None
            print("INFO: Pygame chiuso.")

def convert_state_to_km_for_nn(pos_px, vel_px_s, center_x, center_y, pixels_per_km):
    pos_x_rel_px = pos_px[0] - center_x
    pos_y_rel_px = pos_px[1] - center_y
    pos_km_nn = np.array([pos_x_rel_px / pixels_per_km, -pos_y_rel_px / pixels_per_km])
    vel_kms_nn = np.array([vel_px_s[0] / pixels_per_km, -vel_px_s[1] / pixels_per_km])
    return pos_km_nn, vel_kms_nn

def convert_vel_kms_to_px_s(vel_kms, pixels_per_km):
    return np.array([vel_kms[0] * pixels_per_km, -vel_kms[1] * pixels_per_km])

def predict_nn_trajectory_segment(current_pos_km_nn, current_vel_kms_nn, target_pos_km_nn,
                                  nn_model, scaler_x, scaler_y):
    input_data_km = np.array([[
        current_pos_km_nn[0], current_pos_km_nn[1],
        current_vel_kms_nn[0], current_vel_kms_nn[1],
        target_pos_km_nn[0], target_pos_km_nn[1]
    ]])
    input_scaled = scaler_x.transform(input_data_km)
    pred_scaled = nn_model.predict(input_scaled, verbose=0)
    pred_original_kms = scaler_y.inverse_transform(pred_scaled)[0]
    return pred_original_kms[0], pred_original_kms[1], pred_original_kms[2]

def calculate_total_path_length(waypoints):
    if not waypoints or len(waypoints) < 2:
        return 0.0
    total_length = 0.0
    for i in range(len(waypoints) - 1):
        p1 = np.array(waypoints[i]['pos'])
        p2 = np.array(waypoints[i+1]['pos'])
        total_length += np.linalg.norm(p2 - p1)
    return total_length

def generate_trajectory_from_nn_prediction(current_pos_px, nn_predicted_initial_vel_px_s, nn_predicted_tof_s,
                                        pixels_per_km_env, screen_center_x_env, screen_center_y_env,
                                        moon_radius_px_env, gravity_const_env, time_step =0.1, num_points=10):
    """
    Genera una sequenza di waypoint basata sulla velocità iniziale e TOF predetti dalla NN.
    Simula un volo balistico (solo gravità) per la durata TOF.

    Args:
        current_pos_px (np.array): Posizione iniziale attuale del lander [x, y] in pixel.
        nn_predicted_initial_vel_px_s (np.array): Velocità iniziale [vx, vy] del trasferimento predetta dalla NN, in pixel/s.
        nn_predicted_tof_s (float): Tempo di volo predetto dalla NN in secondi.
        pixels_per_km_env, screen_center_x_env, screen_center_y_env: parametri dell'ambiente.
        moon_radius_px_env (float): Raggio della luna in pixel.
        gravity_const_env (float): Costante gravitazionale dell'ambiente.
        time_step  (float): Passo temporale per la simulazione della traiettoria.
        num_points (int): Numero desiderato di punti nella traiettoria (o massimo).

    Returns:
        list: Lista di dizionari waypoint [{'t': time, 'pos': [x,y], 'vel': [vx,vy]}, ...]
            o None se la generazione fallisce.
    """
    if nn_predicted_tof_s <= 0:
        print("ERRORE (generate_trajectory): TOF predetto non positivo.")
        return None

    trajectory_waypoints  = []
    
    sim_pos = np.array(current_pos_px, dtype=np.float32)
    sim_vel = np.array(nn_predicted_initial_vel_px_s, dtype=np.float32) # USA LA VELOCITÀ INIZIALE PREDETTA DALLA NN
    
    # Calcola il numero di step basato sul TOF e sul time_step desiderato per la traiettoria
    # O un numero fisso di punti distribuiti lungo il TOF
    actual_num_steps = min(num_points, int(nn_predicted_tof_s / time_step ) +1)
    if actual_num_steps < 2: actual_num_steps = 2 # Minimo 2 punti
    
    sim_time_per_step = nn_predicted_tof_s / (actual_num_steps -1) if actual_num_steps > 1 else nn_predicted_tof_s


    current_time = 0.0

    for i in range(actual_num_steps):
        trajectory_waypoints.append({
            't': current_time, # Tempo relativo all'inizio di questa traiettoria simulata
            'pos': [sim_pos[0], sim_pos[1]],
            'vel': [sim_vel[0], sim_vel[1]]
        })

        if i == actual_num_steps - 1: # Ultimo punto
            break

        # Fisica semplificata (solo gravità) per la propagazione
        direction_to_center = np.array([screen_center_x_env, screen_center_y_env]) - sim_pos
        distance_sq = max(np.dot(direction_to_center, direction_to_center), 1.0)
        dist = np.sqrt(distance_sq)
        
        gravity_accel = np.zeros(2)
        if dist > 1e-6 : # Evita divisione per zero se esattamente al centro (improbabile)
            try:
                gravity_force_magnitude = gravity_const_env / distance_sq
                gravity_accel = direction_to_center / dist * gravity_force_magnitude
            except (ValueError, ZeroDivisionError):
                pass # gravity_accel rimane [0,0]

        sim_vel += gravity_accel * sim_time_per_step
        sim_pos += sim_vel * sim_time_per_step
        current_time += sim_time_per_step
        
        # Controllo impatto (opzionale, ma utile per non farla passare attraverso la luna)
        # dist_from_center_after_step = np.linalg.norm(sim_pos - np.array([screen_center_x_env, screen_center_y_env]))
        # if dist_from_center_after_step <= moon_radius_px_env:
        #     # Potresti voler terminare la traiettoria qui o gestirla
        #     trajectory_waypoints.append({'t': current_time, 'pos': list(sim_pos), 'vel': list(sim_vel)})
        #     print(f"WARN (generate_trajectory): Traiettoria interrotta per impatto previsto a t={current_time:.2f}s")
        #     break 
            
    if not trajectory_waypoints:
        return None
        
    return trajectory_waypoints

def load_and_convert_matlab_trajectory(filepath, pixels_per_km, screen_center_x, screen_center_y):
    """
    Carica una traiettoria da un file CSV generato da MATLAB (o simile) e la converte
    in coordinate pixel e sistema di riferimento per Pygame.

    Args:
        filepath (str): Percorso al file CSV della traiettoria.
        pixels_per_km (float): Fattore di scala per convertire km in pixel.
        screen_center_x (int): Coordinata X del centro dello schermo Pygame.
        screen_center_y (int): Coordinata Y del centro dello schermo Pygame.

    Returns:
        list: Una lista di dizionari, dove ogni dizionario rappresenta un waypoint
            con chiavi 't' (tempo), 'pos' ([x_px, y_px]), 'vel' ([vx_px/s, vy_px/s]).
            Restituisce None se il caricamento o la conversione falliscono.
    """
    converted_trajectory = []
    if filepath is None or not os.path.exists(filepath):
        print(f"ERRORE (load_and_convert_matlab_trajectory): File traiettoria '{filepath}' non fornito o non trovato.")
        return None
    try:
        df = pd.read_csv(filepath)
        
        # Verifica la presenza delle colonne necessarie
        required_cols = ['time_s', 'pos_x_km', 'pos_y_km', 'vel_x_kms', 'vel_y_kms']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"ERRORE (load_and_convert_matlab_trajectory): Colonne mancanti nel CSV '{filepath}': {', '.join(missing_cols)}")
            return None

        for index, row in df.iterrows():
            t_matlab = row['time_s']
            pos_x_matlab_km = row['pos_x_km']
            pos_y_matlab_km = row['pos_y_km']
            vel_x_matlab_kms = row['vel_x_kms']
            vel_y_matlab_kms = row['vel_y_kms']

            # Conversione coordinate e scala:
            # Posizione:
            # L'origine MATLAB (0,0) è il centro del corpo celeste.
            # L'origine Pygame per la traiettoria deve essere relativa a (screen_center_x, screen_center_y).
            # L'asse Y di Pygame è invertito (aumenta verso il basso) rispetto a un sistema cartesiano standard.
            pos_x_pixel = (pos_x_matlab_km * pixels_per_km) + screen_center_x
            pos_y_pixel = (-pos_y_matlab_km * pixels_per_km) + screen_center_y # Inverti Y MATLAB e aggiungi centro

            # Velocità:
            vel_x_pixel_s = vel_x_matlab_kms * pixels_per_km
            vel_y_pixel_s = -vel_y_matlab_kms * pixels_per_km # Inverti Y MATLAB

            converted_trajectory.append({
                't': t_matlab,        # Il tempo è relativo all'inizio della manovra
                'pos': [pos_x_pixel, pos_y_pixel],
                'vel': [vel_x_pixel_s, vel_y_pixel_s]
            })
            
        if not converted_trajectory:
            print(f"ATTENZIONE (load_and_convert_matlab_trajectory): Nessun punto caricato da '{filepath}' (file vuoto o solo intestazione?).")
            return None

        print(f"INFO: Traiettoria caricata e convertita con successo da '{filepath}', {len(converted_trajectory)} punti.")
        return converted_trajectory

    except FileNotFoundError: # Già gestito sopra, ma per sicurezza
        print(f"ERRORE (load_and_convert_matlab_trajectory): File traiettoria '{filepath}' non trovato (controllo secondario).")
        return None
    except KeyError as e:
        print(f"ERRORE (load_and_convert_matlab_trajectory): Chiave/Colonna mancante nel file CSV '{filepath}'. Errore: {e}")
        return None
    except Exception as e:
        print(f"ERRORE SCONOSCIUTO durante il caricamento o la conversione della traiettoria da '{filepath}': {type(e).__name__} - {e}")
        import traceback
        traceback.print_exc()
        return None
EARTH_RADIUS_KM_MATLAB = 171.5
DEFAULT_ALTITUDE_TARGET_NN_PX =50
APPROACH_DISTANCE_ALONG_ARC_PX = 70
def normalize_angle_helper(angle_rad):
    """Normalizza un angolo in radianti tra -pi e pi."""
    return (angle_rad + np.pi) % (2 * np.pi) - np.pi
def get_radial_speed_px(waypoint_pos_px, waypoint_vel_px, moon_center_px):
    vector_center_to_ship = waypoint_pos_px - moon_center_px
    distance_to_center = np.linalg.norm(vector_center_to_ship)
    if distance_to_center < 1e-6:
        return 0.0
    radial_unit_vector = vector_center_to_ship / distance_to_center
    radial_speed = np.dot(waypoint_vel_px, radial_unit_vector)
    return radial_speed
if __name__ == '__main__':
    print("INFO: Avvio test manuale ambiente LunarLanderEnv...")

    # Non ridefinire la funzione load_and_convert_matlab_trajectory se è già definita globalmente
    # Se è definita solo dentro la classe LunarLanderEnv (improbabile), allora dovresti renderla
    # una funzione helper esterna o un metodo statico accessibile.
    # Assumiamo che sia una funzione globale definita prima di questo blocco __main__.

    # --- Setup Parametri Ambiente e Autopilota PPO (Opzionale) ---
    log_dir_manual = "ppo_training_runs/PPO_20250509_012618/" # Aggiorna se il percorso è diverso
    model_filename_manual = "ppo_lunar_model.zip"
    vecnormalize_filename_manual = "final_vecnormalize.pkl"
    model_path_manual = os.path.join(log_dir_manual, model_filename_manual)
    vecnormalize_path_manual = os.path.join(log_dir_manual, vecnormalize_filename_manual)
    loaded_ppo_model = None
    vec_env_ap = None

    env_params_manual = {
        "render_mode": 'human', "visualize_hitboxes": False,
        "enable_wind": False, "wind_power": 0.0, "turbulence_power": 0.0,
        "initial_random_angle": False 
    }
    env = LunarLanderEnv(**env_params_manual)

    try: 
        if os.path.exists(model_path_manual):
            print(f"Caricamento modello PPO da: {model_path_manual}")
            loaded_ppo_model = PPO.load(model_path_manual, device='cpu', env=None)
            print(f"Modello PPO caricato.")
            if os.path.exists(vecnormalize_path_manual):
                print(f"Caricamento statistiche VecNormalize da: {vecnormalize_path_manual}")
                dummy_env_for_norm = DummyVecEnv([lambda: LunarLanderEnv(**env_params_manual)])
                vec_env_ap = VecNormalize.load(vecnormalize_path_manual, dummy_env_for_norm)
                vec_env_ap.training = False; vec_env_ap.norm_reward = False
                print("VecNormalize caricato per l'autopilota PPO.")
            else:
                print(f"ATTENZIONE: File VecNormalize '{vecnormalize_path_manual}' non trovato per PPO.")
        else:
            print(f"ATTENZIONE: Modello PPO '{model_path_manual}' non trovato. Autopilota PPO disabilitato.")
    except Exception as load_error:
        print(f"Errore caricamento PPO/VecNormalize: {load_error}")
        traceback.print_exc(); loaded_ppo_model = None; vec_env_ap = None

    # --- Caricamento Modello Neurale e Scaler ---
    neural_model_path = "optimal_trajectory_predictor_2d_csv.keras" # Assicurati che questo sia il nome corretto
    scaler_x_path_nn = "x_scaler_nn.pkl"
    scaler_y_path_nn = "y_scaler_nn.pkl"
    trained_neural_model = None
    x_scaler_nn = None
    y_scaler_nn = None

    if os.path.exists(neural_model_path):
        try:
            trained_neural_model = tf.keras.models.load_model(neural_model_path)
            if os.path.exists(scaler_x_path_nn) and os.path.exists(scaler_y_path_nn):
                with open(scaler_x_path_nn, "rb") as f_x, open(scaler_y_path_nn, "rb") as f_y:
                    x_scaler_nn = pickle.load(f_x)
                    y_scaler_nn = pickle.load(f_y)
                print("INFO: Modello neurale e scaler per predizione traiettoria caricati.")
            else:
                print(f"ATTENZIONE: Uno o entrambi i file scaler ('{scaler_x_path_nn}', '{scaler_y_path_nn}') non trovati. Autopilota neurale disabilitato.")
                trained_neural_model = None
        except Exception as e:
            print(f"ERRORE caricamento modello neurale o scaler: {e}"); trained_neural_model = None
    else:
        print(f"ATTENZIONE: Modello neurale '{neural_model_path}' non trovato. Autopilota neurale disabilitato.")


    obs, info = env.reset()
    total_reward_main = 0.0
    terminated, truncated = False, False
    print("INFO: Ambiente resettato per il test manuale.")

    PIXELS_PER_KM = 1.0 
    if hasattr(env, 'actual_moon_radius') and env.actual_moon_radius and env.actual_moon_radius > 0:
        PIXELS_PER_KM = env.actual_moon_radius / EARTH_RADIUS_KM_MATLAB
        print(f"INFO: Scala: {PIXELS_PER_KM:.3f} px/km (moon_px: {env.actual_moon_radius:.1f}, moon_km: {EARTH_RADIUS_KM_MATLAB:.1f})")
    else:
        print(f"ATTENZIONE: env.actual_moon_radius non valido. Uso scala fallback PIXELS_PER_KM = {PIXELS_PER_KM}.")

    matlab_trajectory_filepath = 'trajectories/optimal_trajectory_lambert.csv' 
    loaded_ext_trajectory = None
    waiting_for_trajectory_start = False

    actual_recording_active = False
    current_episode_transitions = []
    DEMO_SAVE_PATH = "manual_demonstrations.pkl"
    all_demonstrations = []
    if os.path.exists(DEMO_SAVE_PATH):
        try:
            with open(DEMO_SAVE_PATH, "rb") as f: all_demonstrations = pickle.load(f)
            print(f"INFO: Caricate {len(all_demonstrations)} transizioni da '{DEMO_SAVE_PATH}'.")
        except Exception as e: print(f"ATTENZIONE: Errore caricamento demo: {e}")

    print("\n--- CONTROLLI ---")
    print("UP/LEFT/RIGHT: Controllo Manuale")
    print("Q: Esci")
    print("A: Toggle Autopilota PPO (se caricato)")
    print("P: Toggle Autopilota PID (traiettoria esempio interna, se definita)")
    print("N: Toggle Autopilota con Rete Neurale (per traiettoria ottimale)") # NUOVO
    print("L: Carica/Attiva 'Attesa per Traiettoria Esterna' (da file CSV)")
    print("H: Toggle Hitbox")
    print("R: Toggle Registrazione Dimostrazione")
    print("M: Toggle Modalità Mira per Target Manuale")
    print("CLICK (in Modalità Mira): Seleziona Target sulla Luna")
    print("C: Cancella Target Manuale")
    print("-----------------\n")

    running = True
    try:
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT: running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q: running = False
                    if event.key == pygame.K_a:
                        if loaded_ppo_model:
                            env.autopilot_on = not env.autopilot_on
                            if env.autopilot_on: env.trajectory_autopilot_on = False; waiting_for_trajectory_start = False
                            print(f"INFO: Autopilota PPO {'Attivato' if env.autopilot_on else 'Disattivato'}.")
                        else: print("INFO: Modello PPO non caricato.")
                    
                    if event.key == pygame.K_p: # PID con traiettoria interna
                        env.trajectory_autopilot_on = not env.trajectory_autopilot_on
                        if env.trajectory_autopilot_on:
                            env.autopilot_on = False; waiting_for_trajectory_start = False
                            # Qui dovresti chiamare env.set_trajectory() con una traiettoria interna di esempio
                            # Esempio: env.set_trajectory(env.get_example_circular_orbit_trajectory())
                            print("INFO: Autopilota PID (traiettoria interna) Toggle. (Assicurati che una traiettoria sia impostata).")
                        else: print("INFO: Autopilota PID (traiettoria interna) Disattivato.")

                    if event.key == pygame.K_n: # Autopilota con Rete Neurale (Logica a uno o due archi)
                        if trained_neural_model and x_scaler_nn and y_scaler_nn and env.ship_pos is not None and PIXELS_PER_KM > 0:
                            # --- Calcola posizione attuale e target marker ---
                            current_RI_px = env.ship_pos.copy()
                            current_V0_px_s = env.ship_vel.copy()

                            if env.landing_target_display_pos is not None:
                                RF_final_marker_surface_px = np.array(env.landing_target_display_pos)
                            else:
                                RF_final_marker_surface_px = np.array([CENTER_X, CENTER_Y + env.actual_moon_radius]) # Default "polo sud" (Y di Pygame è GIÙ)

                            # --- Calcolo Angoli (Convenzione Matematica Y-SU) ---
                            lander_x_rel = current_RI_px[0] - CENTER_X
                            lander_y_rel_pygame = current_RI_px[1] - CENTER_Y
                            math_lander_y_rel = -lander_y_rel_pygame # Inverti Y per angolo matematico
                            lander_angle_rad = np.arctan2(math_lander_y_rel, lander_x_rel)

                            target_x_rel = RF_final_marker_surface_px[0] - CENTER_X
                            target_y_rel_pygame = RF_final_marker_surface_px[1] - CENTER_Y
                            math_target_y_rel = -target_y_rel_pygame # Inverti Y
                            target_angle_rad = np.arctan2(math_target_y_rel, target_x_rel)
                            
                            angle_diff_rad = normalize_angle_helper(target_angle_rad - lander_angle_rad)
                            angle_diff_deg = np.degrees(angle_diff_rad) # Ora in [-180, 180] con Y-SU

                            # --- CONDIZIONE PER LOGICA A DUE ARCHI ---
                            # Se il target è dietro (ang_diff negativo), fino a quasi 180gradi dietro.
                            # Ex: lander a 0deg. Target a -10deg -> True. Target a -170deg -> True. Target a -180deg (o +180) -> False (è opposto, gestito da caso standard)
                            #     Target a +10deg -> False (è davanti)
                            if -130 < angle_diff_deg < 7: 
                                print(f"INFO: Target ({angle_diff_deg:.1f}° Y-UP) è dietro. Calcolo traiettoria a due archi...")

                                # --- 1. Definisci Punto Intermedio Target ---
                                # Angolo: 180 gradi dall'angolo corrente del lander (nel sistema Y-UP)
                                intermediate_target_math_angle_rad = normalize_angle_helper(lander_angle_rad + np.pi)

                                # Altitudine/Raggio del punto intermedio:
                                current_lander_radius_px = np.linalg.norm(current_RI_px - np.array([CENTER_X, CENTER_Y]))
                                current_alt_lander_px = current_lander_radius_px - env.actual_moon_radius
                                current_alt_lander_px = max(0, current_alt_lander_px) # Evita altitudini negative qui

                                intermediate_point_alt_px = (50.0 + current_alt_lander_px) / 2.0 # Media tra 50px e alt lander
                                #intermediate_point_alt_px = max(20.0, intermediate_point_alt_px) # Minimo di sicurezza per l'alt interm.
                                intermediate_point_radius_px = env.actual_moon_radius + intermediate_point_alt_px
                                
                                # Coordinate cartesiane Pygame del punto intermedio
                                intermediate_target_pos_px = np.array([
                                    CENTER_X + intermediate_point_radius_px * np.cos(intermediate_target_math_angle_rad),
                                    CENTER_Y - intermediate_point_radius_px * np.sin(intermediate_target_math_angle_rad) # Riconverti Y per Pygame
                                ])
                                print(f"    Punto Intermedio calcolato a: {intermediate_target_pos_px} (alt: {intermediate_point_alt_px:.1f}px)")

                                # --- 2. Genera Primo Arco (Lander -> Punto Intermedio) ---
                                current_pos_km_nn_arc1, current_vel_kms_nn_arc1 = convert_state_to_km_for_nn(
                                    current_RI_px, current_V0_px_s, CENTER_X, CENTER_Y, PIXELS_PER_KM
                                )
                                intermediate_target_pos_km_nn_arc1, _ = convert_state_to_km_for_nn(
                                    intermediate_target_pos_px, np.zeros(2), CENTER_X, CENTER_Y, PIXELS_PER_KM
                                )
                                pred_vi_arc1_vx_kms, pred_vi_arc1_vy_kms, pred_tof_arc1_s = predict_nn_trajectory_segment(
                                    current_pos_km_nn_arc1, current_vel_kms_nn_arc1, intermediate_target_pos_km_nn_arc1,
                                    trained_neural_model, x_scaler_nn, y_scaler_nn
                                )

                                arc1_waypoints = []
                                if pred_tof_arc1_s <= 1e-2:
                                    print("ERRORE: NN Arc 1 predicted non-positive/too short TOF.")
                                else:
                                    arc1_initial_vel_px_s = convert_vel_kms_to_px_s(
                                        np.array([pred_vi_arc1_vx_kms, pred_vi_arc1_vy_kms]), PIXELS_PER_KM
                                    )
                                    arc1_waypoints = generate_trajectory_from_nn_prediction(
                                        current_pos_px=current_RI_px,
                                        nn_predicted_initial_vel_px_s=arc1_initial_vel_px_s,
                                        nn_predicted_tof_s=pred_tof_arc1_s,
                                        pixels_per_km_env=PIXELS_PER_KM, screen_center_x_env=CENTER_X, screen_center_y_env=CENTER_Y,
                                        moon_radius_px_env=env.actual_moon_radius, gravity_const_env=GRAVITY_CONST,
                                        time_step=TIME_STEP, num_points=25 # Numero di punti per questo arco
                                    )

                                if not arc1_waypoints or len(arc1_waypoints) < 2:
                                    print("ERRORE: Fallimento generazione waypoint Arco 1. Annullamento.")
                                    env.activate_nn_trajectory_guidance(None)
                                    env.trajectory_autopilot_on = False
                                    # continue # Salta al prossimo evento pygame
                                else:
                                    # --- 3. Definisci Punto di Hover Finale (logica standard) ---
                                    vec_center_to_final_marker_px = RF_final_marker_surface_px - np.array([CENTER_X, CENTER_Y])
                                    norm_vec_final_marker = np.linalg.norm(vec_center_to_final_marker_px)
                                    unit_radial_final_marker = vec_center_to_final_marker_px / (norm_vec_final_marker + 1e-9)
                                    
                                    final_hover_altitude = DEFAULT_ALTITUDE_TARGET_NN_PX 
                                    if final_hover_altitude < 20.0: # Minimo di sicurezza
                                        print(f"WARN: DEFAULT_ALTITUDE_TARGET_NN_PX ({final_hover_altitude}) è basso. Uso 20px.")
                                        final_hover_altitude = 20.0
                                    NN_Target_Hover_Point_Final_px = (np.array([CENTER_X, CENTER_Y]) +
                                                                    unit_radial_final_marker *
                                                                    (env.actual_moon_radius + final_hover_altitude))

                                    # --- 4. Genera Secondo Arco (Punto Intermedio -> Punto Hover Finale) ---
                                    arc1_end_pos_px = np.array(arc1_waypoints[-1]['pos'])
                                    arc1_end_vel_px_s = np.array(arc1_waypoints[-1]['vel'])

                                    current_pos_km_nn_arc2, current_vel_kms_nn_arc2 = convert_state_to_km_for_nn(
                                        arc1_end_pos_px, arc1_end_vel_px_s, CENTER_X, CENTER_Y, PIXELS_PER_KM
                                    )
                                    final_target_hover_pos_km_nn_arc2, _ = convert_state_to_km_for_nn(
                                        NN_Target_Hover_Point_Final_px, np.zeros(2), CENTER_X, CENTER_Y, PIXELS_PER_KM
                                    )
                                    pred_vi_arc2_vx_kms, pred_vi_arc2_vy_kms, pred_tof_arc2_s = predict_nn_trajectory_segment(
                                        current_pos_km_nn_arc2, current_vel_kms_nn_arc2, final_target_hover_pos_km_nn_arc2,
                                        trained_neural_model, x_scaler_nn, y_scaler_nn
                                    )
                                    
                                    arc2_waypoints = []
                                    if pred_tof_arc2_s <= 1e-2:
                                        print("AVVISO: NN Arc 2 predicted non-positive/too short TOF. Si userà solo l'Arco 1.")
                                        final_trajectory_waypoints_px = arc1_waypoints
                                    else:
                                        arc2_initial_vel_px_s = convert_vel_kms_to_px_s(
                                            np.array([pred_vi_arc2_vx_kms, pred_vi_arc2_vy_kms]), PIXELS_PER_KM
                                        )
                                        arc2_waypoints = generate_trajectory_from_nn_prediction(
                                            current_pos_px=arc1_end_pos_px,
                                            nn_predicted_initial_vel_px_s=arc2_initial_vel_px_s,
                                            nn_predicted_tof_s=pred_tof_arc2_s,
                                            pixels_per_km_env=PIXELS_PER_KM, screen_center_x_env=CENTER_X, screen_center_y_env=CENTER_Y,
                                            moon_radius_px_env=env.actual_moon_radius, gravity_const_env=GRAVITY_CONST,
                                            time_step=TIME_STEP, num_points=25 # Numero di punti per questo arco
                                        )

                                        if not arc2_waypoints or len(arc2_waypoints) < 2:
                                            print("AVVISO: Fallimento generazione waypoint Arco 2. Si userà solo l'Arco 1.")
                                            final_trajectory_waypoints_px = arc1_waypoints
                                        else:
                                            # --- 5. Combina Arco1 e Arco2 ---
                                            time_offset_for_arc2 = arc1_waypoints[-1]['t']
                                            # Pulisci il primo punto di arc2 se è troppo vicino all'ultimo di arc1
                                            if np.allclose(arc1_waypoints[-1]['pos'], arc2_waypoints[0]['pos'], atol=1e-1) and \
                                            abs(arc1_waypoints[-1]['t'] - (arc2_waypoints[0]['t'] + time_offset_for_arc2)) > 1e-2 : # Se i tempi sono diversi, ma posizioni vicine
                                                # Mantieni il punto finale di arc1, ma inizia arc2 dal suo secondo punto, aggiustando i tempi di arc2
                                                # basati sul tempo originale del secondo punto di arc2
                                                if len(arc2_waypoints) > 1:
                                                    time_start_arc2_original = arc2_waypoints[0]['t'] # tempo originale del primo punto di arc2
                                                    for wp in arc2_waypoints: # Aggiusta tutti i tempi di arc2 rispetto al tempo originale del suo primo punto
                                                        wp['t'] = (wp['t'] - time_start_arc2_original) + time_offset_for_arc2
                                                    final_trajectory_waypoints_px = arc1_waypoints + arc2_waypoints[1:] # Escludi il primo duplicato di arc2
                                                else: # arc2 ha solo un punto, che è duplicato
                                                    final_trajectory_waypoints_px = arc1_waypoints
                                            else: # Nessun duplicato problematico o tempi allineati
                                                for wp in arc2_waypoints: wp['t'] += time_offset_for_arc2
                                                final_trajectory_waypoints_px = arc1_waypoints + arc2_waypoints
                                    
                                    # --- Attiva la guida RL ---
                                    if final_trajectory_waypoints_px and len(final_trajectory_waypoints_px) >= 1:
                                        print(f"INFO: Traiettoria NN (due archi o fallback) generata con {len(final_trajectory_waypoints_px)} punti.")
                                        env.activate_nn_trajectory_guidance(final_trajectory_waypoints_px)
                                        env.autopilot_on = False 
                                        env.trajectory_autopilot_on = False 
                                        waiting_for_trajectory_start = False
                                        print("INFO: Guida RL (due archi o fallback) ATTIVATA.")
                                    else:
                                        print("ERRORE: Nessuna traiettoria valida generata. Guida RL non attivata.")
                                        env.activate_nn_trajectory_guidance(None)
                                        env.trajectory_autopilot_on = False
                            else: # Target è "davanti" (angle_diff_deg >= 0 o circa +/-180) -> USA LOGICA STANDARD (PRIMO ARCO ACCORCIATO)
                                print(f"\nINFO: Target angolarmente accettabile ({angle_diff_deg:.1f}° Y-UP). Attivazione Autopilota NN (Primo Arco Accorciato)...")
                                
                                # --- INCOLLA QUI LA TUA LOGICA PRECEDENTE PER IL CASO STANDARD (PRIMO ARCO ACCORCIATO) ---
                                # Dalla riga: current_pos_km_nn, current_vel_kms_nn = convert_state_to_km_for_nn(...)
                                # Fino alla riga: env.activate_nn_trajectory_guidance(final_trajectory_waypoints_px)
                                # e la gestione degli errori associata.
                                # Esempio di struttura (DA COMPLETARE CON IL TUO CODICE ESISTENTE):
                                # ---------------------------------------------------------------------
                                current_pos_km_nn, current_vel_kms_nn = convert_state_to_km_for_nn(
                                    current_RI_px, current_V0_px_s, CENTER_X, CENTER_Y, PIXELS_PER_KM
                                )
                                vec_center_to_final_marker_px = RF_final_marker_surface_px - np.array([CENTER_X, CENTER_Y])
                                norm_vec_center_to_final_marker = np.linalg.norm(vec_center_to_final_marker_px)
                                unit_radial_to_final_marker = vec_center_to_final_marker_px / (norm_vec_center_to_final_marker + 1e-9)

                                hover_altitude_standard = DEFAULT_ALTITUDE_TARGET_NN_PX 
                                if hover_altitude_standard < 20: 
                                    hover_altitude_standard = 20.0
                                NN_Target_Hover_Point_Final_px_standard = (np.array([CENTER_X, CENTER_Y]) +
                                                                        unit_radial_to_final_marker *
                                                                        (env.actual_moon_radius + hover_altitude_standard))
                                target_conceptual_hover_pos_km_nn_std, _ = convert_state_to_km_for_nn(
                                    NN_Target_Hover_Point_Final_px_standard, np.zeros(2), CENTER_X, CENTER_Y, PIXELS_PER_KM
                                )
                                pred_VI_std_vx_kms, pred_VI_std_vy_kms, pred_TOF_std_s = predict_nn_trajectory_segment(
                                    current_pos_km_nn, current_vel_kms_nn, target_conceptual_hover_pos_km_nn_std,
                                    trained_neural_model, x_scaler_nn, y_scaler_nn
                                )
                                if pred_TOF_std_s <= 1e-2:
                                    print("ERRORE (Standard Case): NN predicted non-positive/too short TOF. Aborting.")
                                else:
                                    conceptual_nn_initial_vel_std_px_s = convert_vel_kms_to_px_s(
                                        np.array([pred_VI_std_vx_kms, pred_VI_std_vy_kms]), PIXELS_PER_KM
                                    )
                                    num_points_std_arc = 50 
                                    conceptual_full_arc_waypoints_std = generate_trajectory_from_nn_prediction(
                                        current_pos_px=current_RI_px,
                                        nn_predicted_initial_vel_px_s=conceptual_nn_initial_vel_std_px_s,
                                        nn_predicted_tof_s=pred_TOF_std_s,
                                        pixels_per_km_env=PIXELS_PER_KM, screen_center_x_env=CENTER_X, screen_center_y_env=CENTER_Y,
                                        moon_radius_px_env=env.actual_moon_radius, gravity_const_env=GRAVITY_CONST,
                                        time_step=TIME_STEP, num_points=num_points_std_arc 
                                    )
                                    if not conceptual_full_arc_waypoints_std or len(conceptual_full_arc_waypoints_std) < 2:
                                        print("ERRORE (Standard Case): Failed to generate conceptual_full_arc_waypoints_std. Aborting.")
                                    else:
                                        pixels_to_anticipate_std = 40.0
                                        # ... (INCOLLA QUI LA TUA LOGICA DI TRONCAMENTO ESISTENTE per conceptual_full_arc_waypoints_std)
                                        #     che produce 'final_trajectory_waypoints_px_std' ...
                                        # Esempio (devi adattare con la tua logica di troncamento precisa):
                                        positions_on_arc_std = [np.array(wp['pos']) for wp in conceptual_full_arc_waypoints_std]
                                        total_path_length_std = 0
                                        segment_lengths_std = []
                                        for i in range(len(positions_on_arc_std) - 1):
                                            length = np.linalg.norm(positions_on_arc_std[i+1] - positions_on_arc_std[i])
                                            segment_lengths_std.append(length)
                                            total_path_length_std += length
                                        final_trajectory_waypoints_px_std = []
                                        if total_path_length_std <= pixels_to_anticipate_std:
                                            final_trajectory_waypoints_px_std = [conceptual_full_arc_waypoints_std[0].copy()]
                                            if len(conceptual_full_arc_waypoints_std) > 1: final_trajectory_waypoints_px_std.append(conceptual_full_arc_waypoints_std[1].copy())
                                        else:
                                            cumulative_length_from_start_std = 0.0; last_idx_std = 0
                                            for i in range(len(segment_lengths_std)):
                                                if (total_path_length_std - cumulative_length_from_start_std) <= pixels_to_anticipate_std: last_idx_std = i; break
                                                cumulative_length_from_start_std += segment_lengths_std[i]; last_idx_std = i + 1
                                            final_trajectory_waypoints_px_std = [conceptual_full_arc_waypoints_std[j].copy() for j in range(last_idx_std + 1)]
                                        if len(final_trajectory_waypoints_px_std) == 1 and len(conceptual_full_arc_waypoints_std) > 1:
                                            final_trajectory_waypoints_px_std.append(conceptual_full_arc_waypoints_std[1].copy())
                                        elif not final_trajectory_waypoints_px_std and conceptual_full_arc_waypoints_std:
                                            final_trajectory_waypoints_px_std = [conceptual_full_arc_waypoints_std[0].copy()]
                                        # --- Attiva la guida RL con la traiettoria standard accorciata ---
                                        if final_trajectory_waypoints_px_std and len(final_trajectory_waypoints_px_std) >= 1:
                                            print(f"INFO (Standard Case): Traiettoria NN accorciata generata con {len(final_trajectory_waypoints_px_std)} punti.")
                                            env.activate_nn_trajectory_guidance(final_trajectory_waypoints_px_std)
                                            env.autopilot_on = False; env.trajectory_autopilot_on = False; waiting_for_trajectory_start = False
                                            print("INFO (Standard Case): Guida RL su traiettoria NN ACCORCIATA ATTIVATA.")
                                        else:
                                            print("ERRORE (Standard Case): Traiettoria NN accorciata è vuota. Guida RL non attivata.")
                                # ---------------------------------------------------------------------
                                        
                        else: # Fine del blocco if principale (trained_neural_model and x_scaler_nn etc.)
                            print("ATTENZIONE: Modello neurale/scaler non caricati, o stato del lander non valido, o PIXELS_PER_KM non valido.")
                    # ... (altri event.key) ...
                    
                    if event.key == pygame.K_l: # Traiettoria esterna CSV
                        if loaded_ext_trajectory is None:
                            print(f"INFO: Caricamento traiettoria da '{matlab_trajectory_filepath}'...")
                            loaded_ext_trajectory = load_and_convert_matlab_trajectory(
                                matlab_trajectory_filepath, PIXELS_PER_KM, CENTER_X, CENTER_Y
                            )
                        if loaded_ext_trajectory:
                            env.set_trajectory(loaded_ext_trajectory) 
                            waiting_for_trajectory_start = True; env.trajectory_autopilot_on = False; env.autopilot_on = False
                            print("INFO: Modalità 'Attesa Inizio Traiettoria Esterna (CSV)' ATTIVATA.")
                            if len(loaded_ext_trajectory) > 0:
                                print(f"  Lander seguirà quando vicino a: Pos Pix [{loaded_ext_trajectory[0]['pos'][0]:.0f}, {loaded_ext_trajectory[0]['pos'][1]:.0f}]")
                        else: print(f"ERRORE: Traiettoria da '{matlab_trajectory_filepath}' non disponibile.")
                    
                    if event.key == pygame.K_h: env.visualize_hitboxes = not env.visualize_hitboxes; print(f"INFO: Hitbox {'ON' if env.visualize_hitboxes else 'OFF'}")
                    if event.key == pygame.K_r:
                        if not actual_recording_active: 
                            actual_recording_active = True 
                            env.is_recording_hud = True 
                            current_episode_transitions = []
                            print("INFO: REGISTRAZIONE AVVIATA.")
                        else: 
                            actual_recording_active = False; env.is_recording_hud = False; current_episode_transitions = []; print("INFO: Registrazione interrotta.")
                    if event.key == pygame.K_m: 
                        env.is_in_targeting_mode = not env.is_in_targeting_mode 
                        print(f"INFO: Modalità Mira {'ON' if env.is_in_targeting_mode else 'OFF'}") 
                        if not env.is_in_targeting_mode: 
                            env.preview_target_display_pos = None
                    if event.key == pygame.K_c: env.landing_target_circumference_angle = None; env.landing_target_display_pos = None; \
                        env.preview_target_display_pos = None; env.show_target_selected_message = False; \
                        env.is_in_targeting_mode = False; env.episode_outcome_message = ""; print("INFO: Target manuale CANCELLATO.")
                
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    if env.is_in_targeting_mode and env.preview_target_display_pos is not None:
                        vec_preview = np.array(env.preview_target_display_pos) - np.array([CENTER_X, CENTER_Y])
                        fixed_angle = np.arctan2(vec_preview[1], vec_preview[0])
                        env.landing_target_circumference_angle = fixed_angle
                        env.landing_target_display_pos = env.preview_target_display_pos
                        env.show_target_selected_message = True; env.target_selected_message_start_time = time.time()
                        env.is_in_targeting_mode = False; env.preview_target_display_pos = None; env.episode_outcome_message = ""
                        print(f"INFO: Target SELEZIONATO (Ang: {np.degrees(fixed_angle):.1f} deg). Modalità Mira OFF.")

            if env.is_in_targeting_mode and env.is_pygame_initialized:
                mouse_pos_preview = pygame.mouse.get_pos()
                vec_to_mouse_preview = np.array(mouse_pos_preview) - np.array([CENTER_X, CENTER_Y])
                dist_to_mouse_center_preview = np.linalg.norm(vec_to_mouse_preview)
                preview_threshold_on_moon = (env.actual_moon_radius * 0.15) if env.actual_moon_radius else DEFAULT_MOON_RADIUS * 0.15
                current_radius_for_preview = env.actual_moon_radius if env.actual_moon_radius else DEFAULT_MOON_RADIUS
                if abs(dist_to_mouse_center_preview - current_radius_for_preview) < preview_threshold_on_moon:
                    preview_angle = np.arctan2(vec_to_mouse_preview[1], vec_to_mouse_preview[0])
                    env.preview_target_display_pos = (CENTER_X + current_radius_for_preview * np.cos(preview_angle), CENTER_Y + current_radius_for_preview * np.sin(preview_angle))
                else: env.preview_target_display_pos = None
            
            if not running: break

            if waiting_for_trajectory_start and loaded_ext_trajectory and env.ship_pos is not None:
                primo_punto_traj_pos_px = np.array(loaded_ext_trajectory[0]['pos'])
                dist_al_primo_punto_px = np.linalg.norm(env.ship_pos - primo_punto_traj_pos_px)
                SOGLIA_DIST_ATTIVAZIONE_PX = max(20.0, PIXELS_PER_KM * 15.0)
                if env.render_mode == 'human' and env.font and hasattr(env, 'screen') and env.screen:
                    wait_hud_text = (f"WAITING TRAJ: Dist {dist_al_primo_punto_px:.0f}px (Target < {SOGLIA_DIST_ATTIVAZIONE_PX:.0f}px)")
                    text_surface = env.font.render(wait_hud_text, True, YELLOW); text_rect = text_surface.get_rect(centerx=CENTER_X, top=30)
                    env.screen.blit(text_surface, text_rect)
                if dist_al_primo_punto_px < SOGLIA_DIST_ATTIVAZIONE_PX:
                    print(f"INFO: Lander vicino a inizio traiettoria CSV (Dist: {dist_al_primo_punto_px:.1f}px). Attivazione PID.")
                    env.set_trajectory(loaded_ext_trajectory); env.trajectory_autopilot_on = True; waiting_for_trajectory_start = False
            
            current_action_for_step = [0, 0]
            prev_obs_for_demo = obs.copy() 
            keys = pygame.key.get_pressed()
            if env.is_nn_trajectory_guidance_active:
                # QUI L'AGENTE RL DOVREBBE DECIDERE L'AZIONE
                # Per il test manuale, potremmo simulare un agente o usare input manuale
                # Per l'addestramento reale, SB3 chiamerà env.step(action_from_agent)
                # In questo test manuale, se la guida RL è attiva, per ora lasciamo che sia manuale
                # per vedere se le osservazioni e le ricompense di base funzionano.
                # Oppure potremmo mettere un semplice "seguitore" proporzionale qui per test.
                if keys[pygame.K_UP]: current_action_for_step[0] = 1
                if keys[pygame.K_LEFT]: current_action_for_step[1] = 1
                elif keys[pygame.K_RIGHT]: current_action_for_step[1] = 2
            elif env.trajectory_autopilot_on:
                env.ap_trajectory_step_counter += 1
                if env.ap_trajectory_step_counter >= TRAJECTORY_UPDATE_RATE or not hasattr(env, '_cached_trajectory_action') or env._cached_trajectory_action is None:
                    env.ap_trajectory_step_counter = 0; env._cached_trajectory_action = env._get_trajectory_autopilot_action()
                current_action_for_step = env._cached_trajectory_action if hasattr(env, '_cached_trajectory_action') and env._cached_trajectory_action is not None else [0,0]
            elif env.autopilot_on:
                if loaded_ppo_model:
                    try:
                        obs_to_predict = vec_env_ap.normalize_obs(np.array([obs]).astype(np.float32)) if vec_env_ap else obs
                        action_ppo, _ = loaded_ppo_model.predict(obs_to_predict, deterministic=True)
                        current_action_for_step = action_ppo[0] if isinstance(action_ppo, np.ndarray) else action_ppo
                    except Exception as predict_err: print(f"ERRORE PPO predict: {predict_err}"); current_action_for_step = [0,0]
                else: current_action_for_step = [0,0]
            else: # Manual
                manual_thrust_cmd = 1 if keys[pygame.K_UP] else 0
                manual_rotation_cmd = 1 if keys[pygame.K_LEFT] else (2 if keys[pygame.K_RIGHT] else 0)
                current_action_for_step = [manual_thrust_cmd, manual_rotation_cmd]

            if env.fuel_empty: current_action_for_step = [0,0]

            if running:
                obs, reward_this_step, terminated, truncated, info = env.step(current_action_for_step)
                total_reward_main += reward_this_step
                if actual_recording_active:
                    current_episode_transitions.append({"obs": prev_obs_for_demo, "acts": np.array(current_action_for_step, dtype=np.int64), 
                                                        "rews": reward_this_step, "next_obs": obs.copy(), "dones": terminated or truncated, "infos": {}})
            
            if running and env.render_mode == 'human':
                env._render_frame(current_action_for_step)

            if running and (terminated or truncated):
                print(f"\nINFO: Episodio {'Terminated' if terminated else 'Truncated'} @ {info.get('steps',0)}. Score: {env.current_episode_score:.2f}")
                print(f"  Outcome: {env.episode_outcome_message}, Alt: {info.get('altitude_surface',0):.1f}m, V.Rad: {info.get('speed_radial',0):.2f}m/s, V.Tan: {info.get('speed_tangential',0):.2f}m/s, RelAng: {info.get('ship_angle_relative_deg',0):.1f}deg, Fuel: {info.get('fuel_percentage',0):.1f}%")
                if info.get('arc_distance_to_target') is not None: print(f"  ArcDist Target: {info['arc_distance_to_target']:.1f} units")
                
                if actual_recording_active:
                    if info.get("landed_successfully", False): all_demonstrations.extend(current_episode_transitions); print(f"INFO: Demo SUCCESSO ({len(current_episode_transitions)} transizioni) AGGIUNTA.")
                    else: print("INFO: Demo NON salvata (non successo o interrotta).")
                
                actual_recording_active = False; env.is_recording_hud = False; current_episode_transitions = []
                
                print("INFO: Reset automatico tra 2 secondi...")
                time.sleep(2)
                obs, info = env.reset(); total_reward_main = 0.0; terminated, truncated = False, False
                if loaded_ext_trajectory: print("INFO: Traiettoria CSV in memoria. Premi 'L' per riattivare attesa.")
                else: waiting_for_trajectory_start = False; env.trajectory_autopilot_on = False
                if obs is None: print("ERRORE: Reset fallito."); running = False
                else: print("\nINFO: Ambiente resettato.")

            if running and env.render_mode == 'human' and env.clock:
                env.clock.tick(env.target_render_fps)

    except KeyboardInterrupt: print("\nINFO: Uscita da tastiera (Ctrl+C).")
    except Exception as e: print(f"\nERRORE loop principale: {e}"); traceback.print_exc()
    finally:
        print("\nINFO: Test manuale terminato. Pulizia...")
        if all_demonstrations:
            try:
                with open(DEMO_SAVE_PATH, "wb") as f: pickle.dump(all_demonstrations, f)
                print(f"INFO: Demo ({len(all_demonstrations)} transizioni) salvate in '{DEMO_SAVE_PATH}'")
            except Exception as e: print(f"ERRORE salvataggio demo: {e}")
        else: print(f"INFO: Nessuna nuova demo da salvare.")
        if vec_env_ap: print("Chiusura VecNormalize PPO..."); vec_env_ap.close()
        if 'env' in locals() and hasattr(env, 'close') and callable(env.close):
            if env.is_pygame_initialized: print("Chiusura ambiente Pygame..."); env.close()
            else: print("INFO: Pygame non inizializzato, skip chiusura.")
        print("Pulizia completata. Uscita.")