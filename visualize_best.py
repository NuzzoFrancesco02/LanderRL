import os
import time
import pickle
import numpy as np
import tensorflow as tf
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from pathlib import Path

# Importa l'ambiente custom e le funzioni necessarie
# Assicurati che lander_with_traj.py sia accessibile
try:
    from lander_with_traj import (
        LunarLanderEnv, CENTER_X, CENTER_Y, GRAVITY_CONST, TIME_STEP,
        DEFAULT_ALTITUDE_TARGET_NN_PX, EARTH_RADIUS_KM_MATLAB,
        convert_state_to_km_for_nn, convert_vel_kms_to_px_s,
        predict_nn_trajectory_segment, generate_trajectory_from_nn_prediction
    )
    # Importa il wrapper custom
    from trainer import TrajectoryGuidanceWrapper # Assumendo che il wrapper sia in last_rl.py
                                                 # o importalo dallo script originale se si chiama diversamente
except ImportError as e:
    print(f"Errore import: {e}")
    print("Assicurati che 'lander_with_traj.py' e lo script contenente 'TrajectoryGuidanceWrapper' siano accessibili.")
    exit(1)

# --- Configura i Percorsi ---
# Modifica questi percorsi se necessario per puntare al tuo modello specifico
RUN_ID = "20250513_132414" # L'ID del run che vuoi visualizzare
BASE_DIR = Path("lunar_lander_rl")
RUN_DIR = BASE_DIR / "runs" / f"run_{RUN_ID}"
BEST_MODEL_DIR = RUN_DIR / "models" / "best_model"

MODEL_PATH = BEST_MODEL_DIR / "best_model.zip"
VECNORM_PATH = BEST_MODEL_DIR / "vecnormalize.pkl"

# Percorsi Assets (assicurati che esistano)
ASSETS_DIR = BASE_DIR / "assets"
KERAS_MODEL_PATH = ASSETS_DIR / "optimal_trajectory_predictor_2d_csv.keras"
X_SCALER_PATH = ASSETS_DIR / "x_scaler_nn.pkl"
Y_SCALER_PATH = ASSETS_DIR / "y_scaler_nn.pkl"

NUM_EPISODES_TO_VISUALIZE = 3
DELAY_SECONDS = 1 / 60 # Per rallentare la visualizzazione

# --- Funzioni Ausiliarie (semplificate dallo script originale) ---

def load_prediction_assets(keras_path, x_scaler_path, y_scaler_path):
    """Carica il modello Keras e gli scaler"""
    if not all(p.exists() for p in [keras_path, x_scaler_path, y_scaler_path]):
        print(f"ERRORE: File asset mancanti. Controlla i percorsi:\n - {keras_path}\n - {x_scaler_path}\n - {y_scaler_path}")
        return None, None, None
    try:
        keras_model = tf.keras.models.load_model(keras_path)
        with open(x_scaler_path, "rb") as f: x_scaler = pickle.load(f)
        with open(y_scaler_path, "rb") as f: y_scaler = pickle.load(f)
        print("Modello Keras e scaler caricati.")
        return keras_model, x_scaler, y_scaler
    except Exception as e:
        print(f"Errore caricamento assets: {e}")
        return None, None, None

def calculate_pixels_per_km():
    """Calcola pixels_per_km (semplificato)"""
    # Potresti dover adattare questo se il calcolo dinamico è complesso
    # o semplicemente usare un valore fisso se lo conosci.
    # Qui usiamo un fallback o un calcolo semplice se possibile.
    try:
        temp_env = LunarLanderEnv(render_mode=None)
        _, _ = temp_env.reset()
        if hasattr(temp_env, 'actual_moon_radius') and temp_env.actual_moon_radius > 0:
            pixels_per_km = temp_env.actual_moon_radius / EARTH_RADIUS_KM_MATLAB
            print(f"pixels_per_km calcolato: {pixels_per_km:.3f}")
        else:
            pixels_per_km = 1.0 # Fallback
            print(f"Usato fallback pixels_per_km: {pixels_per_km}")
        temp_env.close()
        return pixels_per_km
    except Exception as e:
        print(f"Errore calcolo pixels_per_km: {e}. Uso fallback 1.0")
        return 1.0

# --- Funzione per Creare l'Ambiente di Visualizzazione ---

def make_vis_env(render_mode='human', seed=0):
    """Crea una singola istanza dell'ambiente con i wrapper corretti"""
    # Assicurati che queste variabili siano globali o passate correttamente
    global keras_model, x_scaler, y_scaler, pixels_per_km

    def _init():
        # Crea l'ambiente base
        env = LunarLanderEnv(render_mode=render_mode, random_seed=seed)
        # Applica Monitor (opzionale ma buona pratica)
        env = Monitor(env)
        # Applica il wrapper custom per la traiettoria
        # Assicurati che TrajectoryGuidanceWrapper sia definito correttamente
        if keras_model and x_scaler and y_scaler:
             env = TrajectoryGuidanceWrapper(env, keras_model, x_scaler, y_scaler, pixels_per_km)
        else:
             print("ATTENZIONE: Wrapper TrajectoryGuidance non attivato per mancanza di assets.")
        return env
    return _init

# --- Script Principale di Visualizzazione ---

if __name__ == "__main__":
    print("--- Inizio Visualizzazione Modello ---")

    # 1. Verifica esistenza file modello e stats
    if not MODEL_PATH.exists():
        print(f"ERRORE: File modello non trovato: {MODEL_PATH}")
        exit(1)
    if not VECNORM_PATH.exists():
        print(f"ERRORE: File VecNormalize non trovato: {VECNORM_PATH}")
        exit(1)

    # 2. Carica assets per TrajectoryGuidanceWrapper
    print("Caricamento assets per guida traiettoria...")
    keras_model, x_scaler, y_scaler = load_prediction_assets(KERAS_MODEL_PATH, X_SCALER_PATH, Y_SCALER_PATH)
    if not all([keras_model, x_scaler, y_scaler]):
        print("Impossibile caricare tutti gli assets necessari.")
        # Potresti decidere di uscire o continuare senza guida
        # exit(1)
        pass # Continua senza guida se fallisce

    # 3. Calcola pixels_per_km
    print("Calcolo pixels_per_km...")
    pixels_per_km = calculate_pixels_per_km()

    # 4. Crea l'ambiente vettorizzato (anche per un solo env)
    print("Creazione ambiente di visualizzazione...")
    # Usa un seed diverso per la visualizzazione rispetto al training
    vis_seed = np.random.randint(0, 10000)
    vec_env = DummyVecEnv([make_vis_env(render_mode='human', seed=vis_seed)])

    # 5. Carica le statistiche VecNormalize
    print(f"Caricamento statistiche VecNormalize da: {VECNORM_PATH}")
    try:
        vec_env = VecNormalize.load(str(VECNORM_PATH), vec_env) # Usa str() per compatibilità
        # Imposta in modalità valutazione
        vec_env.training = False
        vec_env.norm_reward = False
        print("Statistiche VecNormalize caricate con successo.")
    except Exception as e:
        print(f"ERRORE durante il caricamento di VecNormalize: {e}")
        print("Verifica che il file .pkl sia corretto e compatibile.")
        vec_env.close()
        exit(1)

    # 6. Carica il modello PPO
    print(f"Caricamento modello PPO da: {MODEL_PATH}")
    try:
        # Passa l'ambiente GIA' normalizzato
        model = PPO.load(MODEL_PATH, env=vec_env, device="auto")
        print("Modello PPO caricato con successo.")
    except ValueError as e:
        print(f"\nERRORE durante il caricamento del modello PPO: {e}")
        print("Questo di solito accade se l'observation space dell'ambiente creato ora")
        print("non corrisponde a quello salvato nel modello (spesso dovuto a wrapper mancanti o diversi).")
        print("Verifica che la funzione 'make_vis_env' applichi ESATTAMENTE gli stessi wrapper usati in training.")
        vec_env.close()
        exit(1)
    except Exception as e:
        print(f"\nERRORE generico durante il caricamento del modello PPO: {e}")
        vec_env.close()
        exit(1)


    # 7. Esegui Episodi di Visualizzazione
    print(f"\nEsecuzione di {NUM_EPISODES_TO_VISUALIZE} episodi...")
    for episode in range(NUM_EPISODES_TO_VISUALIZE):
        print(f"\n--- Episodio {episode + 1} ---")
        obs = vec_env.reset()
        terminated = np.array([False]) # Necessario per VecEnv
        truncated = np.array([False]) # Necessario per VecEnv
        total_reward = 0.0
        step = 0
        while not (terminated[0] or truncated[0]):
            # Usa deterministic=True per vedere la policy "migliore" appresa
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, info = vec_env.step(action)

            # Accedi all'info dal wrapper DummyVecEnv se necessario
            # info è una lista, prendi il primo elemento per env singolo
            # actual_info = info[0]

            # Renderizza l'ambiente (il render_mode='human' dovrebbe farlo)
            # vec_env.render() # Potrebbe non essere necessario se render_mode='human' è attivo

            total_reward += reward[0] # reward è un array con VecEnv
            step += 1
            time.sleep(DELAY_SECONDS) # Rallenta per vedere meglio

        print(f"Episodio {episode + 1} terminato dopo {step} steps.")
        print(f"Reward totale: {total_reward:.2f}")
        # Prova a stampare l'outcome se disponibile nell'info
        if 'final_info' in info[0] and info[0]['final_info'] is not None and 'outcome_message' in info[0]['final_info']:
             print(f"Outcome: {info[0]['final_info']['outcome_message']}")
        elif 'outcome_message' in info[0]: # Potrebbe essere direttamente in info
             print(f"Outcome: {info[0]['outcome_message']}")


    # 8. Cleanup
    print("\nChiusura ambiente...")
    vec_env.close()
    print("--- Visualizzazione Completata ---")