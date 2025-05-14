import gymnasium as gym
from gymnasium.core import Wrapper
import numpy as np

class LunarLanderRewardWrapper(Wrapper):
    """
    Wrapper che modifica la funzione di reward del LunarLander per evitare
    problemi di oscillazione nella policy durante l'addestramento.
    """
    def __init__(self, env, fuel_efficiency_weight=0.5, time_penalty_factor=0.01):
        super().__init__(env)
        self.fuel_efficiency_weight = fuel_efficiency_weight
        self.time_penalty_factor = time_penalty_factor
        self.step_count = 0
        
    def reset(self, **kwargs):
        self.step_count = 0
        return self.env.reset(**kwargs)
    
    def step(self, action):
        self.step_count += 1
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Modifichiamo il reward per bilanciare l'uso del carburante e il tempo
        # Analisi dell'azione: [0] è main engine, [1] è left/right engine
        engine_usage = abs(action[0]) + abs(action[1])
        
        # 1. Penalizza leggermente episodi troppo lunghi (tempo)
        time_penalty = self.time_penalty_factor * self.step_count if self.step_count > 100 else 0
        
        # 2. Bilancia la penalità del carburante per evitare che l'agente diventi troppo passivo
        fuel_adjustment = 0
        if engine_usage > 0.1:  # C'è un uso significativo dei motori
            # Riduci un po' la penalità del carburante per non scoraggiare troppo
            fuel_adjustment = engine_usage * self.fuel_efficiency_weight
        
        # Applichiamo la penalità del tempo e l'aggiustamento del carburante
        modified_reward = reward + fuel_adjustment - time_penalty
        
        # Aggiungiamo informazioni utili per il debug
        if 'original_reward' not in info:
            info['original_reward'] = reward
        info['modified_reward'] = modified_reward
        info['time_penalty'] = time_penalty
        info['fuel_adjustment'] = fuel_adjustment
        
        return obs, modified_reward, terminated, truncated, info
