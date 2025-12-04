"""RL-Based Dynamic Parameter Controller for GA."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import random

class PopulationStateEncoder:
    """Encode population state into features for RL agent."""
    
    def __init__(self):
        self.history_window = 5
        self.state_history = deque(maxlen=self.history_window)
    
    def encode_population_state(self, population_stats: dict, generation: int) -> np.ndarray:
       
        features = []
        
        # Current population metrics
        features.extend([
            population_stats.get('diversity', 0.0) / 20.0,  # Normalize diversity
            population_stats.get('reliability_mean', 0.0) / 100.0,
            population_stats.get('reliability_std', 0.0) / 50.0,
            population_stats.get('accuracy_mean', 0.0) / 100.0,
            population_stats.get('sparsity_mean', 0.0) / 100.0,
            generation / 100.0,  # Normalize generation
        ])
        
        # FAULT DEGRADATION PATTERN ANALYSIS
        fault_degradation_features = self._analyze_fault_degradation_patterns(population_stats)
        features.extend(fault_degradation_features)
        
        # Progress tracking
        if len(self.state_history) >= 2:
            prev_rel = self.state_history[-1].get('reliability_mean', 0.0)
            curr_rel = population_stats.get('reliability_mean', 0.0)
            rel_trend = (curr_rel - prev_rel) / 10.0  # Normalize trend
        else:
            rel_trend = 0.0
            
        features.extend([
            rel_trend,
            population_stats.get('generations_without_improvement', 0) / 10.0,
        ])
        
        # Convergence indicators
        best_fitness = population_stats.get('best_reliability', 0.0)
        avg_fitness = population_stats.get('reliability_mean', 0.0)
        convergence = (best_fitness - avg_fitness) / 100.0 if best_fitness > 0 else 0.0
        
        features.extend([
            convergence,
            population_stats.get('pareto_front_size', 0) / 50.0,  # Normalize front size
        ])
        
        # Historical context (trend over last 3 generations)
        if len(self.state_history) >= 3:
            rel_history = [s.get('reliability_mean', 0) for s in list(self.state_history)[-3:]]
            rel_trend_3gen = (rel_history[-1] - rel_history[0]) / 30.0
        else:
            rel_trend_3gen = 0.0
            
        features.append(rel_trend_3gen)
        
        # Update history
        self.state_history.append(population_stats)
        
        return np.array(features, dtype=np.float32)
    
    def _analyze_fault_degradation_patterns(self, population_stats: dict) -> list:
       
        features = []
        
        # Extract multi-level reliability scores (10, 25, 50, 100, 150 faults)
        rel_10 = population_stats.get('reliability_10_faults_mean', 0.0)
        rel_25 = population_stats.get('reliability_25_faults_mean', 0.0)
        rel_50 = population_stats.get('reliability_50_faults_mean', 0.0)
        rel_100 = population_stats.get('reliability_100_faults_mean', 0.0)
        rel_150 = population_stats.get('reliability_150_faults_mean', 0.0)
        
        fault_levels = [rel_10, rel_25, rel_50, rel_100, rel_150]
        
        if any(rel > 0 for rel in fault_levels):
            # CLIFF COLLAPSE DETECTION (like your 48% → 14% drop)
            max_single_drop = 0.0
            cliff_location = 0.0  # Where the cliff happens (normalized)
            
            for i in range(len(fault_levels) - 1):
                if fault_levels[i] > 0 and fault_levels[i+1] > 0:
                    drop = fault_levels[i] - fault_levels[i+1]
                    if drop > max_single_drop:
                        max_single_drop = drop
                        cliff_location = (i + 1) / len(fault_levels)  # 0-1 normalized
            
            features.extend([
                max_single_drop / 100.0,  # Normalize max drop (0-1)
                cliff_location,           # Where cliff occurs (0-1)
            ])
            
            # DEGRADATION CURVE SHAPE ANALYSIS
            # Calculate Area Under Curve (AUC) for degradation
            if rel_10 > 0:
                normalized_levels = [10, 25, 50, 100, 150]
                auc = np.trapz(fault_levels, normalized_levels) / (150 * 100)  # Normalize by max area
                
                # Consistency ratio: min/max reliability across fault levels
                valid_rels = [r for r in fault_levels if r > 0]
                if len(valid_rels) > 1:
                    consistency = min(valid_rels) / max(valid_rels)
                else:
                    consistency = 1.0
                    
                # Early collapse indicator: reliability at 25 faults vs 10 faults
                if rel_10 > 10.0:  # Avoid division by very small numbers
                    early_degradation = 1.0 - (rel_25 / rel_10)  # 0 = no degradation, 1 = total collapse
                else:
                    early_degradation = 0.5  # Neutral
                    
                features.extend([
                    auc,                    # Overall degradation AUC
                    consistency,            # Consistency across fault levels
                    early_degradation,      # Early degradation indicator
                ])
                
                # BRITTLENESS SCORE: High brittleness = cliff collapse pattern
                # Your example (64→48→14) would score high brittleness
                brittleness = 0.0
                if max_single_drop > 25.0:  # Large single drop
                    brittleness += 0.4
                if consistency < 0.3:       # Low consistency
                    brittleness += 0.3
                if early_degradation > 0.5: # Early collapse
                    brittleness += 0.3
                    
                features.append(min(1.0, brittleness))  # Normalize to 0-1
            else:
                # No valid reliability data - neutral features
                features.extend([0.5, 0.5, 0.5, 0.5])  # AUC, consistency, early_deg, brittleness
        else:
            # No multi-level data available - use neutral values
            features.extend([0.0, 0.0, 0.5, 0.5, 0.5, 0.5])  # All degradation features neutral
            
        return features


class DQNParameterController(nn.Module):
    """Deep Q-Network for GA parameter control."""
    
    def __init__(self, state_dim=17, action_dim=6, hidden_dim=64):  # Fixed: 17 features from state encoder
     
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(), 
            nn.Linear(hidden_dim, action_dim)
        )
        
    def forward(self, state):
        return self.network(state)


class RLParameterController:
    
    
    def __init__(self, config, learning_rate=0.001):
        """
        Initialize RL parameter controller.
        
        Args:
            config: GA configuration object
            learning_rate: Learning rate for neural network
        """
        self.config = config
        self.state_encoder = PopulationStateEncoder()
        
        # Enhanced degradation tracking
        self.degradation_patterns = deque(maxlen=20)  # Store last 20 patterns for analysis
        self.cliff_collapse_history = []
        
        # Moving window global tracking for adaptive penalties
        self.reliability_history = deque(maxlen=20)  # Last 20 generations of best reliability
        self.global_best_reliability = 0.0
        
        # DQN setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dqn = DQNParameterController().to(self.device)
        self.target_dqn = DQNParameterController().to(self.device)
        self.optimizer = torch.optim.Adam(self.dqn.parameters(), lr=learning_rate)
        
        # RL parameters
        self.epsilon = 0.3  # Exploration rate
        self.epsilon_decay = 0.998
        self.epsilon_min = 0.05
        self.gamma = 0.95  # Discount factor
        self.update_target_freq = 10  # Update target network frequency
        
        # Experience replay
        self.memory = deque(maxlen=2000)
        self.batch_size = 32
        
        # Action mapping (parameter adjustments)
        self.actions = {
            0: 'increase_mutation',      # Increase exploration
            1: 'decrease_mutation',      # Decrease exploration  
            2: 'increase_diversity',     # More diversity injection
            3: 'decrease_diversity',     # Less diversity injection
            4: 'restart_population',     # Population restart
            5: 'no_change'              # Keep current parameters
        }
        
        # Parameter bounds
        self.param_bounds = {
            'mutation_prob': (0.1, 0.9),
            'mutation_sigma': (5.0, 40.0),
            'diversity_threshold': (3.0, 15.0)
        }
        
        # Performance tracking
        self.reward_history = deque(maxlen=100)
        self.action_history = deque(maxlen=50)
        
        # Current adapted parameters
        self.current_params = {
            'mutation_prob': config.mutation_prob,
            'mutation_sigma': config.mutation_sigma,
            'diversity_threshold': 8.0,
        }
        
    def select_action(self, population_stats: dict, generation: int) -> int:
      
        state = self.state_encoder.encode_population_state(population_stats, generation)
        
        if random.random() < self.epsilon:
            # Exploration: random action
            action = random.randint(0, len(self.actions) - 1)
        else:
            # Exploitation: best action from DQN
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.dqn(state_tensor)
                action = q_values.argmax().item()
                
        self.action_history.append(action)
        return action
    
    def apply_action(self, action: int) -> dict:
       
        action_name = self.actions[action]
        param_changes = {}
        
        if action_name == 'increase_mutation':
            # Boost exploration when stuck
            new_prob = min(self.param_bounds['mutation_prob'][1], 
                          self.current_params['mutation_prob'] * 1.3)
            new_sigma = min(self.param_bounds['mutation_sigma'][1],
                           self.current_params['mutation_sigma'] * 1.4)
            
            self.current_params['mutation_prob'] = new_prob
            self.current_params['mutation_sigma'] = new_sigma
            param_changes = {'mutation_boost': True}
            
        elif action_name == 'decrease_mutation':
            # Fine-tune when converging well
            new_prob = max(self.param_bounds['mutation_prob'][0],
                          self.current_params['mutation_prob'] * 0.8)
            new_sigma = max(self.param_bounds['mutation_sigma'][0],
                           self.current_params['mutation_sigma'] * 0.7)
                           
            self.current_params['mutation_prob'] = new_prob
            self.current_params['mutation_sigma'] = new_sigma
            param_changes = {'mutation_reduction': True}
            
        elif action_name == 'increase_diversity':
            # More aggressive diversity injection
            new_threshold = min(self.param_bounds['diversity_threshold'][1],
                               self.current_params['diversity_threshold'] * 1.5)
            self.current_params['diversity_threshold'] = new_threshold
            param_changes = {'diversity_boost': True}
            
        elif action_name == 'decrease_diversity':
            # Less diversity injection (for refinement)
            new_threshold = max(self.param_bounds['diversity_threshold'][0],
                               self.current_params['diversity_threshold'] * 0.7)
            self.current_params['diversity_threshold'] = new_threshold
            param_changes = {'diversity_reduction': True}
            
        elif action_name == 'restart_population':
            # Trigger population restart
            param_changes = {'population_restart': True}
            
        else:  # no_change
            param_changes = {'no_change': True}
            
        return param_changes
    
    def calculate_reward(self, prev_stats: dict, current_stats: dict, 
                        applied_action: int) -> float:
        
        reward = 0.0
        
        # Primary reward: reliability improvement (use RAW reliability percentages)
        prev_rel = prev_stats.get('raw_reliability_mean', prev_stats.get('reliability_mean', 0.0))
        curr_rel = current_stats.get('raw_reliability_mean', current_stats.get('reliability_mean', 0.0))
        rel_improvement = curr_rel - prev_rel
        
        # PRIMARY OBJECTIVE: RELIABILITY IMPROVEMENT (This is the MAIN goal!)
        if rel_improvement > 5.0:
            reward += 50.0  # MASSIVE reward for excellent reliability improvement
        elif rel_improvement > 2.0:
            reward += 25.0  # HIGH reward for good reliability improvement  
        elif rel_improvement > 0.0:
            reward += 10.0  # Significant reward for ANY reliability improvement
        elif rel_improvement < -2.0:
            reward -= 30.0  # HEAVY penalty for reliability degradation
            
        # FAULT DEGRADATION ANALYSIS (HELPER for better reliability, not replacement)
        # This SUPPORTS the main reliability objective by ensuring stability
        degradation_reward = self._calculate_degradation_reward(prev_stats, current_stats)
        # Scale down degradation rewards to be SECONDARY to primary reliability
        reward += degradation_reward * 0.5  # 50% weight - helper function only
        
        # RELIABILITY BREAKTHROUGH BONUSES (Primary Success Metric)
        if prev_rel < 30.0 and curr_rel > 45.0:
            reward += 100.0  # MASSIVE breakthrough bonus - escaping local optima
        elif prev_rel < 40.0 and curr_rel > 50.0:
            reward += 75.0   # Major reliability breakthrough
        elif prev_rel < 50.0 and curr_rel > 60.0:
            reward += 50.0   # Good reliability progress
        
        # ABSOLUTE RELIABILITY ACHIEVEMENT BONUSES (with accuracy constraints)
        # Use RAW accuracy (not fitness values) for proper thresholds
        curr_acc = current_stats.get('raw_accuracy_mean', current_stats.get('accuracy_mean', 0.0))
        if curr_rel > 70.0 and curr_acc >= 85.0:  # Must maintain decent accuracy
            reward += 40.0   # Excellent reliability achievement
        elif curr_rel > 60.0 and curr_acc >= 83.0:
            reward += 25.0   # Good reliability achievement
        elif curr_rel > 50.0 and curr_acc >= 80.0:
            reward += 15.0   # Decent reliability achievement
        
        # ACCURACY PROTECTION: Hard constraint (not competing objective)  
        # Use raw accuracy percentages for proper thresholds
        if curr_acc < 65.0:  # Below 65% accuracy is HARD CONSTRAINT VIOLATION
            reward = -200.0  # OVERRIDE all rewards - this violates the constraint
            print(f"   RL: HARD CONSTRAINT VIOLATED - Accuracy {curr_acc:.1f}% < 65.0% threshold")
        elif curr_acc < 75.0:  # Below 75% accuracy is concerning but acceptable
            reward *= 0.3    # Reduce ALL rewards (70% penalty) 
            print(f"   RL: Accuracy warning - {curr_acc:.1f}% < 75.0%, rewards reduced by 70%")
            
        # Diversity maintenance (HELPER for reliability exploration)
        diversity = current_stats.get('diversity', 0.0)
        if 8.0 <= diversity <= 15.0:
            reward += 3.0   # Modest reward - diversity HELPS reliability exploration
        elif diversity < 3.0:
            reward -= 8.0   # Reduced penalty - focus on reliability, not diversity per se
            
        # Action-specific rewards with ADAPTIVE context
        action_name = self.actions[applied_action]
        adaptive_threshold = self.get_adaptive_penalty_threshold()
        
        if action_name == 'increase_mutation' and rel_improvement > 1.0:
            reward += 8.0   # Good reward for exploration that IMPROVES RELIABILITY
        elif action_name == 'restart_population' and rel_improvement > 3.0:
            reward += 20.0  # Strong reward for restart that IMPROVES RELIABILITY
        elif action_name == 'no_change':
            # Adaptive reward for no_change based on current performance vs moving window
            if curr_rel >= adaptive_threshold:
                reward += 3.0  # Good to maintain when above adaptive threshold
            else:
                reward -= 1.0  # Should explore more when below threshold
            
        # Update global tracking for adaptive penalties
        self._update_global_tracking(current_stats)
        
        # Adaptive stagnation penalty based on moving window
        stagnation = current_stats.get('generations_without_improvement', 0)
        moving_window_avg = np.mean(list(self.reliability_history)) if self.reliability_history else curr_rel
        
        # Adaptive stagnation penalty - stricter if we know better performance is possible
        if stagnation > 8:
            if curr_rel < moving_window_avg * 0.8:  # Significantly below recent performance
                reward -= stagnation * 2.5  # Heavier penalty
            else:
                reward -= stagnation * 1.0  # Lighter penalty if near recent performance
            
        return reward
    
    def _update_global_tracking(self, current_stats: dict):
        """Update moving window tracking for adaptive penalties."""
        curr_rel = current_stats.get('reliability_mean', 0.0)
        
        # Update global best
        if curr_rel > self.global_best_reliability:
            self.global_best_reliability = curr_rel
            
        # Update moving window
        self.reliability_history.append(curr_rel)
    
    def get_adaptive_penalty_threshold(self) -> float:
        """Calculate adaptive penalty threshold based on moving window."""
        if len(self.reliability_history) < 5:
            return 25.0  # Default threshold early in evolution
            
        # Use 80% of moving window average as threshold
        window_avg = np.mean(list(self.reliability_history))
        adaptive_threshold = window_avg * 0.8
        
        # Don't let threshold go too low (minimum 15%) or too high (maximum 50%)
        return max(15.0, min(50.0, adaptive_threshold))
    
    def _calculate_degradation_reward(self, prev_stats: dict, current_stats: dict) -> float:
       
        degradation_reward = 0.0
        
        # Extract current multi-level reliability
        curr_rel_10 = current_stats.get('reliability_10_faults_mean', 0.0)
        curr_rel_25 = current_stats.get('reliability_25_faults_mean', 0.0) 
        curr_rel_50 = current_stats.get('reliability_50_faults_mean', 0.0)
        curr_rel_100 = current_stats.get('reliability_100_faults_mean', 0.0)
        curr_rel_150 = current_stats.get('reliability_150_faults_mean', 0.0)
        
        curr_levels = [curr_rel_10, curr_rel_25, curr_rel_50, curr_rel_100, curr_rel_150]
        
        # Extract previous multi-level reliability
        prev_rel_10 = prev_stats.get('reliability_10_faults_mean', 0.0)
        prev_rel_25 = prev_stats.get('reliability_25_faults_mean', 0.0)
        prev_rel_50 = prev_stats.get('reliability_50_faults_mean', 0.0)
        prev_rel_100 = prev_stats.get('reliability_100_faults_mean', 0.0) 
        prev_rel_150 = prev_stats.get('reliability_150_faults_mean', 0.0)
        
        prev_levels = [prev_rel_10, prev_rel_25, prev_rel_50, prev_rel_100, prev_rel_150]
        
        if any(curr > 0 for curr in curr_levels):
            # INDIVIDUAL FAULT LEVEL REWARDS - Focus on absolute performance at each level
            
            # Define realistic thresholds based on your expectations
            fault_thresholds = {
                10:  {'excellent': 60, 'good': 40, 'acceptable': 20},
                25:  {'excellent': 50, 'good': 35, 'acceptable': 15}, 
                50:  {'excellent': 40, 'good': 25, 'acceptable': 10},
                100: {'excellent': 30, 'good': 20, 'acceptable': 8},
                150: {'excellent': 20, 'good': 15, 'acceptable': 5}  # Your expectations: 15-20% nice
            }
            
            # ENHANCED REWARDS: Stronger signals for high absolute reliability at each fault level
            for fault_level, curr_rel in zip([10, 25, 50, 100, 150], curr_levels):
                if curr_rel > 0 and fault_level in fault_thresholds:
                    thresholds = fault_thresholds[fault_level]
                    
                    if curr_rel >= thresholds['excellent']:
                        degradation_reward += 12.0  # STRONGER reward for excellent absolute reliability
                    elif curr_rel >= thresholds['good']:
                        degradation_reward += 8.0   # STRONGER reward for good absolute reliability  
                    elif curr_rel >= thresholds['acceptable']:
                        degradation_reward += 4.0   # STRONGER reward for acceptable performance
                    else:
                        degradation_reward -= 6.0   # STRONGER penalty for poor reliability
            
            # MULTI-LEVEL CONSISTENCY BONUS - Reward solutions good across ALL fault levels
            excellent_count = sum(1 for fault_level, curr_rel in zip([10, 25, 50, 100, 150], curr_levels) 
                                if curr_rel >= fault_thresholds.get(fault_level, {}).get('excellent', 100))
            good_count = sum(1 for fault_level, curr_rel in zip([10, 25, 50, 100, 150], curr_levels) 
                           if curr_rel >= fault_thresholds.get(fault_level, {}).get('good', 100))
            
            # ENHANCED CONSISTENCY BONUSES: Stronger rewards for high absolute reliability across ALL levels
            if excellent_count >= 4:  # Excellent at 4+ fault levels
                degradation_reward += 25.0  # MASSIVE bonus for excellent reliability across multiple levels
            elif excellent_count >= 3:  # Excellent at 3+ fault levels
                degradation_reward += 18.0  # MAJOR bonus for excellent consistency  
            elif good_count >= 4:  # Good at 4+ fault levels
                degradation_reward += 15.0  # STRONG bonus for good consistency across multiple levels
            elif good_count >= 3:  # Good at 3+ fault levels
                degradation_reward += 10.0  # DECENT bonus for good consistency
                
        return degradation_reward
    
    def _calculate_max_single_drop(self, reliability_levels: list) -> float:
        """Calculate maximum single drop between consecutive fault levels."""
        max_drop = 0.0
        for i in range(len(reliability_levels) - 1):
            if reliability_levels[i] > 0 and reliability_levels[i+1] > 0:
                drop = reliability_levels[i] - reliability_levels[i+1]
                max_drop = max(max_drop, drop)
        return max_drop
    
    def _calculate_degradation_smoothness(self, reliability_levels: list) -> float:
        """Calculate how smooth the degradation curve is (0-1 score)."""
        if len(reliability_levels) < 3:
            return 0.5
            
        valid_levels = [r for r in reliability_levels if r > 0]
        if len(valid_levels) < 3:
            return 0.5
            
        # Calculate variance in consecutive drops (lower = smoother)
        drops = []
        for i in range(len(valid_levels) - 1):
            drop = valid_levels[i] - valid_levels[i+1]
            drops.append(max(0, drop))  # Only count decreases
            
        if len(drops) < 2:
            return 0.5
            
        drop_variance = np.var(drops)
        # Convert variance to smoothness score (high variance = low smoothness)
        smoothness = max(0.0, 1.0 - (drop_variance / 400.0))  # Normalize by reasonable variance
        return smoothness
    
    def store_experience(self, state, action, reward, next_state, done=False):
        """Store experience in replay buffer."""
        self.memory.append((state, action, reward, next_state, done))
        
    def train_dqn(self):
        """Train DQN using experience replay."""
        if len(self.memory) < self.batch_size:
            return
            
        # Sample batch from memory
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device) 
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)
        
        # Current Q values
        current_q_values = self.dqn(states).gather(1, actions.unsqueeze(1))
        
        # Next Q values from target network
        with torch.no_grad():
            next_q_values = self.target_dqn(next_states).max(1)[0].detach()
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)
            
        # Compute loss and update
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        return loss.item()
    
    def update_target_network(self):
        """Update target network with main network weights."""
        self.target_dqn.load_state_dict(self.dqn.state_dict())
        
    def get_current_parameters(self) -> dict:
        """Get current adapted parameters."""
        return self.current_params.copy()
        
    def get_learning_stats(self) -> dict:
        """Get RL learning statistics."""
        action_counts = {}
        for action in self.action_history:
            action_name = self.actions[action]
            action_counts[action_name] = action_counts.get(action_name, 0) + 1
            
        # Analyze degradation patterns learned
        recent_patterns = list(self.degradation_patterns)[-10:] if self.degradation_patterns else []
        avg_brittleness = np.mean([p.get('brittleness', 0.5) for p in recent_patterns]) if recent_patterns else 0.5
        
        return {
            'current_params': self.current_params,
            'epsilon': self.epsilon,
            'avg_reward': np.mean(self.reward_history) if self.reward_history else 0.0,
            'action_distribution': action_counts,
            'memory_size': len(self.memory),
            # INDIVIDUAL FAULT LEVEL LEARNING STATS
            'global_best_reliability': self.global_best_reliability,
            'moving_window_avg': np.mean(list(self.reliability_history)) if self.reliability_history else 0.0,
            'adaptive_penalty_threshold': self.get_adaptive_penalty_threshold(),
            'window_size': len(self.reliability_history),
            'degradation_patterns_analyzed': len(self.degradation_patterns)
        }