"""RL-Guided Adaptive Genetic Operators for Reliability-Aware Pruning."""

import numpy as np
import torch
import torch.nn as nn
import random
from typing import Dict, List, Tuple, Any
from collections import deque

class OperatorSelectionAgent:
    """Q-Learning agent for selecting optimal genetic operators."""
    
    def __init__(self, state_dim=8, action_dim=5, lr=0.01):
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr
        self.epsilon = 0.3  # Exploration rate
        self.epsilon_decay = 0.995
        self.gamma = 0.9  # Discount factor
        
        # Simple Q-table for operator selection
        self.q_table = np.zeros((100, action_dim))  # 100 discretized states
        self.experience_buffer = deque(maxlen=1000)
        
        # Operator mapping
        self.operators = {
            0: 'gaussian_mutation',
            1: 'island_hopping', 
            2: 'anti_convergence',
            3: 'reliability_guided',
            4: 'diversity_injection'
        }
        
        # Performance tracking
        self.operator_rewards = {op: deque(maxlen=50) for op in range(action_dim)}
        self.selection_history = []
        
    def encode_state(self, population_stats: Dict) -> int:
        
        # Extract key features
        diversity = population_stats.get('diversity', 0.0)
        rel_mean = population_stats.get('reliability_mean', 0.0)
        rel_std = population_stats.get('reliability_std', 0.0)
        generation = population_stats.get('generation', 0)
        stagnation = population_stats.get('generations_without_improvement', 0)
        
        # Normalize and discretize features
        div_bucket = min(9, int(diversity))  # 0-9
        rel_bucket = min(9, int(rel_mean / 10))  # 0-9 (0-90% reliability)
        
        # Combine into state index
        state_index = div_bucket * 10 + rel_bucket
        return min(99, state_index)
    
    def select_operator(self, population_stats: Dict) -> int:
       
        state = self.encode_state(population_stats)
        
        if random.random() < self.epsilon:
            # Exploration: random operator
            action = random.randint(0, self.action_dim - 1)
        else:
            # Exploitation: best known operator for this state
            action = np.argmax(self.q_table[state])
        
        self.selection_history.append((state, action))
        return action
    
    def update_q_values(self, reward: float, next_stats: Dict = None):
       
        if not self.selection_history:
            return
            
        # Get last state-action pair
        state, action = self.selection_history[-1]
        
        # Calculate reward-based update
        if next_stats:
            next_state = self.encode_state(next_stats)
            max_next_q = np.max(self.q_table[next_state])
            target = reward + self.gamma * max_next_q
        else:
            target = reward
            
        # Q-learning update
        current_q = self.q_table[state, action]
        self.q_table[state, action] += self.lr * (target - current_q)
        
        # Store performance
        self.operator_rewards[action].append(reward)
        
        # Decay exploration
        self.epsilon = max(0.05, self.epsilon * self.epsilon_decay)
    
    def calculate_reward(self, prev_stats: Dict, current_stats: Dict) -> float:
      
        reward = 0.0
        
        # Primary reward: reliability improvement
        prev_rel = prev_stats.get('reliability_mean', 0.0)
        curr_rel = current_stats.get('reliability_mean', 0.0)
        rel_improvement = curr_rel - prev_rel
        
        if rel_improvement > 2.0:
            reward += 10.0  # Strong positive reward for good improvement
        elif rel_improvement > 0.5:
            reward += 5.0   # Moderate reward
        elif rel_improvement < -1.0:
            reward -= 5.0   # Penalty for degradation
            
        # Diversity maintenance reward
        diversity = current_stats.get('diversity', 0.0)
        if diversity > 8.0:
            reward += 2.0  # Bonus for maintaining diversity
        elif diversity < 3.0:
            reward -= 3.0  # Penalty for over-convergence
            
        # Breakthrough reward: massive bonus for escaping local optima
        if prev_rel < 35.0 and curr_rel > 45.0:
            reward += 50.0  # Breakthrough bonus
            
        # Stagnation penalty
        stagnation = current_stats.get('generations_without_improvement', 0)
        if stagnation > 5:
            reward -= stagnation * 2.0
            
        return reward
    
    def get_operator_performance(self) -> Dict:
        """Get performance statistics for each operator."""
        performance = {}
        for op_idx, rewards in self.operator_rewards.items():
            if rewards:
                performance[self.operators[op_idx]] = {
                    'mean_reward': np.mean(rewards),
                    'success_rate': sum(1 for r in rewards if r > 0) / len(rewards),
                    'usage_count': len(rewards)
                }
        return performance


class RLAdaptiveGeneticOperators:
    """RL-enhanced genetic operators with adaptive selection."""
    
    def __init__(self, config, base_operators):
       
        self.config = config
        self.base_operators = base_operators
        self.rl_agent = OperatorSelectionAgent()
        self.prev_population_stats = None
        
    def adaptive_mutation(self, individual: List[float], 
                         population_stats: Dict,
                         **kwargs) -> Tuple[List[float]]:
       
        # Select operator using RL agent
        selected_op = self.rl_agent.select_operator(population_stats)
        
        # Apply selected operator - ensure all return DEAP-compatible tuples
        if selected_op == 0:  # gaussian_mutation
            result = self.base_operators.gaussian_mutation(individual, sigma=15.0, indpb=0.3)
            return (result,) if not isinstance(result, tuple) else result
            
        elif selected_op == 1:  # island_hopping
            result = self.base_operators.reliability_island_mutation(
                individual, 
                reliability_score=population_stats.get('reliability_mean', 0),
                population_diversity=population_stats.get('diversity', 0)
            )
            return (result,) if not isinstance(result, tuple) else result
            
        elif selected_op == 2:  # anti_convergence
            return self._anti_convergence_mutation(individual, population_stats)
            
        elif selected_op == 3:  # reliability_guided
            reliability = population_stats.get('reliability_mean', 0)
            if reliability < 30.0:
                # Aggressive exploration for low reliability
                result = self.base_operators.gaussian_mutation(individual, sigma=25.0, indpb=0.6)
            else:
                # Fine-tuning for good reliability
                result = self.base_operators.gaussian_mutation(individual, sigma=8.0, indpb=0.2)
            return (result,) if not isinstance(result, tuple) else result
                
        else:  # diversity_injection (operator 4)
            return self._diversity_focused_mutation(individual)
    
    def _anti_convergence_mutation(self, individual: List[float], 
                                  population_stats: Dict) -> Tuple[List[float]]:
        """Anti-convergence mutation that pushes away from population centroid."""
        mutated = list(individual)
        centroid = population_stats.get('centroid', [50.0] * len(individual))
        
        for i in range(len(mutated)):
            if random.random() < 0.4:
                # Push away from centroid
                if mutated[i] > centroid[i]:
                    mutated[i] += random.uniform(10, 25)  # Push further away
                else:
                    mutated[i] -= random.uniform(10, 25)
                    
                mutated[i] = max(30.0, min(85.0, mutated[i]))
                
        return mutated,
    
    def _diversity_focused_mutation(self, individual: List[float]) -> Tuple[List[float]]:
        """Mutation focused on maintaining population diversity."""
        mutated = list(individual)
        
        # Random walk with large steps
        for i in range(len(mutated)):
            if random.random() < 0.5:
                mutated[i] += random.uniform(-30, 30)
                mutated[i] = max(30.0, min(85.0, mutated[i]))
                
        return mutated,
    
    def update_performance(self, current_population_stats: Dict):
        """Update RL agent based on population performance."""
        if self.prev_population_stats is not None:
            # Calculate reward and update Q-values
            reward = self.rl_agent.calculate_reward(
                self.prev_population_stats, 
                current_population_stats
            )
            self.rl_agent.update_q_values(reward, current_population_stats)
            
        self.prev_population_stats = current_population_stats.copy()
    
    def get_learning_stats(self) -> Dict:
        """Get RL learning statistics."""
        return {
            'operator_performance': self.rl_agent.get_operator_performance(),
            'exploration_rate': self.rl_agent.epsilon,
            'q_table_entropy': self._calculate_q_entropy()
        }
    
    def _calculate_q_entropy(self) -> float:
        """Calculate entropy of Q-table for learning analysis."""
        q_probs = np.abs(self.rl_agent.q_table) + 1e-8
        q_probs = q_probs / np.sum(q_probs, axis=1, keepdims=True)
        entropy = -np.sum(q_probs * np.log(q_probs + 1e-8))
        return entropy / q_probs.shape[0]  # Average entropy per state