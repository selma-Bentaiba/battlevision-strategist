import numpy as np
import nashpy as nash
import matplotlib.pyplot as plt
from scipy.optimize import linprog

def generate_payoff_matrix(attack_effectiveness, defense_strength):
    """
    Generate payoff matrix based on attack and defense parameters
    
    Args:
        attack_effectiveness: List of 3 values [0,1] for each attack type
        defense_strength: List of 3 values [0,1] for each defense type
    
    Returns:
        3x3 numpy array representing payoff matrix (attacker's perspective)
    """
    # Initialize matrix
    matrix = np.zeros((3, 3))
    
    # Basic interaction: attack effectiveness vs defense strength
    for i in range(3):
        for j in range(3):
            # Base payoff from effectiveness
            attack_power = attack_effectiveness[i]
            defense_power = defense_strength[j]
            
            # Calculate payoff
            # Positive = attacker success, Negative = defender success
            payoff = attack_power - defense_power
            
            # Add some asymmetry based on matchups
            # Some attacks work better against certain defenses
            matchup_bonus = get_matchup_bonus(i, j)
            
            matrix[i, j] = payoff + matchup_bonus
    
    # Normalize to [-1, 1]
    max_val = np.abs(matrix).max()
    if max_val > 0:
        matrix = matrix / max_val
    
    return matrix


def get_matchup_bonus(attack_idx, defense_idx):
    """
    Get matchup-specific bonuses (rock-paper-scissors style)
    """
    # Matchup bonuses
    # 0: Camouflage, 1: Geometric, 2: Texture
    # vs
    # 0: Denoise, 1: Ensemble, 2: Attention
    
    bonuses = np.array([
        [0.1, -0.2, -0.1],   # Camouflage: good vs denoise, bad vs ensemble
        [-0.1, 0.1, -0.2],   # Geometric: good vs ensemble, bad vs attention
        [-0.2, -0.1, 0.2]    # Texture: good vs attention, bad vs denoise
    ])
    
    return bonuses[attack_idx, defense_idx]


def calculate_nash_equilibrium(payoff_matrix):
    """
    Calculate Nash equilibrium using linear programming
    
    Args:
        payoff_matrix: numpy array (m x n) representing payoff matrix
    
    Returns:
        attacker_strategy: optimal mixed strategy for attacker
        defender_strategy: optimal mixed strategy for defender
        game_value: value of the game
    """
    m, n = payoff_matrix.shape
    
    # Use nashpy library
    game = nash.Game(payoff_matrix)
    
    # Find Nash equilibrium using support enumeration
    equilibria = list(game.support_enumeration())
    
    if len(equilibria) > 0:
        # Take first equilibrium
        attacker_strategy, defender_strategy = equilibria[0]
        
        # Calculate game value
        game_value = np.dot(attacker_strategy, np.dot(payoff_matrix, defender_strategy))
        
        return attacker_strategy, defender_strategy, game_value
    else:
        # Fallback to uniform strategy if no equilibrium found
        attacker_strategy = np.ones(m) / m
        defender_strategy = np.ones(n) / n
        game_value = np.dot(attacker_strategy, np.dot(payoff_matrix, defender_strategy))
        
        return attacker_strategy, defender_strategy, game_value


def calculate_nash_equilibrium_lp(payoff_matrix):
    """
    Alternative implementation using linear programming directly
    """
    m, n = payoff_matrix.shape
    
    # Solve for attacker (row player, maximizer)
    # Variables: p_1, ..., p_m, v
    # Maximize v
    # Subject to: sum(p_i * A_ij) >= v for all j
    #            sum(p_i) = 1
    #            p_i >= 0
    
    c = np.zeros(m + 1)
    c[-1] = -1  # Maximize v (minimize -v)
    
    # Inequality constraints: -sum(p_i * A_ij) + v <= 0
    A_ub = np.column_stack([-payoff_matrix.T, np.ones(n)])
    b_ub = np.zeros(n)
    
    # Equality constraint: sum(p_i) = 1
    A_eq = np.zeros((1, m + 1))
    A_eq[0, :m] = 1
    b_eq = np.array([1])
    
    # Bounds
    bounds = [(0, None) for _ in range(m)] + [(None, None)]
    
    # Solve
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, 
                     bounds=bounds, method='highs')
    
    if result.success:
        attacker_strategy = result.x[:m]
        game_value_max = result.x[-1]
    else:
        attacker_strategy = np.ones(m) / m
        game_value_max = 0
    
    # Solve for defender (column player, minimizer)
    # Variables: q_1, ..., q_n, v
    # Minimize v
    # Subject to: sum(q_j * A_ij) <= v for all i
    #            sum(q_j) = 1
    #            q_j >= 0
    
    c = np.zeros(n + 1)
    c[-1] = 1  # Minimize v
    
    # Inequality constraints: sum(q_j * A_ij) - v <= 0
    A_ub = np.column_stack([payoff_matrix, -np.ones(m)])
    b_ub = np.zeros(m)
    
    # Equality constraint: sum(q_j) = 1
    A_eq = np.zeros((1, n + 1))
    A_eq[0, :n] = 1
    b_eq = np.array([1])
    
    # Bounds
    bounds = [(0, None) for _ in range(n)] + [(None, None)]
    
    # Solve
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                     bounds=bounds, method='highs')
    
    if result.success:
        defender_strategy = result.x[:n]
        game_value_min = result.x[-1]
    else:
        defender_strategy = np.ones(n) / n
        game_value_min = 0
    
    # Average the two game values (should be equal at equilibrium)
    game_value = (game_value_max + game_value_min) / 2
    
    return attacker_strategy, defender_strategy, game_value


def plot_strategy_evolution(payoff_matrix, num_iterations=50):
    """
    Plot the evolution of strategies using replicator dynamics
    
    Args:
        payoff_matrix: numpy array representing payoff matrix
        num_iterations: number of iterations to simulate
    
    Returns:
        matplotlib figure
    """
    m, n = payoff_matrix.shape
    
    # Initialize random strategies
    attacker_strategy = np.random.dirichlet(np.ones(m))
    defender_strategy = np.random.dirichlet(np.ones(n))
    
    # Track evolution
    attacker_history = [attacker_strategy.copy()]
    defender_history = [defender_strategy.copy()]
    
    # Learning rate
    alpha = 0.1
    
    for _ in range(num_iterations):
        # Calculate expected payoffs
        attacker_payoffs = payoff_matrix @ defender_strategy
        defender_payoffs = -payoff_matrix.T @ attacker_strategy
        
        # Average payoffs
        avg_attacker_payoff = attacker_strategy @ attacker_payoffs
        avg_defender_payoff = defender_strategy @ defender_payoffs
        
        # Replicator dynamics update
        attacker_strategy = attacker_strategy * (1 + alpha * (attacker_payoffs - avg_attacker_payoff))
        defender_strategy = defender_strategy * (1 + alpha * (defender_payoffs - avg_defender_payoff))
        
        # Normalize
        attacker_strategy = attacker_strategy / attacker_strategy.sum()
        defender_strategy = defender_strategy / defender_strategy.sum()
        
        attacker_history.append(attacker_strategy.copy())
        defender_history.append(defender_strategy.copy())
    
    # Create plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot attacker strategy evolution
    attacker_history = np.array(attacker_history)
    for i in range(m):
        ax1.plot(attacker_history[:, i], label=f'Patch {chr(65+i)}', linewidth=2)
    
    ax1.set_xlabel('Iteration', fontsize=12)
    ax1.set_ylabel('Strategy Probability', fontsize=12)
    ax1.set_title('Attacker Strategy Evolution', fontsize=14, fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1])
    
    # Plot defender strategy evolution
    defender_history = np.array(defender_history)
    for i in range(n):
        ax2.plot(defender_history[:, i], label=f'Defense {chr(88+i)}', linewidth=2)
    
    ax2.set_xlabel('Iteration', fontsize=12)
    ax2.set_ylabel('Strategy Probability', fontsize=12)
    ax2.set_title('Defender Strategy Evolution', fontsize=14, fontweight='bold')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1])
    
    plt.tight_layout()
    
    return fig


def calculate_best_response(payoff_matrix, opponent_strategy, player='attacker'):
    """
    Calculate best response to opponent's strategy
    
    Args:
        payoff_matrix: numpy array
        opponent_strategy: opponent's mixed strategy
        player: 'attacker' or 'defender'
    
    Returns:
        best_response: best pure strategy index
        expected_payoffs: expected payoffs for each pure strategy
    """
    if player == 'attacker':
        # Attacker's expected payoffs for each pure strategy
        expected_payoffs = payoff_matrix @ opponent_strategy
        best_response = np.argmax(expected_payoffs)
    else:  # defender
        # Defender's expected payoffs (negative of attacker's)
        expected_payoffs = -payoff_matrix.T @ opponent_strategy
        best_response = np.argmax(expected_payoffs)
    
    return best_response, expected_payoffs


def calculate_regret(payoff_matrix, attacker_strategy, defender_strategy):
    """
    Calculate regret for both players
    """
    # Attacker's expected payoff
    attacker_payoff = attacker_strategy @ payoff_matrix @ defender_strategy
    
    # Best response payoff
    _, attacker_payoffs = calculate_best_response(payoff_matrix, defender_strategy, 'attacker')
    best_attacker_payoff = np.max(attacker_payoffs)
    
    attacker_regret = best_attacker_payoff - attacker_payoff
    
    # Defender's expected payoff
    defender_payoff = -attacker_payoff
    
    # Best response payoff
    _, defender_payoffs = calculate_best_response(payoff_matrix, attacker_strategy, 'defender')
    best_defender_payoff = np.max(defender_payoffs)
    
    defender_regret = best_defender_payoff - defender_payoff
    
    return attacker_regret, defender_regret


def plot_payoff_landscape(payoff_matrix):
    """
    Create 3D visualization of payoff landscape (for 2x2 games)
    """
    if payoff_matrix.shape != (2, 2):
        return None
    
    # Create grid of mixed strategies
    p_values = np.linspace(0, 1, 50)
    q_values = np.linspace(0, 1, 50)
    P, Q = np.meshgrid(p_values, q_values)
    
    # Calculate payoffs
    Z = np.zeros_like(P)
    for i in range(len(p_values)):
        for j in range(len(q_values)):
            p = p_values[i]
            q = q_values[j]
            attacker_strategy = np.array([p, 1-p])
            defender_strategy = np.array([q, 1-q])
            Z[j, i] = attacker_strategy @ payoff_matrix @ defender_strategy
    
    # Create 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    surf = ax.plot_surface(P, Q, Z, cmap='RdYlGn', alpha=0.8)
    
    ax.set_xlabel('Attacker: P(Patch A)', fontsize=10)
    ax.set_ylabel('Defender: P(Defense X)', fontsize=10)
    ax.set_zlabel('Attacker Payoff', fontsize=10)
    ax.set_title('Payoff Landscape', fontsize=12, fontweight='bold')
    
    fig.colorbar(surf, ax=ax, label='Payoff')
    
    return fig
