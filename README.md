# BattleVision Strategist

**Game Theory Analysis of Adversarial Attacks on Battlefield Computer Vision**

A comprehensive Streamlit application demonstrating the strategic dynamics between adversarial attacks on AI vision systems and defensive countermeasures, analyzed through the lens of zero-sum game theory.

---

## Overview

BattleVision Strategist provides an interactive platform for analyzing adversarial warfare through game-theoretic modeling and computer vision simulation. The application demonstrates how low-cost adversarial patches can defeat sophisticated AI detection systems, and how optimal defense strategies can be derived using Nash equilibrium analysis.

---

## Features

### Four Interactive Modules

**1. THE BATTLEFIELD**
Real-world problem introduction with scenario-based analysis
- Conflict scenarios from Ukraine, Gaza, Syria, and urban warfare
- Visual comparison of clean vs. attacked detection systems
- Impact statistics and cost-effectiveness analysis
- Demonstrates the asymmetric advantage of cheap adversarial patches

**2. GAME THEORY WAR ROOM**
Core strategic analysis using game theory
- Interactive payoff matrix builder with customizable parameters
- Nash equilibrium calculator using linear programming
- Best response analysis for both attackers and defenders
- Strategy evolution visualization through replicator dynamics
- Complete mathematical formulation and interpretation

**3. VISION SIMULATOR**
Practical demonstration of attacks and defenses
- Upload custom images or use sample battlefield imagery
- Apply adversarial patches: Camouflage, Geometric, Texture, Random
- Test defense mechanisms: Gaussian Denoising, Median Filter, Bilateral Filter, JPEG Compression
- Real-time metrics showing attack success and defense recovery rates
- Before/after/defended image comparisons

**4. STRATEGIC INSIGHTS**
Comprehensive analysis and recommendations
- Key findings for attackers (asymmetric forces)
- Key findings for defenders (military/surveillance)
- Nash equilibrium strategic interpretation
- Real-world implementation recommendations
- Performance metrics and simulation results

---

## Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager
- 4GB RAM minimum (8GB recommended)

### Installation

1. Clone or download this repository

```bash
cd battlevision_app
```

2. Install dependencies

```bash
pip install -r requirements.txt
```

3. Run the application

```bash
streamlit run app.py
```

Alternatively, use the provided launcher scripts:
- **Linux/Mac**: `./start.sh`
- **Windows**: `start.bat`

4. Access the application

Open your browser and navigate to `http://localhost:8501`

The application will automatically open in your default browser.

---

## Project Structure

```
battlevision_app/
├── app.py                      # Main Streamlit application (~730 lines)
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
├── start.sh                    # Linux/Mac launcher script
├── start.bat                   # Windows launcher script
├── test_components.py          # Component testing suite
└── utils/
    ├── cv_functions.py         # Computer vision utilities (~250 lines)
    └── game_theory.py          # Game theory calculations (~300 lines)
```

---

## How to Use

### Tab 1: The Battlefield

**Explore Real-World Scenarios**
1. Select a conflict scenario from the dropdown menu
2. Review the scenario description, effectiveness rates, and costs
3. Compare clean detection vs. adversarial patch application
4. Understand the fundamental asymmetry in adversarial warfare

**Key Insights**
- Detection drops of 65-80% are achievable with simple patches
- Patch costs range from $5-200 vs. $2M+ detection systems
- Real-world examples demonstrate practical applicability

### Tab 2: Game Theory War Room

**Build the Payoff Matrix**
1. Adjust the 6 sliders to configure attack and defense parameters
   - 3 attack effectiveness sliders (Patches A, B, C)
   - 3 defense strength sliders (Defenses X, Y, Z)
2. Observe the payoff matrix update in real-time
3. Review the matchup bonuses (rock-paper-scissors dynamics)

**Calculate Nash Equilibrium**
1. Click "Calculate Optimal Strategies"
2. Review the optimal mixed strategies for both players
   - Attacker: Probability distribution over 3 patch types
   - Defender: Probability distribution over 3 defense types
3. Interpret the game value:
   - Positive: Attacker advantage
   - Negative: Defender advantage
   - Near-zero: Balanced game

**Analyze Strategic Dynamics**
1. View the strategy evolution plot (replicator dynamics)
2. Test best responses to pure strategies
3. Examine the complete mathematical formulation

### Tab 3: Vision Simulator

**Run Baseline Detection**
1. Upload an image or select a sample image
2. Click "Run Baseline Detection"
3. Observe detected objects with bounding boxes and confidence scores

**Launch Adversarial Attack**
1. Select patch type (Camouflage, Geometric, Texture, Random)
2. Adjust attack intensity (0.0 to 1.0)
3. Choose patch placement (Center, Top-Left, Bottom-Right, Random)
4. Click "Launch Attack!"
5. Observe the detection suppression (typically 60-80% reduction)

**Apply Defense Mechanism**
1. Select defense type (Gaussian, Median, Bilateral, JPEG)
2. Adjust defense strength (0.0 to 1.0)
3. Click "Apply Defense!"
4. Observe detection recovery (typically 70-90% restoration)

**Metrics Analysis**
- Compare object counts: Baseline vs. Attacked vs. Defended
- Review attack success rate percentage
- Evaluate defense recovery rate percentage

### Tab 4: Strategic Insights

**Review Findings**
1. Examine attacker insights and optimal strategies
2. Review defender recommendations and countermeasures
3. Analyze Nash equilibrium interpretation based on game value
4. Study real-world implementation advice

**Performance Metrics**
- Baseline detection count
- Attack success rate with confidence delta
- Defense recovery rate with positive delta
- Net defense effect calculation

**Strategic Recommendations**
- For attackers: Deployment tactics, cost optimization, randomization strategies
- For defenders: Ensemble models, continuous retraining, human-AI teaming
- For both: Adaptation strategies and monitoring protocols

---

## Game Theory Concepts

### Zero-Sum Game

BattleVision models adversarial computer vision as a **zero-sum game**:
- **Attacker's gain = Defender's loss**
- Total payoff always equals zero
- Pure conflict scenario with no possibility of cooperation
- One player's success necessarily comes at the other's expense

### Nash Equilibrium

The **Nash equilibrium** represents the strategic balance point:
- Neither player can improve their expected payoff by unilaterally changing strategy
- Both players are playing optimally given the opponent's strategy
- Represents the "stable" outcome of rational strategic interaction
- Calculated using linear programming (support enumeration method)

**Key Properties**:
- Guarantees existence in finite games (via mixed strategies)
- May not be unique (multiple equilibria possible)
- Provides a predictive model of rational behavior
- Forms the basis for strategic recommendations

### Mixed Strategies

Both players should **randomize** their choices:
- **Attacker**: Probability distribution over Camouflage, Geometric, and Texture patches
- **Defender**: Probability distribution over Denoising, Ensemble, and Attention defenses
- **Rationale**: Predictability allows exploitation; randomization prevents it
- **Implementation**: Use optimal probabilities from Nash equilibrium

**Why Pure Strategies Fail**:
- If attacker always uses Camouflage, defender can optimize against it
- If defender always uses Denoising, attacker can exploit that knowledge
- Mixed strategies ensure neither player can be exploited

### Game Value

The **game value** indicates structural advantage at equilibrium:
- **v > +0.1**: Strong attacker advantage (even optimal defense struggles)
- **-0.1 ≤ v ≤ +0.1**: Balanced game (execution quality determines outcome)
- **v < -0.1**: Strong defender advantage (robust defenses prevail)

---

## Technical Implementation

### Computer Vision Components

**Object Detection System**
- Simulated detection using Canny edge detection and contour analysis
- Returns bounding boxes with coordinates [x, y, width, height]
- Confidence scores based on detected area and characteristics
- Easily replaceable with YOLO, Faster R-CNN, or other production models

**Adversarial Patch Generation**

1. **Camouflage Pattern**
   - Multi-colored blobs in earth tones (greens, browns, tans)
   - Gaussian blur for natural blending
   - Most effective in outdoor/natural environments
   - Mimics military camouflage design principles

2. **Geometric Shapes**
   - Stripes, circles, and rectangles in contrasting colors
   - High-contrast edges confuse boundary detection
   - Effective against shape-based detectors
   - Easier to fabricate than complex patterns

3. **Texture Noise**
   - Perlin-like noise with high-frequency components
   - Upscaled from low resolution for coherent patterns
   - Disrupts gradient-based features
   - Computationally optimized attacks

4. **Random Pixels**
   - Pure random noise as baseline
   - Least sophisticated but still somewhat effective
   - Useful for ablation studies and comparisons

**Defense Mechanisms**

1. **Gaussian Denoising**
   - Applies Gaussian blur to smooth adversarial perturbations
   - Strength parameter controls kernel size and sigma
   - Effective against high-frequency noise
   - May reduce legitimate edge sharpness

2. **Median Filter**
   - Non-linear filter that replaces pixels with neighborhood median
   - Excellent at removing salt-and-pepper noise
   - Preserves edges better than Gaussian blur
   - Computationally efficient

3. **Bilateral Filter**
   - Edge-preserving smoothing filter
   - Considers both spatial distance and intensity similarity
   - Best balance between denoising and edge preservation
   - More computationally expensive

4. **JPEG Compression**
   - Simulates lossy compression artifacts
   - Removes high-frequency adversarial components
   - Quality parameter controls compression strength
   - Real-world applicable (actual JPEG pipeline)

### Game Theory Implementation

**Payoff Matrix Generation**
```python
payoff = attack_effectiveness - defense_strength + matchup_bonus
```

Components:
- **attack_effectiveness**: Inherent strength of each patch type (0.0 to 1.0)
- **defense_strength**: Inherent robustness of each defense (0.0 to 1.0)
- **matchup_bonus**: Rock-paper-scissors style interactions (-0.2 to +0.2)

**Nash Equilibrium Calculation**

Primary method: Support enumeration via `nashpy` library
- Enumerates all possible support sets
- Solves system of equations for each support
- Validates equilibrium conditions
- Returns first valid equilibrium found

Fallback method: Linear programming
- Solves dual formulation for maxmin and minmax
- Guarantees solution for all finite games
- Uses `scipy.optimize.linprog` with interior-point method
- More robust for degenerate cases

**Replicator Dynamics**

Models evolutionary strategy adaptation:
- Strategies that perform above average increase in frequency
- Strategies that perform below average decrease in frequency
- Simulates learning and adaptation over time
- Converges to Nash equilibrium (for certain game classes)

Update rule:
```
p[i](t+1) = p[i](t) * (1 + α * (u[i] - ū))
```
where:
- `p[i](t)` = probability of strategy i at time t
- `u[i]` = expected payoff of strategy i
- `ū` = average payoff across all strategies
- `α` = learning rate

---

## Interpretation Guide

### Game Value Analysis

| Game Value | Interpretation | Strategic Implication |
|------------|----------------|----------------------|
| v > +0.3 | Overwhelming attacker advantage | Defenders need technological breakthrough |
| +0.1 < v ≤ +0.3 | Moderate attacker advantage | Current defenses insufficient |
| -0.1 ≤ v ≤ +0.1 | Balanced game | Execution quality matters most |
| -0.3 ≤ v < -0.1 | Moderate defender advantage | Defenses effective with proper deployment |
| v < -0.3 | Overwhelming defender advantage | Attackers need new attack vectors |

### Success Metrics

**Attack Success Rate**
- **Excellent**: > 80% (detection suppression)
- **Good**: 60-80%
- **Moderate**: 40-60%
- **Poor**: < 40%

**Defense Recovery Rate**
- **Excellent**: > 90% (of baseline capability)
- **Good**: 70-90%
- **Moderate**: 50-70%
- **Poor**: < 50%

### Strategy Interpretation

**Highly Mixed Strategies** (e.g., [0.35, 0.33, 0.32])
- Game is well-balanced
- All strategies have similar expected values
- Small changes in opponent strategy don't dramatically favor one response
- Unpredictability is crucial

**Strongly Biased Strategies** (e.g., [0.70, 0.20, 0.10])
- One strategy is clearly superior in expectation
- Still requires mixing to prevent exploitation
- Dominant strategy should be played more frequently
- Indicates asymmetry in matchup quality

---

## Educational Applications

### Game Theory Courses

**Concepts Demonstrated**:
- Zero-sum games and minimax theorem
- Nash equilibrium existence and computation
- Mixed strategies and expected value calculation
- Best response dynamics and strategic stability
- Linear programming formulation of games
- Replicator dynamics and evolutionary game theory

**Learning Objectives**:
- Understand when and why mixed strategies are necessary
- Calculate Nash equilibria using support enumeration
- Interpret game values and strategic implications
- Apply game theory to real-world adversarial scenarios

### Computer Vision Courses

**Concepts Demonstrated**:
- Adversarial examples and robustness
- Attack generation and transferability
- Defense mechanisms and their limitations
- Object detection and confidence scores
- Image preprocessing and filtering
- Evaluation metrics for detection systems

**Learning Objectives**:
- Understand vulnerabilities in AI vision systems
- Design and implement adversarial attacks
- Develop and test defense mechanisms
- Evaluate robustness quantitatively

### Military/Security Studies

**Concepts Demonstrated**:
- Asymmetric warfare and cost-effectiveness analysis
- Technology advantage and strategic implications
- Offensive and defensive strategies in modern warfare
- Decision-making under uncertainty
- Intelligence and counter-intelligence dynamics

**Learning Objectives**:
- Analyze modern battlefield technology
- Evaluate strategic options using formal methods
- Understand AI's role in military operations
- Assess cost-benefit tradeoffs in defense systems

---

## Customization and Extension

### Adding New Attack Types

1. Create patch generation function in `utils/cv_functions.py`:

```python
def create_your_patch(size, intensity):
    """
    Create your custom adversarial patch
    
    Args:
        size: Patch dimensions (size x size)
        intensity: Attack strength (0.0 to 1.0)
    
    Returns:
        patch: numpy array (size, size, 3)
    """
    patch = np.zeros((size, size, 3), dtype=np.uint8)
    # Your patch generation logic here
    return patch
```

2. Update `apply_patch()` function to include new type
3. Add to UI dropdown in `app.py`
4. Update game theory parameters to include 4th strategy

### Adding New Defense Types

1. Extend `defend_image()` function in `utils/cv_functions.py`:

```python
def defend_image(image, defense_type, strength=0.6):
    result = image.copy()
    
    if defense_type == "Your Defense":
        # Your defense implementation
        result = your_defense_function(result, strength)
    
    return result
```

2. Add to UI dropdown in `app.py`
3. Update game theory parameters to include 4th defense

### Modifying Payoff Matrix

Edit `generate_payoff_matrix()` in `utils/game_theory.py`:

```python
def generate_payoff_matrix(attack_effectiveness, defense_strength):
    matrix = np.zeros((len(attack_effectiveness), len(defense_strength)))
    
    for i in range(len(attack_effectiveness)):
        for j in range(len(defense_strength)):
            # Your custom payoff calculation
            payoff = your_formula(attack_effectiveness[i], defense_strength[j])
            matrix[i, j] = payoff
    
    return matrix
```
---

## Future Enhancements

### Planned Features

**Computer Vision**
- Integration with YOLOv8, Faster R-CNN, and other state-of-the-art detectors
- Multi-spectral imaging support (thermal, infrared, radar)
- Video stream processing and temporal analysis
- 3D scene understanding and depth-aware attacks
- Transfer learning across different detection models

**Game Theory**
- Multi-agent scenarios (drone swarms with distributed strategies)
- Temporal dynamics and sequential games (attacker moves first, defender responds)
- Bayesian games with incomplete information (unknown opponent capabilities)
- Repeated games with reputation effects
- Extensive-form games with game trees
- Evolutionary game theory with population dynamics

**Machine Learning**
- Reinforcement learning agents that learn optimal strategies
- Deep Q-learning for strategy optimization
- Policy gradient methods for continuous strategy spaces
- Adversarial training for robust detection models
- Meta-learning for rapid adaptation

**User Interface**
- Real-time video processing
- Batch analysis of multiple images
- Parameter sweep and sensitivity analysis
- Export results to JSON, CSV, Excel
- 3D visualization of payoff landscapes
- Interactive game tree exploration

**Network Analysis**
- Networked surveillance systems
- Information sharing and coordination protocols
- Distributed Nash equilibrium computation
- Communication constraints and latency effects
- Byzantine adversaries and trust models

---

## Real-World Applications

### Military and Defense

**Drone Surveillance Optimization**
- Design camouflage patterns that defeat AI detection
- Test detection system robustness before deployment
- Develop countermeasures against adversarial attacks
- Optimize resource allocation (cheap patches vs. expensive sensors)

**Strategic Planning**
- Evaluate technological advantage in adversarial scenarios
- Cost-benefit analysis of defense system upgrades
- Predict opponent strategies using game-theoretic models
- Design mixed-strategy deployment protocols

**Training and Education**
- Teach officers about AI vulnerabilities and defenses
- Demonstrate asymmetric warfare principles
- Practice strategic decision-making under uncertainty

### Security Systems

**Adversarial Robustness Testing**
- Evaluate AI security cameras against adversarial attacks
- Test facial recognition systems for vulnerability
- Design robust authentication and access control
- Certify system resilience to adversarial perturbations

**Security System Design**
- Choose detection algorithms with best robustness guarantees
- Design defense-in-depth architectures with multiple layers
- Implement anomaly detection for adversarial attacks
- Balance false positive and false negative rates

### Research Applications

**Adversarial Machine Learning**
- Study fundamental limits of adversarial robustness
- Develop new attack and defense algorithms
- Theoretical analysis of certified defenses
- Transferability of adversarial examples

**Game-Theoretic Security**
- Apply game theory to cybersecurity problems
- Model attacker-defender interactions formally
- Compute optimal security policies
- Analyze market equilibria in security economics

**Computer Vision Robustness**
- Benchmark detection systems under adversarial conditions
- Develop robust training methods
- Study the geometry of adversarial examples
- Design provably robust architectures

---

## Mathematical Formulation

### Zero-Sum Game Representation

A zero-sum game is defined by:
- **Players**: Attacker (row player) and Defender (column player)
- **Strategies**: Attacker has m strategies, Defender has n strategies
- **Payoff Matrix**: A ∈ ℝ^(m×n) where A[i,j] is attacker's payoff when attacker plays i and defender plays j
- **Zero-Sum**: Defender's payoff is -A[i,j]

### Mixed Strategy Nash Equilibrium

A mixed strategy Nash equilibrium is a pair (p*, q*) where:
- p* ∈ Δ^m (probability distribution over attacker's strategies)
- q* ∈ Δ^n (probability distribution over defender's strategies)
- p* is a best response to q*: p* ∈ argmax_p p^T A q*
- q* is a best response to p*: q* ∈ argmin_q (p*)^T A q

### Linear Programming Formulation

**Attacker's Problem (Maxmin)**:
```
maximize v
subject to:
    Σ_i A[i,j] * p[i] ≥ v    for all j ∈ {1,...,n}
    Σ_i p[i] = 1
    p[i] ≥ 0                 for all i ∈ {1,...,m}
```

**Defender's Problem (Minmax)**:
```
minimize v
subject to:
    Σ_j A[i,j] * q[j] ≤ v    for all i ∈ {1,...,m}
    Σ_j q[j] = 1
    q[j] ≥ 0                 for all j ∈ {1,...,n}
```

By the Minimax Theorem (von Neumann, 1928):
```
max_p min_q p^T A q = min_q max_p p^T A q = v*
```

The game value v* represents the expected payoff at equilibrium.

### Support Enumeration Algorithm

For each possible support S_A ⊆ {1,...,m} and S_D ⊆ {1,...,n}:

1. Solve for p with support S_A:
   ```
   A[S_A, S_D] q = v * 1
   Σ_j q[j] = 1
   q[j] ≥ 0
   ```

2. Check if p is a best response to q (equilibrium condition)

3. If valid equilibrium found, return (p, q, v)

### Replicator Dynamics

The continuous-time replicator equation:
```
dp[i]/dt = p[i] * ((A q)[i] - p^T A q)
```

Discrete-time update:
```
p[i](t+1) = p[i](t) * (1 + α * ((A q)[i] - p^T A q))
p(t+1) = p(t+1) / Σ_i p[i](t+1)
```

Properties:
- Nash equilibria are fixed points
- Interior Nash equilibria are asymptotically stable
- Pure Nash equilibria attract nearby trajectories

---

## Performance Considerations

### Computational Complexity

**Nash Equilibrium Computation**:
- Support enumeration: O(2^m * 2^n * poly(m,n))
- Linear programming: O(max(m,n)^3)
- For 3×3 games: < 0.1 seconds

**Object Detection**:
- Simulated detection: O(image_size)
- Real YOLO: O(image_size), ~30-60ms per image on GPU

**Image Processing**:
- Patch application: O(patch_size^2)
- Defense mechanisms: O(image_size * kernel_size^2)

### Optimization Tips

1. **Caching**: Use Streamlit's `@st.cache_data` for expensive computations
2. **Batch Processing**: Process multiple images in parallel
3. **GPU Acceleration**: Use CUDA for real YOLO inference
4. **Downsampling**: Reduce image resolution for faster processing
5. **Lazy Loading**: Only compute visualizations when tabs are accessed

### Scalability

**Current Limits**:
- Game size: Up to 10×10 matrices (computational)
- Image size: Up to 4K resolution (memory)
- Batch size: Limited by available RAM

**Scaling to Large Games**:
- Use approximate Nash equilibrium algorithms
- Sample support sets instead of enumerating all
- Employ iterative methods (fictitious play, no-regret learning)
- Distributed computation for massive games



### Troubleshooting

**Import Errors**:
- Ensure all packages in `requirements.txt` are installed
- Try `pip install -r requirements.txt --upgrade`
- Check Python version (3.8+)

**Display Issues**:
- Clear browser cache
- Try a different browser
- Disable browser extensions

**Port Already in Use**:
- Kill existing Streamlit process
- Use alternative port: `streamlit run app.py --server.port 8502`

**Performance Issues**:
- Reduce image resolution
- Use simpler patch types
- Decrease number of replicator dynamics iterations

---

## Contributing

Contributions are welcome! This project benefits from:

**Code Contributions**:
- New attack types and adversarial patch designs
- Additional defense mechanisms
- Better detection models (YOLO integration)
- Performance optimizations
- Bug fixes and code improvements

**Documentation**:
- Additional examples and tutorials
- Improved mathematical explanations
- Real-world case studies
- Translation to other languages

**Research**:
- Novel game-theoretic analyses
- Empirical studies of attack/defense effectiveness
- Theoretical results on robustness
- Benchmarking against state-of-the-art

**UI/UX**:
- Design improvements
- Better visualizations
- More intuitive controls
- Accessibility enhancements

### Contribution Guidelines

1. Fork the repository
2. Create a feature branch
3. Make your changes with clear commits
4. Add tests for new functionality
5. Update documentation as needed
6. Submit a pull request with detailed description

---

## Disclaimer

**IMPORTANT**: This application is strictly for **educational and research purposes only**.

**Restrictions**:
- The scenarios and analyses are hypothetical and simplified
- Not intended for actual military operations or adversarial use
- Demonstrates game theory and computer vision concepts, not operational tactics
- Users are responsible for ethical and legal use of this software

**Ethical Considerations**:
- AI security research has dual-use implications
- Knowledge of vulnerabilities can be used for both attack and defense
- Responsible disclosure of security flaws is essential
- Consider societal impact of adversarial AI research

**Legal Compliance**:
- Ensure compliance with local laws and regulations
- Respect intellectual property rights
- Do not use for illegal purposes
- Follow institutional ethics guidelines for research

Use this software responsibly and ethically.

---

## References

### Game Theory

**Books**:
- Tadelis, S. (2013). *Game Theory: An Introduction*. Princeton University Press.
- Dixit, A., & Nalebuff, B. (2008). *The Art of Strategy*. W. W. Norton & Company.
- Osborne, M. J., & Rubinstein, A. (1994). *A Course in Game Theory*. MIT Press.

**Papers**:
- Von Neumann, J. (1928). "Zur Theorie der Gesellschaftsspiele". *Mathematische Annalen*, 100(1), 295-320.
- Nash, J. (1951). "Non-Cooperative Games". *Annals of Mathematics*, 54(2), 286-295.

### Adversarial Machine Learning

**Papers**:
- Ilyas, A., et al. (2019). "Adversarial Examples Are Not Bugs, They Are Features". *NeurIPS*.
- Brown, T. B., et al. (2017). "Adversarial Patch". *arXiv:1712.09665*.
- Goodfellow, I. J., et al. (2015). "Explaining and Harnessing Adversarial Examples". *ICLR*.
- Carlini, N., & Wagner, D. (2017). "Towards Evaluating the Robustness of Neural Networks". *IEEE S&P*.

**Surveys**:
- Yuan, X., et al. (2019). "Adversarial Examples: Attacks and Defenses for Deep Learning". *IEEE TNNLS*.
- Akhtar, N., & Mian, A. (2018). "Threat of Adversarial Attacks on Deep Learning in Computer Vision: A Survey". *IEEE Access*.

### Computer Vision

**Papers**:
- Redmon, J., et al. (2016). "You Only Look Once: Unified, Real-Time Object Detection". *CVPR*.
- Ren, S., et al. (2015). "Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks". *NeurIPS*.
- He, K., et al. (2017). "Mask R-CNN". *ICCV*.

### Game-Theoretic Security

**Papers**:
- Tambe, M. (2011). *Security and Game Theory*. Cambridge University Press.
- Grossklags, J., et al. (2008). "Security and Games". *IEEE Security & Privacy*.

