# Formate Beads Experiment Designer

A comprehensive tool for designing and optimizing bacterial culture experiments using slow-release formate beads. This project combines biological modeling (Monod kinetics), Model Predictive Control (MPC), and practical laboratory constraints to help researchers plan experiments efficiently.

---

## 🧬 Scientific Background

### The Experimental Challenge

Growing bacterial cultures with formate as the carbon source presents unique challenges:

1. **Formate Toxicity**: High concentrations can inhibit bacterial growth
2. **Alkalinization**: Formate consumption releases OH⁻, raising pH and disrupting membrane permeability
3. **Depletion Risk**: Bacteria consume formate continuously; if depleted, growth stops
4. **Manual Labor**: Traditional methods require frequent substrate additions and pH monitoring

**Solution**: Slow-release formate beads provide sustained substrate delivery while maintaining optimal concentrations.

---

### Monod Kinetics: Modeling Bacterial Growth

This tool uses the **Monod equation** to predict how bacteria grow based on substrate availability:

```
μ = μ_max × [S] / (K_s + [S])
```

Where:
- **μ** = Specific growth rate (1/day)
- **μ_max** = Maximum growth rate (1/day) - how fast bacteria can grow with unlimited substrate
- **[S]** = Substrate concentration (mmol/L) - formate available in the medium
- **K_s** = Half-saturation constant (mmol/L) - substrate concentration at which growth is half-maximal

**Key Insight**: When [S] >> K_s, bacteria grow at maximum rate. When [S] << K_s, growth slows dramatically.

**Yield Coefficient (Y_xs)**:
```
ΔOD = Y_xs × Δ[Substrate consumed]
```
- Converts substrate consumption (mmol/L) to bacterial growth (Optical Density units)
- For formate: Y_xs ≈ 0.0067 OD per mmol/L (much lower than glucose due to lower energy content)

---

### Slow-Release Bead Technology

Two bead types provide complementary release profiles:

#### **M07 Beads: Fast Initial Release**
- High release on Day 1 (0.444 mmol/day)
- Rapid decay over 7 days
- **Use case**: Quick substrate boost when concentration drops rapidly
- Total release: ~0.739 mmol per bead

#### **M03 Beads: Sustained Release**
- More uniform release over 7 days (0.28 mmol/day initially)
- Slower decay
- **Use case**: Maintenance and steady substrate supply
- Total release: ~0.507 mmol per bead

**Bead Strategy**: The algorithm combines both types intelligently:
- Initial beads: 70% M03 / 30% M07 (balanced for fast response with sustained release)
- OD-aware scaling: Initial beads scale with (OD/0.02)^1.2 for high starting densities
- Later days: Dynamic M07/M03 ratios (60/40 urgent, 50/50 moderate, 45/55 normal)

---

### pH Control: HCl Requirements

**The Alkalinization Problem**:
```
HCOO⁻ + O₂ → CO₂ + OH⁻ (bacterial metabolism)
```

Each mmol of formate consumed generates 1 mmol of OH⁻, raising pH and reducing membrane permeability.

**Solution**: Add HCl in a 1:1 molar ratio to neutralize alkalinization:
```
HCl + OH⁻ → H₂O + Cl⁻
```

The tool automatically calculates:
- Total HCl needed throughout the experiment
- Daily HCl addition rates
- Practical volumes for different stock concentrations (12M, 6M, 1M)

⚠️ **Add HCl continuously or in small increments** to prevent pH spikes.

---

## 🎯 Gentle Model Predictive Control (MPC)

The tool uses **Model Predictive Control** with very conservative buffer factors and realistic growth prediction.

### How It Works

1. **Realistic Growth Prediction** (1.5-2 days ahead):
   - Uses **average substrate** (current + target)/2 for growth rate calculation
   - Prevents under-prediction when substrate is low at experiment start
   - Accounts for substrate increase from bead release

2. **OD-Aware Buffer Factors**:
   - Days 0-2: **0.05×** base (extremely conservative, bacteria just starting)
   - Days 3-4: **0.10×** base (still very gentle)
   - Days 5+: **0.2×** base (reduced from 0.5 to prevent explosion)
   - **OD scaling**: Buffer multiplied by (OD/0.02)^0.75 for adaptive control
   - **Hard cap**: Maximum buffer factor of 0.5 to prevent overshoot
   - Buffer reduced further (×0.7) when close to target

3. **Balanced Strategy**:
   - **Initial beads**: Scale with consumption rate × 2 days × (OD/0.02)^1.2
   - **Initial ratio**: 70% M03 / 30% M07 for mixed fast+sustained release
   - **Existing bead accounting**: 0.95 (trusts release profiles strongly)
   - **Dynamic M07/M03 ratios**: 60/40 (urgent), 50/50 (moderate), 45/55 (normal)

4. **Smart Control**:
   - **No upper bound check**: Bacteria naturally consume excess
   - **Asymmetric**: Prevents drops (critical), tolerates temporary highs

### Key Parameters

**Intervention Interval** (0.1-7.0 days):
- How often you check and add beads/HCl
- Examples: 0.5 = twice daily, 1.0 = daily, 2.0 = every 2 days

**Lower Threshold** (80-99% of target):
- When to trigger bead addition
- 95% = add when 5% below target (tight control, early intervention)
- 85% = add when 15% below target (moderate intervention)
- 80% = add when 20% below target (relaxed control)

**Important Interplay**:
```
Frequent checks (0.5 days) + Lower threshold (80-85%) = Prevents overshoots, tight control
Infrequent checks (2 days) + Higher threshold (90-95%) = Must act early, can't check often
```

---

## 🖥️ Two User Interfaces

### 1. Streamlit GUI (`formate_beads_gui.py`)

**Best for**: Quick experiment design, non-programmers, presentation-ready outputs

#### Features:
- ✅ **No coding required** - Pure point-and-click interface
- ✅ **Real-time visualization** - Adjust parameters, see results instantly
- ✅ **Interactive controls** - Sliders and input fields in left sidebar
- ✅ **Export ready** - Download bead schedule and HCl requirements as CSV
- ✅ **Professional plots** - 6-subplot figure with all experiment dynamics
- ✅ **Easy sharing** - Can be deployed to web for remote access

#### Running the GUI:

# Activate your Python environment
conda activate evya_venv

# Run Streamlit
streamlit run formate_beads_gui.py
```

The interface opens in your browser at `http://localhost:8501`

#### Using the GUI:

**Step 1: Configure Parameters (Left Sidebar)**
- **Culture volume** (L): Your bioreactor or flask volume
- **Initial OD**: Starting bacterial density
- **Target concentration** (mmol/L): Desired formate level (typically 20-50)
- **Lower threshold** (80-99%): When to add beads (% of target that triggers addition)
- **Intervention interval** (0.1-7.0 days): How often you'll check
- **Experiment days**: Total duration
- **Simulation dt**: Time resolution (leave at 0.01 for accuracy)

**Step 2: Run Simulation**
- Click **"🚀 Run Simulation"** button
- Wait 1-5 seconds for calculations

**Step 3: Explore Results (Tabs)**
- **📈 Plots**: All 6 plots showing substrate, growth, beads, rates, HCl
- **📋 Bead Schedule**: Day-by-day table of what to add (download CSV)
- **🧪 HCl Requirements**: Total and daily HCl needs with practical volumes
- **📊 Statistics**: Performance metrics and mass balance verification

**Step 4: Export Data**
- Download CSV files for lab notebook
- Right-click plots to save images

---

### 2. Jupyter Notebook (`formate_beads_notebook.ipynb`)

**Best for**: Detailed analysis, parameter exploration, understanding the algorithm, customization

#### Features:
- ✅ **Full transparency** - See every step of the calculation
- ✅ **Extensive documentation** - Comments explain the science and code
- ✅ **Modifiable** - Easy to test different parameters or modify algorithms
- ✅ **Educational** - Great for learning how MPC works
- ✅ **Advanced analysis** - Mass balance checks, sanity tests included

#### Running the Notebook:

```
# Activate your Python environment
conda activate evya_venv

# Launch Jupyter
jupyter notebook formate_beads_notebook.ipynb

# Or use VS Code's built-in notebook support
```

#### Structure of the Notebook:

**Cell 1: Configuration**
- Edit all experiment parameters here (volume, OD, target, threshold, interval)
- Displays current configuration

**Cells 2-3: Bead Release Profiles**
- Corrects empirical data for linear interpolation
- Creates M07 and M03 release functions

**Cells 4-5: Monod Kinetics & Calculator Class**
- Implements bacterial growth equations
- Defines the `ConstantSubstrateCalculator` class

**Cell 6: Run Simulation**
- Executes MPC algorithm with your parameters
- Shows timeline and prints progress

**Cell 7: Visualization**
- Generates all 6 plots with detailed annotations
- Same output as GUI but customizable

**Cell 8: Bead Schedule Table**
- Prints day-by-day additions with HCl requirements
- Shows totals and practical volumes

**Cell 9: Summary Statistics**
- Performance metrics (avg substrate, OD growth, etc.)
- HCl requirements for different stock concentrations

**Cell 10: Mass Balance Verification**
- Sanity check: Does consumed substrate match OD increase?
- Verifies yield coefficient accuracy

---

## ⚙️ Modifying Hardcoded Parameters

Some parameters rarely change and are hardcoded for simplicity. If you need to modify them:

### In the GUI (`formate_beads_gui.py`):

**Lines 13-23: Bead Properties**
```python
FORMATE_MW = 68  # Molecular weight (mg/mmol)
M07_FORMATE_CONTENT = 50.252  # mg per bead
M03_FORMATE_CONTENT = 35.1356  # mg per bead
```

**Lines 25-48: Bead Release Profiles**
```python
M07_EMPIRICAL = {
    1: 0.444,  # Day 1 release (mmol/day)
    2: 0.135,
    # ... etc
}

M03_EMPIRICAL = {
    1: 0.279909614,
    # ... etc
}
```
⚠️ If you modify these, ensure the total release matches bead formate content!

**Lines 50-59: Monod Kinetics**
```python
DEFAULT_MONOD_PARAMS = {
    'mu_max': 1,        # Maximum growth rate (1/day)
    'K_s': 20.0,        # Half-saturation constant (mmol/L)
    'Y_xs': 0.0067      # Yield coefficient (OD per mmol/L)
}
```

**Lines 145-280: Control Strategy**
```python
# Initial beads: Scale with OD (superlinear)
od_boost = max(1.0, (initial_od / 0.02) ** 1.2)
total_formate_needed = consumption_rate * volume * 2.0 * od_boost

# Split 70% M03, 30% M07
avg_release_per_bead = 0.7 * m03_total + 0.3 * m07_total
total_beads = max(3, int(np.ceil(total_formate_needed / avg_release_per_bead)))
initial_m07 = int(np.round(total_beads * 0.3))
initial_m03 = int(np.round(total_beads * 0.7))

# OD-aware buffer factors by day
od_buffer_scale = max(1.0, (od / 0.02) ** 0.75)
if current_day <= 2:
    base_buffer = 0.05
elif current_day <= 4:
    base_buffer = 0.10
else:
    base_buffer = 0.2  # Reduced from 0.5
buffer_factor = min(0.5, base_buffer * od_buffer_scale)  # Hard cap

# Reduce buffer more if close to target
if abs(deficit) < target * 0.15:
    buffer_factor *= 0.7

# Calculate total needed with MPC
total_needed = deficit + (predicted_consumption × buffer_factor)

# Existing bead accounting: 0.95
total_needed = max(0, total_needed - existing_supply * 0.95)

# Balanced M07/M03 ratios
if deficit_ratio > 0.5:
    m07_weight, m03_weight = 0.6, 0.4  # Urgent
elif deficit_ratio > 0.3:
    m07_weight, m03_weight = 0.5, 0.5  # Moderate
else:
    m07_weight, m03_weight = 0.45, 0.55  # Normal
```

---

### In the Notebook (`formate_beads_notebook.ipynb`):

**Cell 1 (Lines 45-65): All Configuration**
- Edit `FORMATE_MW`, bead properties, release profiles
- Modify `MONOD_PARAMS` dictionary
- Change `TARGET_CONCENTRATION`, `LOWER_THRESHOLD`, `INTERVENTION_INTERVAL`

**Cell 4 (Lines 331-443): Calculator Class**
- Buffer factors in the `calculate_bead_schedule` method (around line 380-390)
- Initial bead calculation logic around line 340-355

---

## 📊 Interpreting Results

### Substrate Concentration Plot
- **Blue line**: Actual substrate over time
- **Red dashed line**: Your target
- **Purple dashed line**: Action threshold (when beads are added)

**Good control**: Substrate oscillates near target, stays above action threshold

### Bacterial Growth (OD) Plot
- **Green line**: Bacterial density increasing over time
- **Purple star**: Inoculation point (Day 0)

**Expected**: Exponential-like growth, slowing as experiment progresses

### Bead Addition Schedule
- **Blue bars**: M07 beads added
- **Red bars**: M03 beads added
- **Purple line**: Bacterial inoculation time

**Pattern**: Large initial addition (Day 0), then smaller maintenance additions

### Release vs Consumption Rates
- **Cyan line**: Formate released from all active beads
- **Magenta line**: Formate consumed by bacteria

**Balance**: Release should slightly exceed consumption to maintain target

### Cumulative Consumption & HCl
- **Red line**: Total formate consumed
- **Blue dashed line**: Total HCl needed (1:1 ratio)

**Linear growth**: Indicates steady bacterial consumption

### HCl Addition Rate
- **Orange line**: How much HCl to add per day

**Profile**: Peaks mid-experiment when bacterial growth is fastest

---

## 📦 Installation

### Requirements

```bash
# Install dependencies
pip install -r requirements_gui.txt

# Contents: streamlit, numpy, matplotlib, pandas, scipy
```

### For Notebook:

```bash
pip install jupyter notebook
# or use VS Code with Jupyter extension
```

---

## 🧪 Typical Use Cases

### Standard Daily Protocol (Small Scale)
```
Volume: 0.1 L (100 mL flask)
Initial OD: 0.02
Target: 30 mmol/L
Threshold: 95%
Interval: 1.0 day
Duration: 7 days
```
**Result**: Check once per day, minimal beads needed, tight control

### Large Scale Bioreactor
```
Volume: 5 L
Initial OD: 0.02
Target: 30 mmol/L
Threshold: 90%
Interval: 1.0 day
Duration: 7 days
```
**Result**: Daily checks, more beads required, excellent control

### High-Frequency Critical Experiment
```
Volume: 0.1 L
Initial OD: 0.02
Target: 30 mmol/L
Threshold: 90%
Interval: 0.5 days (12 hours)
Duration: 7 days
```
**Result**: Check twice daily, minimal oscillations, excellent control

---

## 🎓 Educational Value

This tool demonstrates:
1. **Biological modeling**: Translating Monod kinetics into predictions
2. **Control theory**: MPC for optimal resource allocation
3. **Experimental design**: Balancing control quality vs. labor
4. **Data-driven decisions**: Using release profiles to optimize bead selection

Perfect for:
- Graduate students learning bioprocess engineering
- Researchers planning formate-based experiments
- Teaching control theory applications in biology

---

## 🔧 Troubleshooting

### GUI won't start
```bash
# Check Streamlit installation
streamlit --version

# Reinstall
pip install streamlit --upgrade
```

### Simulation errors
- Check that initial OD > 0
- Ensure target concentration > 0
- Verify intervention_interval ≤ experiment_days

### Unrealistic results
- Lower threshold too low? Try 85-95%
- Intervention interval too long? Try 1.0 day
- Check Monod parameters match your bacterial strain

---

## 📝 Citation

If you use this tool in your research, please cite:
- The Monod equation: Monod, J. (1949). "The Growth of Bacterial Cultures"
- Model Predictive Control principles in your methods section
- Include experiment designer parameters in supplementary materials

---

## 📧 Support

For questions or issues:
1. Check this README carefully
2. Review the notebook documentation (extensive comments)
3. Examine console output for error messages
4. Test with default parameters first

---

## 🚀 Quick Start Summary

**Want to design an experiment RIGHT NOW?**

```bash
# 1. Install
pip install streamlit numpy matplotlib pandas scipy

# 2. Run GUI
streamlit run formate_beads_gui.py

# 3. Configure in sidebar (or use defaults)

# 4. Click "Run Simulation"

# 5. Download CSV files from tabs

# 6. Start your experiment!
```

---

## License

Academic use encouraged. Cite appropriately in publications.

---

**Version**: 2.0 (January 2026)  
**Author**: Evyatar  
**Last Updated**: 2026-01-17
