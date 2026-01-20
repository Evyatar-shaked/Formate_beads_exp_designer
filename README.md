# Formate Beads Experiment Designer

A comprehensive tool for designing and optimizing bacterial culture experiments using slow-release formate beads. This project combines biological modeling (Monod kinetics), Model Predictive Control (MPC), and practical laboratory constraints to help researchers plan experiments efficiently.

**Two interfaces available:**
- 🖥️ **Streamlit GUI** - Point-and-click web interface (no coding required)
- 📓 **Jupyter Notebook** - Full transparency and customization

---

## 🧬 Scientific Background

### The Experimental Challenge

Growing bacterial cultures with formate as the energy source presents unique challenges:

1. **Formate Toxicity**: High concentrations can inhibit bacterial growth
2. **Alkalinization**: Formate consumption depletes protons from the medium, increasing pH and disrupting membrane permeability.
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

### 1. 🚀 Quick Start: Streamlit GUI (Recommended for Most Users)

**Best for**: Quick experiment design, non-programmers, presentation-ready outputs

#### Installation & Launch:

```bash
# first clone this repo
# Install dependencies
pip install -r requirements_gui.txt

# Launch the GUI
streamlit run formate_beads_gui.py
```

The interface opens automatically in your browser at `http://localhost:8501`

#### GUI Features:
- ✅ **No coding required** - Pure point-and-click interface
- ✅ **Real-time visualization** - Adjust parameters, see results instantly
- ✅ **Bead Release Profile Plots** - Visual reference for M07 and M03 characteristics
- ✅ **8 Interactive plots** - Complete experiment dynamics
- ✅ **Combined intervention schedule** - Beads + HCl in single CSV (minimizes lab visits)
- ✅ **Export ready** - Download complete protocol as CSV
- ✅ **Professional outputs** - Publication-quality figures

#### Using the GUI:

**Step 1: Configure Parameters (Left Sidebar)**
- **Culture volume** (L): Your bioreactor or flask volume (0.05-10 L)
- **Initial OD**: Starting bacterial density (0.01-0.1 typical)
- **Target concentration** (mmol/L): Desired formate level (20-50 typical)
- **Lower threshold** (80-99%): When to add beads (95% = tight control)
- **Intervention interval** (0.1-7.0 days): How often you'll check (1.0 = daily)
- **Experiment days**: Total duration (3-14 days typical)
- **Simulation dt**: Time resolution (0.01 recommended, 0.001 for high precision)

**Step 2: Run Simulation**
- Click **"🚀 Run Simulation"** button
- Wait 1-5 seconds for calculations
- See summary metrics at top (avg substrate, total beads, final OD, HCl needed)

**Step 3: Navigate Results (3 Tabs)**

**📈 Plots Tab:**
- **Bead Release Profiles** - M07 (fast initial) and M03 (sustained) release curves
- **Substrate Concentration** - With target and action threshold lines
- **Bacterial Growth** - OD trajectory over time
- **Bead Addition Schedule** - When and how many beads to add
- **Release vs Consumption** - Real-time balance of supply and demand
- **Cumulative Consumption & HCl** - Total formate used and pH control needs
- **HCl Addition Rate** - When to add HCl for pH control

**📋 Intervention Schedule Tab:**
- **Combined table** with: Time (days), M07 Beads, M03 Beads, Total Beads, HCl (mmol), HCl (mg)
- **Summary metrics**: Total interventions, total beads (M07/M03), total HCl
- **Practical HCl volumes**: For 12M, 6M, and 1M stock solutions
- **Single CSV download**: Complete protocol for the lab bench

**📊 Statistics Tab:**
- **Substrate control performance**: Average, range, standard deviation
- **Bacterial growth**: Initial/final OD, growth factor, max growth rate
- **Bead usage**: Total beads, addition events, max release rate
- **Mass balance verification**: Checks if consumed substrate matches OD increase

**Step 4: Export & Execute**
- Download intervention schedule CSV
- Print or save for lab notebook
- Follow schedule: At each time point, add both beads AND HCl together
- One lab visit per row = minimal work!


### 2. 📓 Advanced: Jupyter Notebook (For Detailed Analysis)

**Best for**: Understanding the algorithm, parameter exploration, customization, research

#### Launch:

```bash
# clone the repo if you didnt before :) 
# Option 1: Jupyter Notebook
jupyter notebook formate_beads_notebook.ipynb

# Option 2: VS Code (with Jupyter extension)
# Just open the .ipynb file
```

#### Notebook Features:
- ✅ **Full transparency** - See every calculation step
- ✅ **Extensive documentation** - Comments explain the science
- ✅ **Modifiable** - Easy to customize algorithms
- ✅ **Educational** - Learn how MPC works
- ✅ **Advanced analysis** - Mass balance checks, sanity tests

#### Notebook Structure:

**Section 1: Configuration (Cell 1)**
```python
# Edit all parameters here
VOLUME = 0.1  # L
INITIAL_OD = 0.02
TARGET_CONCENTRATION = 30  # mmol/L
LOWER_THRESHOLD = 0.95  # 95% of target
INTERVENTION_INTERVAL = 1.0  # days
EXPERIMENT_DAYS = 7
```

**Section 2: Bead Release Profiles (Cells 2-3)**
- Defines M07 and M03 empirical release data
- Corrects for linear interpolation
- Visualizes release curves with area under curve

**Section 3: Monod Kinetics (Cell 4)**
- Implements bacterial growth equations
- Growth rate: μ = μ_max × [S] / (K_s + [S])
- Consumption rate: consumption = μ × OD / Y_xs

**Section 4: Calculator Class (Cell 5)**
- Complete MPC algorithm implementation
- Bead scheduling logic
- Intervention timing control

**Section 5: Run Simulation (Cell 6)**
- Execute with your parameters
- Shows progress timeline
- Generates results dictionary

**Section 6: Comprehensive Plots (Cell 7)**
- All 6 plots (same as GUI)
- Customizable matplotlib figures
- Publication-quality outputs

**Section 7: Bead Schedule (Cell 8)**
- Formatted table of interventions
- HCl requirements
- Practical volumes for stock solutions

**Section 8: Statistics (Cell 9)**
- Performance metrics
- Mass balance verification
- Detailed analysis

**To use**: Just run all cells (Cell → Run All) or step through one by one.

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

### Quick Install:

```bash
# Install all dependencies
pip install -r requirements_gui.txt

# This installs: streamlit, numpy, matplotlib, pandas, scipy
```

### For Notebook Only:

```bash
# If you only want the notebook
pip install jupyter notebook numpy matplotlib pandas scipy

# Or use VS Code with Jupyter extension (recommended)
```

### Verify Installation:

```bash
# Check Streamlit
streamlit --version

# Check Jupyter (if using notebook)
jupyter --version
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

## 🔧 Troubleshooting

### GUI Issues

**GUI won't start:**
```bash
# Check Streamlit installation
streamlit --version

# Reinstall if needed
pip install streamlit --upgrade

# Try running in a new terminal
```

**Plots not showing:**
- Refresh the page (F5)
- Check browser console for errors
- Try a different browser (Chrome/Firefox work best)
- Clear browser cache

**Simulation is slow:**
- Increase dt from 0.001 to 0.01 (faster, still accurate)
- Reduce experiment duration
- Close other browser tabs

### Notebook Issues

**Kernel errors:**
```bash
# Restart kernel: Kernel → Restart
# Or reinstall packages in the notebook environment
```

**Plots don't display:**
- Make sure you ran the import cells
- Try `%matplotlib inline` at the top
- Restart kernel and run all cells

### General Issues

**Simulation errors:**
- Check that initial OD > 0
- Ensure target concentration > 0
- Verify intervention_interval ≤ experiment_days
- Make sure dt < intervention_interval

**Unrealistic results:**
- Lower threshold too low? Try 85-95%
- Intervention interval too long? Try 1.0 day
- Check Monod parameters match your bacterial strain
- Verify bead release profiles are correct

**Import errors:**
```bash
# Make sure all packages are installed
pip install streamlit numpy matplotlib pandas scipy jupyter

# Check Python version (3.8+ required)
python --version
```
---

## 📁 Project Files

- `formate_beads_gui.py` - Streamlit web interface (main GUI application)
- `formate_beads_notebook.ipynb` - Jupyter notebook (detailed analysis)
- `test_formate_beads.py` - Comprehensive test suite for validation
- `requirements_gui.txt` - Python dependencies
- `README.md` - This comprehensive guide
- `README_GUI.md` - GUI-specific documentation (deprecated, now merged here)

---

## 🧪 Testing

A comprehensive test suite is provided to validate the functionality of the module.

### Running Tests:

```bash
# Option 1: Using pytest (recommended)
pip install pytest
pytest test_formate_beads.py -v

# Option 2: Direct execution
python test_formate_beads.py
```

### Test Coverage:

The test suite includes:
- **Bead Release Profiles**: Validates release data integrity and profiles
- **Bead Class**: Tests individual bead behavior and release calculations
- **BeadManager**: Tests multi-bead management and total release
- **Monod Kinetics**: Validates growth and consumption rate calculations
- **Calculator**: Tests the main MPC algorithm and bead scheduling
- **Intervention Intervals**: Tests different intervention timing scenarios
- **Edge Cases**: Validates boundary conditions (small volumes, high OD, etc.)
- **Mass Balance**: Verifies conservation of mass (yield coefficient consistency)

### Example Output:

```
test_bead_release_before_addition ... ok
test_calculate_bead_schedule_basic ... ok
test_growth_rate_at_half_saturation ... ok
test_mass_balance_verification ... ok
...

======================================================================
TEST SUMMARY
======================================================================
Tests run: 45
Successes: 45
Failures: 0
Errors: 0

✅ ALL TESTS PASSED!
```
---

- **Author**: Evyatar Shaked
- **Last Updated**: 2026-01-20  
- **Interfaces**: Streamlit GUI + Jupyter Notebook
- **This module was built as the final project for the Basic Programming (Python) course at the Weizmann Institute of Science (WIS)**

