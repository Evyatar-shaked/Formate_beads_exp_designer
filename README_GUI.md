# Formate Beads Experiment Designer - GUI Application

A modern, interactive web-based GUI for designing formate bead experiments with bacterial cultures. Built with Streamlit for a professional, user-friendly interface.

## Features

### 🎯 Interactive Configuration
- **Adjustable Parameters:**
  - Culture volume (L)
  - Initial bacterial OD (optical density)
  - Target substrate concentration (mmol/L)
  - **Lower bound threshold (80-99%)** - When to add beads (default: 95% = add when 5% below target)
  - Experiment duration (days)
  - Simulation time step
  - **Intervention interval (0.1-7.0 days)** - How often to check and add beads/HCl (default: 1.0 day)
  - **Note:** These two parameters work together - lower thresholds pair well with less frequent interventions

### 🔒 Hardcoded Parameters (Rarely Changed)
- Bead properties (M07/M03 release profiles)
- Monod kinetics (μ_max, K_s, Y_xs)
- Formate molecular weight

### 📊 Comprehensive Outputs
- **6 Interactive Plots:**
  1. Substrate concentration with control bounds
  2. Bacterial growth trajectory
  3. Bead addition schedule
  4. Release vs consumption rates
  5. Cumulative consumption & HCl needs
  6. HCl addition rate over time

- **Data Tables:**
  - Bead addition schedule (downloadable CSV)
  - HCl requirements (downloadable CSV)
  - Detailed statistics

- **Key Metrics:**
  - Average substrate achieved
  - Total beads needed (M07 + M03)
  - Final bacterial OD
  - Total HCl required (mmol, mg, practical volumes)

## Installation

### 1. Install Dependencies

```bash
# Using pip
pip install -r requirements_gui.txt

# Or install individually
pip install streamlit numpy matplotlib pandas
```

### 2. Verify Installation

```bash
streamlit --version
```

## Running the Application

### Quick Start

```bash
streamlit run formate_beads_gui.py
```

The application will automatically open in your default web browser at `http://localhost:8501`

### Command Line Options

```bash
# Specify a different port
streamlit run formate_beads_gui.py --server.port 8080

# Run in headless mode (no browser auto-open)
streamlit run formate_beads_gui.py --server.headless true

# Enable wide mode by default
streamlit run formate_beads_gui.py --theme.base light
```

## Usage Guide

### Step 1: Configure Experiment
1. Use the **left sidebar** to set your experiment parameters
2. Adjust culture volume, initial OD, target concentration
3. Set **lower bound threshold** - controls when to add beads (80-99%, e.g., 95% = add when 5% below target)
4. Set experiment duration and simulation precision (dt)
5. **Configure intervention interval** - how frequently you can add beads/HCl (e.g., 0.5 = twice daily, 1.0 = daily, 2.0 = every 2 days)

### Step 2: Run Simulation
1. Click the **"🚀 Run Simulation"** button
2. Wait for the calculation to complete (typically 1-5 seconds)

### Step 3: Analyze Results
Navigate through the tabs:
- **📈 Plots:** Visual representation of all experiment dynamics
- **📋 Bead Schedule:** Day-by-day bead addition requirements
- **🧪 HCl Requirements:** pH control needs with practical volumes
- **📊 Statistics:** Detailed performance metrics and mass balance

### Step 4: Export Data
- Download bead schedule as CSV
- Download HCl requirements as CSV
- Save plots (right-click → Save Image)

## Understanding the Algorithm

### MPC (Model Predictive Control) Features
- **Multi-step lookahead:** Predicts 1.5-2 days ahead
- **OD-aware scaling:** Initial beads scale with (OD/0.02)^1.2 for high starting densities
- **Initial bead mix:** 70% M03 / 30% M07 for balanced fast+sustained release
- **Configurable threshold:** User controls when to add beads (80-99%, default: 95%)
- **Adaptive buffers:** 0.05× (days 0-2), 0.10× (days 3-4), 0.2× (days 5+), scaled by (OD/0.02)^0.75
- **Hard cap:** Maximum buffer factor of 0.5 to prevent day 5+ explosion
- **Balanced M07/M03:** Dynamic ratios (60/40 urgent, 50/50 moderate, 45/55 normal)
- **Flexible intervention:** Configurable check intervals (default: daily)
- **Smart control:** No action when substrate is high - bacteria naturally consume excess

### Control Strategy
- **Action threshold:** User-configurable (80-99%, default: 95% of target)
  - 95% = add beads when 5% below target (tight control, early intervention)
  - 90% = add beads when 10% below target (balanced approach)
  - 85% = add beads when 15% below target (moderate intervention)
  - 80% = add beads when 20% below target (relaxed control)
  - Higher values = earlier intervention, tighter control
  - **Interplay with intervention interval:** Lower thresholds work well with less frequent interventions (e.g., 80% threshold + 2-day interval)
- **Hard limits (visualization only):**
  - Lower limit: 88% of target (-12%)
  - Upper limit: 108% of target (+8%)
- **Philosophy:** Act early to prevent depletion, wait when excess (bacteria will consume)

### Bead Types
- **M07:** High initial release (0.444 mmol/day peak), good for quick substrate boost
- **M03:** Sustained release (0.28 mmol/day sustained), good for maintenance
- **Initial strategy:** 70% M03 / 30% M07 for balanced startup
- **Later additions:** Dynamic ratios based on urgency (60/40 to 45/55)

## Typical Use Cases

### Intervention Interval Guidelines
The **intervention interval** determines how often you can physically add beads and HCl:

**Daily Intervention (1.0 day)** - Standard protocol
```
Suitable for: Regular lab operations
Commitment: Check once per day
Control: Good substrate stability
Recommended threshold: 90-95% (tight control)
```

**Twice Daily (0.5 days)** - Enhanced control
```
Suitable for: Critical experiments, high-growth cultures
Commitment: Check every 12 hours
Control: Excellent substrate stability, minimal oscillations
Benefit: Smoother substrate concentration profile
Recommended threshold: 85-95% (can afford lower with frequent checks)
```

**Every 2 Days (2.0 days)** - Minimal intervention
```
Suitable for: Weekend/holiday coverage, automated systems
Commitment: Check every other day
Control: Acceptable with some oscillations
Note: Requires more beads per addition
Recommended threshold: 70-80% (allow larger deviations, add more when needed)
```

**Flexible intervals (0.1-7.0 days)** - Custom protocols
```
Match your lab availability and experiment requirements
Shorter intervals → Higher thresholds (95%+) = tight control, more work
Longer intervals → Lower thresholds (60-80%) = fewer checks, larger additions
Key principle: Let it drop more if you can't check often
```

### Small Scale Laboratory (50-500 mL)
```
Volume: 0.05-0.5 L
Initial OD: 0.01-0.05
Target: 20-40 mmol/L
Duration: 3-7 days
```

### Bioreactor Scale (1-10 L)
```
Volume: 1-10 L
Initial OD: 0.02-0.1
Target: 30-50 mmol/L
Duration: 7-14 days
```

## Troubleshooting

### Application won't start
```bash
# Check if streamlit is installed
pip show streamlit

# Reinstall dependencies
pip install -r requirements_gui.txt --upgrade
```

### Simulation is slow
- Increase dt (time step) from 0.001 to 0.01
- Reduce experiment duration
- Close other browser tabs

### Plots not showing
- Check matplotlib backend
- Try refreshing the page (F5)
- Clear browser cache

## Advantages over Jupyter Notebook

✅ **No coding required** - Pure GUI interface  
✅ **Instant visualization** - Real-time updates  
✅ **Export ready** - Download CSVs with one click  
✅ **Professional look** - Clean, modern interface  
✅ **Easy sharing** - Can be deployed to cloud  
✅ **Interactive** - Adjust parameters and re-run instantly  

## Deployment (Optional)

### Deploy to Streamlit Cloud (Free)
1. Push code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repository
4. Deploy!

### Local Network Access
```bash
# Run with network access
streamlit run formate_beads_gui.py --server.address 0.0.0.0
```
Then access from other devices: `http://YOUR_IP:8501`

## Technical Details

### Hardcoded Constants
Located in the `formate_beads_gui.py` file:
- Lines 25-35: Bead properties and formate MW
- Lines 37-48: Empirical bead release profiles
- Lines 50-53: Monod kinetics parameters

To modify these, edit the Python file directly.

### Algorithm
The simulation uses the same enhanced MPC algorithm from the notebook:
- Monod kinetics for bacterial growth
- Linear interpolation for bead release
- Predictive bead scheduling
- Automatic HCl calculation

## Support

For issues or questions:
1. Check this README
2. Review the notebook documentation
3. Inspect console output for error messages

## Version History

- **v1.0** (2026-01-16): Initial GUI release
  - Streamlit-based interface
  - All notebook features ported
  - CSV export functionality
  - MPC-enhanced algorithm

## License

Same as the parent project.
