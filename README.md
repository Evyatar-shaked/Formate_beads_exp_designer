# Formate Beads Experiment Design - Integrated Model

This repository contains a computational model designed to automate the scheduling and dosage of formate-releasing beads (M07 and M03) in bacterial culture experiments. The model ensures a constant substrate concentration by balancing bead release kinetics with bacterial consumption rates modeled via Monod kinetics.

## 🚀 Overview
The primary goal of this notebook is to calculate **how many beads to add each day** to maintain a target formate concentration. It solves the mismatch between the non-linear release rates of polymer beads and the exponential growth demand of bacteria.

## 🛠️ Code Structure & Classes

The notebook is modularized into configuration, core logic classes, and simulation loops.

### 1. Configuration (`Experiment Configuration`)
A centralized section defines all physical and biological constants. No code changes are needed elsewhere if these parameters change.
* **Bead Properties:** Formate content (mg) and molecular weights.
* **Empirical Data:** Daily release rates for M07 (larger content) and M03.
* **Biological Parameters:** $\mu_{max}$, $K_s$, and Yield coefficients ($Y_{xs}$).

### 2. Core Classes

#### `Bead`
Represents a single physical bead added to the culture.
* **Attributes:** Bead type (M07/M03), `day_added`, and `current_age`.
* **Key Method:** `get_release_rate(current_time)`
    * Returns the specific mmol/day release rate for that bead based on its specific age using linear interpolation of the empirical data.

#### `MonodKinetics`
Encapsulates the biological constraints of the experiment.
* **Logic:** Implements the Monod equation: $\mu = \mu_{max} \frac{S}{K_s + S}$
* **Key Method:** `calculate_rates(OD, S)`
    * Computes the instantaneous growth rate and substrate consumption rate ($d(Substrate)/dt$) based on current cell density and formate concentration.

#### `ExperimentManager`
The "Controller" class that links bead supply with bacterial demand.
* **Role:** Manages the list of active `Bead` objects.
* **Key Method:** `get_total_release()`
    * Sums the release rates of *all* active beads in the culture at the current time step.
* **Key Method:** `update_beads()`
    * Ages existing beads and handles the addition of new beads to the virtual culture.

### 3. Helper Functions
* `correct_for_linear_interpolation()`: A pre-processing step that adjusts empirical daily measurements to ensure the area under the curve (integral) in the simulation matches the physical total mass of formate in the real beads.

## 📈 Workflow Logic

1.  **Initialization:** Set target concentration (e.g., 30mM) and initial OD.
2.  **Daily Loop:**
    * Calculate bacterial demand for the next 24 hours based on predicted growth.
    * Calculate current supply from existing ("old") beads.
    * **Decision Algorithm:** Determine how many *new* beads (M07 or M03) are required to fill the deficit.
3.  **Integration:** Solve the system of ODEs for the 24-hour period using `scipy.integrate.odeint`.
4.  **Update:** Record state, age beads, and repeat for the next day.

## 🔮 Future Improvements
* **Refined Substrate Control:** Optimization of the bead addition algorithm to minimize fluctuations and achieve a tighter, more stable constant substrate concentration.
* **Manual Mode (Custom User Choice):** Implementation of a manual override feature, allowing researchers to input specific bead counts manually to simulate "what-if" scenarios, rather than relying solely on the automated optimizer.
* **GUI Implementation:** Development of a Graphical User Interface (GUI) to allow non-coders to easily configure parameters (like volume or target concentration) and generate schedules without interacting directly with the Jupyter Notebook code.

## 📊 Outputs
* **Daily Schedule:** A printed log specifying exactly how many beads to add at $t=0, 24h, 48h$, etc.
* **Plots:**
    * *Substrate vs. Target:* Visual verification that concentration stays within tolerance.
    * *Bacterial Growth:* Predicted OD curve.
    * *Release Profile:* Aggregate formate release over time.

## 📦 Requirements
* Python 3.x
* `numpy` (Array manipulation)
* `scipy` (ODE integration)
* `matplotlib` (Plotting)

---
*Based on empirical release kinetics of M07/M03 beads and standard Monod growth modeling.*
