"""
Formate Beads Experiment Designer - Interactive GUI Application
===============================================================
A Streamlit-based GUI for designing formate bead experiments with bacterial cultures.

Run with: streamlit run formate_beads_gui.py
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict

# Configure page
st.set_page_config(
    page_title="Formate Beads Experiment Designer",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# HARDCODED CONSTANTS (rarely changed)
# ============================================================================

# Formate Properties
FORMATE_MW = 68  # mg/mmol

# Bead Physical Properties
M07_FORMATE_CONTENT = 50.252  # mg formate per M07 bead
M03_FORMATE_CONTENT = 35.1356  # mg formate per M03 bead

# Empirical Bead Release Data (mmol/day)
M07_EMPIRICAL = {
    1: 0.444, 2: 0.135, 3: 0.05, 4: 0.04,
    5: 0.03, 6: 0.02, 7: 0.02
}

M03_EMPIRICAL = {
    1: 0.279909614, 2: 0.085731303, 3: 0.068230318,
    4: 0.033414337, 5: 0.02, 6: 0.02, 7: 0.01
}

# Monod Kinetics Parameters (bacterial strain specific)
MU_MAX = 1.0  # Maximum growth rate (1/day)
K_S = 20.0    # Half-saturation constant (mmol/L)
Y_XS = 0.0067 # Yield coefficient (OD per mmol/L substrate)

# ============================================================================
# CORE CLASSES (from notebook)
# ============================================================================

def correct_for_linear_interpolation(empirical_profile):
    """Adjust release rates for linear interpolation."""
    days = sorted(empirical_profile.keys())
    integral = 0.0
    for i in range(len(days) - 1):
        day1, day2 = days[i], days[i + 1]
        rate1, rate2 = empirical_profile[day1], empirical_profile[day2]
        integral += 0.5 * (rate1 + rate2) * (day2 - day1)
    integral += 0.5 * empirical_profile[days[-1]] * 1
    discrete_sum = sum(empirical_profile.values())
    scaling_factor = discrete_sum / integral if integral > 0 else 1.0
    return {day: rate * scaling_factor for day, rate in empirical_profile.items()}, scaling_factor

# Apply correction
M07_BEAD_RELEASE, _ = correct_for_linear_interpolation(M07_EMPIRICAL)
M03_BEAD_RELEASE, _ = correct_for_linear_interpolation(M03_EMPIRICAL)


class Bead:
    """Represents a single formate-releasing bead."""
    def __init__(self, bead_type, day_added=0):
        self.bead_type = bead_type
        self.day_added = day_added
        self.release_profile = M07_BEAD_RELEASE if bead_type == 'M07' else M03_BEAD_RELEASE
    
    def get_release_rate(self, current_time):
        """Get release rate at specific time using linear interpolation."""
        bead_age = current_time - self.day_added
        if bead_age <= 0:
            return 0.0
        max_day = max(self.release_profile.keys())
        if bead_age > max_day:
            return 0.0
        
        day_floor = int(np.floor(bead_age))
        day_ceil = int(np.ceil(bead_age))
        
        if day_floor == day_ceil:
            return self.release_profile.get(day_floor, 0.0)
        
        if day_floor < 1:
            day_floor = 1
        if day_ceil > max_day:
            return 0.0
        
        rate_floor = self.release_profile.get(day_floor, 0.0)
        rate_ceil = self.release_profile.get(day_ceil, 0.0)
        fraction = bead_age - day_floor
        return rate_floor + fraction * (rate_ceil - rate_floor)


class ExperimentManager:
    """Manages all beads in the experiment."""
    def __init__(self):
        self.beads = []
    
    def add_bead(self, bead_type, day_added):
        self.beads.append(Bead(bead_type, day_added))
    
    def get_total_release(self, current_time):
        return sum(bead.get_release_rate(current_time) for bead in self.beads)


class MonodKinetics:
    """Models bacterial growth and substrate consumption."""
    def __init__(self, mu_max, K_s, Y_xs):
        self.mu_max = mu_max
        self.K_s = K_s
        self.Y_xs = Y_xs
    
    def growth_rate(self, S):
        if S <= 0:
            return 0
        return self.mu_max * S / (self.K_s + S)
    
    def consumption_rate(self, S, X):
        mu = self.growth_rate(S)
        return mu * X / self.Y_xs


class ConstantSubstrateCalculator:
    """Enhanced MPC-based substrate controller."""
    def __init__(self, volume, monod_params, target_concentration):
        self.volume = volume
        self.monod = MonodKinetics(**monod_params)
        self.target_concentration = target_concentration
    
    def calculate_bead_schedule(self, initial_od, experiment_days=7, dt=0.01, intervention_interval=1.0, lower_threshold=0.95):
        """Enhanced MPC version with configurable lower threshold."""
        manager = ExperimentManager()
        
        # Initial beads - scale with volume and initial OD
        # Higher initial OD = more bacteria = much higher consumption rate from day 0
        # Calculate based on actual consumption rate, not just static target
        
        # First, estimate initial consumption rate with higher OD
        initial_substrate = self.target_concentration  # Assume we want to reach target
        initial_consumption_rate = self.monod.consumption_rate(initial_substrate, initial_od)
        
        # How much substrate do we need for first ~2 days at this consumption rate?
        # Scale with a "boost" for higher OD since consumption accelerates with growth
        days_to_cover = 2.0  # Cover first 2 days more aggressively
        od_boost = max(1.0, (initial_od / 0.02) ** 1.2)  # Superlinear scaling for high OD
        total_formate_needed = initial_consumption_rate * self.volume * days_to_cover * od_boost
        
        # Add base minimum (3% of target × volume)
        base_minimum = self.target_concentration * self.volume * 0.03
        total_formate_needed = max(total_formate_needed, base_minimum)
        
        m07_total_release = sum(M07_BEAD_RELEASE.values())
        m03_total_release = sum(M03_BEAD_RELEASE.values())
        
        # Split 70% M03, 30% M07 by count for same total formate but mixed release
        # Weighted average release per bead
        avg_release_per_bead = 0.7 * m03_total_release + 0.3 * m07_total_release
        total_beads = max(3, int(np.ceil(total_formate_needed / avg_release_per_bead)))
        
        initial_m07 = int(np.round(total_beads * 0.3))
        initial_m03 = int(np.round(total_beads * 0.7))
        
        for _ in range(initial_m07):
            manager.add_bead('M07', 0)
        for _ in range(initial_m03):
            manager.add_bead('M03', 0)
        
        # Arrays
        times = np.arange(0, experiment_days + dt, dt)
        substrate = np.zeros_like(times)
        od = np.zeros_like(times)
        release_rates = np.zeros_like(times)
        cumulative_consumed = np.zeros_like(times)
        
        substrate[0] = 0.0
        od[0] = initial_od
        bead_schedule = {0: {'M07': initial_m07, 'M03': initial_m03, 'HCl_mmol': 0}}
        last_intervention_consumption = 0.0  # Track consumption for HCl calculation
        
        # Control - configurable lower threshold for intervention
        action_threshold = self.target_concentration * lower_threshold  # User-configurable intervention point
        
        for i in range(1, len(times)):
            current_time = times[i]
            current_day = int(current_time)
            
            release_rate = manager.get_total_release(current_time)
            release_rates[i] = release_rate
            
            mu = self.monod.growth_rate(substrate[i-1])
            consumption_rate = self.monod.consumption_rate(substrate[i-1], od[i-1])
            consumption_amount = consumption_rate * self.volume * dt
            cumulative_consumed[i] = cumulative_consumed[i-1] + consumption_amount
            
            dOD_dt = mu * od[i-1]
            od[i] = od[i-1] + dOD_dt * dt
            
            dS_dt = (release_rate / self.volume) - consumption_rate
            substrate[i] = max(0, substrate[i-1] + dS_dt * dt)
            
            # Check at intervention intervals for bead additions
            intervention_time = np.round(current_time / intervention_interval) * intervention_interval
            if intervention_time > 0 and abs(current_time - intervention_time) < dt/2:
                current_consumption = self.monod.consumption_rate(substrate[i], od[i]) * self.volume
                
                # Add beads if below action threshold
                if substrate[i] < action_threshold:
                    deficit = (self.target_concentration - substrate[i]) * self.volume
                    days_remaining = experiment_days - current_day
                    
                    # Gentle MPC with very conservative buffer factors
                    # Use average substrate for growth prediction (not current low substrate)
                    avg_substrate_for_prediction = (substrate[i] + self.target_concentration) / 2
                    mu_current = self.monod.growth_rate(avg_substrate_for_prediction)
                    
                    # Scale buffer with current OD - higher OD needs more aggressive control
                    # because consumption accelerates faster
                    od_buffer_scale = max(1.0, (od[i] / 0.02) ** 0.75)  # Middle ground scaling (between sqrt and linear)
                    
                    if current_day <= 2:
                        projection_window = min(1.5, days_remaining)
                        base_buffer = 0.05
                    elif current_day <= 4:
                        projection_window = min(2.0, days_remaining)
                        base_buffer = 0.10
                    else:
                        projection_window = min(2.0, days_remaining)
                        base_buffer = 0.2  # Reduced from 0.5 to prevent explosion
                    
                    # Apply OD scaling and cap at 0.5 maximum
                    buffer_factor = min(0.5, base_buffer * od_buffer_scale)
                    
                    # Project consumption with lookahead
                    time_points = np.linspace(0, projection_window, 20)
                    projected_ods = od[i] * np.exp(mu_current * time_points)
                    projected_consumptions = []
                    for idx, od_proj in enumerate(projected_ods):
                        time_frac = time_points[idx] / projection_window if projection_window > 0 else 0
                        est_substrate = substrate[i] + (self.target_concentration - substrate[i]) * time_frac
                        est_substrate = max(0, min(est_substrate, self.target_concentration * 1.1))
                        projected_consumptions.append(
                            self.monod.consumption_rate(est_substrate, od_proj)
                        )
                    
                    avg_consumption = np.mean(projected_consumptions)
                    daily_consumption = avg_consumption * self.volume
                    
                    # Reduce buffer even more if close to target
                    if abs(deficit) < self.target_concentration * self.volume * 0.15:
                        buffer_factor *= 0.7
                    
                    total_needed = deficit + daily_consumption * projection_window * buffer_factor
                    
                    # Account for existing beads very strongly
                    upcoming_release_1day = manager.get_total_release(current_time + 1.0)
                    upcoming_release_2day = manager.get_total_release(current_time + 2.0)
                    avg_upcoming = (upcoming_release_1day + upcoming_release_2day) / 2
                    existing_supply = avg_upcoming * projection_window
                    total_needed = max(0, total_needed - existing_supply * 0.95)
                    
                    # Balance M07/M03 for similar overall utilization
                    # M07: Fast response (high peak), M03: Sustained release
                    deficit_ratio = deficit / (daily_consumption + 1e-6)
                    
                    if deficit_ratio > 0.5 or days_remaining < 2:
                        # Urgent: Need fast response
                        m07_weight, m03_weight = 0.6, 0.4
                    elif deficit_ratio > 0.3:
                        # Moderate: Balanced approach
                        m07_weight, m03_weight = 0.5, 0.5
                    else:
                        # Normal: Slightly favor sustained release
                        m07_weight, m03_weight = 0.45, 0.55
                    
                    m07_share = total_needed * m07_weight
                    m03_share = total_needed * m03_weight
                    
                    m07_needed = int(np.ceil(m07_share / m07_total_release)) if m07_share > 0.05 else 0
                    m03_needed = int(np.ceil(m03_share / m03_total_release)) if m03_share > 0.05 else 0
                    
                    # Ensure at least 1 bead if significant deficit
                    if total_needed > 0.15 and m07_needed == 0 and m03_needed == 0:
                        m03_needed = 1
                    
                    for _ in range(m07_needed):
                        manager.add_bead('M07', intervention_time)
                    for _ in range(m03_needed):
                        manager.add_bead('M03', intervention_time)
                    
                    # Only add beads and HCl together when intervention is needed
                    if m07_needed > 0 or m03_needed > 0:
                        # Calculate HCl needed since last intervention (or start)
                        hcl_needed = cumulative_consumed[i] - last_intervention_consumption
                        last_intervention_consumption = cumulative_consumed[i]
                        
                        if intervention_time in bead_schedule:
                            bead_schedule[intervention_time]['M07'] += m07_needed
                            bead_schedule[intervention_time]['M03'] += m03_needed
                            bead_schedule[intervention_time]['HCl_mmol'] += hcl_needed
                        else:
                            bead_schedule[intervention_time] = {'M07': m07_needed, 'M03': m03_needed, 'HCl_mmol': hcl_needed}
        
        # Calculate HCl aligned with intervention intervals (not continuous)
        consumption_rates = np.zeros_like(times)
        hcl_needed_daily = np.zeros_like(times)
        for i in range(len(times)):
            consumption_rates[i] = self.monod.consumption_rate(substrate[i], od[i]) * self.volume
            # Only show HCl addition at intervention times
            current_time = times[i]
            intervention_time = np.round(current_time / intervention_interval) * intervention_interval
            if abs(current_time - intervention_time) < dt/2 and intervention_time in bead_schedule:
                # Distribute the HCl for this intervention over the dt interval for visualization
                hcl_needed_daily[i] = bead_schedule[intervention_time]['HCl_mmol'] / dt
            else:
                hcl_needed_daily[i] = 0
        hcl_needed_cumulative = cumulative_consumed.copy()
        
        return {
            'times': times,
            'substrate': substrate,
            'od': od,
            'bead_schedule': bead_schedule,
            'release_rates': release_rates,
            'consumption_rates': consumption_rates,
            'cumulative_consumed': cumulative_consumed,
            'hcl_needed_daily': hcl_needed_daily,
            'hcl_needed_cumulative': hcl_needed_cumulative
        }


# ============================================================================
# STREAMLIT GUI
# ============================================================================

def main():
    st.title("🧪 Formate Beads Experiment Designer")
    st.markdown("### Interactive MPC-Enhanced Substrate Control System")
    
    # Sidebar for configuration
    st.sidebar.header("⚙️ Experiment Configuration")
    
    st.sidebar.subheader("📊 Culture Parameters")
    volume = st.sidebar.number_input(
        "Culture Volume (L)",
        min_value=0.01,
        max_value=100.0,
        value=0.1,
        step=0.1,
        help="Total volume of bacterial culture"
    )
    
    initial_od = st.sidebar.number_input(
        "Initial OD (Optical Density)",
        min_value=0.001,
        max_value=5.0,
        value=0.02,
        step=0.01,
        format="%.3f",
        help="Starting bacterial density at inoculation"
    )
    
    st.sidebar.subheader("🎯 Target Control")
    target_concentration = st.sidebar.number_input(
        "Target Substrate Concentration (mmol/L)",
        min_value=1.0,
        max_value=100.0,
        value=30.0,
        step=1.0,
        help="Desired formate concentration to maintain"
    )
    
    lower_threshold = st.sidebar.slider(
        "Lower Bound Threshold (%)",
        min_value=80,
        max_value=99,
        value=95,
        help="The minimum substrate concentration (as % of target) that triggers bead addition. For example, at 95%, beads are added when substrate drops to 95% of the target concentration (28.5 mmol/L for 30 mmol/L target)."
    ) / 100.0
    
    st.sidebar.subheader("⏱️ Simulation Parameters")
    experiment_days = st.sidebar.number_input(
        "Experiment Duration (days)",
        min_value=1,
        max_value=30,
        value=7,
        step=1,
        help="Total duration of the experiment"
    )
    
    dt = st.sidebar.select_slider(
        "Simulation Time Step (days)",
        options=[0.001, 0.005, 0.01, 0.02, 0.05],
        value=0.001,
        help="Smaller = more accurate but slower"
    )
    
    intervention_interval = st.sidebar.number_input(
        "Intervention Interval (days)",
        min_value=0.1,
        max_value=7.0,
        value=1.0,
        step=0.1,
        help="How often to check and add beads/HCl (e.g., 0.5 = twice per day, 1.0 = daily)"
    )
    
    # Display hardcoded parameters
    with st.sidebar.expander("🔒 Fixed Parameters (Hardcoded)"):
        st.write("**Monod Kinetics:**")
        st.write(f"• μ_max: {MU_MAX} 1/day")
        st.write(f"• K_s: {K_S} mmol/L")
        st.write(f"• Y_xs: {Y_XS} OD per mmol/L")
        st.write("\n**Bead Properties:**")
        st.write(f"• M07 content: {M07_FORMATE_CONTENT:.2f} mg")
        st.write(f"• M03 content: {M03_FORMATE_CONTENT:.2f} mg")
    
    # Run simulation button
    run_button = st.sidebar.button("🚀 Run Simulation", type="primary")
    
    # Main content area
    if run_button:
        with st.spinner("Running MPC-enhanced simulation..."):
            # Create calculator and run simulation
            calculator = ConstantSubstrateCalculator(
                volume=volume,
                monod_params={'mu_max': MU_MAX, 'K_s': K_S, 'Y_xs': Y_XS},
                target_concentration=target_concentration
            )
            
            results = calculator.calculate_bead_schedule(
                initial_od=initial_od,
                experiment_days=experiment_days,
                dt=dt,
                intervention_interval=intervention_interval,
                lower_threshold=lower_threshold
            )
            
            # Display success message
            st.success("✅ Simulation completed successfully!")
            
            # Summary statistics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Average Substrate",
                    f"{np.mean(results['substrate']):.2f} mmol/L",
                    delta=f"{np.mean(results['substrate']) - target_concentration:.2f}"
                )
            
            with col2:
                total_m07 = sum(b.get('M07', 0) for b in results['bead_schedule'].values())
                total_m03 = sum(b.get('M03', 0) for b in results['bead_schedule'].values())
                st.metric("Total Beads", f"{total_m07 + total_m03}", delta=f"M07:{total_m07} M03:{total_m03}")
            
            with col3:
                st.metric(
                    "Final OD",
                    f"{results['od'][-1]:.3f}",
                    delta=f"{(results['od'][-1]/initial_od):.1f}x growth"
                )
            
            with col4:
                st.metric(
                    "Total HCl Needed",
                    f"{results['hcl_needed_cumulative'][-1]:.2f} mmol",
                    delta=f"{results['hcl_needed_cumulative'][-1] * 36.46:.2f} mg"
                )
            
            # Tabs for different views
            tab1, tab2, tab3 = st.tabs(["📈 Plots", "📋 Intervention Schedule", "📊 Statistics"])
            
            with tab1:
                st.subheader("Simulation Results - Comprehensive Plots")
                
                # Bead Release Profiles (static reference plots)
                st.markdown("### 🔬 Bead Release Profiles (Reference)")
                st.info("These plots show the release characteristics of M07 and M03 beads used in the simulation.")
                
                fig_beads, axes_beads = plt.subplots(1, 2, figsize=(14, 6))
                fig_beads.suptitle('Bead Release Profiles: Linear Interpolation with Area Under Curve', 
                                   fontsize=14, fontweight='bold')
                
                def plot_bead_release_profile(ax, release_profile, empirical_profile, bead_name, 
                                             formate_content, color_fill, color_line):
                    """Plot the interpolated release rate curve with shaded area."""
                    days = sorted(release_profile.keys())
                    max_day = max(days)
                    
                    # Create high-resolution time array for smooth curve
                    time_fine = np.linspace(0, max_day + 1, 1000)
                    rates_fine = []
                    
                    for t in time_fine:
                        if t <= 0 or t > max_day:
                            rates_fine.append(0)
                        else:
                            # Linear interpolation
                            day_floor = int(np.floor(t))
                            day_ceil = int(np.ceil(t))
                            
                            if day_floor == day_ceil or day_floor < 1:
                                day_floor = max(1, day_floor)
                                rates_fine.append(release_profile.get(day_floor, 0))
                            elif day_ceil > max_day:
                                rates_fine.append(0)
                            else:
                                rate_floor = release_profile.get(day_floor, 0)
                                rate_ceil = release_profile.get(day_ceil, 0)
                                fraction = t - day_floor
                                rates_fine.append(rate_floor + fraction * (rate_ceil - rate_floor))
                    
                    # Plot the smooth interpolated curve
                    ax.plot(time_fine, rates_fine, color=color_line, linewidth=2.5, 
                           label='Interpolated release rate', zorder=3)
                    
                    # Fill area under curve
                    ax.fill_between(time_fine, 0, rates_fine, alpha=0.3, color=color_fill, 
                                   label='Area = Total release', zorder=1)
                    
                    # Plot corrected daily values as points
                    day_values = [release_profile[d] for d in days]
                    ax.scatter(days, day_values, color=color_line, s=100, zorder=4, 
                             edgecolors='black', linewidths=1.5, label='Corrected daily rates')
                    
                    # Plot empirical values as comparison
                    empirical_values = [empirical_profile[d] for d in days]
                    ax.scatter(days, empirical_values, color='red', s=80, zorder=5, 
                             marker='x', linewidths=2, label='Empirical measurements')
                    
                    # Add vertical lines at day boundaries
                    for day in range(1, max_day + 1):
                        ax.axvline(x=day, color='gray', linestyle=':', alpha=0.4, linewidth=1)
                    
                    # Calculate integral
                    integral_mmol = 0.0
                    for i in range(len(days) - 1):
                        day1, day2 = days[i], days[i + 1]
                        rate1, rate2 = release_profile[day1], release_profile[day2]
                        integral_mmol += 0.5 * (rate1 + rate2) * (day2 - day1)
                    integral_mmol += 0.5 * release_profile[days[-1]] * 1
                    
                    integral_mg = integral_mmol * FORMATE_MW
                    empirical_sum_mmol = sum(empirical_profile.values())
                    empirical_sum_mg = empirical_sum_mmol * FORMATE_MW
                    
                    # Add text box with summary
                    textstr = f'{bead_name} Beads\n'
                    textstr += f'─────────────────\n'
                    textstr += f'Empirical total:\n'
                    textstr += f'  {empirical_sum_mmol:.4f} mmol\n'
                    textstr += f'  {empirical_sum_mg:.2f} mg\n\n'
                    textstr += f'Integral (area):\n'
                    textstr += f'  {integral_mmol:.4f} mmol\n'
                    textstr += f'  {integral_mg:.2f} mg\n\n'
                    textstr += f'Bead capacity:\n'
                    textstr += f'  {formate_content} mg\n\n'
                    textstr += f'Match: {integral_mg/empirical_sum_mg*100:.1f}%'
                    
                    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
                    ax.text(0.98, 0.97, textstr, transform=ax.transAxes, fontsize=9,
                           verticalalignment='top', horizontalalignment='right', 
                           bbox=props, family='monospace')
                    
                    ax.set_xlabel('Time (days)', fontsize=11, fontweight='bold')
                    ax.set_ylabel('Release Rate (mmol/day)', fontsize=11, fontweight='bold')
                    ax.set_title(f'{bead_name} Bead Release Profile', fontsize=12, fontweight='bold')
                    ax.legend(loc='upper left', fontsize=8)
                    ax.set_xlim(0, 8)
                    ax.set_ylim(0, max(release_profile.values()) * 1.1)
                
                # Plot M07 and M03 beads
                plot_bead_release_profile(axes_beads[0], M07_BEAD_RELEASE, M07_EMPIRICAL, 'M07', 
                                         M07_FORMATE_CONTENT, 'skyblue', 'darkblue')
                plot_bead_release_profile(axes_beads[1], M03_BEAD_RELEASE, M03_EMPIRICAL, 'M03', 
                                         M03_FORMATE_CONTENT, 'lightcoral', 'darkred')
                
                plt.tight_layout()
                st.pyplot(fig_beads)
                
                st.markdown("---")
                st.markdown("### 📊 Experiment Simulation Results")
                
                # Create comprehensive plots
                fig, axes = plt.subplots(3, 2, figsize=(14, 16))
                fig.suptitle('Formate Beads Experiment: MPC-Enhanced Substrate Control', 
                            fontsize=16, fontweight='bold')
                
                # Plot 1: Substrate Concentration
                ax1 = axes[0, 0]
                ax1.plot(results['times'], results['substrate'], 'b-', linewidth=2, label='Substrate')
                ax1.axhline(y=target_concentration, color='r', linestyle='--', linewidth=2, label='Target')
                action_line = target_concentration * lower_threshold
                ax1.axhline(y=action_line, color='purple', linestyle='--', linewidth=1.5, label=f'Action threshold ({int(lower_threshold*100)}%)', alpha=0.7)
                ax1.set_xlabel('Time (days)', fontsize=11)
                ax1.set_ylabel('Substrate Concentration (mmol/L)', fontsize=11)
                ax1.set_title('Substrate Concentration - MPC Enhanced Control', fontsize=12, fontweight='bold')
                ax1.legend(loc='best', fontsize=8)
                ax1.grid(True, alpha=0.3)
                
                # Plot 2: Bacterial Growth
                ax2 = axes[0, 1]
                ax2.plot(results['times'], results['od'], 'g-', linewidth=2, label='Bacterial OD')
                ax2.scatter([0], [initial_od], color='purple', s=150, zorder=5, 
                          marker='*', edgecolors='black', linewidths=1.5, label='Inoculation')
                ax2.set_xlabel('Time (days)', fontsize=11)
                ax2.set_ylabel('Optical Density (OD)', fontsize=11)
                ax2.set_title('Bacterial Growth Trajectory', fontsize=12, fontweight='bold')
                ax2.legend(loc='best', fontsize=9)
                ax2.grid(True, alpha=0.3)
                
                # Plot 3: Bead Addition Schedule
                ax3 = axes[1, 0]
                if results['bead_schedule']:
                    days = sorted(results['bead_schedule'].keys())
                    m07_counts = [results['bead_schedule'][d].get('M07', 0) for d in days]
                    m03_counts = [results['bead_schedule'][d].get('M03', 0) for d in days]
                    x = np.arange(len(days))
                    width = 0.35
                    ax3.bar(x - width/2, m07_counts, width, label='M07 beads', color='skyblue', edgecolor='black')
                    ax3.bar(x + width/2, m03_counts, width, label='M03 beads', color='lightcoral', edgecolor='black')
                    ax3.set_xlabel('Day', fontsize=11)
                    ax3.set_ylabel('Number of Beads Added', fontsize=11)
                    ax3.set_title('Bead Addition Schedule', fontsize=12, fontweight='bold')
                    ax3.set_xticks(x)
                    # Format labels based on intervention interval
                    if intervention_interval < 1.0:
                        ax3.set_xticklabels([f'{d:.1f}' for d in days])
                    else:
                        ax3.set_xticklabels([f'{d:.0f}' for d in days])
                    ax3.legend(loc='best')
                    ax3.grid(True, alpha=0.3, axis='y')
                
                # Plot 4: Release vs Consumption
                ax4 = axes[1, 1]
                release_rates_per_L = results['release_rates'] / volume
                consumption_rates_per_L = results['consumption_rates'] / volume
                ax4.plot(results['times'], release_rates_per_L, 'm-', linewidth=2.5, 
                        label='Bead Release Rate', alpha=0.8)
                ax4.plot(results['times'], consumption_rates_per_L, 'c-', linewidth=2.5, 
                        label='Bacterial Consumption Rate', alpha=0.8)
                ax4.set_xlabel('Time (days)', fontsize=11)
                ax4.set_ylabel('Rate (mmol/L/day)', fontsize=11)
                ax4.set_title('Formate Release vs Bacterial Consumption', fontsize=12, fontweight='bold')
                ax4.legend(loc='best', fontsize=9)
                ax4.grid(True, alpha=0.3)
                
                # Plot 5: Cumulative Consumption
                ax5 = axes[2, 0]
                ax5.plot(results['times'], results['cumulative_consumed'], 'b-', linewidth=2.5, 
                        label='Cumulative Formate Consumed')
                ax5.plot(results['times'], results['hcl_needed_cumulative'], 'r--', linewidth=2.5, 
                        label='Cumulative HCl Needed (1:1)', alpha=0.8)
                ax5.set_xlabel('Time (days)', fontsize=11)
                ax5.set_ylabel('Amount (mmol)', fontsize=11)
                ax5.set_title('Cumulative Formate Consumption & HCl Requirement', fontsize=12, fontweight='bold')
                ax5.legend(loc='best', fontsize=9)
                ax5.grid(True, alpha=0.3)
                
                # Plot 6: HCl Addition per Intervention
                ax6 = axes[2, 1]
                if results['bead_schedule']:
                    days = sorted(results['bead_schedule'].keys())
                    hcl_amounts = [results['bead_schedule'][d].get('HCl_mmol', 0) for d in days]
                    x = np.arange(len(days))
                    ax6.bar(x, hcl_amounts, color='red', alpha=0.7, edgecolor='darkred', linewidth=1.5)
                    ax6.set_xlabel('Intervention Time (days)', fontsize=11)
                    ax6.set_ylabel('HCl Added (mmol)', fontsize=11)
                    ax6.set_title('HCl Addition per Intervention', fontsize=12, fontweight='bold')
                    ax6.set_xticks(x)
                    # Format labels based on intervention interval
                    if intervention_interval < 1.0:
                        ax6.set_xticklabels([f'{d:.1f}' for d in days])
                    else:
                        ax6.set_xticklabels([f'{d:.0f}' for d in days])
                    ax6.grid(True, alpha=0.3, axis='y')
                
                plt.tight_layout()
                st.pyplot(fig)
            
            with tab2:
                st.subheader("📋 Intervention Schedule - Beads & HCl")
                
                st.info("**Combined intervention schedule** - Add both beads and HCl at each time point. HCl amounts shown neutralize formate consumed since last intervention (minimizes lab visits)")
                
                # Create DataFrame for combined bead and HCl schedule
                schedule_data = []
                for time_point in sorted(results['bead_schedule'].keys()):
                    m07 = results['bead_schedule'][time_point].get('M07', 0)
                    m03 = results['bead_schedule'][time_point].get('M03', 0)
                    hcl_mmol = results['bead_schedule'][time_point].get('HCl_mmol', 0)
                    
                    # Format time based on intervention interval
                    if intervention_interval < 1.0:
                        time_str = f"{time_point:.2f}"
                    else:
                        time_str = f"{time_point:.1f}"
                    
                    schedule_data.append({
                        'Time (days)': time_str,
                        'M07 Beads': m07,
                        'M03 Beads': m03,
                        'Total Beads': m07 + m03,
                        'HCl Added (mmol)': f"{hcl_mmol:.3f}",
                        'HCl Added (mg)': f"{hcl_mmol * 36.46:.2f}"
                    })
                
                if schedule_data:
                    df_schedule = pd.DataFrame(schedule_data)
                    st.dataframe(df_schedule, use_container_width=True)
                    
                    # Summary statistics
                    col1, col2, col3 = st.columns(3)
                    
                    total_m07 = sum(b.get('M07', 0) for b in results['bead_schedule'].values())
                    total_m03 = sum(b.get('M03', 0) for b in results['bead_schedule'].values())
                    total_hcl = sum([results['bead_schedule'][t].get('HCl_mmol', 0) for t in results['bead_schedule'].keys()])
                    
                    with col1:
                        st.metric("Total Interventions", len(schedule_data))
                        st.metric("Total Beads", total_m07 + total_m03)
                    
                    with col2:
                        st.metric("M07 Beads", total_m07)
                        st.metric("M03 Beads", total_m03)
                    
                    with col3:
                        st.metric("Total HCl", f"{total_hcl:.2f} mmol")
                        st.metric("Total HCl (mg)", f"{total_hcl * 36.46:.2f}")
                    
                    # Practical HCl volumes
                    st.markdown("---")
                    st.write("**Practical HCl Volumes (for entire experiment):**")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        vol_ml = (total_hcl / 1000) / 12.0 * 1000
                        st.write(f"• **12M HCl:** {vol_ml:.2f} mL" if vol_ml >= 1 else f"• **12M HCl:** {vol_ml*1000:.0f} μL")
                    with col2:
                        vol_ml = (total_hcl / 1000) / 6.0 * 1000
                        st.write(f"• **6M HCl:** {vol_ml:.2f} mL")
                    with col3:
                        vol_ml = (total_hcl / 1000) / 1.0 * 1000
                        st.write(f"• **1M HCl:** {vol_ml:.2f} mL")
                    
                    # Download button
                    csv = df_schedule.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Complete Schedule (CSV)",
                        data=csv,
                        file_name="intervention_schedule.csv",
                        mime="text/csv"
                    )
                else:
                    st.info("No intervention schedule generated.")
            
            with tab3:
                st.subheader("📊 Detailed Statistics")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Substrate Control Performance:**")
                    st.write(f"• Target: {target_concentration} mmol/L")
                    st.write(f"• Average achieved: {np.mean(results['substrate']):.2f} mmol/L")
                    st.write(f"• Range: {np.min(results['substrate']):.2f} - {np.max(results['substrate']):.2f} mmol/L")
                    st.write(f"• Standard deviation: {np.std(results['substrate']):.2f} mmol/L")
                    st.write(f"• Deviation from target: {abs(np.mean(results['substrate']) - target_concentration):.2f} mmol/L ({abs(np.mean(results['substrate']) - target_concentration)/target_concentration*100:.1f}%)")
                
                with col2:
                    st.write("**Bacterial Growth:**")
                    st.write(f"• Initial OD: {initial_od:.3f}")
                    st.write(f"• Final OD: {results['od'][-1]:.3f}")
                    st.write(f"• Growth factor: {results['od'][-1]/initial_od:.1f}x")
                    st.write(f"• Max growth rate: {np.max(np.diff(results['od'])/dt):.4f} OD/day")
                
                st.write("**Bead Usage:**")
                total_m07 = sum(b.get('M07', 0) for b in results['bead_schedule'].values())
                total_m03 = sum(b.get('M03', 0) for b in results['bead_schedule'].values())
                st.write(f"• Total M07 beads: {total_m07}")
                st.write(f"• Total M03 beads: {total_m03}")
                st.write(f"• Total beads: {total_m07 + total_m03}")
                st.write(f"• Addition events: {len([d for d in results['bead_schedule'].keys() if d > 0])}")
                st.write(f"• Max release rate: {np.max(results['release_rates']):.4f} mmol/day")
                
                # Mass balance verification
                st.write("**Mass Balance Verification:**")
                total_consumed_mmol = results['cumulative_consumed'][-1]
                total_consumed_mmol_per_L = total_consumed_mmol / volume
                yield_coefficient = Y_XS
                od_increase_expected = total_consumed_mmol_per_L * yield_coefficient
                final_od_expected = initial_od + od_increase_expected
                relative_error = abs(final_od_expected - results['od'][-1]) / results['od'][-1] * 100
                
                st.write(f"• Expected final OD: {final_od_expected:.4f}")
                st.write(f"• Actual final OD: {results['od'][-1]:.4f}")
                st.write(f"• Relative error: {relative_error:.2f}%")
                if relative_error < 0.1:
                    st.success("✅ Mass balance verified (error < 0.1%)")
                elif relative_error < 1.0:
                    st.success("✅ Mass balance acceptable (error < 1%)")
                else:
                    st.warning(f"⚠️ Mass balance error: {relative_error:.2f}%")
    
    else:
        # Initial view before simulation
        st.info("""
        👈 **Configure your experiment in the sidebar and click "Run Simulation" to start!**
        
        This tool uses an advanced **Model Predictive Control (MPC)** algorithm to:
        - Predict bacterial growth and substrate consumption
        - Automatically schedule bead additions to maintain target substrate levels
        - Calculate HCl requirements for pH control
        - Optimize M07/M03 bead ratios for efficient substrate delivery
        """)
        
        # Show example configuration
        st.subheader("📝 Typical Configuration")
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Small Scale (Lab):**")
            st.write("• Volume: 0.05-0.5 L")
            st.write("• Initial OD: 0.01-0.05")
            st.write("• Duration: 3-7 days")
        
        with col2:
            st.write("**Large Scale (Bioreactor):**")
            st.write("• Volume: 1-10 L")
            st.write("• Initial OD: 0.02-0.1")
            st.write("• Duration: 7-14 days")


if __name__ == "__main__":
    main()
