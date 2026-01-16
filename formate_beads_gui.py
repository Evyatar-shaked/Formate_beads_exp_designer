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
    def __init__(self, volume, monod_params, target_concentration, tolerance=0.1):
        self.volume = volume
        self.monod = MonodKinetics(**monod_params)
        self.target_concentration = target_concentration
        self.tolerance = tolerance
    
    def calculate_bead_schedule(self, initial_od, experiment_days=7, dt=0.01):
        """Enhanced MPC version with tighter control."""
        manager = ExperimentManager()
        
        # Initial beads
        total_formate_needed = self.target_concentration * self.volume * 1.1
        m07_total_release = sum(M07_BEAD_RELEASE.values())
        m03_total_release = sum(M03_BEAD_RELEASE.values())
        
        initial_m07 = max(1, int(np.ceil(total_formate_needed * 0.7 / m07_total_release)))
        initial_m03 = max(1, int(np.ceil(total_formate_needed * 0.3 / m03_total_release)))
        
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
        bead_schedule = {0: {'M07': initial_m07, 'M03': initial_m03}}
        
        # Tighter control bounds
        lower_bound = self.target_concentration * 0.92
        upper_bound = self.target_concentration * 1.08
        action_threshold = self.target_concentration * 0.95
        
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
            
            # Check daily for bead additions
            if abs(current_time - current_day) < dt/2 and current_day > 0:
                current_consumption = self.monod.consumption_rate(substrate[i], od[i]) * self.volume
                
                if substrate[i] < action_threshold and substrate[i] < upper_bound:
                    deficit = (self.target_concentration - substrate[i]) * self.volume
                    days_remaining = experiment_days - current_day
                    
                    mu_current = self.monod.growth_rate(substrate[i])
                    if current_day <= 3:
                        projection_window = min(2.5, days_remaining)
                        buffer_factor = 0.7
                    else:
                        projection_window = min(1.8, days_remaining)
                        buffer_factor = 0.85
                    
                    # Project consumption
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
                    
                    if abs(deficit) < self.target_concentration * self.volume * 0.15:
                        buffer_factor *= 0.8
                    
                    total_needed = deficit + daily_consumption * projection_window * buffer_factor
                    
                    # Account for existing beads
                    upcoming_release_1day = manager.get_total_release(current_time + 1.0)
                    upcoming_release_2day = manager.get_total_release(current_time + 2.0)
                    avg_upcoming = (upcoming_release_1day + upcoming_release_2day) / 2
                    existing_supply = avg_upcoming * projection_window
                    total_needed = max(0, total_needed - existing_supply * 0.6)
                    
                    # Optimize bead ratio
                    deficit_ratio = deficit / (daily_consumption + 1e-6)
                    if deficit_ratio > 0.4 or days_remaining < 3:
                        m07_weight, m03_weight = 0.6, 0.4
                    else:
                        m07_weight, m03_weight = 0.45, 0.55
                    
                    m07_share = total_needed * m07_weight
                    m03_share = total_needed * m03_weight
                    
                    m07_needed = int(np.ceil(m07_share / m07_total_release)) if m07_share > 0.05 else 0
                    m03_needed = int(np.ceil(m03_share / m03_total_release)) if m03_share > 0.05 else 0
                    
                    if total_needed > 0.15 and m07_needed == 0 and m03_needed == 0:
                        if deficit_ratio > 0.3:
                            m07_needed = 1
                        else:
                            m03_needed = 1
                    
                    for _ in range(m07_needed):
                        manager.add_bead('M07', current_day)
                    for _ in range(m03_needed):
                        manager.add_bead('M03', current_day)
                    
                    if m07_needed > 0 or m03_needed > 0:
                        if current_day in bead_schedule:
                            bead_schedule[current_day]['M07'] += m07_needed
                            bead_schedule[current_day]['M03'] += m03_needed
                        else:
                            bead_schedule[current_day] = {'M07': m07_needed, 'M03': m03_needed}
        
        # Calculate HCl
        consumption_rates = np.zeros_like(times)
        hcl_needed_daily = np.zeros_like(times)
        for i in range(len(times)):
            consumption_rates[i] = self.monod.consumption_rate(substrate[i], od[i]) * self.volume
            hcl_needed_daily[i] = consumption_rates[i]
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
        value=5.0,
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
    
    tolerance = st.sidebar.slider(
        "Control Tolerance (±%)",
        min_value=1,
        max_value=20,
        value=10,
        help="Acceptable deviation from target"
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
                target_concentration=target_concentration,
                tolerance=tolerance
            )
            
            results = calculator.calculate_bead_schedule(
                initial_od=initial_od,
                experiment_days=experiment_days,
                dt=dt
            )
            
            # Calculate daily HCl
            daily_hcl = {}
            daily_hcl[0] = 0.0
            for day in range(1, experiment_days + 1):
                mask = (results['times'] >= (day - 1)) & (results['times'] < day)
                if np.any(mask):
                    daily_hcl[day] = np.sum(results['hcl_needed_daily'][mask] * dt)
                else:
                    daily_hcl[day] = 0.0
            
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
            tab1, tab2, tab3, tab4 = st.tabs(["📈 Plots", "📋 Bead Schedule", "🧪 HCl Requirements", "📊 Statistics"])
            
            with tab1:
                st.subheader("Simulation Results - Comprehensive Plots")
                
                # Create comprehensive plots
                fig, axes = plt.subplots(3, 2, figsize=(14, 16))
                fig.suptitle('Formate Beads Experiment: MPC-Enhanced Substrate Control', 
                            fontsize=16, fontweight='bold')
                
                # Plot 1: Substrate Concentration
                ax1 = axes[0, 0]
                ax1.plot(results['times'], results['substrate'], 'b-', linewidth=2, label='Substrate')
                ax1.axhline(y=target_concentration, color='r', linestyle='--', linewidth=2, label='Target')
                lower_bound = target_concentration * 0.92
                upper_bound = target_concentration * 1.08
                ax1.axhline(y=lower_bound, color='orange', linestyle=':', linewidth=1)
                ax1.axhline(y=upper_bound, color='orange', linestyle=':', linewidth=1)
                ax1.fill_between(results['times'], lower_bound, upper_bound, 
                                alpha=0.2, color='green', label='Target range')
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
                    ax3.set_xticklabels([f'Day {d}' for d in days])
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
                
                # Plot 6: HCl Addition Rate
                ax6 = axes[2, 1]
                ax6.plot(results['times'], results['hcl_needed_daily'], 'r-', linewidth=2.5, 
                        label='HCl Addition Rate', alpha=0.8)
                ax6.fill_between(results['times'], 0, results['hcl_needed_daily'], 
                                alpha=0.3, color='red')
                ax6.set_xlabel('Time (days)', fontsize=11)
                ax6.set_ylabel('HCl Rate (mmol/day)', fontsize=11)
                ax6.set_title('HCl Addition Rate for pH Control', fontsize=12, fontweight='bold')
                ax6.legend(loc='best', fontsize=9)
                ax6.grid(True, alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig)
            
            with tab2:
                st.subheader("📋 Bead Addition Schedule")
                
                # Create DataFrame for bead schedule
                schedule_data = []
                for day in range(experiment_days + 1):
                    if day in results['bead_schedule']:
                        m07 = results['bead_schedule'][day].get('M07', 0)
                        m03 = results['bead_schedule'][day].get('M03', 0)
                    else:
                        m07 = 0
                        m03 = 0
                    
                    if m07 > 0 or m03 > 0:
                        schedule_data.append({
                            'Day': day,
                            'M07 Beads': m07,
                            'M03 Beads': m03,
                            'Total Beads': m07 + m03
                        })
                
                if schedule_data:
                    df_schedule = pd.DataFrame(schedule_data)
                    st.dataframe(df_schedule, use_container_width=True)
                    
                    # Download button
                    csv = df_schedule.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Bead Schedule (CSV)",
                        data=csv,
                        file_name="bead_schedule.csv",
                        mime="text/csv"
                    )
                else:
                    st.info("No additional beads needed beyond initial beads.")
            
            with tab3:
                st.subheader("🧪 HCl Requirements for pH Control")
                
                st.info("""
                **Why HCl is needed:** Formate consumption alkalinizes the medium. 
                HCl addition maintains pH and cell membrane permeability.
                """)
                
                # Create DataFrame for HCl requirements
                hcl_data = []
                for day in range(experiment_days + 1):
                    hcl_mmol = daily_hcl.get(day, 0.0)
                    hcl_mg = hcl_mmol * 36.46
                    if hcl_mmol > 0.001:
                        hcl_data.append({
                            'Day': day,
                            'HCl Needed (mmol)': f"{hcl_mmol:.3f}",
                            'HCl Needed (mg)': f"{hcl_mg:.2f}"
                        })
                
                df_hcl = pd.DataFrame(hcl_data)
                st.dataframe(df_hcl, use_container_width=True)
                
                # Summary metrics
                col1, col2 = st.columns(2)
                
                total_hcl = results['hcl_needed_cumulative'][-1]
                with col1:
                    st.metric("Total HCl (mmol)", f"{total_hcl:.3f}")
                    st.metric("Total HCl (mg)", f"{total_hcl * 36.46:.2f}")
                    st.metric("Total HCl (mol)", f"{total_hcl / 1000:.6f}")
                
                with col2:
                    st.write("**Practical Volumes:**")
                    for conc, label in [(12.0, "12M HCl"), (6.0, "6M HCl"), (1.0, "1M HCl")]:
                        vol_ml = (total_hcl / 1000) / conc * 1000
                        if vol_ml >= 1:
                            st.write(f"• {label}: **{vol_ml:.2f} mL**")
                        else:
                            st.write(f"• {label}: **{vol_ml*1000:.0f} μL**")
                
                # Download button
                csv = df_hcl.to_csv(index=False)
                st.download_button(
                    label="📥 Download HCl Schedule (CSV)",
                    data=csv,
                    file_name="hcl_requirements.csv",
                    mime="text/csv"
                )
            
            with tab4:
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
        
        **Features:**
        - ✅ Tighter control bounds (±8% vs ±15%)
        - ✅ Proactive action threshold (95% of target)
        - ✅ Conservative buffer factors to prevent overshoot
        - ✅ Advanced consumption forecasting with exponential growth projection
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
