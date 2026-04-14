#!/usr/bin/env python3
"""
Hodgkin-Huxley simulation tool for the OpenWorm MCP server.

Uses the well-tested implementation from:
https://github.com/openworm/hodgkin_huxley_tutorial/blob/master/Tutorial/Source/HodgkinHuxley.py

This pure-Python/scipy implementation does NOT require the NEURON simulator.

Originally from neuroml-ai: openworm_mcp/tools/hh_tools.py
"""

from dataclasses import asdict
from textwrap import dedent
from typing import Any, Dict

from openworm_mcp.sandbox.sandbox import RunPythonCode
from openworm_mcp.sandbox import openworm_mcp_sandbox

sbox = openworm_mcp_sandbox


async def run_hh_simulation_tool(
    current_injection: float = 0.1,
    duration: float = 300.0,
    delay: float = 50.0,
    temperature: float = 6.3,
    g_Na: float = 120.0,
    g_K: float = 36.0,
    g_L: float = 0.3,
    E_Na: float = 50.0,
    E_K: float = -77.0,
    E_L: float = -54.387,
    C_m: float = 1.0,
) -> Dict[str, Any]:
    """Run a Hodgkin-Huxley single compartment neuron simulation.

    Uses the standard Hodgkin-Huxley squid giant axon model (1952) implemented
    with scipy's odeint, based on the well-tested openworm/hodgkin_huxley_tutorial.
    Does NOT require the NEURON simulator.

    Inputs:

    - current_injection (float, default 0.1): injected current amplitude in uA/cm^2.
      Increase to make the neuron fire more frequently.
      Decrease below threshold to observe subthreshold behaviour.
    - duration (float, default 300.0): total simulation duration in milliseconds.
    - delay (float, default 50.0): delay in milliseconds before current injection begins.
      Must be less than duration.
    - temperature (float, default 6.3): simulation temperature in Celsius.
      The original Hodgkin-Huxley model was recorded at 6.3C.
    - g_Na (float, default 120.0): max sodium conductance in mS/cm^2.
      Hodgkin & Huxley (1952) measured 120.0 in squid giant axon.
      Reduce to simulate sodium channel blockers (e.g. TTX).
    - g_K (float, default 36.0): max potassium conductance in mS/cm^2.
      Hodgkin & Huxley (1952) measured 36.0. Reduce to simulate
      potassium channel blockers (e.g. TEA).
    - g_L (float, default 0.3): leak conductance in mS/cm^2.
    - E_Na (float, default 50.0): sodium reversal potential in mV.
    - E_K (float, default -77.0): potassium reversal potential in mV.
      Shift to simulate different extracellular potassium concentrations.
    - E_L (float, default -54.387): leak reversal potential in mV.
    - C_m (float, default 1.0): membrane capacitance in uF/cm^2.

    Output:

    Dictionary with keys:
    - stdout (str): JSON string containing:
        - firing_rate_hz (float): number of action potentials per second
        - num_action_potentials (int): total action potentials during simulation
        - peak_voltage_mv (float): maximum membrane voltage reached
        - resting_voltage_mv (float): initial resting membrane potential
        - simulation_duration_ms (float): total simulation duration
        - current_injection (float): injected current used
        - temperature_c (float): temperature used
        - voltage_trace (dict): downsampled trace with keys t_ms and v_mv
    - stderr (str): any errors
    - returncode (int): 0 if successful, non-zero if error occurred
    - data (dict): additional metadata

    Examples:

    - Default simulation: run_hh_simulation_tool()
    - Strong stimulation: run_hh_simulation_tool(current_injection=0.5)
    - Subthreshold (no firing expected): run_hh_simulation_tool(current_injection=0.01)
    - Long simulation: run_hh_simulation_tool(duration=1000.0)
    - Temperature effect: call twice with temperature=6.3 and temperature=25.0
    - Sodium channel block (TTX): run_hh_simulation_tool(g_Na=0.0, current_injection=10.0)
    - Potassium channel block (TEA): run_hh_simulation_tool(g_K=0.0, current_injection=10.0)
    - High extracellular K+: run_hh_simulation_tool(E_K=-50.0)
    """
    # Validate: duration must be long enough for current injection after delay.
    # If not, auto-correct to: delay + stimulus + recovery (equal periods).
    if duration <= delay:
        duration = delay * 3  # delay + equal stimulus + equal recovery

    code = dedent(f"""
import json
import numpy as np
from scipy.integrate import odeint

# ── Hodgkin-Huxley model parameters ──────────────────────────────────
# Based on: openworm/hodgkin_huxley_tutorial
# Reference: Hodgkin & Huxley (1952) J Physiol 117:500-544

C_m  = {C_m}      # membrane capacitance (uF/cm^2)
g_Na = {g_Na}    # max sodium conductance (mS/cm^2)
g_K  = {g_K}     # max potassium conductance (mS/cm^2)
g_L  = {g_L}      # leak conductance (mS/cm^2)
E_Na = {E_Na}     # sodium reversal potential (mV)
E_K  = {E_K}    # potassium reversal potential (mV)
E_L  = {E_L}  # leak reversal potential (mV)

# Temperature correction factor (Q10)
T = {temperature}
phi = 3.0 ** ((T - 6.3) / 10.0)

# ── Gating variable kinetics (Hodgkin & Huxley 1952) ─────────────────
def alpha_m(V):
    return phi * 0.1 * (V + 40.0) / (1.0 - np.exp(-(V + 40.0) / 10.0))

def beta_m(V):
    return phi * 4.0 * np.exp(-(V + 65.0) / 18.0)

def alpha_h(V):
    return phi * 0.07 * np.exp(-(V + 65.0) / 20.0)

def beta_h(V):
    return phi * 1.0 / (1.0 + np.exp(-(V + 35.0) / 10.0))

def alpha_n(V):
    return phi * 0.01 * (V + 55.0) / (1.0 - np.exp(-(V + 55.0) / 10.0))

def beta_n(V):
    return phi * 0.125 * np.exp(-(V + 65.0) / 80.0)

# ── Ionic currents ───────────────────────────────────────────────────
def I_Na(V, m, h):
    return g_Na * m**3 * h * (V - E_Na)

def I_K(V, n):
    return g_K * n**4 * (V - E_K)

def I_L(V):
    return g_L * (V - E_L)

# ── Injected current with delay ──────────────────────────────────────
def I_inj(t):
    if {delay} <= t <= {duration}:
        return {current_injection}
    return 0.0

# ── ODE system ───────────────────────────────────────────────────────
def dALLdt(X, t):
    V, m, h, n = X
    dVdt = (I_inj(t) - I_Na(V, m, h) - I_K(V, n) - I_L(V)) / C_m
    dmdt = alpha_m(V) * (1.0 - m) - beta_m(V) * m
    dhdt = alpha_h(V) * (1.0 - h) - beta_h(V) * h
    dndt = alpha_n(V) * (1.0 - n) - beta_n(V) * n
    return [dVdt, dmdt, dhdt, dndt]

# ── Initial conditions at resting potential ──────────────────────────
V0 = -65.0
m0 = alpha_m(V0) / (alpha_m(V0) + beta_m(V0))
h0 = alpha_h(V0) / (alpha_h(V0) + beta_h(V0))
n0 = alpha_n(V0) / (alpha_n(V0) + beta_n(V0))

# ── Run simulation ───────────────────────────────────────────────────
t = np.arange(0.0, {duration}, 0.01)  # 0.01 ms timestep
X = odeint(dALLdt, [V0, m0, h0, n0], t)
V = X[:, 0]

# ── Count action potentials (upward zero crossings) ──────────────────
crossings = sum(
    1 for i in range(1, len(V))
    if V[i-1] < 0 and V[i] >= 0
)

stimulus_duration_s = ({duration} - {delay}) / 1000.0
firing_rate = crossings / stimulus_duration_s if stimulus_duration_s > 0 else 0

# ── Generate voltage trace plot ──────────────────────────────────────
plot_base64 = ""
plot_path = ""
try:
    import base64, tempfile
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(t, V, color='#2563eb', linewidth=0.8)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Membrane Potential (mV)')
    ax.set_title(
        f'Hodgkin-Huxley Simulation  |  '
        f'I = {current_injection} uA/cm^2, '
        f'{{crossings}} APs, '
        f'{{round(firing_rate, 1)}} Hz'
    )
    ax.axhline(y=0, color='grey', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.axvspan({delay}, {duration}, alpha=0.06, color='orange',
               label=f'Stimulus ({current_injection} uA/cm^2)')
    ax.legend(loc='upper right', fontsize=8)
    ax.set_xlim(0, {duration})
    fig.tight_layout()

    plot_file = tempfile.NamedTemporaryFile(
        suffix='.png', prefix='hh_trace_', delete=False, dir=tempfile.gettempdir()
    )
    plot_path = plot_file.name
    plot_file.close()
    fig.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    with open(plot_path, 'rb') as pf:
        plot_base64 = base64.b64encode(pf.read()).decode('utf-8')
except Exception as plot_err:
    import sys
    print(f"Warning: plot generation failed: {{plot_err}}", file=sys.stderr)

# ── Downsample and output ────────────────────────────────────────────
step = max(1, len(t) // 300)  # ~300 points for the trace
print(json.dumps({{
    "firing_rate_hz": round(firing_rate, 2),
    "num_action_potentials": crossings,
    "peak_voltage_mv": round(float(np.max(V)), 2),
    "resting_voltage_mv": round(float(V[0]), 2),
    "simulation_duration_ms": {duration},
    "current_injection": {current_injection},
    "temperature_c": {temperature},
    "voltage_trace": {{
        "t_ms": [round(float(x), 2) for x in t[::step]],
        "v_mv": [round(float(x), 2) for x in V[::step]]
    }},
    "plot_base64": plot_base64,
    "plot_path": plot_path
}}))
""")

    request = RunPythonCode(code=code)
    async with sbox(".") as f:
        result = await f.run(request)
    return asdict(result)
