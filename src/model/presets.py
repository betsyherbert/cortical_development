"""Developmental parameter presets for the cortical circuit simulation, updated.

Note: All spatial parameters are in μm.
Default anatomical grid size is 1000 μm × 1000 μm.
"""

# ----------------- P0 PRESET (Early 1st Week) -----------------
P0_PRESET = {
    # Approximate time constants (ms).
    "time_constants": {
        "E": 50.0,  # immature pyramids, high τm
        "SST": 60.0,  # very integrative early SST
        "PV": 40.0,  # PV-lineage not yet fast-spiking
    },
    # Constant background input for each cell type.
    "background_input": {"E": 0.0, "SST": 0.0, "PV": 0.0},
    # Thalamic input widths: how broadly thalamic input is spatially spread (μm).
    "thalamic_widths": {"E": 200.0, "SST": 200.0, "PV": 150.0},
    # Outgoing widths: how far each cell type's lateral connections spread (μm).
    "outgoing_widths": {"E": 300.0, "SST": 400.0, "PV": 250.0},
    # Strength scaling factors: overall scaling of connection strengths.
    "strength_scaling": {
        "E": 2.0,  # weak excitatory network
        "SST": 3.0,  # slightly stronger SST modulation
        "PV": 0.2,  # proto-PV inhibition
        "thalamus": 2.0,  # baseline thalamic drive
    },
    # Thalamic alpha: developmental time parameter [0=early, 1=late]
    "thalamic_alpha": 0.1,
    "connection_strengths": {
        # ---------- L2/3 (within L2/3) ----------
        "L23_E_to_L23_E": 0.05,
        "L23_E_to_L23_SST": 0.1,
        "L23_E_to_L23_PV": 0.0,
        "L23_SST_to_L23_E": 0.0,
        "L23_SST_to_L23_PV": 0.0,
        "L23_PV_to_L23_E": 0.0,
        "L23_PV_to_L23_SST": 0.0,
        "L23_PV_to_L23_PV": 0.0,
        # ---------- L2/3 to L4 ----------
        "L23_E_to_L4_E": 0.0,
        "L23_E_to_L4_SST": 0.0,
        "L23_E_to_L4_PV": 0.5,
        "L23_SST_to_L4_E": 0.0,
        "L23_SST_to_L4_PV": 0.0,
        "L23_PV_to_L4_E": 0.0,
        "L23_PV_to_L4_SST": 0.0,
        "L23_PV_to_L4_PV": 0.0,
        # ---------- L2/3 to L5 ----------
        "L23_E_to_L5_E": 0.1,
        "L23_E_to_L5_SST": 0.0,
        "L23_E_to_L5_PV": 0.0,
        "L23_SST_to_L5_E": 0.0,
        "L23_SST_to_L5_PV": 0.0,
        "L23_PV_to_L5_E": 0.0,
        "L23_PV_to_L5_SST": 0.0,
        "L23_PV_to_L5_PV": 0.0,
        # ---------- L4 (within L4) ----------
        "L4_E_to_L4_E": 0.05,
        "L4_E_to_L4_SST": 0.1,
        "L4_E_to_L4_PV": 0.0,
        "L4_SST_to_L4_E": 0.0,
        "L4_SST_to_L4_PV": 0.0,
        "L4_PV_to_L4_E": 0.0,
        "L4_PV_to_L4_SST": 0.0,
        "L4_PV_to_L4_PV": 0.0,
        # ---------- L4 to L2/3 ----------
        "L4_E_to_L23_E": 0.1,
        "L4_E_to_L23_SST": 0.0,
        "L4_E_to_L23_PV": 0.0,
        "L4_SST_to_L23_E": 0.0,
        "L4_SST_to_L23_PV": 0.0,
        "L4_PV_to_L23_E": 0.0,
        "L4_PV_to_L23_SST": 0.0,
        "L4_PV_to_L23_PV": 0.0,
        # ---------- L4 to L5 ----------
        "L4_E_to_L5_E": 0.1,
        "L4_E_to_L5_SST": 0.4,
        "L4_E_to_L5_PV": 0.0,
        "L4_SST_to_L5_E": 0.0,
        "L4_SST_to_L5_PV": 0.0,
        "L4_PV_to_L5_E": 0.0,
        "L4_PV_to_L5_SST": 0.0,
        "L4_PV_to_L5_PV": 0.0,
        # ---------- L5 (within L5) ----------
        "L5_E_to_L5_E": 0.05,
        "L5_E_to_L5_SST": 0.1,
        "L5_E_to_L5_PV": 0.1,
        "L5_SST_to_L5_E": 0.0,
        "L5_SST_to_L5_PV": 0.0,
        "L5_PV_to_L5_E": 0.0,
        "L5_PV_to_L5_SST": 0.0,
        "L5_PV_to_L5_PV": 0.0,
        # ---------- L5 to L2/3 ----------
        "L5_E_to_L23_E": 0.5,
        "L5_E_to_L23_SST": 0.3,
        "L5_E_to_L23_PV": 0.3,
        "L5_SST_to_L23_E": 0.0,
        "L5_SST_to_L23_PV": 0.0,
        "L5_PV_to_L23_E": 0.0,
        "L5_PV_to_L23_SST": 0.0,
        "L5_PV_to_L23_PV": 0.0,
        # ---------- L5 to L4 ----------
        "L5_E_to_L4_E": 0.1,
        "L5_E_to_L4_SST": 0.1,
        "L5_E_to_L4_PV": 0.0,
        "L5_SST_to_L4_E": -0.4,
        "L5_SST_to_L4_PV": 0.0,
        "L5_PV_to_L4_E": 0.0,
        "L5_PV_to_L4_SST": 0.0,
        "L5_PV_to_L4_PV": 0.0,
        # ---------- Thalamic connections ----------
        "thalamus_to_L23_E": 0.2,
        "thalamus_to_L23_SST": 0.2,
        "thalamus_to_L23_PV": 0.2,
        "thalamus_to_L4_E": 0.3,
        "thalamus_to_L4_SST": 0.3,
        "thalamus_to_L4_PV": 0.1,  # reduced
        "thalamus_to_L5_E": 0.3,
        "thalamus_to_L5_SST": 0.6,  # boosted transient deep SST drive
        "thalamus_to_L5_PV": 0.1,  # reduced
    },
}

# ----------------- P5 PRESET (Late 1st Week) -----------------
P5_PRESET = {
    "time_constants": {"E": 45.0, "SST": 55.0, "PV": 35.0},
    "background_input": {"E": 0.0, "SST": 0.0, "PV": 0.0},
    # Thalamic input widths (μm)
    "thalamic_widths": {"E": 150.0, "SST": 150.0, "PV": 120.0},
    # Outgoing widths (μm)
    "outgoing_widths": {"E": 200.0, "SST": 400.0, "PV": 250.0},
    "strength_scaling": {"E": 3.2, "SST": 4.0, "PV": 0.7, "thalamus": 2.0},
    "thalamic_alpha": 0.3,
    "connection_strengths": {
        # ---------- L2/3 (within L2/3) ----------
        "L23_E_to_L23_E": 0.3,
        "L23_E_to_L23_SST": 0.3,
        "L23_E_to_L23_PV": 0.2,
        "L23_SST_to_L23_E": -0.2,
        "L23_SST_to_L23_PV": -0.1,
        "L23_PV_to_L23_E": 0.0,
        "L23_PV_to_L23_SST": -0.2,
        "L23_PV_to_L23_PV": -0.2,
        # ---------- L2/3 to L4 ----------
        "L23_E_to_L4_E": 0.1,
        "L23_E_to_L4_SST": 0.1,
        "L23_E_to_L4_PV": 0.4,
        "L23_SST_to_L4_E": 0.0,
        "L23_SST_to_L4_PV": 0.0,
        "L23_PV_to_L4_E": -0.1,
        "L23_PV_to_L4_SST": 0.0,
        "L23_PV_to_L4_PV": 0.0,
        # ---------- L2/3 to L5 ----------
        "L23_E_to_L5_E": 0.2,
        "L23_E_to_L5_SST": 0.0,
        "L23_E_to_L5_PV": 0.1,
        "L23_SST_to_L5_E": -0.1,
        "L23_SST_to_L5_PV": 0.0,
        "L23_PV_to_L5_E": -0.1,
        "L23_PV_to_L5_SST": 0.0,
        "L23_PV_to_L5_PV": 0.0,
        # ---------- L4 (within L4) ----------
        "L4_E_to_L4_E": 0.4,
        "L4_E_to_L4_SST": 0.4,
        "L4_E_to_L4_PV": 0.2,
        "L4_SST_to_L4_E": -0.4,
        "L4_SST_to_L4_PV": -0.1,
        "L4_PV_to_L4_E": 0.0,
        "L4_PV_to_L4_SST": -0.2,
        "L4_PV_to_L4_PV": -0.3,
        # ---------- L4 to L2/3 ----------
        "L4_E_to_L23_E": 0.3,
        "L4_E_to_L23_SST": 0.1,
        "L4_E_to_L23_PV": 0.2,
        "L4_SST_to_L23_E": 0.0,
        "L4_SST_to_L23_PV": 0.0,
        "L4_PV_to_L23_E": -0.1,
        "L4_PV_to_L23_SST": 0.0,
        "L4_PV_to_L23_PV": -0.1,
        # ---------- L4 to L5 ----------
        "L4_E_to_L5_E": 0.1,
        "L4_E_to_L5_SST": 0.4,
        "L4_E_to_L5_PV": 0.1,
        "L4_SST_to_L5_E": 0.0,
        "L4_SST_to_L5_PV": 0.0,
        "L4_PV_to_L5_E": -0.1,
        "L4_PV_to_L5_SST": 0.0,
        "L4_PV_to_L5_PV": -0.1,
        # ---------- L5 (within L5) ----------
        "L5_E_to_L5_E": 0.4,
        "L5_E_to_L5_SST": 0.3,
        "L5_E_to_L5_PV": 0.4,
        "L5_SST_to_L5_E": -0.3,
        "L5_SST_to_L5_PV": -0.4,  # boosted early SST→PV
        "L5_PV_to_L5_E": 0.0,
        "L5_PV_to_L5_SST": -0.2,
        "L5_PV_to_L5_PV": -0.3,
        # ---------- L5 to L2/3 ----------
        "L5_E_to_L23_E": 0.5,
        "L5_E_to_L23_SST": 0.3,
        "L5_E_to_L23_PV": 0.4,
        "L5_SST_to_L23_E": 0.0,
        "L5_SST_to_L23_PV": 0.0,
        "L5_PV_to_L23_E": 0.0,
        "L5_PV_to_L23_SST": 0.0,
        "L5_PV_to_L23_PV": 0.0,
        # ---------- L5 to L4 ----------
        "L5_E_to_L4_E": 0.2,
        "L5_E_to_L4_SST": 0.2,
        "L5_E_to_L4_PV": 0.2,
        "L5_SST_to_L4_E": -0.4,
        "L5_SST_to_L4_PV": -0.5,
        "L5_PV_to_L4_E": -0.1,
        "L5_PV_to_L4_SST": 0.0,
        "L5_PV_to_L4_PV": 0.0,
        # ---------- Thalamic connections ----------
        "thalamus_to_L23_E": 0.1,
        "thalamus_to_L23_SST": 0.1,
        "thalamus_to_L23_PV": 0.1,
        "thalamus_to_L4_E": 0.6,
        "thalamus_to_L4_SST": 0.2,
        "thalamus_to_L4_PV": 0.2,  # reduced
        "thalamus_to_L5_E": 0.3,
        "thalamus_to_L5_SST": 0.4,
        "thalamus_to_L5_PV": 0.1,  # reduced
    },
}

# ----------------- P10 PRESET (Mid 2nd Week) -----------------
P10_PRESET = {
    "time_constants": {"E": 30.0, "SST": 50.0, "PV": 20.0},
    # Thalamic input widths (μm)
    "thalamic_widths": {"E": 120.0, "SST": 120.0, "PV": 100.0},
    # Outgoing widths (μm)
    "outgoing_widths": {"E": 200.0, "SST": 220.0, "PV": 120.0},
    "strength_scaling": {"E": 4.5, "SST": 4.0, "PV": 3.0, "thalamus": 2.0},
    "thalamic_alpha": 0.7,
    "background_input": {"E": 0.05, "SST": 0.0, "PV": 0.0},
    "connection_strengths": {
        # ---------- L2/3 (within L2/3) ----------
        "L23_E_to_L23_E": 0.4,
        "L23_E_to_L23_SST": 0.6,
        "L23_E_to_L23_PV": 0.5,
        "L23_SST_to_L23_E": -0.4,
        "L23_SST_to_L23_PV": -0.2,
        "L23_PV_to_L23_E": -0.2,
        "L23_PV_to_L23_SST": -0.4,
        "L23_PV_to_L23_PV": -0.4,
        # ---------- L2/3 to L4 ----------
        "L23_E_to_L4_E": 0.2,
        "L23_E_to_L4_SST": 0.2,
        "L23_E_to_L4_PV": 0.3,
        "L23_SST_to_L4_E": 0.0,
        "L23_SST_to_L4_PV": 0.0,
        "L23_PV_to_L4_E": -0.2,
        "L23_PV_to_L4_SST": 0.0,
        "L23_PV_to_L4_PV": 0.0,
        # ---------- L2/3 to L5 ----------
        "L23_E_to_L5_E": 0.8,
        "L23_E_to_L5_SST": 0.1,
        "L23_E_to_L5_PV": 0.6,
        "L23_SST_to_L5_E": -0.2,
        "L23_SST_to_L5_PV": 0.0,
        "L23_PV_to_L5_E": -0.1,
        "L23_PV_to_L5_SST": 0.0,
        "L23_PV_to_L5_PV": 0.0,
        # ---------- L4 (within L4) ----------
        "L4_E_to_L4_E": 0.5,
        "L4_E_to_L4_SST": 0.5,
        "L4_E_to_L4_PV": 0.8,
        "L4_SST_to_L4_E": -0.4,
        "L4_SST_to_L4_PV": -0.2,
        "L4_PV_to_L4_E": -0.2,
        "L4_PV_to_L4_SST": -0.2,
        "L4_PV_to_L4_PV": -0.4,
        # ---------- L4 to L2/3 ----------
        "L4_E_to_L23_E": 0.6,
        "L4_E_to_L23_SST": 0.2,
        "L4_E_to_L23_PV": 0.4,
        "L4_SST_to_L23_E": 0.0,
        "L4_SST_to_L23_PV": 0.0,
        "L4_PV_to_L23_E": -0.2,
        "L4_PV_to_L23_SST": 0.0,
        "L4_PV_to_L23_PV": -0.2,
        # ---------- L4 to L5 ----------
        "L4_E_to_L5_E": 0.1,
        "L4_E_to_L5_SST": 0.4,
        "L4_E_to_L5_PV": 0.2,
        "L4_SST_to_L5_E": -0.1,
        "L4_SST_to_L5_PV": 0.0,
        "L4_PV_to_L5_E": -0.2,
        "L4_PV_to_L5_SST": 0.0,
        "L4_PV_to_L5_PV": -0.2,
        # ---------- L5 (within L5) ----------
        "L5_E_to_L5_E": 0.6,
        "L5_E_to_L5_SST": 0.4,
        "L5_E_to_L5_PV": 0.8,
        "L5_SST_to_L5_E": -0.5,
        "L5_SST_to_L5_PV": -0.3,
        "L5_PV_to_L5_E": -0.4,
        "L5_PV_to_L5_SST": -0.3,
        "L5_PV_to_L5_PV": -0.6,
        # ---------- L5 to L2/3 ----------
        "L5_E_to_L23_E": 0.5,
        "L5_E_to_L23_SST": 0.3,
        "L5_E_to_L23_PV": 0.5,
        "L5_SST_to_L23_E": -0.3,
        "L5_SST_to_L23_PV": 0.0,
        "L5_PV_to_L23_E": 0.0,
        "L5_PV_to_L23_SST": 0.0,
        "L5_PV_to_L23_PV": -0.2,
        # ---------- L5 to L4 ----------
        "L5_E_to_L4_E": 0.2,
        "L5_E_to_L4_SST": 0.1,
        "L5_E_to_L4_PV": 0.3,
        "L5_SST_to_L4_E": -0.2,
        "L5_SST_to_L4_PV": 0.0,
        "L5_PV_to_L4_E": -0.2,
        "L5_PV_to_L4_SST": 0.0,
        "L5_PV_to_L4_PV": 0.0,
        # ---------- Thalamic connections ----------
        "thalamus_to_L23_E": 0.1,
        "thalamus_to_L23_SST": 0.0,
        "thalamus_to_L23_PV": 0.1,
        "thalamus_to_L4_E": 0.8,
        "thalamus_to_L4_SST": 0.1,
        "thalamus_to_L4_PV": 0.5,  # reduced
        "thalamus_to_L5_E": 0.2,
        "thalamus_to_L5_SST": 0.2,
        "thalamus_to_L5_PV": 0.2,
    },
}

# ----------------- P15 PRESET (Late 2nd Week) -----------------
P15_PRESET = {
    "time_constants": {"E": 25.0, "SST": 30.0, "PV": 12.0},
    # Thalamic input widths (μm)
    "thalamic_widths": {"E": 100.0, "SST": 100.0, "PV": 100.0},
    # Outgoing widths (μm)
    "outgoing_widths": {
        "E": 200.0,
        "SST": 200.0,
        "PV": 100.0,
    },
    "strength_scaling": {"E": 6.0, "SST": 6.0, "PV": 5.0, "thalamus": 2.0},
    "thalamic_alpha": 0.9,
    "background_input": {"E": 0.05, "SST": 0.0, "PV": 0.0},
    "connection_strengths": {
        # ---------- L2/3 (within L2/3) ----------
        "L23_E_to_L23_E": 0.6,
        "L23_E_to_L23_SST": 0.8,
        "L23_E_to_L23_PV": 1.0,
        "L23_SST_to_L23_E": -0.3,
        "L23_SST_to_L23_PV": -0.3,
        "L23_PV_to_L23_E": -0.8,
        "L23_PV_to_L23_SST": -0.8,
        "L23_PV_to_L23_PV": -0.8,
        # ---------- L2/3 to L4 ----------
        "L23_E_to_L4_E": 0.3,
        "L23_E_to_L4_SST": 0.3,
        "L23_E_to_L4_PV": 0.3,
        "L23_SST_to_L4_E": 0.0,
        "L23_SST_to_L4_PV": 0.0,
        "L23_PV_to_L4_E": -0.3,
        "L23_PV_to_L4_SST": 0.0,
        "L23_PV_to_L4_PV": 0.0,
        # ---------- L2/3 to L5 ----------
        "L23_E_to_L5_E": 1.0,
        "L23_E_to_L5_SST": 0.3,
        "L23_E_to_L5_PV": 0.6,
        "L23_SST_to_L5_E": -0.3,
        "L23_SST_to_L5_PV": 0.0,
        "L23_PV_to_L5_E": -0.1,
        "L23_PV_to_L5_SST": -0.1,
        "L23_PV_to_L5_PV": 0.0,
        # ---------- L4 (within L4) ----------
        "L4_E_to_L4_E": 0.6,
        "L4_E_to_L4_SST": 0.7,
        "L4_E_to_L4_PV": 1.0,
        "L4_SST_to_L4_E": -0.4,
        "L4_SST_to_L4_PV": -0.3,
        "L4_PV_to_L4_E": -0.6,
        "L4_PV_to_L4_SST": -0.4,
        "L4_PV_to_L4_PV": -0.6,
        # ---------- L4 to L2/3 ----------
        "L4_E_to_L23_E": 1.0,
        "L4_E_to_L23_SST": 0.2,
        "L4_E_to_L23_PV": 0.6,
        "L4_SST_to_L23_E": 0.0,
        "L4_SST_to_L23_PV": 0.0,
        "L4_PV_to_L23_E": -0.3,
        "L4_PV_to_L23_SST": 0.0,
        "L4_PV_to_L23_PV": -0.3,
        # ---------- L4 to L5 ----------
        "L4_E_to_L5_E": 0.1,
        "L4_E_to_L5_SST": 0.4,
        "L4_E_to_L5_PV": 0.3,
        "L4_SST_to_L5_E": -0.3,
        "L4_SST_to_L5_PV": 0.0,
        "L4_PV_to_L5_E": -0.3,
        "L4_PV_to_L5_SST": -0.2,
        "L4_PV_to_L5_PV": -0.3,
        # ---------- L5 (within L5) ----------
        "L5_E_to_L5_E": 0.8,
        "L5_E_to_L5_SST": 0.7,
        "L5_E_to_L5_PV": 1.0,
        "L5_SST_to_L5_E": -0.4,
        "L5_SST_to_L5_PV": -0.3,
        "L5_PV_to_L5_E": -1.2,
        "L5_PV_to_L5_SST": -0.3,
        "L5_PV_to_L5_PV": -0.8,
        # ---------- L5 to L2/3 ----------
        "L5_E_to_L23_E": 0.4,
        "L5_E_to_L23_SST": 0.1,
        "L5_E_to_L23_PV": 0.2,
        "L5_SST_to_L23_E": -0.5,
        "L5_SST_to_L23_PV": 0.0,
        "L5_PV_to_L23_E": 0.0,
        "L5_PV_to_L23_SST": 0.0,
        "L5_PV_to_L23_PV": -0.3,
        # ---------- L5 to L4 ----------
        "L5_E_to_L4_E": 0.1,
        "L5_E_to_L4_SST": 0.1,
        "L5_E_to_L4_PV": 0.3,
        "L5_SST_to_L4_E": -0.1,
        "L5_SST_to_L4_PV": -0.1,
        "L5_PV_to_L4_E": -0.3,
        "L5_PV_to_L4_SST": 0.0,
        "L5_PV_to_L4_PV": 0.0,
        # ---------- Thalamic connections ----------
        "thalamus_to_L23_E": 0.1,
        "thalamus_to_L23_SST": 0.0,
        "thalamus_to_L23_PV": 0.1,
        "thalamus_to_L4_E": 1.0,
        "thalamus_to_L4_SST": 0.0,
        "thalamus_to_L4_PV": 0.6,  # slightly reduced vs original
        "thalamus_to_L5_E": 0.2,
        "thalamus_to_L5_SST": 0.0,
        "thalamus_to_L5_PV": 0.1,  # reduced
    },
}
