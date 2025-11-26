"""Developmental parameter presets for the cortical circuit simulation, updated.

Note: All spatial parameters (thalamic_widths, outgoing_widths) are in μm.
Default anatomical grid size is 1000 μm × 1000 μm.
"""

# ----------------- P0 PRESET (Early 1st Week) -----------------
P0_PRESET = {
    # Approximate time constants (ms). 
    'time_constants': {
        'E': 15.0,    # excitatory cells slower than adult
        'SST': 25.0,  # SST cells already somewhat active, but still slower
        'PV': 25.0    # prospective PV cells, minimal functional synapses
    },
    
    # Constant background input for each cell type.
    'background_input': {
        'E': 0.0,
        'SST': 0.0,
        'PV': 0.0
    },

    # Thalamic input widths: how broadly thalamic input is spatially spread (μm). 
    'thalamic_widths': {
        'E': 100.0,
        'SST': 75.0,
        'PV': 100.0
    },

    # Outgoing widths: how far each cell type's lateral connections spread (μm).
    'outgoing_widths': {
        'E': 300.0,
        'SST': 300.0,
        'PV': 300.0
    },
    
    # Strength scaling factors: overall scaling of connection strengths.
    'strength_scaling': {
        'E': 0.8,      # E cells have weak but present connections
        'SST': 0.5,    # SST cells have weaker connections
        'PV': 0.5,     # PV cells have very weak connections 
        'thalamus': 1.0  # Thalamic input is present but not fully developed
    },
    
    # Thalamic alpha: ratio between sensory-driven vs. intrinsic input.
    'thalamic_alpha': 0.1,
    
    'connection_strengths': {

            # ---------- L2/3 (within L2/3) ----------
            'L23_E_to_L23_E': 0.05,
            'L23_E_to_L23_SST': 0.1,
            'L23_E_to_L23_PV': 0.0,
            'L23_SST_to_L23_E': 0.0,   
            'L23_SST_to_L23_PV': 0.0,
            'L23_PV_to_L23_E': 0.0,    
            'L23_PV_to_L23_SST': 0.0,  
            'L23_PV_to_L23_PV': 0.0,   

            # ---------- L2/3 to L4 ----------
            'L23_E_to_L4_E': 0.0,        
            'L23_E_to_L4_SST': 0.0,
            'L23_E_to_L4_PV': 0.5,
            'L23_SST_to_L4_E': 0.0,
            'L23_SST_to_L4_PV': 0.0,
            'L23_PV_to_L4_E': 0.0,
            'L23_PV_to_L4_SST': 0.0,
            'L23_PV_to_L4_PV': 0.0,

            # ---------- L2/3 to L5 ----------
            'L23_E_to_L5_E': 0.1,
            'L23_E_to_L5_SST': 0.0,
            'L23_E_to_L5_PV': 0.0,
            'L23_SST_to_L5_E': 0.0,
            'L23_SST_to_L5_PV': 0.0,
            'L23_PV_to_L5_E': 0.0,
            'L23_PV_to_L5_SST': 0.0,
            'L23_PV_to_L5_PV': 0.0,

            # ---------- L4 (within L4) ----------
            'L4_E_to_L4_E': 0.05,
            'L4_E_to_L4_SST': 0.1,
            'L4_E_to_L4_PV': 0.0,
            'L4_SST_to_L4_E': 0.0,
            'L4_SST_to_L4_PV': 0.0,
            'L4_PV_to_L4_E': 0.0,
            'L4_PV_to_L4_SST': 0.0,
            'L4_PV_to_L4_PV': 0.0,

            # ---------- L4 to L2/3 ----------
            'L4_E_to_L23_E': 0.1,
            'L4_E_to_L23_SST': 0.0,
            'L4_E_to_L23_PV': 0.0,
            'L4_SST_to_L23_E': 0.0,
            'L4_SST_to_L23_PV': 0.0,
            'L4_PV_to_L23_E': 0.0,
            'L4_PV_to_L23_SST': 0.0,
            'L4_PV_to_L23_PV': 0.0,

            # ---------- L4 to L5 ----------
            'L4_E_to_L5_E': 0.1,
            'L4_E_to_L5_SST': 0.4,
            'L4_E_to_L5_PV': 0.0,
            'L4_SST_to_L5_E': 0.0,
            'L4_SST_to_L5_PV': 0.0,
            'L4_PV_to_L5_E': 0.0,
            'L4_PV_to_L5_SST': 0.0,
            'L4_PV_to_L5_PV': 0.0,

            # ---------- L5 (within L5) ----------
            'L5_E_to_L5_E': 0.05,
            'L5_E_to_L5_SST': 0.1,
            'L5_E_to_L5_PV': 0.1,
            'L5_SST_to_L5_E': 0.0,
            'L5_SST_to_L5_PV': 0.0,
            'L5_PV_to_L5_E': 0.0,
            'L5_PV_to_L5_SST': 0.0,
            'L5_PV_to_L5_PV': 0.0,

            # ---------- L5 to L2/3 ----------
            'L5_E_to_L23_E': 0.5,  
            'L5_E_to_L23_SST': 0.3,
            'L5_E_to_L23_PV': 0.3,
            'L5_SST_to_L23_E': 0.0,
            'L5_SST_to_L23_PV': 0.0,
            'L5_PV_to_L23_E': 0.0,
            'L5_PV_to_L23_SST': 0.0,
            'L5_PV_to_L23_PV': 0.0,

            # ---------- L5 to L4 ----------
            'L5_E_to_L4_E': 0.1,
            'L5_E_to_L4_SST': 0.1,
            'L5_E_to_L4_PV': 0.0,
            'L5_SST_to_L4_E': -0.4,
            'L5_SST_to_L4_PV': 0.0,
            'L5_PV_to_L4_E': 0.0,
            'L5_PV_to_L4_SST': 0.0,
            'L5_PV_to_L4_PV': 0.0,

            # ---------- Thalamic connections ----------
            'thalamus_to_L23_E': 0.2,
            'thalamus_to_L23_SST': 0.2,
            'thalamus_to_L23_PV': 0.2,
            'thalamus_to_L4_E': 0.3,
            'thalamus_to_L4_SST': 0.3,
            'thalamus_to_L4_PV': 0.2,
            'thalamus_to_L5_E': 0.3,
            'thalamus_to_L5_SST': 0.5,
            'thalamus_to_L5_PV': 0.3
        }
}

# ----------------- P5 PRESET (Late 1st Week) -----------------
P5_PRESET = {
    'time_constants': {
        'E': 14.0,
        'SST': 20.0,
        'PV': 20.0
    },
    
    'background_input': {
        'E': 0.0,
        'SST': 0.0,
        'PV': 0.0
    },

    # Thalamic input widths (μm)
    'thalamic_widths': {
        'E': 55.0,
        'SST': 75.0,
        'PV': 100.0
    },

    # Outgoing widths (μm)
    'outgoing_widths': {
        'E': 125.0,
        'SST': 300.0,
        'PV': 300.0
    },

    'strength_scaling': {
        'E': 1.5,    
        'SST': 1.5,  
        'PV': 1.0,   
        'thalamus': 3.0
    },

    'thalamic_alpha': 0.3,
        
    'connection_strengths': {

            # ---------- L2/3 (within L2/3) ----------
            'L23_E_to_L23_E': 0.3,
            'L23_E_to_L23_SST': 0.3,
            'L23_E_to_L23_PV': 0.2,
            'L23_SST_to_L23_E': -0.2,   
            'L23_SST_to_L23_PV': -0.1,
            'L23_PV_to_L23_E': 0.0,    
            'L23_PV_to_L23_SST': -0.2,  
            'L23_PV_to_L23_PV': -0.2,   

            # ---------- L2/3 to L4 ----------
            'L23_E_to_L4_E': 0.1,        
            'L23_E_to_L4_SST': 0.1,
            'L23_E_to_L4_PV': 0.4,
            'L23_SST_to_L4_E': 0.0,
            'L23_SST_to_L4_PV': 0.0,
            'L23_PV_to_L4_E': -0.1,
            'L23_PV_to_L4_SST': 0.0,
            'L23_PV_to_L4_PV': 0.0,

            # ---------- L2/3 to L5 ----------
            'L23_E_to_L5_E': 0.2,
            'L23_E_to_L5_SST': 0.0,
            'L23_E_to_L5_PV': 0.1,
            'L23_SST_to_L5_E': -0.1,
            'L23_SST_to_L5_PV': 0.0,
            'L23_PV_to_L5_E': -0.1,
            'L23_PV_to_L5_SST': 0.0,
            'L23_PV_to_L5_PV': 0.0,

            # ---------- L4 (within L4) ----------
            'L4_E_to_L4_E': 0.4,
            'L4_E_to_L4_SST': 0.4,
            'L4_E_to_L4_PV': 0.2,
            'L4_SST_to_L4_E': -0.4,
            'L4_SST_to_L4_PV': -0.1,
            'L4_PV_to_L4_E': 0.0,
            'L4_PV_to_L4_SST': -0.2,
            'L4_PV_to_L4_PV': -0.3,

            # ---------- L4 to L2/3 ----------
            'L4_E_to_L23_E': 0.3,
            'L4_E_to_L23_SST': 0.1,
            'L4_E_to_L23_PV': 0.2,
            'L4_SST_to_L23_E': 0.0,
            'L4_SST_to_L23_PV': 0.0,
            'L4_PV_to_L23_E': -0.1,
            'L4_PV_to_L23_SST': 0.0,
            'L4_PV_to_L23_PV': -0.1,

            # ---------- L4 to L5 ----------
            'L4_E_to_L5_E': 0.1,
            'L4_E_to_L5_SST': 0.6,
            'L4_E_to_L5_PV': 0.1,
            'L4_SST_to_L5_E': 0.0,
            'L4_SST_to_L5_PV': 0.0,
            'L4_PV_to_L5_E': -0.1,
            'L4_PV_to_L5_SST': 0.0,
            'L4_PV_to_L5_PV': -0.1,

            # ---------- L5 (within L5) ----------
            'L5_E_to_L5_E': 0.4,
            'L5_E_to_L5_SST': 0.3,
            'L5_E_to_L5_PV': 0.4,
            'L5_SST_to_L5_E': -0.5,
            'L5_SST_to_L5_PV': -0.2,
            'L5_PV_to_L5_E': 0.0,
            'L5_PV_to_L5_SST': -0.2,
            'L5_PV_to_L5_PV': -0.3,

            # ---------- L5 to L2/3 ----------
            'L5_E_to_L23_E': 0.5,  
            'L5_E_to_L23_SST': 0.3,
            'L5_E_to_L23_PV': 0.4,
            'L5_SST_to_L23_E': 0.0,
            'L5_SST_to_L23_PV': 0.0,
            'L5_PV_to_L23_E': 0.0,
            'L5_PV_to_L23_SST': 0.0,
            'L5_PV_to_L23_PV': 0.0,

            # ---------- L5 to L4 ----------
            'L5_E_to_L4_E': 0.2,
            'L5_E_to_L4_SST': 0.2,
            'L5_E_to_L4_PV': 0.2,
            'L5_SST_to_L4_E': -0.6,
            'L5_SST_to_L4_PV': -0.5,
            'L5_PV_to_L4_E': -0.1,
            'L5_PV_to_L4_SST': 0.0,
            'L5_PV_to_L4_PV': 0.0,

            # ---------- Thalamic connections ----------
            'thalamus_to_L23_E': 0.1,
            'thalamus_to_L23_SST': 0.1,
            'thalamus_to_L23_PV': 0.1,
            'thalamus_to_L4_E': 0.5,
            'thalamus_to_L4_SST': 0.2,
            'thalamus_to_L4_PV': 0.3,
            'thalamus_to_L5_E': 0.3,
            'thalamus_to_L5_SST': 0.5,
            'thalamus_to_L5_PV': 0.2
        }
}

# ----------------- P10 PRESET (Mid 2nd Week) -----------------
P10_PRESET = {
    'time_constants': {
        'E': 9.0,  
        'SST': 20.0,
        'PV': 20.0
    },

    # Thalamic input widths (μm)
    'thalamic_widths': {
        'E': 50.0,
        'SST': 90.0,
        'PV': 75.0
    },

    # Outgoing widths (μm)
    'outgoing_widths': {
        'E': 140.0,   
        'SST': 75.0,
        'PV': 100.0
    },
    
    'strength_scaling': {
        'E': 2.5,      
        'SST': 1.5,    
        'PV': 1.5,     
        'thalamus': 3.0
    },

    'thalamic_alpha': 0.6, 
    
    'background_input': {
        'E': 0.0,
        'SST': 0.0,
        'PV': 0.0
    },

    'connection_strengths': {

            # ---------- L2/3 (within L2/3) ----------
            'L23_E_to_L23_E': 0.4,
            'L23_E_to_L23_SST': 0.6,
            'L23_E_to_L23_PV': 0.8,
            'L23_SST_to_L23_E': -0.4,   
            'L23_SST_to_L23_PV': -0.2,
            'L23_PV_to_L23_E': -0.2,    
            'L23_PV_to_L23_SST': -0.4,  
            'L23_PV_to_L23_PV': -0.4,   

            # ---------- L2/3 to L4 ----------
            'L23_E_to_L4_E': 0.2,        
            'L23_E_to_L4_SST': 0.2,
            'L23_E_to_L4_PV': 0.3,
            'L23_SST_to_L4_E': 0.0,
            'L23_SST_to_L4_PV': 0.0,
            'L23_PV_to_L4_E': -0.2,
            'L23_PV_to_L4_SST': 0.0,
            'L23_PV_to_L4_PV': 0.0,

            # ---------- L2/3 to L5 ----------
            'L23_E_to_L5_E': 0.5,
            'L23_E_to_L5_SST': 0.1,
            'L23_E_to_L5_PV': 0.3,
            'L23_SST_to_L5_E': -0.2,
            'L23_SST_to_L5_PV': 0.0,
            'L23_PV_to_L5_E': -0.1,
            'L23_PV_to_L5_SST': 0.0,
            'L23_PV_to_L5_PV': 0.0,

            # ---------- L4 (within L4) ----------
            'L4_E_to_L4_E': 0.5,
            'L4_E_to_L4_SST': 0.5,
            'L4_E_to_L4_PV': 0.8,
            'L4_SST_to_L4_E': -0.4,
            'L4_SST_to_L4_PV': -0.2,
            'L4_PV_to_L4_E': -0.2,
            'L4_PV_to_L4_SST': -0.2,
            'L4_PV_to_L4_PV': -0.4,

            # ---------- L4 to L2/3 ----------
            'L4_E_to_L23_E': 0.6,
            'L4_E_to_L23_SST': 0.2,
            'L4_E_to_L23_PV': 0.4,
            'L4_SST_to_L23_E': 0.0,
            'L4_SST_to_L23_PV': 0.0,
            'L4_PV_to_L23_E': -0.2,
            'L4_PV_to_L23_SST': 0.0,
            'L4_PV_to_L23_PV': -0.2,

            # ---------- L4 to L5 ----------
            'L4_E_to_L5_E': 0.1,
            'L4_E_to_L5_SST': 0.4,
            'L4_E_to_L5_PV': 0.2,
            'L4_SST_to_L5_E': -0.1,
            'L4_SST_to_L5_PV': 0.0,
            'L4_PV_to_L5_E': -0.2,
            'L4_PV_to_L5_SST': 0.0,
            'L4_PV_to_L5_PV': -0.2,

            # ---------- L5 (within L5) ----------
            'L5_E_to_L5_E': 0.4,
            'L5_E_to_L5_SST': 0.4,
            'L5_E_to_L5_PV': 0.8,
            'L5_SST_to_L5_E': -0.6,
            'L5_SST_to_L5_PV': -0.3,
            'L5_PV_to_L5_E': -0.4,
            'L5_PV_to_L5_SST': -0.3,
            'L5_PV_to_L5_PV': -0.6,

            # ---------- L5 to L2/3 ----------
            'L5_E_to_L23_E': 0.5,  
            'L5_E_to_L23_SST': 0.3,
            'L5_E_to_L23_PV': 0.5,
            'L5_SST_to_L23_E': -0.3,
            'L5_SST_to_L23_PV': 0.0,
            'L5_PV_to_L23_E': 0.0,
            'L5_PV_to_L23_SST': 0.0,
            'L5_PV_to_L23_PV': -0.2,

            # ---------- L5 to L4 ----------
            'L5_E_to_L4_E': 0.1,
            'L5_E_to_L4_SST': 0.1,
            'L5_E_to_L4_PV': 0.3,
            'L5_SST_to_L4_E': -0.4,
            'L5_SST_to_L4_PV': 0.0,
            'L5_PV_to_L4_E': -0.2,
            'L5_PV_to_L4_SST': 0.0,
            'L5_PV_to_L4_PV': 0.0,

            # ---------- Thalamic connections ----------
            'thalamus_to_L23_E': 0.1,
            'thalamus_to_L23_SST': 0.0,
            'thalamus_to_L23_PV': 0.1,
            'thalamus_to_L4_E': 0.8,
            'thalamus_to_L4_SST': 0.1,
            'thalamus_to_L4_PV': 0.5,
            'thalamus_to_L5_E': 0.2,
            'thalamus_to_L5_SST': 0.2,
            'thalamus_to_L5_PV': 0.2
        }

}

# ----------------- P15 PRESET (Late 2nd Week) -----------------
P15_PRESET = {
    'time_constants': {
        'E': 7.0,  
        'SST': 10.0,
        'PV': 5.0
    },

    # Thalamic input widths (μm)
    'thalamic_widths': {
        'E': 50.0,
        'SST': 60.0,
        'PV': 60.0
    },

    # Outgoing widths (μm)
    'outgoing_widths': {
        'E': 40.0,
        'SST': 50.0,  
        'PV': 60.0, 
    },
    
    'strength_scaling': {
        'E': 4.0,      
        'SST': 1.5,    
        'PV': 3.5,     
        'thalamus': 4.0
    },

    'thalamic_alpha': 0.8,
    
    'background_input': {
        'E': 0.1,
        'SST': 0.0,
        'PV': 0.0
    },

    'connection_strengths': {

            # ---------- L2/3 (within L2/3) ----------
            'L23_E_to_L23_E': 0.6,
            'L23_E_to_L23_SST': 0.8,
            'L23_E_to_L23_PV': 1.0,
            'L23_SST_to_L23_E': -0.3,   
            'L23_SST_to_L23_PV': -0.3,
            'L23_PV_to_L23_E': -0.8,    
            'L23_PV_to_L23_SST': -0.6,  
            'L23_PV_to_L23_PV': -0.8,   

            # ---------- L2/3 to L4 ----------
            'L23_E_to_L4_E': 0.3,        
            'L23_E_to_L4_SST': 0.3,
            'L23_E_to_L4_PV': 0.3,
            'L23_SST_to_L4_E': 0.0,
            'L23_SST_to_L4_PV': 0.0,
            'L23_PV_to_L4_E': -0.3,
            'L23_PV_to_L4_SST': 0.0,
            'L23_PV_to_L4_PV': 0.0,

            # ---------- L2/3 to L5 ----------
            'L23_E_to_L5_E': 1.0,
            'L23_E_to_L5_SST': 0.3,
            'L23_E_to_L5_PV': 0.6,
            'L23_SST_to_L5_E': -0.3,
            'L23_SST_to_L5_PV': 0.0,
            'L23_PV_to_L5_E': -0.1,
            'L23_PV_to_L5_SST': -0.1,
            'L23_PV_to_L5_PV': 0.0,

            # ---------- L4 (within L4) ----------
            'L4_E_to_L4_E': 0.6,
            'L4_E_to_L4_SST': 0.7,
            'L4_E_to_L4_PV': 1.0,
            'L4_SST_to_L4_E': -0.4,
            'L4_SST_to_L4_PV': -0.3,
            'L4_PV_to_L4_E': -0.6,
            'L4_PV_to_L4_SST': -0.4,
            'L4_PV_to_L4_PV': -0.6,

            # ---------- L4 to L2/3 ----------
            'L4_E_to_L23_E': 1.0,
            'L4_E_to_L23_SST': 0.2,
            'L4_E_to_L23_PV': 0.6,
            'L4_SST_to_L23_E': 0.0,
            'L4_SST_to_L23_PV': 0.0,
            'L4_PV_to_L23_E': -0.3,
            'L4_PV_to_L23_SST': 0.0,
            'L4_PV_to_L23_PV': -0.3,

            # ---------- L4 to L5 ----------
            'L4_E_to_L5_E': 0.1,
            'L4_E_to_L5_SST': 0.4,
            'L4_E_to_L5_PV': 0.3,
            'L4_SST_to_L5_E': -0.2,
            'L4_SST_to_L5_PV': 0.0,
            'L4_PV_to_L5_E': -0.3,
            'L4_PV_to_L5_SST': -0.1,
            'L4_PV_to_L5_PV': -0.3,

            # ---------- L5 (within L5) ----------
            'L5_E_to_L5_E': 0.8,
            'L5_E_to_L5_SST': 0.7,
            'L5_E_to_L5_PV': 1.0,
            'L5_SST_to_L5_E': -0.4,
            'L5_SST_to_L5_PV': -0.3,
            'L5_PV_to_L5_E': -1.2,
            'L5_PV_to_L5_SST': -0.2,
            'L5_PV_to_L5_PV': -0.8,

            # ---------- L5 to L2/3 ----------
            'L5_E_to_L23_E': 0.4,  
            'L5_E_to_L23_SST': 0.1,
            'L5_E_to_L23_PV': 0.2,
            'L5_SST_to_L23_E': -0.5,
            'L5_SST_to_L23_PV': 0.0,
            'L5_PV_to_L23_E': 0.0,
            'L5_PV_to_L23_SST': 0.0,
            'L5_PV_to_L23_PV': -0.3,

            # ---------- L5 to L4 ----------
            'L5_E_to_L4_E': 0.1,
            'L5_E_to_L4_SST': 0.1,
            'L5_E_to_L4_PV': 0.3,
            'L5_SST_to_L4_E': -0.1,
            'L5_SST_to_L4_PV': -0.1,
            'L5_PV_to_L4_E': -0.3,
            'L5_PV_to_L4_SST': 0.0,
            'L5_PV_to_L4_PV': 0.0,

            # ---------- Thalamic connections ----------
            'thalamus_to_L23_E': 0.1,
            'thalamus_to_L23_SST': 0.0,
            'thalamus_to_L23_PV': 0.1,
            'thalamus_to_L4_E': 1.0,
            'thalamus_to_L4_SST': 0.0,
            'thalamus_to_L4_PV': 0.6,
            'thalamus_to_L5_E': 0.2,
            'thalamus_to_L5_SST': 0.0,
            'thalamus_to_L5_PV': 0.2
        }   

}