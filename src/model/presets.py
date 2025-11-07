"""Developmental parameter presets for the cortical circuit simulation, updated."""

# ----------------- P4 PRESET (Early 1st Week) -----------------
P4_PRESET = {
    # Approximate time constants (ms). 
    'time_constants': {
        'E': 15.0,    # excitatory cells slower than adult
        'SST': 25.0,  # SST cells already somewhat active, but still slower
        'PV': 25.0    # prospective PV cells, minimal functional synapses
    },

    # Gains: relative excitability within [0,1]. 
    'gains': {
        'E': 1.0,
        'SST': 1.0,
        'PV': 1.0
    },
    
    # Constant background input for each cell type.
    'background_input': {
        'E': 0.0,
        'SST': 0.0,
        'PV': 0.0
    },

    # Thalamic input widths: how broadly thalamic input is spatially spread. 
    'thalamic_widths': {
        'E': 2.0,
        'SST': 1.5,
        'PV': 2.0
    },

    # Outgoing widths: how far each cell type's lateral connections spread.
    'outgoing_widths': {
        'E': 6.0,
        'SST': 6.0,
        'PV': 6.0
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

# ----------------- P8 PRESET (Late 1st Week) -----------------
P8_PRESET = {
    'time_constants': {
        'E': 14.0,
        'SST': 20.0,
        'PV': 20.0
    },
    'gains': {
        'E': 1.0,
        'SST': 1.0,
        'PV': 1.0
    },
    
    'background_input': {
        'E': 0.0,
        'SST': 0.0,
        'PV': 0.0
    },

    'thalamic_widths': {
        'E': 1.1,
        'SST': 1.5,
        'PV': 2.0
    },

    'outgoing_widths': {
        'E': 2.5,
        'SST': 6.0,
        'PV': 6.0
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

# ----------------- P12 PRESET (Mid 2nd Week) -----------------
P12_PRESET = {
    'time_constants': {
        'E': 7.0,  
        'SST': 20.0,
        'PV': 20.0
    },

    'gains': {
        'E': 1.0,    
        'SST': 1.0,  
        'PV': 1.0    
    },

    'thalamic_widths': {
        'E': 1.0,
        'SST': 1.8,
        'PV': 1.5
    },

    'outgoing_widths': {
        'E': 2.8,   
        'SST': 1.5,
        'PV': 2.0
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

# ----------------- P16 PRESET (Late 2nd Week) -----------------
P16_PRESET = {
    'time_constants': {
        'E': 9.0,  
        'SST': 10.0,
        'PV': 5.0
    },

    'gains': {
        'E': 1.0,   
        'SST': 1.0,
        'PV': 1.0   
    },

    'thalamic_widths': {
        'E': 1.0,
        'SST': 1.2,
        'PV': 1.2
    },

    'outgoing_widths': {
        'E': 0.8,
        'SST': 1.0,  
        'PV': 1.2, 
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