"""Developmental parameter presets for the cortical circuit simulation, updated."""

# ----------------- P4 PRESET (Early 1st Week) -----------------
P4_PRESET = {
    # Approximate time constants (ms). 
    'time_constants': {
        'E': 20.0,    # excitatory cells slower than adult
        'SST': 20.0,  # SST cells already somewhat active, but still slower
        'PV': 20.0    # prospective PV cells, minimal functional synapses
    },

    # Gains: relative excitability within [0,1]. 
    'gains': {
        'E': 1.0,
        'SST': 1.0,
        'PV': 1.0
    },
    
    # Input noise parameters for each cell type.
    'noise_params': {
        'E': {
            'mean': 0.0,
            'std': 0.2,  
            'c': 0.4     
        },
        'SST': {
            'mean': 0.0, 
            'std': 0.2,  
            'c': 0.4     
        },
        'PV': {
            'mean': 0.0,   
            'std': 0.2,    
            'c': 0.4     
        }
    },

    # Thalamic input widths: how broadly thalamic input is spatially spread. 
    'thalamic_widths': {
        'E': 4.0,
        'SST': 4.0,
        'PV': 4.0
    },

    # Outgoing widths: how far each cell type's lateral connections spread.
    'outgoing_widths': {
        'E': 3.0,
        'SST': 4.0,
        'PV': 4.0
    },
    
    # Strength scaling factors: overall scaling of connection strengths.
    'strength_scaling': {
        'E': 1.0,      # E cells have weak but present connections
        'SST': 1.0,    # SST cells have weaker connections
        'PV': 1.0,     # PV cells have very weak connections 
        'thalamus': 0.0  # Thalamic input is present but not fully developed
    },
    
    # Sparsity factors: fraction of connections present (1 = all, 0 = none).
    'sparsity': {
        'E': 1.0,      # E cells have moderately sparse connectivity
        'SST': 1.0,    # SST cells have sparser connectivity
        'PV': 1.0,     # PV cells have very sparse connectivity
        'thalamus': 1.0  # Thalamic connectivity is more complete but still developing
    },

    # Thalamic alpha: ratio between sensory-driven vs. intrinsic input.
    'thalamic_alpha': 0.2,
    
    'connection_strengths': {

            # ---------- L2/3 (within L2/3) ----------
            'L23_E_to_L23_E': 0.1,
            'L23_E_to_L23_SST': 0.1,
            'L23_E_to_L23_PV': 0.2,
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
            'L4_E_to_L4_E': 0.1,
            'L4_E_to_L4_SST': 0.0,
            'L4_E_to_L4_PV': 0.1,
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
            'L4_SST_to_L5_PV': 0.3,
            'L4_PV_to_L5_E': 0.0,
            'L4_PV_to_L5_SST': 0.0,
            'L4_PV_to_L5_PV': 0.0,

            # ---------- L5 (within L5) ----------
            'L5_E_to_L5_E': 0.1,
            'L5_E_to_L5_SST': 0.1,
            'L5_E_to_L5_PV': 0.1,
            'L5_SST_to_L5_E': 0.2,
            'L5_SST_to_L5_PV': 0.5,
            'L5_PV_to_L5_E': 0.0,
            'L5_PV_to_L5_SST': 0.0,
            'L5_PV_to_L5_PV': 0.0,

            # ---------- L5 to L2/3 ----------
            'L5_E_to_L23_E': 1.0,  
            'L5_E_to_L23_SST': 0.1,
            'L5_E_to_L23_PV': 0.0,
            'L5_SST_to_L23_E': 0.0,
            'L5_SST_to_L23_PV': 0.2,
            'L5_PV_to_L23_E': 0.0,
            'L5_PV_to_L23_SST': 0.0,
            'L5_PV_to_L23_PV': 0.0,

            # ---------- L5 to L4 ----------
            'L5_E_to_L4_E': 0.1,
            'L5_E_to_L4_SST': 0.1,
            'L5_E_to_L4_PV': 0.0,
            'L5_SST_to_L4_E': 0.5,
            'L5_SST_to_L4_PV': 0.5,
            'L5_PV_to_L4_E': 0.0,
            'L5_PV_to_L4_SST': 0.0,
            'L5_PV_to_L4_PV': 0.0,

            # ---------- Thalamic connections ----------
            'thalamus_to_L23_E': 0.0,
            'thalamus_to_L23_SST': 0.2,
            'thalamus_to_L23_PV': 0.0,
            'thalamus_to_L4_E': 0.2,
            'thalamus_to_L4_SST': 0.5,
            'thalamus_to_L4_PV': 0.0,
            'thalamus_to_L5_E': 0.2,
            'thalamus_to_L5_SST': 1.0,
            'thalamus_to_L5_PV': 0.1
        }
}

# ----------------- P8 PRESET (Late 1st Week) -----------------
P8_PRESET = {
    'time_constants': {
        'E': 10.0,
        'SST': 20.0,
        'PV': 20.0
    },
    'gains': {
        'E': 1.0,
        'SST': 1.0,
        'PV': 1.0
    },
    
    # Input noise parameters for each cell type.
    'noise_params': {
        'E': {
            'mean': 0.0,
            'std': 0.2,  
            'c': 0.4     
        },
        'SST': {
            'mean': 0.0, 
            'std': 0.2,  
            'c': 0.4     
        },
        'PV': {
            'mean': 0.0,   
            'std': 0.2,    
            'c': 0.4     
        }
    },

    'thalamic_widths': {
        'E': 3.0,
        'SST': 3.0,
        'PV': 3.0
    },

    'outgoing_widths': {
        'E': 2.0,
        'SST': 4.0,
        'PV': 4.0
    },

    'strength_scaling': {
        'E': 1.0,    
        'SST': 1.0,  
        'PV': 1.0,   
        'thalamus': 0.0
    },
    
    'sparsity': {
        'E': 1.0,      
        'SST': 1.0,    
        'PV': 1.0,     
        'thalamus': 1.0 
    },

    'thalamic_alpha': 0.3,
        
    'connection_strengths': {

            # ---------- L2/3 (within L2/3) ----------
            'L23_E_to_L23_E': 0.3,
            'L23_E_to_L23_SST': 0.3,
            'L23_E_to_L23_PV': 0.3,
            'L23_SST_to_L23_E': -0.3,   
            'L23_SST_to_L23_PV': -0.7,
            'L23_PV_to_L23_E': -0.3,    
            'L23_PV_to_L23_SST': -0.1,  
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
            'L23_E_to_L5_PV': 0.0,
            'L23_SST_to_L5_E': -0.1,
            'L23_SST_to_L5_PV': 0.0,
            'L23_PV_to_L5_E': -0.1,
            'L23_PV_to_L5_SST': 0.0,
            'L23_PV_to_L5_PV': 0.0,

            # ---------- L4 (within L4) ----------
            'L4_E_to_L4_E': 0.3,
            'L4_E_to_L4_SST': 0.2,
            'L4_E_to_L4_PV': 0.4,
            'L4_SST_to_L4_E': -0.2,
            'L4_SST_to_L4_PV': -0.1,
            'L4_PV_to_L4_E': -0.3,
            'L4_PV_to_L4_SST': -0.1,
            'L4_PV_to_L4_PV': -0.3,

            # ---------- L4 to L2/3 ----------
            'L4_E_to_L23_E': 0.3,
            'L4_E_to_L23_SST': 0.1,
            'L4_E_to_L23_PV': 0.2,
            'L4_SST_to_L23_E': 0.0,
            'L4_SST_to_L23_PV': 0.0,
            'L4_PV_to_L23_E': -0.1,
            'L4_PV_to_L23_SST': 0.0,
            'L4_PV_to_L23_PV': 0.0,

            # ---------- L4 to L5 ----------
            'L4_E_to_L5_E': 0.1,
            'L4_E_to_L5_SST': 0.6,
            'L4_E_to_L5_PV': 0.0,
            'L4_SST_to_L5_E': 0.0,
            'L4_SST_to_L5_PV': -0.5,
            'L4_PV_to_L5_E': -0.1,
            'L4_PV_to_L5_SST': 0.0,
            'L4_PV_to_L5_PV': -0.1,

            # ---------- L5 (within L5) ----------
            'L5_E_to_L5_E': 0.3,
            'L5_E_to_L5_SST': 0.3,
            'L5_E_to_L5_PV': 0.2,
            'L5_SST_to_L5_E': -0.3,
            'L5_SST_to_L5_PV': -0.7,
            'L5_PV_to_L5_E': -0.3,
            'L5_PV_to_L5_SST': -0.1,
            'L5_PV_to_L5_PV': -0.3,

            # ---------- L5 to L2/3 ----------
            'L5_E_to_L23_E': 0.7,  
            'L5_E_to_L23_SST': 0.2,
            'L5_E_to_L23_PV': 0.1,
            'L5_SST_to_L23_E': -0.1,
            'L5_SST_to_L23_PV': 0.0,
            'L5_PV_to_L23_E': 0.0,
            'L5_PV_to_L23_SST': 0.0,
            'L5_PV_to_L23_PV': 0.0,

            # ---------- L5 to L4 ----------
            'L5_E_to_L4_E': 0.2,
            'L5_E_to_L4_SST': 0.2,
            'L5_E_to_L4_PV': 0.3,
            'L5_SST_to_L4_E': -0.5,
            'L5_SST_to_L4_PV': -0.5,
            'L5_PV_to_L4_E': -0.1,
            'L5_PV_to_L4_SST': 0.0,
            'L5_PV_to_L4_PV': 0.0,

            # ---------- Thalamic connections ----------
            'thalamus_to_L23_E': 0.0,
            'thalamus_to_L23_SST': 0.1,
            'thalamus_to_L23_PV': 0.0,
            'thalamus_to_L4_E': 0.4,
            'thalamus_to_L4_SST': 0.4,
            'thalamus_to_L4_PV': 0.3,
            'thalamus_to_L5_E': 0.2,
            'thalamus_to_L5_SST': 0.5,
            'thalamus_to_L5_PV': 0.1
        }
}

# ----------------- P12 PRESET (Mid 2nd Week) -----------------
P12_PRESET = {
    'time_constants': {
        'E': 10.0,  
        'SST': 15.0,
        'PV': 15.0
    },

    'gains': {
        'E': 1.0,    
        'SST': 1.0,  
        'PV': 1.0    
    },

    'thalamic_widths': {
        'E': 2.0,
        'SST': 2.0,
        'PV': 2.0
    },

    'outgoing_widths': {
        'E': 2.0,   
        'SST': 3.0,
        'PV': 3.0
    },
    
    'strength_scaling': {
        'E': 1.0,      
        'SST': 1.0,    
        'PV': 1.0,     
        'thalamus': 0.0
    },
    
    'sparsity': {
        'E': 1.0,     
        'SST': 1.0,   
        'PV': 1.0,    
        'thalamus': 1.0 
    },

    'thalamic_alpha': 0.5, 
    
    'noise_params': {
        'E': {
            'mean': 0.0,
            'std': 0.2,  
            'c': 0.4     
        },
        'SST': {
            'mean': 0.0, 
            'std': 0.2,  
            'c': 0.4     
        },
        'PV': {
            'mean': 0.0,   
            'std': 0.2,    
            'c': 0.4     
        }
    },

    'connection_strengths': {

            # ---------- L2/3 (within L2/3) ----------
            'L23_E_to_L23_E': 0.6,
            'L23_E_to_L23_SST': 0.5,
            'L23_E_to_L23_PV': 0.5,
            'L23_SST_to_L23_E': -0.7,   
            'L23_SST_to_L23_PV': -0.5,
            'L23_PV_to_L23_E': -0.7,    
            'L23_PV_to_L23_SST': -0.2,  
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
            'L23_E_to_L5_PV': 0.1,
            'L23_SST_to_L5_E': -0.2,
            'L23_SST_to_L5_PV': 0.0,
            'L23_PV_to_L5_E': -0.2,
            'L23_PV_to_L5_SST': 0.0,
            'L23_PV_to_L5_PV': 0.0,

            # ---------- L4 (within L4) ----------
            'L4_E_to_L4_E': 0.4,
            'L4_E_to_L4_SST': 0.6,
            'L4_E_to_L4_PV': 0.8,
            'L4_SST_to_L4_E': -0.5,
            'L4_SST_to_L4_PV': -0.2,
            'L4_PV_to_L4_E': -0.8,
            'L4_PV_to_L4_SST': -0.2,
            'L4_PV_to_L4_PV': -0.5,

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
            'L4_E_to_L5_PV': 0.1,
            'L4_SST_to_L5_E': -0.1,
            'L4_SST_to_L5_PV': -0.3,
            'L4_PV_to_L5_E': -0.2,
            'L4_PV_to_L5_SST': 0.0,
            'L4_PV_to_L5_PV': -0.2,

            # ---------- L5 (within L5) ----------
            'L5_E_to_L5_E': 0.6,
            'L5_E_to_L5_SST': 0.6,
            'L5_E_to_L5_PV': 0.6,
            'L5_SST_to_L5_E': -0.6,
            'L5_SST_to_L5_PV': -0.3,
            'L5_PV_to_L5_E': -0.8,
            'L5_PV_to_L5_SST': -0.2,
            'L5_PV_to_L5_PV': -0.5,

            # ---------- L5 to L2/3 ----------
            'L5_E_to_L23_E': 0.4,  
            'L5_E_to_L23_SST': 0.3,
            'L5_E_to_L23_PV': 0.2,
            'L5_SST_to_L23_E': -0.2,
            'L5_SST_to_L23_PV': 0.0,
            'L5_PV_to_L23_E': -0.3,
            'L5_PV_to_L23_SST': 0.0,
            'L5_PV_to_L23_PV': -0.2,

            # ---------- L5 to L4 ----------
            'L5_E_to_L4_E': 0.1,
            'L5_E_to_L4_SST': 0.1,
            'L5_E_to_L4_PV': 0.4,
            'L5_SST_to_L4_E': -0.3,
            'L5_SST_to_L4_PV': -0.2,
            'L5_PV_to_L4_E': -0.2,
            'L5_PV_to_L4_SST': 0.0,
            'L5_PV_to_L4_PV': 0.0,

            # ---------- Thalamic connections ----------
            'thalamus_to_L23_E': 0.1,
            'thalamus_to_L23_SST': 0.0,
            'thalamus_to_L23_PV': 0.0,
            'thalamus_to_L4_E': 0.8,
            'thalamus_to_L4_SST': 0.2,
            'thalamus_to_L4_PV': 0.5,
            'thalamus_to_L5_E': 0.2,
            'thalamus_to_L5_SST': 0.3,
            'thalamus_to_L5_PV': 0.2
        }

}

# ----------------- P16 PRESET (Late 2nd Week) -----------------
P16_PRESET = {
    'time_constants': {
        'E': 10.0,  
        'SST': 10.0,
        'PV': 10.0
    },

    'gains': {
        'E': 1.0,   
        'SST': 1.0,
        'PV': 1.0   
    },

    'thalamic_widths': {
        'E': 1.0,
        'SST': 1.0,
        'PV': 1.0
    },

    'outgoing_widths': {
        'E': 1.0,
        'SST': 2.0,  
        'PV': 2.0
    },
    
    'strength_scaling': {
        'E': 1.0,      
        'SST': 1.0,    
        'PV': 1.0,     
        'thalamus': 0.0
    },
    
    'sparsity': {
        'E': 1.0,      
        'SST': 1.0,    
        'PV': 1.0,     
        'thalamus': 1.0  
    },

    'thalamic_alpha': 0.7,
    
    'noise_params': {
        'E': {
            'mean': 0.0,
            'std': 0.2,  
            'c': 0.4     
        },
        'SST': {
            'mean': 0.0, 
            'std': 0.2,  
            'c': 0.4     
        },
        'PV': {
            'mean': 0.0,   
            'std': 0.2,    
            'c': 0.4     
        }
    },

    'connection_strengths': {

            # ---------- L2/3 (within L2/3) ----------
            'L23_E_to_L23_E': 0.8,
            'L23_E_to_L23_SST': 1.0,
            'L23_E_to_L23_PV': 1.0,
            'L23_SST_to_L23_E': -0.7,   
            'L23_SST_to_L23_PV': -0.3,
            'L23_PV_to_L23_E': -0.6,    
            'L23_PV_to_L23_SST': -0.2,  
            'L23_PV_to_L23_PV': -0.5,   

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
            'L23_E_to_L5_SST': 0.2,
            'L23_E_to_L5_PV': 0.2,
            'L23_SST_to_L5_E': -0.3,
            'L23_SST_to_L5_PV': 0.0,
            'L23_PV_to_L5_E': -0.3,
            'L23_PV_to_L5_SST': -0.1,
            'L23_PV_to_L5_PV': 0.0,

            # ---------- L4 (within L4) ----------
            'L4_E_to_L4_E': 0.5,
            'L4_E_to_L4_SST': 1.0,
            'L4_E_to_L4_PV': 1.0,
            'L4_SST_to_L4_E': -0.7,
            'L4_SST_to_L4_PV': -0.3,
            'L4_PV_to_L4_E': -1.0,
            'L4_PV_to_L4_SST': -0.2,
            'L4_PV_to_L4_PV': -0.7,

            # ---------- L4 to L2/3 ----------
            'L4_E_to_L23_E': 1.0,
            'L4_E_to_L23_SST': 0.3,
            'L4_E_to_L23_PV': 0.5,
            'L4_SST_to_L23_E': 0.0,
            'L4_SST_to_L23_PV': 0.0,
            'L4_PV_to_L23_E': -0.3,
            'L4_PV_to_L23_SST': 0.0,
            'L4_PV_to_L23_PV': -0.4,

            # ---------- L4 to L5 ----------
            'L4_E_to_L5_E': 0.1,
            'L4_E_to_L5_SST': 0.3,
            'L4_E_to_L5_PV': 0.2,
            'L4_SST_to_L5_E': -0.2,
            'L4_SST_to_L5_PV': -0.1,
            'L4_PV_to_L5_E': -0.3,
            'L4_PV_to_L5_SST': -0.1,
            'L4_PV_to_L5_PV': -0.3,

            # ---------- L5 (within L5) ----------
            'L5_E_to_L5_E': 0.8,
            'L5_E_to_L5_SST': 1.0,
            'L5_E_to_L5_PV': 1.0,
            'L5_SST_to_L5_E': -1.0,
            'L5_SST_to_L5_PV': -0.3,
            'L5_PV_to_L5_E': -1.0,
            'L5_PV_to_L5_SST': -0.2,
            'L5_PV_to_L5_PV': -0.7,

            # ---------- L5 to L2/3 ----------
            'L5_E_to_L23_E': 0.2,  
            'L5_E_to_L23_SST': 0.1,
            'L5_E_to_L23_PV': 0.2,
            'L5_SST_to_L23_E': 0.0,
            'L5_SST_to_L23_PV': 0.0,
            'L5_PV_to_L23_E': 0.0,
            'L5_PV_to_L23_SST': 0.0,
            'L5_PV_to_L23_PV': -0.3,

            # ---------- L5 to L4 ----------
            'L5_E_to_L4_E': 0.1,
            'L5_E_to_L4_SST': 0.1,
            'L5_E_to_L4_PV': 0.5,
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
            'thalamus_to_L4_SST': 0.1,
            'thalamus_to_L4_PV': 0.7,
            'thalamus_to_L5_E': 0.2,
            'thalamus_to_L5_SST': 0.1,
            'thalamus_to_L5_PV': 0.2
        }

}
