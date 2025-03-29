"""Developmental parameter presets for the cortical circuit simulation, updated."""

# ----------------- P4 PRESET (Early 1st Week) -----------------
P4_PRESET = {
    # Approximate time constants (ms). Neonatal neurons have relatively large tau.
    # PV is quite immature; many PV cells are essentially silent.
    'time_constants': {
        'E': 40.0,    # excitatory cells slower than adult
        'SST': 30.0,  # SST cells already somewhat active, but still slower
        'PV': 15.0    # prospective PV cells, minimal functional synapses
    },

    # Gains: relative excitability within [0..1]. 
    # E is fairly excitable early on, SST also active (gets strong early thalamic drive). 
    # PV has low functional gain at P4.
    'gains': {
        'E': 0.8,
        'SST': 0.7,
        'PV': 0.6
    },

    # Thalamic input widths: how broadly thalamic input is spatially spread. 
    # For P4, still fairly diffuse. 
    'thalamic_widths': {
        'E': 4.0,
        'SST': 4.0,
        'PV': 2.0
    },

    # Outgoing widths: how far each cell type’s lateral connections spread.
    # E and SST can be relatively broad. PV is minimal at P4.
    'outgoing_widths': {
        'E': 7.0,
        'SST': 4.0,
        'PV': 2.0
    },

    # Thalamic alpha: ratio between sensory-driven vs. intrinsic input.
    # In neonates, spontaneous (intrinsic) drive is still dominant, so alpha is small.
    'thalamic_alpha': 0.2,
    
    'connection_strengths': {

            # ---------- L2/3 (within L2/3) ----------
            'L23_E_to_L23_E': 0.1,
            'L23_E_to_L23_SST': 0.1,
            'L23_E_to_L23_PV': 0.1,
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
            'L5_SST_to_L23_PV': 0.0,
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
    # Time constants reduce somewhat. PV is still not fully FS, but more active than P4.
    'time_constants': {
        'E': 30.0,
        'SST': 20.0,
        'PV': 12.0
    },

    # Gains: E and SST remain fairly high, PV is rising now.
    'gains': {
        'E': 0.7,
        'SST': 0.6,
        'PV': 0.6
    },

    # Thalamic input widths shrink slightly as barrels refine; still moderate at P8.
    'thalamic_widths': {
        'E': 3.0,
        'SST': 3.0,
        'PV': 2.0
    },

    # Outgoing widths: L2/3 & L4 excitatory still fairly broad, but narrower than at P4. 
    'outgoing_widths': {
        'E': 6.0,
        'SST': 5.0,
        'PV': 3.0
    },

    # Thalamic alpha. Sensory-driven input is more important by P8, but intrinsic still present.
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
            'L4_E_to_L4_SST': 0.0,
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
            'L5_PV_to_L23_E': -0.3,
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
        'E': 15.0,   # faster now
        'SST': 12.0,
        'PV': 8.0
    },

    'gains': {
        'E': 0.5,    # excitability lower than at P8
        'SST': 0.6,  # moderate
        'PV': 0.8    # PV is more fully active
    },

    'thalamic_widths': {
        'E': 2.0,
        'SST': 2.0,
        'PV': 2.0
    },

    'outgoing_widths': {
        'E': 5.0,    # still moderate horizontal connectivity
        'SST': 6.0,
        'PV': 4.0
    },

    'thalamic_alpha': 0.5,  # more weighting on sensory input
        
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
            'L4_E_to_L4_E': 0.6,
            'L4_E_to_L4_SST': 0.2,
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
            'L4_E_to_L5_SST': 0.3,
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
        'E': 12.0,  # near adult
        'SST': 10.0,
        'PV': 6.0
    },

    'gains': {
        'E': 0.4,   # lowered excitability
        'SST': 0.6,
        'PV': 0.8   # PV is quite potent now
    },

    # Thalamic input narrower, strongly columnar
    'thalamic_widths': {
        'E': 2.0,
        'SST': 1.0,
        'PV': 1.0
    },

    # Outgoing widths: more local for E, though L2/3 horizontal can remain moderate
    'outgoing_widths': {
        'E': 4.0,
        'SST': 7.0,  # Martinotti can be broad
        'PV': 5.0
    },

    'thalamic_alpha': 0.7,  # mostly sensory-driven at P16
    
    'connection_strengths': {

            # ---------- L2/3 (within L2/3) ----------
            'L23_E_to_L23_E': 0.8,
            'L23_E_to_L23_SST': 1.0,
            'L23_E_to_L23_PV': 1.0,
            'L23_SST_to_L23_E': -1.0,   
            'L23_SST_to_L23_PV': -0.3,
            'L23_PV_to_L23_E': -1.0,    
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
            'L4_E_to_L4_E': 0.8,
            'L4_E_to_L4_SST': 0.4,
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
            'L4_E_to_L5_SST': 0.1,
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
            'L5_SST_to_L23_E': -0.2,
            'L5_SST_to_L23_PV': 0.0,
            'L5_PV_to_L23_E': -0.3,
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
