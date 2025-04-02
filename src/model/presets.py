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
        'E': 1.0,
        'SST': 1.0,
        'PV': 1.0
    },

    # Thalamic input widths: how broadly thalamic input is spatially spread. 
    # For P4, still fairly diffuse. 
    'thalamic_widths': {
        'E': 4.0,
        'SST': 4.0,
        'PV': 2.0
    },

    # Outgoing widths: how far each cell type's lateral connections spread.
    # E and SST can be relatively broad. PV is minimal at P4.
    'outgoing_widths': {
        'E': 7.0,
        'SST': 4.0,
        'PV': 2.0
    },
    
    # Strength scaling factors: overall scaling of connection strengths.
    # At P4, connections are generally weaker and more variable.
    'strength_scaling': {
        'E': 1.0,      # E cells have weak but present connections
        'SST': 1.0,    # SST cells have weaker connections
        'PV': 1.0,     # PV cells have very weak connections 
        'thalamus': 1.0  # Thalamic input is present but not fully developed
    },
    
    # Sparsity factors: fraction of connections present (1 = all, 0 = none).
    # At P4, connectivity is quite sparse.
    'sparsity': {
        'E': 1.0,      # E cells have moderately sparse connectivity
        'SST': 1.0,    # SST cells have sparser connectivity
        'PV': 1.0,     # PV cells have very sparse connectivity
        'thalamus': 1.0  # Thalamic connectivity is more complete but still developing
    },

    # Thalamic alpha: ratio between sensory-driven vs. intrinsic input.
    # In neonates, spontaneous (intrinsic) drive is still dominant, so alpha is small.
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
    
    # Strength scaling factors: connections becoming stronger.
    # At P8, connections are strengthening.
    'strength_scaling': {
        'E': 2.0,      # E cells have stronger connections
        'SST': 2.0,    # SST cells have stronger connections
        'PV': 3.0,     # PV cells have developing connections
        'thalamus': 2.0  # Thalamic input is stronger
    },
    
    # Sparsity factors: connectivity increasing.
    # At P8, connectivity is becoming less sparse.
    'sparsity': {
        'E': 1.0,      # E cells have less sparse connectivity
        'SST': 0.2,    # SST cells have less sparse connectivity
        'PV': 1.0,     # PV cells have developing connectivity
        'thalamus': 1.0  # Thalamic connectivity is more complete
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
        'E': 20.0,   # faster now
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
        'E': 4.0,    # still moderate horizontal connectivity
        'SST': 6.0,
        'PV': 4.0
    },
    
    # Strength scaling factors: connections now close to mature strength.
    # At P12, connections are significantly strengthened.
    'strength_scaling': {
        'E': 3.0,      # E cells have almost mature strength
        'SST': 2.0,    # SST cells have strong connections
        'PV': 4.0,     # PV cells now with significant connectivity
        'thalamus': 2.0  # Thalamic input close to mature levels
    },
    
    # Sparsity factors: connectivity more complete.
    # At P12, connectivity is more complete.
    'sparsity': {
        'E': 1.0,      # E cells have more complete connectivity
        'SST': 1.0,    # SST cells have more complete connectivity
        'PV': 1.0,     # PV cells have developing but substantial connectivity
        'thalamus': 1.0  # Thalamic connectivity almost complete
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
        'E': 15.0,  # near adult
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
        'E': 2.0,
        'SST': 7.0,  # Martinotti can be broad
        'PV': 5.0
    },
    
    # Strength scaling factors: mature connection strength.
    # At P16, connections are at mature strength.
    'strength_scaling': {
        'E': 4.0,      # E cells have mature strength
        'SST': 2.0,    # SST cells have mature strength
        'PV': 5.0,     # PV cells have mature strength
        'thalamus': 2.0  # Thalamic input at mature levels
    },
    
    # Sparsity factors: mature connectivity pattern.
    # At P16, connectivity is mature.
    'sparsity': {
        'E': 1.0,      # E cells have mature connectivity
        'SST': 1.0,    # SST cells have nearly complete connectivity
        'PV': 1.0,     # PV cells have mature connectivity pattern
        'thalamus': 1.0  # Thalamic connectivity complete
    },

    'thalamic_alpha': 0.7,  # mostly sensory-driven at P16
    
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
