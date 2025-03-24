"""Parameter presets for the cortical circuit simulation."""

# P4 preset configuration based on the image values
P4_PRESET = {
    # Time constants (ms)
    'time_constants': {
        'E': 80.0,
        'SST': 60.0,
        'PV': 40.0
    },
    
    # Firing thresholds
    'firing_thresholds': {
        'E': 0.05,
        'SST': 0.1,
        'PV': 0.05
    },
    
    # Thalamic input widths (controlled by Thalamic Input Width)
    'thalamic_widths': {
        'E': 6.0,
        'SST': 6.0,
        'PV': 6.0
    },
    
    # Outgoing widths
    'outgoing_widths': {
        'E': 4.0,
        'SST': 4.0,
        'PV': 4.0
    },
    
    # Thalamic alpha (intrinsic/sensory tradeoff)
    'thalamic_alpha': 0.2,

    # Connection strengths from the matrix
    'connection_strengths': {
        # L2/3 connections
        'L23_E_to_L23_E': 0.2,
        'L23_E_to_L23_SST': 0.1,
        'L23_E_to_L23_PV': 0.2,
        'L23_SST_to_L23_E': 0.1,
        'L23_SST_to_L23_PV': 0.0,
        'L23_PV_to_L23_E': 0.0,
        'L23_PV_to_L23_SST': 0.0,
        'L23_PV_to_L23_PV': 0.0,

        # L2/3 to L4 connections
        'L23_E_to_L4_E': 0.1,
        'L23_E_to_L4_SST': 0.1,
        'L23_E_to_L4_PV': 0.1,
        'L23_SST_to_L4_E': 0.0,
        'L23_SST_to_L4_PV': 0.0,
        'L23_PV_to_L4_E': 0.0,
        'L23_PV_to_L4_SST': 0.0,
        'L23_PV_to_L4_PV': 0.0,

        # L2/3 to L5 connections
        'L23_E_to_L5_E': 0.5,
        'L23_E_to_L5_SST': 0.1,
        'L23_E_to_L5_PV': 0.1,
        'L23_SST_to_L5_E': 0.2,
        'L23_SST_to_L5_PV': 0.0,
        'L23_PV_to_L5_E': 0.0,
        'L23_PV_to_L5_SST': 0.0,
        'L23_PV_to_L5_PV': 0.0,

        # L4 connections
        'L4_E_to_L4_E': 0.1,
        'L4_E_to_L4_SST': 0.1,
        'L4_E_to_L4_PV': 0.1,
        'L4_SST_to_L4_E': 0.0,
        'L4_SST_to_L4_PV': 0.0,
        'L4_PV_to_L4_E': 0.0,
        'L4_PV_to_L4_SST': 0.0,
        'L4_PV_to_L4_PV': 0.0,

        # L4 to L2/3 connections
        'L4_E_to_L23_E': 0.1,
        'L4_E_to_L23_SST': 0.1,
        'L4_E_to_L23_PV': 0.1,
        'L4_SST_to_L23_E': 0.1,
        'L4_SST_to_L23_PV': 0.0,
        'L4_PV_to_L23_E': 0.0,
        'L4_PV_to_L23_SST': 0.0,
        'L4_PV_to_L23_PV': 0.0,

        # L4 to L5 connections
        'L4_E_to_L5_E': 0.1,
        'L4_E_to_L5_SST': 0.1,
        'L4_E_to_L5_PV': 0.1,
        'L4_SST_to_L5_E': 0.5,
        'L4_SST_to_L5_PV': 0.5,
        'L4_PV_to_L5_E': 0.0,
        'L4_PV_to_L5_SST': 0.0,
        'L4_PV_to_L5_PV': 0.0,

        # L5 connections
        'L5_E_to_L5_E': 0.1,
        'L5_E_to_L5_SST': 0.1,
        'L5_E_to_L5_PV': 0.1,
        'L5_SST_to_L5_E': 0.5,
        'L5_SST_to_L5_PV': 0.5,
        'L5_PV_to_L5_E': 0.0,
        'L5_PV_to_L5_SST': 0.0,
        'L5_PV_to_L5_PV': 0.0,

        # L5 to L2/3 connections
        'L5_E_to_L23_E': 0.5,
        'L5_E_to_L23_SST': 0.1,
        'L5_E_to_L23_PV': 0.1,
        'L5_SST_to_L23_E': 0.2,
        'L5_SST_to_L23_PV': 0.0,
        'L5_PV_to_L23_E': 0.0,
        'L5_PV_to_L23_SST': 0.0,
        'L5_PV_to_L23_PV': 0.0,

        # L5 to L4 connections
        'L5_E_to_L4_E': 0.1,
        'L5_E_to_L4_SST': 0.1,
        'L5_E_to_L4_PV': 0.1,
        'L5_SST_to_L4_E': 0.5,
        'L5_SST_to_L4_PV': 0.5,
        'L5_PV_to_L4_E': 0.0,
        'L5_PV_to_L4_SST': 0.0,
        'L5_PV_to_L4_PV': 0.0,

        # Thalamic connections
        'thalamus_to_L23_E': 0.1,
        'thalamus_to_L23_SST': 0.0,
        'thalamus_to_L23_PV': 0.0,
        'thalamus_to_L4_E': 0.5,
        'thalamus_to_L4_SST': 0.3,
        'thalamus_to_L4_PV': 0.1,
        'thalamus_to_L5_E': 0.1,
        'thalamus_to_L5_SST': 0.5,
        'thalamus_to_L5_PV': 0.1
    }
}

# P8 preset - emphasizing excitatory connections
P8_PRESET = {
    # Time constants (ms)
    'time_constants': {
        'E': 60.0,
        'SST': 70.0,
        'PV': 50.0
    },
    
    # Firing thresholds
    'firing_thresholds': {
        'E': 0.1,
        'SST': 0.15,
        'PV': 0.1
    },
    
    # Thalamic input widths
    'thalamic_widths': {
        'E': 5.0,
        'SST': 5.0,
        'PV': 5.0
    },
    
    # Outgoing widths
    'outgoing_widths': {
        'E': 5.0,
        'SST': 3.0,
        'PV': 3.0
    },
    
    # Thalamic alpha (intrinsic/sensory tradeoff)
    'thalamic_alpha': 0.4,

    # Connection strengths - copied from P4 but can be customized later
    'connection_strengths': P4_PRESET['connection_strengths'].copy()
}

# P12 preset - emphasizing inhibitory control
P12_PRESET = {
    # Time constants (ms)
    'time_constants': {
        'E': 70.0,
        'SST': 40.0,
        'PV': 30.0
    },
    
    # Firing thresholds
    'firing_thresholds': {
        'E': 0.15,
        'SST': 0.05,
        'PV': 0.05
    },
    
    # Thalamic input widths
    'thalamic_widths': {
        'E': 4.0,
        'SST': 8.0,
        'PV': 8.0
    },
    
    # Outgoing widths
    'outgoing_widths': {
        'E': 3.0,
        'SST': 6.0,
        'PV': 6.0
    },
    
    # Thalamic alpha (intrinsic/sensory tradeoff)
    'thalamic_alpha': 0.6,

    # Connection strengths - copied from P4 but can be customized later
    'connection_strengths': P4_PRESET['connection_strengths'].copy()
}

# P16 preset - emphasizing thalamic influence
P16_PRESET = {
    # Time constants (ms)
    'time_constants': {
        'E': 90.0,
        'SST': 80.0,
        'PV': 60.0
    },
    
    # Firing thresholds
    'firing_thresholds': {
        'E': 0.03,
        'SST': 0.08,
        'PV': 0.03
    },
    
    # Thalamic input widths
    'thalamic_widths': {
        'E': 8.0,
        'SST': 7.0,
        'PV': 7.0
    },
    
    # Outgoing widths
    'outgoing_widths': {
        'E': 3.0,
        'SST': 3.0,
        'PV': 3.0
    },
    
    # Thalamic alpha (intrinsic/sensory tradeoff)
    'thalamic_alpha': 0.8,

    # Connection strengths - copied from P4 but can be customized later
    'connection_strengths': P4_PRESET['connection_strengths'].copy()
} 