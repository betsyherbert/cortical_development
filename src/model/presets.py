"""Parameter presets for the cortical circuit simulation."""

# P4 preset configuration based on the image values
P4_PRESET = {
    # Time constants (ms)
    'time_constants': {
        'E': 80.0,
        'SST': 60.0,
        'PV': 40.0
    },
    
    # Gains
    'gains': {
        'E': 0.9,
        'SST': 0.8,
        'PV': 0.5
    },
    
    # Thalamic input widths (controlled by Thalamic Input Width)
    'thalamic_widths': {
        'E': 10.0,
        'SST': 10.0,
        'PV': 10.0
    },
    
    # Outgoing widths
    'outgoing_widths': {
        'E': 2.0,
        'SST': 2.0,
        'PV': 2.0
    },
    
    # Thalamic alpha (intrinsic/sensory tradeoff)
    'thalamic_alpha': 0.2,

    # Connection strengths from the matrix
    'connection_strengths': {
        # L2/3 connections
        'L23_E_to_L23_E': 0.1,
        'L23_E_to_L23_SST': 0.2,
        'L23_E_to_L23_PV': 0.0,
        'L23_SST_to_L23_E': 0.2,
        'L23_SST_to_L23_PV': 0.1,
        'L23_PV_to_L23_E': 0.0,
        'L23_PV_to_L23_SST': 0.0,
        'L23_PV_to_L23_PV': 0.0,

        # L2/3 to L4 connections
        'L23_E_to_L4_E': 0.1,
        'L23_E_to_L4_SST': 0.1,
        'L23_E_to_L4_PV': 0.0,
        'L23_SST_to_L4_E': 0.2,
        'L23_SST_to_L4_PV': 0.0,
        'L23_PV_to_L4_E': 0.0,
        'L23_PV_to_L4_SST': 0.0,
        'L23_PV_to_L4_PV': 0.0,

        # L2/3 to L5 connections
        'L23_E_to_L5_E': 0.1,
        'L23_E_to_L5_SST': 0.2,
        'L23_E_to_L5_PV': 0.0,
        'L23_SST_to_L5_E': 0.1,
        'L23_SST_to_L5_PV': 0.0,
        'L23_PV_to_L5_E': 0.0,
        'L23_PV_to_L5_SST': 0.0,
        'L23_PV_to_L5_PV': 0.0,

        # L4 connections
        'L4_E_to_L4_E': 0.1,
        'L4_E_to_L4_SST': 0.2,
        'L4_E_to_L4_PV': 0.0,
        'L4_SST_to_L4_E': 0.2,
        'L4_SST_to_L4_PV': 0.0,
        'L4_PV_to_L4_E': 0.0,
        'L4_PV_to_L4_SST': 0.0,
        'L4_PV_to_L4_PV': 0.0,

        # L4 to L2/3 connections
        'L4_E_to_L23_E': 0.1,
        'L4_E_to_L23_SST': 0.2,
        'L4_E_to_L23_PV': 0.0,
        'L4_SST_to_L23_E': 0.1,
        'L4_SST_to_L23_PV': 0.0,
        'L4_PV_to_L23_E': 0.0,
        'L4_PV_to_L23_SST': 0.0,
        'L4_PV_to_L23_PV': 0.0,

        # L4 to L5 connections
        'L4_E_to_L5_E': 0.1,
        'L4_E_to_L5_SST': 0.2,
        'L4_E_to_L5_PV': 0.0,
        'L4_SST_to_L5_E': 0.1,
        'L4_SST_to_L5_PV': 0.0,
        'L4_PV_to_L5_E': 0.0,
        'L4_PV_to_L5_SST': 0.0,
        'L4_PV_to_L5_PV': 0.0,

        # L5 connections
        'L5_E_to_L5_E': 0.1,
        'L5_E_to_L5_SST': 0.2,
        'L5_E_to_L5_PV': 0.0,
        'L5_SST_to_L5_E': 0.1,
        'L5_SST_to_L5_PV': 0.0,
        'L5_PV_to_L5_E': 0.0,
        'L5_PV_to_L5_SST': 0.0,
        'L5_PV_to_L5_PV': 0.0,

        # L5 to L2/3 connections
        'L5_E_to_L23_E': 0.1,
        'L5_E_to_L23_SST': 0.2,
        'L5_E_to_L23_PV': 0.0,
        'L5_SST_to_L23_E': 0.2,
        'L5_SST_to_L23_PV': 0.0,
        'L5_PV_to_L23_E': 0.0,
        'L5_PV_to_L23_SST': 0.0,
        'L5_PV_to_L23_PV': 0.0,

        # L5 to L4 connections
        'L5_E_to_L4_E': 0.1,
        'L5_E_to_L4_SST': 0.2,
        'L5_E_to_L4_PV': 0.0,
        'L5_SST_to_L4_E': 0.2,
        'L5_SST_to_L4_PV': 0.0,
        'L5_PV_to_L4_E': 0.0,
        'L5_PV_to_L4_SST': 0.0,
        'L5_PV_to_L4_PV': 0.0,

        # Thalamic connections
        'thalamus_to_L23_E': 0.3,
        'thalamus_to_L23_SST': 1.0,
        'thalamus_to_L23_PV': 0.0,
        'thalamus_to_L4_E': 0.2,
        'thalamus_to_L4_SST': 1.0,
        'thalamus_to_L4_PV': 0.0,
        'thalamus_to_L5_E': 0.1,
        'thalamus_to_L5_SST': 1.0,
        'thalamus_to_L5_PV': 0.0
    }
}

# P8 preset - emphasizing excitatory connections
P8_PRESET = {
    # Time constants (ms)
    'time_constants': {
        'E': 60.0,
        'SST': 40.0,
        'PV': 30.0
    },
    
    # Gains
    'gains': {
        'E': 0.7,
        'SST': 0.6,
        'PV': 0.4
    },
    
    # Thalamic input widths
    'thalamic_widths': {
        'E': 8.0,
        'SST': 8.0,
        'PV': 8.0
    },
    
    # Outgoing widths
    'outgoing_widths': {
        'E': 4.0,
        'SST': 4.0,
        'PV': 4.0
    },
    
    # Thalamic alpha (intrinsic/sensory tradeoff)
    'thalamic_alpha': 0.4,

    # Connection strengths - copied from P4 but can be customized later
    'connection_strengths': {
        # L2/3 connections
        'L23_E_to_L23_E': 0.4,
        'L23_E_to_L23_SST': 0.5,
        'L23_E_to_L23_PV': 0.3,
        'L23_SST_to_L23_E': -0.5,
        'L23_SST_to_L23_PV': -0.3,
        'L23_PV_to_L23_E': -0.4,
        'L23_PV_to_L23_SST': -0.2,
        'L23_PV_to_L23_PV': 0.5,

        # L2/3 to L4 connections
        'L23_E_to_L4_E': 0.5,
        'L23_E_to_L4_SST': 0.4,
        'L23_E_to_L4_PV': 0.3,
        'L23_SST_to_L4_E': -0.5,
        'L23_SST_to_L4_PV': -0.2,
        'L23_PV_to_L4_E': -0.4,
        'L23_PV_to_L4_SST': -0.2,
        'L23_PV_to_L4_PV': 0.5,

        # L2/3 to L5 connections
        'L23_E_to_L5_E': 0.4,
        'L23_E_to_L5_SST': 0.5,
        'L23_E_to_L5_PV': 0.3,
        'L23_SST_to_L5_E': -0.4,
        'L23_SST_to_L5_PV': -0.2,
        'L23_PV_to_L5_E': -0.4,
        'L23_PV_to_L5_SST': -0.2,
        'L23_PV_to_L5_PV': 0.5,

        # L4 connections
        'L4_E_to_L4_E': 0.6,
        'L4_E_to_L4_SST': 0.5,
        'L4_E_to_L4_PV': 0.5,
        'L4_SST_to_L4_E': -0.6,
        'L4_SST_to_L4_PV': -0.3,
        'L4_PV_to_L4_E': -0.7,
        'L4_PV_to_L4_SST': -0.3,
        'L4_PV_to_L4_PV': 0.6,

        # L4 to L2/3 connections
        'L4_E_to_L23_E': 0.7,
        'L4_E_to_L23_SST': 0.6,
        'L4_E_to_L23_PV': 0.6,
        'L4_SST_to_L23_E': -0.5,
        'L4_SST_to_L23_PV': -0.3,
        'L4_PV_to_L23_E': -0.8,
        'L4_PV_to_L23_SST': -0.3,
        'L4_PV_to_L23_PV': 0.7,

        # L4 to L5 connections
        'L4_E_to_L5_E': 0.6,
        'L4_E_to_L5_SST': 0.5,
        'L4_E_to_L5_PV': 0.5,
        'L4_SST_to_L5_E': -0.5,
        'L4_SST_to_L5_PV': -0.3,
        'L4_PV_to_L5_E': -0.7,
        'L4_PV_to_L5_SST': -0.3,
        'L4_PV_to_L5_PV': 0.6,

        # L5 connections
        'L5_E_to_L5_E': 0.5,
        'L5_E_to_L5_SST': 0.5,
        'L5_E_to_L5_PV': 0.4,
        'L5_SST_to_L5_E': -0.4,
        'L5_SST_to_L5_PV': -0.3,
        'L5_PV_to_L5_E': -0.6,
        'L5_PV_to_L5_SST': -0.3,
        'L5_PV_to_L5_PV': 0.5,

        # L5 to L2/3 connections
        'L5_E_to_L23_E': 0.6,
        'L5_E_to_L23_SST': 0.5,
        'L5_E_to_L23_PV': 0.4,
        'L5_SST_to_L23_E': -0.5,
        'L5_SST_to_L23_PV': -0.3,
        'L5_PV_to_L23_E': -0.6,
        'L5_PV_to_L23_SST': -0.3,
        'L5_PV_to_L23_PV': 0.5,

        # L5 to L4 connections
        'L5_E_to_L4_E': 0.7,
        'L5_E_to_L4_SST': 0.6,
        'L5_E_to_L4_PV': 0.5,
        'L5_SST_to_L4_E': -0.5,
        'L5_SST_to_L4_PV': -0.3,
        'L5_PV_to_L4_E': -0.7,
        'L5_PV_to_L4_SST': -0.3,
        'L5_PV_to_L4_PV': 0.6,

        # Thalamic connections
        'thalamus_to_L23_E': 0.5,
        'thalamus_to_L23_SST': 0.4,
        'thalamus_to_L23_PV': 0.3,
        'thalamus_to_L4_E': 0.8,
        'thalamus_to_L4_SST': 0.2,
        'thalamus_to_L4_PV': 0.5,
        'thalamus_to_L5_E': 0.4,
        'thalamus_to_L5_SST': 0.2,
        'thalamus_to_L5_PV': 0.3
    }
}

# P12 preset - emphasizing inhibitory control
P12_PRESET = {
    # Time constants (ms)
    'time_constants': {
        'E': 40.0,
        'SST': 20.0,
        'PV': 10.0
    },
    
    # Gains
    'gains': {
        'E': 0.5,
        'SST': 0.5,
        'PV': 0.25
    },
    
    # Thalamic input widths
    'thalamic_widths': {
        'E': 4.0,
        'SST': 4.0,
        'PV': 4.0
    },
    
    # Outgoing widths
    'outgoing_widths': {
        'E': 8.0,
        'SST': 8.0,
        'PV': 8.0
    },
    
    # Thalamic alpha (intrinsic/sensory tradeoff)
    'thalamic_alpha': 0.6,

    # Connection strengths 
    'connection_strengths': {
        # L2/3 connections
        'L23_E_to_L23_E': 0.8,
        'L23_E_to_L23_SST': 0.9,
        'L23_E_to_L23_PV': 0.8,
        'L23_SST_to_L23_E': -0.7,
        'L23_SST_to_L23_PV': -0.5,
        'L23_PV_to_L23_E': -0.9,
        'L23_PV_to_L23_SST': -0.5,
        'L23_PV_to_L23_PV': 0.8,

        # L2/3 to L4 connections
        'L23_E_to_L4_E': 0.8,
        'L23_E_to_L4_SST': 0.8,
        'L23_E_to_L4_PV': 0.7,
        'L23_SST_to_L4_E': -0.7,
        'L23_SST_to_L4_PV': -0.5,
        'L23_PV_to_L4_E': -0.9,
        'L23_PV_to_L4_SST': -0.5,
        'L23_PV_to_L4_PV': 0.8,

        # L2/3 to L5 connections
        'L23_E_to_L5_E': 0.7,
        'L23_E_to_L5_SST': 0.8,
        'L23_E_to_L5_PV': 0.7,
        'L23_SST_to_L5_E': -0.6,
        'L23_SST_to_L5_PV': -0.4,
        'L23_PV_to_L5_E': -0.8,
        'L23_PV_to_L5_SST': -0.4,
        'L23_PV_to_L5_PV': 0.7,

        # L4 connections
        'L4_E_to_L4_E': 0.9,
        'L4_E_to_L4_SST': 0.9,
        'L4_E_to_L4_PV': 0.9,
        'L4_SST_to_L4_E': -0.8,
        'L4_SST_to_L4_PV': -0.5,
        'L4_PV_to_L4_E': -1.0,
        'L4_PV_to_L4_SST': -0.5,
        'L4_PV_to_L4_PV': 0.9,

        # L4 to L2/3 connections
        'L4_E_to_L23_E': 0.9,
        'L4_E_to_L23_SST': 0.9,
        'L4_E_to_L23_PV': 0.9,
        'L4_SST_to_L23_E': -0.7,
        'L4_SST_to_L23_PV': -0.5,
        'L4_PV_to_L23_E': -1.0,
        'L4_PV_to_L23_SST': -0.5,
        'L4_PV_to_L23_PV': 0.9,

        # L4 to L5 connections
        'L4_E_to_L5_E': 0.8,
        'L4_E_to_L5_SST': 0.8,
        'L4_E_to_L5_PV': 0.8,
        'L4_SST_to_L5_E': -0.7,
        'L4_SST_to_L5_PV': -0.5,
        'L4_PV_to_L5_E': -0.9,
        'L4_PV_to_L5_SST': -0.5,
        'L4_PV_to_L5_PV': 0.8,

        # L5 connections
        'L5_E_to_L5_E': 0.8,
        'L5_E_to_L5_SST': 0.9,
        'L5_E_to_L5_PV': 0.8,
        'L5_SST_to_L5_E': -0.7,
        'L5_SST_to_L5_PV': -0.5,
        'L5_PV_to_L5_E': -0.9,
        'L5_PV_to_L5_SST': -0.5,
        'L5_PV_to_L5_PV': 0.8,

        # L5 to L2/3 connections
        'L5_E_to_L23_E': 0.9,
        'L5_E_to_L23_SST': 0.9,
        'L5_E_to_L23_PV': 0.8,
        'L5_SST_to_L23_E': -0.7,
        'L5_SST_to_L23_PV': -0.5,
        'L5_PV_to_L23_E': -0.9,
        'L5_PV_to_L23_SST': -0.5,
        'L5_PV_to_L23_PV': 0.8,

        # L5 to L4 connections
        'L5_E_to_L4_E': 0.9,
        'L5_E_to_L4_SST': 0.9,
        'L5_E_to_L4_PV': 0.8,
        'L5_SST_to_L4_E': -0.7,
        'L5_SST_to_L4_PV': -0.5,
        'L5_PV_to_L4_E': -0.9,
        'L5_PV_to_L4_SST': -0.5,
        'L5_PV_to_L4_PV': 0.8,

        # Thalamic connections
        'thalamus_to_L23_E': 1.0,
        'thalamus_to_L23_SST': 0.1,
        'thalamus_to_L23_PV': 0.8,
        'thalamus_to_L4_E': 1.0,
        'thalamus_to_L4_SST': 0.0,
        'thalamus_to_L4_PV': 0.8,
        'thalamus_to_L5_E': 0.8,
        'thalamus_to_L5_SST': 0.0,
        'thalamus_to_L5_PV': 0.8
    }
}

# P16 preset - emphasizing thalamic influence
P16_PRESET = {
    # Time constants (ms)
    'time_constants': {
        'E': 20.0,
        'SST': 10.0,
        'PV': 5.0
    },
    
    # Gains
    'gains': {
        'E': 0.35,
        'SST': 0.4,
        'PV': 0.15
    },
    
    # Thalamic input widths
    'thalamic_widths': {
        'E': 2.0,
        'SST': 2.0,
        'PV': 2.0
    },
    
    # Outgoing widths
    'outgoing_widths': {
        'E': 10.0,
        'SST': 10.0,
        'PV': 10.0
    },
    
    # Thalamic alpha (intrinsic/sensory tradeoff)
    'thalamic_alpha': 0.8,

    # Connection strengths 
    'connection_strengths': {
        # L2/3 connections
        'L23_E_to_L23_E': 1.0,
        'L23_E_to_L23_SST': 1.0,
        'L23_E_to_L23_PV': 1.0,
        'L23_SST_to_L23_E': -0.8,
        'L23_SST_to_L23_PV': -0.5,
        'L23_PV_to_L23_E': -1.0,
        'L23_PV_to_L23_SST': -0.5,
        'L23_PV_to_L23_PV': 1.0,

        # L2/3 to L4 connections
        'L23_E_to_L4_E': 1.0,
        'L23_E_to_L4_SST': 1.0,
        'L23_E_to_L4_PV': 1.0,
        'L23_SST_to_L4_E': -0.8,
        'L23_SST_to_L4_PV': -0.5,
        'L23_PV_to_L4_E': -1.0,
        'L23_PV_to_L4_SST': -0.5,
        'L23_PV_to_L4_PV': 1.0,

        # L2/3 to L5 connections
        'L23_E_to_L5_E': 1.0,
        'L23_E_to_L5_SST': 1.0,
        'L23_E_to_L5_PV': 1.0,
        'L23_SST_to_L5_E': -0.8,
        'L23_SST_to_L5_PV': -0.5,
        'L23_PV_to_L5_E': -1.0,
        'L23_PV_to_L5_SST': -0.5,
        'L23_PV_to_L5_PV': 1.0,

        # L4 connections
        'L4_E_to_L4_E': 1.0,
        'L4_E_to_L4_SST': 1.0,
        'L4_E_to_L4_PV': 1.0,
        'L4_SST_to_L4_E': -0.8,
        'L4_SST_to_L4_PV': -0.5,
        'L4_PV_to_L4_E': -1.0,
        'L4_PV_to_L4_SST': -0.5,
        'L4_PV_to_L4_PV': 1.0,

        # L4 to L2/3 connections
        'L4_E_to_L23_E': 1.0,
        'L4_E_to_L23_SST': 1.0,
        'L4_E_to_L23_PV': 1.0,
        'L4_SST_to_L23_E': -0.8,
        'L4_SST_to_L23_PV': -0.5,
        'L4_PV_to_L23_E': -1.0,
        'L4_PV_to_L23_SST': -0.5,
        'L4_PV_to_L23_PV': 1.0,

        # L4 to L5 connections
        'L4_E_to_L5_E': 1.0,
        'L4_E_to_L5_SST': 1.0,
        'L4_E_to_L5_PV': 1.0,
        'L4_SST_to_L5_E': -0.8,
        'L4_SST_to_L5_PV': -0.5,
        'L4_PV_to_L5_E': -1.0,
        'L4_PV_to_L5_SST': -0.5,
        'L4_PV_to_L5_PV': 1.0,

        # L5 connections
        'L5_E_to_L5_E': 1.0,
        'L5_E_to_L5_SST': 1.0,
        'L5_E_to_L5_PV': 1.0,
        'L5_SST_to_L5_E': -0.8,
        'L5_SST_to_L5_PV': -0.5,
        'L5_PV_to_L5_E': -1.0,
        'L5_PV_to_L5_SST': -0.5,
        'L5_PV_to_L5_PV': 1.0,

        # L5 to L2/3 connections
        'L5_E_to_L23_E': 1.0,
        'L5_E_to_L23_SST': 1.0,
        'L5_E_to_L23_PV': 1.0,
        'L5_SST_to_L23_E': -0.8,
        'L5_SST_to_L23_PV': -0.5,
        'L5_PV_to_L23_E': -1.0,
        'L5_PV_to_L23_SST': -0.5,
        'L5_PV_to_L23_PV': 1.0,

        # L5 to L4 connections
        'L5_E_to_L4_E': 1.0,
        'L5_E_to_L4_SST': 1.0,
        'L5_E_to_L4_PV': 1.0,
        'L5_SST_to_L4_E': -0.8,
        'L5_SST_to_L4_PV': -0.5,
        'L5_PV_to_L4_E': -1.0,
        'L5_PV_to_L4_SST': -0.5,
        'L5_PV_to_L4_PV': 1.0,

        # Thalamic connections
        'thalamus_to_L23_E': 1.0,
        'thalamus_to_L23_SST': 0.0,
        'thalamus_to_L23_PV': 1.0,
        'thalamus_to_L4_E': 1.0,
        'thalamus_to_L4_SST': 0.0,
        'thalamus_to_L4_PV': 1.0,
        'thalamus_to_L5_E': 1.0,
        'thalamus_to_L5_SST': 0.0,
        'thalamus_to_L5_PV': 1.0
    }
} 