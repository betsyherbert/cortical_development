"""Core stability analysis implementation."""

import numpy as np

from src.main import CorticalSimulation

from .config import (
    ANALYSIS_PARAMS,
    CELL_TYPES,
    DEVELOPMENTAL_STAGES,
    DT,
    LAYERS,
    PRESETS,
    REGIMES,
)


class StabilityAnalysis:
    """Main class for stability analysis of cortical circuits."""

    def __init__(self):
        """Initialize analysis with simulation instance."""
        self.simulation = CorticalSimulation()
        self.params = ANALYSIS_PARAMS

        self.snapshots = {}
        self.lambda_max_results = {}

    def collect_snapshots(self, stage: str) -> dict[str, list[dict]]:
        """Collect snapshots for idle and driven regimes."""
        print(f"Collecting snapshots for {stage}...")

        # Reset simulation state and random seed to ensure reproducible inputs across stages
        self.simulation.reset()

        preset = PRESETS[stage]
        self._apply_preset(preset)

        print("  Running adaptive snapshot collection...")

        n_steps = int(self.params["duration"] * 1000 / DT)
        sample_interval = max(1, n_steps // 200)

        thalamic_buffer = []
        candidate_states = {"idle": [], "driven": []}

        snapshots = {"idle": [], "driven": []}
        thresholds_computed = False
        idle_threshold = driven_threshold = 0

        for step in range(n_steps):
            activities = self.simulation.update()
            thalamic_sum = np.sum(activities["thalamus"])

            thalamic_buffer.append(thalamic_sum)

            if step % sample_interval == 0 and step < n_steps // 2:
                state = self._capture_current_state(step, thalamic_sum, activities["thalamus"])
                candidate_states["early_buffer"] = candidate_states.get("early_buffer", [])
                candidate_states["early_buffer"].append((step, thalamic_sum, state))

            if not thresholds_computed and step > n_steps // 4:
                idle_threshold = np.percentile(thalamic_buffer, self.params["percentiles"][0])
                driven_threshold = np.percentile(thalamic_buffer, self.params["percentiles"][1])
                thresholds_computed = True

                if "early_buffer" in candidate_states:
                    for frame_idx, thal_sum, state in candidate_states["early_buffer"]:
                        if (
                            thal_sum <= idle_threshold
                            and len(snapshots["idle"]) < self.params["n_snapshots"]
                        ):
                            snapshots["idle"].append(state)
                        elif (
                            thal_sum >= driven_threshold
                            and len(snapshots["driven"]) < self.params["n_snapshots"]
                        ):
                            snapshots["driven"].append(state)
                    del candidate_states["early_buffer"]

                print(
                    f"    Thresholds: idle ≤ {idle_threshold:.2f}, driven ≥ {driven_threshold:.2f}"
                )

            if thresholds_computed:
                if (
                    thalamic_sum <= idle_threshold
                    and len(snapshots["idle"]) < self.params["n_snapshots"]
                    and step % max(1, sample_interval // 2) == 0
                ):

                    state = self._capture_current_state(step, thalamic_sum, activities["thalamus"])
                    snapshots["idle"].append(state)
                    print(
                        f"    Captured idle snapshot {len(snapshots['idle'])}/{self.params['n_snapshots']} at step {step}"
                    )

                elif (
                    thalamic_sum >= driven_threshold
                    and len(snapshots["driven"]) < self.params["n_snapshots"]
                    and step % max(1, sample_interval // 2) == 0
                ):

                    state = self._capture_current_state(step, thalamic_sum, activities["thalamus"])
                    snapshots["driven"].append(state)
                    print(
                        f"    Captured driven snapshot {len(snapshots['driven'])}/{self.params['n_snapshots']} at step {step}"
                    )

                if (
                    len(snapshots["idle"]) >= self.params["n_snapshots"]
                    and len(snapshots["driven"]) >= self.params["n_snapshots"]
                ):
                    print(f"    Early termination at step {step}/{n_steps}")
                    break

        print(
            f"  Collected {len(snapshots.get('idle', []))} idle and {len(snapshots.get('driven', []))} driven snapshots"
        )
        return snapshots

    def _apply_preset(self, preset: dict):
        """Apply developmental preset to simulation."""
        for key, value in preset["connection_strengths"].items():
            self.simulation.circuit.connectivity.layer_params[key] = {
                "amplitude": value,
                "sigma": 2.0,
            }

        self.simulation.circuit.connectivity.update_weights()

        for cell_type in CELL_TYPES:
            if cell_type in preset["time_constants"]:
                self.simulation.set_time_constant(cell_type, preset["time_constants"][cell_type])

        # Update thalamic developmental parameters
        self.simulation.update_thalamic_params(preset)

    def _capture_current_state(
        self, frame_idx: int, thalamic_sum: float, thalamic_input: np.ndarray
    ) -> dict:
        """Capture current network state for analysis."""
        voltages = {}
        for layer in LAYERS:
            voltages[layer] = {}
            for cell_type in CELL_TYPES:
                voltages[layer][cell_type] = (
                    self.simulation.circuit.layers[layer].V[cell_type].copy()
                )

        return {
            "frame_idx": frame_idx,
            "thalamic_sum": thalamic_sum,
            "voltages": voltages,
            "thalamic_input": thalamic_input.copy(),
            "time_constants": self.simulation.get_time_constants(),
            "gains": self.simulation.get_gains(),
        }

    def compute_jacobian(
        self,
        patch_voltages: np.ndarray,
        patch_connections: np.ndarray,
        time_constants: np.ndarray,
        gains: np.ndarray,
    ) -> np.ndarray:
        """Compute voltage Jacobian matrix for a patch."""
        heaviside = (patch_voltages > 0).astype(float)

        gains_heaviside = gains * heaviside
        jacobian = (
            patch_connections * gains_heaviside[np.newaxis, :] / time_constants[:, np.newaxis]
        )

        np.fill_diagonal(jacobian, -1.0 / time_constants)

        return jacobian

    def analyze_patch(
        self, snapshot: dict, patch_coords: tuple, analysis_type: str, condition: str = None
    ):
        """
        Analyze a single patch for stability.

        If condition is provided, returns lambda_max for that condition.
        If condition is None, returns dict with both lambda values and regime classification.
        """
        layers_to_analyze = LAYERS if analysis_type == "layer" else [LAYERS]
        patch_size = self.params[f"{analysis_type}_patch_size"]

        if condition is not None:
            # Legacy mode: return lambda_max for specific condition
            results = {}
            for layer_group in layers_to_analyze:
                if analysis_type == "layer":
                    layer = layer_group
                    patch_data = self._extract_patch_data(
                        snapshot, [layer], patch_coords, patch_size, condition
                    )
                    lambda_max = self._compute_lambda_max(patch_data)
                    results[layer] = lambda_max
                else:
                    patch_data = self._extract_patch_data(
                        snapshot, LAYERS, patch_coords, patch_size, condition
                    )
                    lambda_max = self._compute_lambda_max(patch_data)
                    results = lambda_max
            return results
        else:
            # New mode: return full analysis with regime classification
            results = {}
            for layer_group in layers_to_analyze:
                if analysis_type == "layer":
                    layer = layer_group
                    # Get lambda values for all conditions
                    patch_data_full = self._extract_patch_data(
                        snapshot, [layer], patch_coords, patch_size, "full"
                    )
                    lambda_full = self._compute_lambda_max(patch_data_full)

                    patch_data_e_only = self._extract_patch_data(
                        snapshot, [layer], patch_coords, patch_size, "e_only"
                    )
                    lambda_e_only = self._compute_lambda_max(patch_data_e_only)

                    patch_data_e_pv_only = self._extract_patch_data(
                        snapshot, [layer], patch_coords, patch_size, "e_pv_only"
                    )
                    lambda_e_pv_only = self._compute_lambda_max(patch_data_e_pv_only)

                    patch_data_e_sst_only = self._extract_patch_data(
                        snapshot, [layer], patch_coords, patch_size, "e_sst_only"
                    )
                    lambda_e_sst_only = self._compute_lambda_max(patch_data_e_sst_only)

                    # Classify regime
                    regime = self._classify_stability_regime(lambda_full, lambda_e_only)

                    results[layer] = {
                        "lambda_full": lambda_full,
                        "lambda_e_only": lambda_e_only,
                        "lambda_e_pv_only": lambda_e_pv_only,
                        "lambda_e_sst_only": lambda_e_sst_only,
                        "regime": regime,
                    }
                else:
                    # Column-wise analysis
                    patch_data_full = self._extract_patch_data(
                        snapshot, LAYERS, patch_coords, patch_size, "full"
                    )
                    lambda_full = self._compute_lambda_max(patch_data_full)

                    patch_data_e_only = self._extract_patch_data(
                        snapshot, LAYERS, patch_coords, patch_size, "e_only"
                    )
                    lambda_e_only = self._compute_lambda_max(patch_data_e_only)

                    patch_data_e_pv_only = self._extract_patch_data(
                        snapshot, LAYERS, patch_coords, patch_size, "e_pv_only"
                    )
                    lambda_e_pv_only = self._compute_lambda_max(patch_data_e_pv_only)

                    patch_data_e_sst_only = self._extract_patch_data(
                        snapshot, LAYERS, patch_coords, patch_size, "e_sst_only"
                    )
                    lambda_e_sst_only = self._compute_lambda_max(patch_data_e_sst_only)

                    # Classify regime
                    regime = self._classify_stability_regime(lambda_full, lambda_e_only)

                    results = {
                        "lambda_full": lambda_full,
                        "lambda_e_only": lambda_e_only,
                        "lambda_e_pv_only": lambda_e_pv_only,
                        "lambda_e_sst_only": lambda_e_sst_only,
                        "regime": regime,
                    }

            return results

    def _compute_lambda_max(self, patch_data: tuple) -> float:
        """Compute lambda_max from patch data."""
        patch_voltages, patch_connections, time_constants, gains = patch_data
        jacobian = self.compute_jacobian(patch_voltages, patch_connections, time_constants, gains)
        eigenvalues = np.linalg.eigvals(jacobian)
        return np.max(np.real(eigenvalues))

    def _extract_patch_data(
        self,
        snapshot: dict,
        layers: list[str],
        patch_coords: tuple,
        patch_size: int,
        condition: str,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Extract data for patch analysis."""
        start_x, start_y = patch_coords

        n_layers = len(layers)
        n_neurons = n_layers * patch_size * patch_size * len(CELL_TYPES)
        patch_voltages = np.zeros(n_neurons)
        time_constants = np.zeros(n_neurons)
        gains = np.zeros(n_neurons)

        idx = 0
        for layer in layers:
            for dx in range(patch_size):
                for dy in range(patch_size):
                    x, y = start_x + dx, start_y + dy
                    for cell_type in CELL_TYPES:
                        patch_voltages[idx] = snapshot["voltages"][layer][cell_type][x, y]
                        time_constants[idx] = snapshot["time_constants"][cell_type]
                        gains[idx] = snapshot["gains"][cell_type]
                        idx += 1

        # Include a hash of current connection strengths in cache key to ensure
        # different developmental stages get different cached matrices
        conn_strengths = tuple(
            sorted(
                (k, v["amplitude"])
                for k, v in self.simulation.circuit.connectivity.layer_params.items()
            )
        )
        connectivity_hash = hash(conn_strengths)
        cache_key = (tuple(layers), patch_size, condition, connectivity_hash)
        if not hasattr(self, "_connection_cache"):
            self._connection_cache = {}

        if cache_key not in self._connection_cache:
            self._connection_cache[cache_key] = self._build_connection_matrix(
                layers, patch_size, condition
            )

        patch_connections = self._connection_cache[cache_key]

        return patch_voltages, patch_connections, time_constants, gains

    def _build_connection_matrix(
        self, layers: list[str], patch_size: int, condition: str
    ) -> np.ndarray:
        """Build connection matrix for patch."""
        n_neurons = len(layers) * patch_size * patch_size * len(CELL_TYPES)
        patch_connections = np.zeros((n_neurons, n_neurons))
        connectivity = self.simulation.circuit.connectivity

        def get_neuron_idx(layer_idx, dx, dy, cell_type_idx):
            return (
                layer_idx * patch_size * patch_size * len(CELL_TYPES)
                + dx * patch_size * len(CELL_TYPES)
                + dy * len(CELL_TYPES)
                + cell_type_idx
            )

        inhibitory_mask = {}
        if condition in ["e_only", "e_pv_only", "e_sst_only"]:
            for src_cell in CELL_TYPES:
                for tgt_cell in CELL_TYPES:
                    inhibitory_mask[(src_cell, tgt_cell)] = self._is_inhibitory_connection(
                        src_cell, tgt_cell, condition
                    )

        for layer_idx, layer in enumerate(layers):
            for dx in range(patch_size):
                for dy in range(patch_size):
                    for src_cell_idx, src_cell in enumerate(CELL_TYPES):
                        src_idx = get_neuron_idx(layer_idx, dx, dy, src_cell_idx)

                        for tgt_layer_idx, tgt_layer in enumerate(layers):
                            for tgt_dx in range(patch_size):
                                for tgt_dy in range(patch_size):
                                    for tgt_cell_idx, tgt_cell in enumerate(CELL_TYPES):
                                        tgt_idx = get_neuron_idx(
                                            tgt_layer_idx, tgt_dx, tgt_dy, tgt_cell_idx
                                        )

                                        conn_key = f"{layer}_{src_cell}_to_{tgt_layer}_{tgt_cell}"

                                        if conn_key in connectivity.layer_params:
                                            strength = connectivity.layer_params[conn_key][
                                                "amplitude"
                                            ]

                                            if condition in [
                                                "e_only",
                                                "e_pv_only",
                                                "e_sst_only",
                                            ] and inhibitory_mask.get((src_cell, tgt_cell), False):
                                                strength = 0.0

                                            patch_connections[tgt_idx, src_idx] = strength

        return patch_connections

    def _is_inhibitory_connection(
        self, source_cell: str, target_cell: str, condition: str = "e_only"
    ) -> bool:
        """Check if connection should be zeroed based on condition."""
        if condition == "e_only":
            # Original E-only condition: remove all inhibitory connections
            inhibitory_cells = ["SST", "PV"]

            if source_cell in inhibitory_cells and target_cell == "E":
                return True
            if source_cell in inhibitory_cells and target_cell in inhibitory_cells:
                return True
            if source_cell == "E" and target_cell in inhibitory_cells:
                return True

        elif condition == "e_pv_only":
            # E + PV only: remove SST connections
            if source_cell == "SST":
                return True
            if target_cell == "SST":
                return True

        elif condition == "e_sst_only":
            # E + SST only: remove PV connections
            if source_cell == "PV":
                return True
            if target_cell == "PV":
                return True

        return False

    def _classify_stability_regime(self, lambda_full: float, lambda_e_only: float) -> str:
        """
        Classify patch into interpretable stability regime.

        Args:
            lambda_full: Lambda_max for full network (with inhibition)
            lambda_e_only: Lambda_max for E-only network (without inhibition)

        Returns:
            Stability regime category (matching colorbar order from bottom to top)
        """
        if lambda_full < 0 and lambda_e_only > 0:
            return "inhibition \n stabilised"  # Inhibition rescues unstable E-circuit
        elif lambda_full < 0 and lambda_e_only < 0:
            return "intrinsically \n stable"  # Stable even without inhibition
        elif lambda_full > 0 and lambda_e_only > 0:
            return "intrinsically \n unstable"  # Unstable despite inhibition
        elif lambda_full > 0 and lambda_e_only < 0:
            return "inhibition \n destabilised"  # Inhibition makes it worse (rare!)
        else:
            return "unknown"  # Edge case

    def run_analysis(self) -> dict:
        """Run complete stability analysis for all conditions."""
        print("Starting stability analysis...")

        results = {}

        for stage in DEVELOPMENTAL_STAGES:
            print(f"\nAnalyzing {stage}...")
            results[stage] = {}

            snapshots = self.collect_snapshots(stage)
            self.snapshots[stage] = snapshots

            for regime in REGIMES:
                if regime not in snapshots:
                    continue

                results[stage][regime] = {}

                for snap_idx, snapshot in enumerate(snapshots[regime]):
                    print(f"  Processing {regime} snapshot {snap_idx}...")
                    results[stage][regime][snap_idx] = self._analyze_snapshot(snapshot)

        self.lambda_max_results = results
        return results

    def _generate_patch_coords(self, patch_size: int) -> list[tuple[int, int]]:
        """Generate valid patch coordinates."""
        cache_key = patch_size
        if not hasattr(self, "_coord_cache"):
            self._coord_cache = {}

        if cache_key not in self._coord_cache:
            boundary = self.params["boundary_exclude"]
            grid_size = self.simulation.grid_size

            x_range = np.arange(boundary, grid_size - boundary - patch_size + 1)
            y_range = np.arange(boundary, grid_size - boundary - patch_size + 1)
            xx, yy = np.meshgrid(x_range, y_range)
            coords = list(zip(xx.flatten(), yy.flatten(), strict=False))

            self._coord_cache[cache_key] = coords

        return self._coord_cache[cache_key]

    def _analyze_snapshot(self, snapshot: dict) -> dict:
        """Analyze a single snapshot for all patches and conditions."""
        layer_coords = self._generate_patch_coords(self.params["layer_patch_size"])
        column_coords = self._generate_patch_coords(self.params["column_patch_size"])

        results = {
            "layer_wise": {
                "full": {layer: [] for layer in LAYERS},
                "e_only": {layer: [] for layer in LAYERS},
                "e_pv_only": {layer: [] for layer in LAYERS},  # E + PV (no SST)
                "e_sst_only": {layer: [] for layer in LAYERS},  # E + SST (no PV)
                "regimes": {layer: [] for layer in LAYERS},  # Store regimes
            },
            "column_wise": {
                "full": [],
                "e_only": [],
                "e_pv_only": [],  # E + PV (no SST)
                "e_sst_only": [],  # E + SST (no PV)
                "regimes": [],  # Store regimes
            },
            "layer_coords": layer_coords,
            "column_coords": column_coords,
            "thalamic_input": snapshot["thalamic_input"],
        }

        # Layer-wise analysis with regime classification
        for coord in layer_coords:
            layer_results = self.analyze_patch(snapshot, coord, "layer", condition=None)
            for layer in LAYERS:
                results["layer_wise"]["full"][layer].append(layer_results[layer]["lambda_full"])
                results["layer_wise"]["e_only"][layer].append(layer_results[layer]["lambda_e_only"])
                results["layer_wise"]["e_pv_only"][layer].append(
                    layer_results[layer]["lambda_e_pv_only"]
                )
                results["layer_wise"]["e_sst_only"][layer].append(
                    layer_results[layer]["lambda_e_sst_only"]
                )
                results["layer_wise"]["regimes"][layer].append(layer_results[layer]["regime"])

        # Column-wise analysis with regime classification
        for coord in column_coords:
            column_result = self.analyze_patch(snapshot, coord, "column", condition=None)
            results["column_wise"]["full"].append(column_result["lambda_full"])
            results["column_wise"]["e_only"].append(column_result["lambda_e_only"])
            results["column_wise"]["e_pv_only"].append(column_result["lambda_e_pv_only"])
            results["column_wise"]["e_sst_only"].append(column_result["lambda_e_sst_only"])
            results["column_wise"]["regimes"].append(column_result["regime"])

        return results
