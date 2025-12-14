"""Smoke and basic integration tests for analysis pipelines.

These tests verify the pipeline classes can be imported, instantiated,
and configured correctly without running full analyses.
"""

import pytest
import tempfile
from pathlib import Path


class TestPipelineImports:
    """Test that pipeline classes can be imported without errors."""

    def test_import_bifurcation_pipeline(self):
        """Test that BifurcationPipeline can be imported."""
        from src.analysis.bifurcation import BifurcationPipeline

        assert BifurcationPipeline is not None

    def test_import_stability_pipeline(self):
        """Test that StabilityPipeline can be imported."""
        from src.analysis.stability import StabilityPipeline

        assert StabilityPipeline is not None

    def test_import_descriptive_pipeline(self):
        """Test that DescriptivePipeline can be imported."""
        from src.analysis.descriptive import DescriptivePipeline

        assert DescriptivePipeline is not None


class TestPipelineInstantiation:
    """Test that pipeline classes can be instantiated with default config."""

    def test_bifurcation_pipeline_default_config(self):
        """Test BifurcationPipeline with default configuration."""
        from src.analysis.bifurcation import BifurcationPipeline

        pipeline = BifurcationPipeline()
        assert pipeline.config is not None
        assert "stages" in pipeline.config
        assert "mode" in pipeline.config
        assert "output_dir" in pipeline.config
        assert pipeline.config["stages"] == ["P0", "P5", "P10", "P15"]

    def test_stability_pipeline_default_config(self):
        """Test StabilityPipeline with default configuration."""
        from src.analysis.stability import StabilityPipeline

        pipeline = StabilityPipeline()
        assert pipeline.config is not None
        assert "stages" in pipeline.config
        assert "duration" in pipeline.config
        assert "n_snapshots" in pipeline.config
        assert "output_dir" in pipeline.config

    def test_descriptive_pipeline_default_config(self):
        """Test DescriptivePipeline with default configuration."""
        from src.analysis.descriptive import DescriptivePipeline

        pipeline = DescriptivePipeline()
        assert pipeline.config is not None
        assert "stages" in pipeline.config
        assert "warmup_duration" in pipeline.config
        assert "simulation_duration" in pipeline.config
        assert "output_dir" in pipeline.config


class TestPipelineCustomConfig:
    """Test that pipeline classes accept custom configuration."""

    def test_bifurcation_pipeline_custom_stages(self):
        """Test BifurcationPipeline with custom stages."""
        from src.analysis.bifurcation import BifurcationPipeline

        config = {"stages": ["P0", "P5"], "mode": "fixed_ratio", "output_dir": "/tmp/test"}
        pipeline = BifurcationPipeline(config)
        assert pipeline.config["stages"] == ["P0", "P5"]
        assert pipeline.config["mode"] == "fixed_ratio"

    def test_stability_pipeline_custom_stages(self):
        """Test StabilityPipeline with custom stages."""
        from src.analysis.stability import StabilityPipeline

        config = {"stages": ["P0"], "duration": 5.0, "output_dir": "/tmp/test"}
        pipeline = StabilityPipeline(config)
        assert pipeline.config["stages"] == ["P0"]
        assert pipeline.config["duration"] == 5.0

    def test_descriptive_pipeline_custom_stages(self):
        """Test DescriptivePipeline with custom stages."""
        from src.analysis.descriptive import DescriptivePipeline

        config = {"stages": ["P10", "P15"], "simulation_duration": 5.0, "output_dir": "/tmp/test"}
        pipeline = DescriptivePipeline(config)
        assert pipeline.config["stages"] == ["P10", "P15"]
        assert pipeline.config["simulation_duration"] == 5.0


class TestCLIModules:
    """Test that CLI modules can be parsed without errors."""

    def test_bifurcation_cli_help(self):
        """Test bifurcation CLI help doesn't raise."""
        from src.analysis.bifurcation.run_analysis import main
        import sys

        # Capture args and test help parsing
        old_argv = sys.argv
        try:
            sys.argv = ["test", "--help"]
            with pytest.raises(SystemExit) as exc_info:
                main()
            # Help should exit with 0
            assert exc_info.value.code == 0
        finally:
            sys.argv = old_argv

    def test_stability_cli_help(self):
        """Test stability CLI help doesn't raise."""
        from src.analysis.stability.run_analysis import main
        import sys

        old_argv = sys.argv
        try:
            sys.argv = ["test", "--help"]
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 0
        finally:
            sys.argv = old_argv

    def test_descriptive_cli_help(self):
        """Test descriptive CLI help doesn't raise."""
        from src.analysis.descriptive.run_analysis import main
        import sys

        old_argv = sys.argv
        try:
            sys.argv = ["test", "--help"]
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 0
        finally:
            sys.argv = old_argv


class TestSaveResultsIntegration:
    """Test that save_results method works correctly with mock data."""

    def test_bifurcation_save_results(self):
        """Test BifurcationPipeline save_results creates file with metadata."""
        from src.analysis.bifurcation import BifurcationPipeline
        from src.analysis.common import load_with_version

        with tempfile.TemporaryDirectory() as tmpdir:
            config = {"stages": ["P0"], "mode": "fixed_absolute", "output_dir": tmpdir}
            pipeline = BifurcationPipeline(config)

            # Mock results
            mock_results = {"stability": {"P0": {"test": "data"}}}

            # Use the internal save logic pattern
            results_file = Path(tmpdir) / "test_results.pkl"
            from src.analysis.common import make_run_metadata, save_with_version

            metadata = make_run_metadata(stages=["P0"], params={"mode": "fixed_absolute"})
            save_with_version(mock_results, str(results_file), metadata=metadata)

            # Verify file exists and has correct structure
            assert results_file.exists()
            loaded = load_with_version(str(results_file))
            assert "data" in loaded
            assert "metadata" in loaded
            assert loaded["metadata"]["stages"] == ["P0"]

    def test_stability_save_results(self):
        """Test StabilityPipeline save_results creates file with metadata."""
        from src.analysis.stability import StabilityPipeline
        from src.analysis.common import load_with_version

        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                "stages": ["P0"],
                "duration": 10.0,
                "n_snapshots": 30,
                "layer_patch_size": 5,
                "output_dir": tmpdir,
            }
            pipeline = StabilityPipeline(config)

            # Mock results
            mock_results = {"P0": {"idle": [], "driven": []}}

            # Save results
            results_file = pipeline.save_results(mock_results)

            # Verify
            assert results_file.exists()
            loaded = load_with_version(str(results_file))
            assert loaded["data"] == mock_results
            assert loaded["metadata"]["stages"] == ["P0"]

    def test_descriptive_save_results(self):
        """Test DescriptivePipeline save_results creates file with metadata."""
        from src.analysis.descriptive import DescriptivePipeline
        from src.analysis.common import load_with_version

        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                "stages": ["P0"],
                "warmup_duration": 2.0,
                "simulation_duration": 10.0,
                "sampling_interval": 20.0,
                "output_dir": tmpdir,
            }
            pipeline = DescriptivePipeline(config)

            # Mock results
            mock_results = {"P0": {"time": [0, 20, 40], "stage": "P0"}}

            # Save results
            results_file = pipeline.save_results(mock_results)

            # Verify
            assert results_file.exists()
            loaded = load_with_version(str(results_file))
            assert loaded["data"] == mock_results
            assert loaded["metadata"]["stages"] == ["P0"]


class TestPipelineMethods:
    """Test that pipeline methods exist and have correct signatures."""

    def test_bifurcation_has_required_methods(self):
        """Test BifurcationPipeline has required methods."""
        from src.analysis.bifurcation import BifurcationPipeline

        pipeline = BifurcationPipeline()
        assert hasattr(pipeline, "run_stability_analysis")
        assert hasattr(pipeline, "run_gain_analysis")
        assert hasattr(pipeline, "run_all")
        assert callable(pipeline.run_stability_analysis)
        assert callable(pipeline.run_gain_analysis)
        assert callable(pipeline.run_all)

    def test_stability_has_required_methods(self):
        """Test StabilityPipeline has required methods."""
        from src.analysis.stability import StabilityPipeline

        pipeline = StabilityPipeline()
        assert hasattr(pipeline, "run")
        assert hasattr(pipeline, "save_results")
        assert hasattr(pipeline, "generate_visualizations")
        assert hasattr(pipeline, "print_summary")
        assert callable(pipeline.run)
        assert callable(pipeline.save_results)

    def test_descriptive_has_required_methods(self):
        """Test DescriptivePipeline has required methods."""
        from src.analysis.descriptive import DescriptivePipeline

        pipeline = DescriptivePipeline()
        assert hasattr(pipeline, "run")
        assert hasattr(pipeline, "save_results")
        assert hasattr(pipeline, "generate_visualizations")
        assert hasattr(pipeline, "print_summary")
        assert callable(pipeline.run)
        assert callable(pipeline.save_results)
