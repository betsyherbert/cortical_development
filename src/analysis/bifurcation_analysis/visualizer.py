"""Visualization module for bifurcation analysis.

This module creates Figure 5A-style bifurcation diagrams and related plots
for visualizing stability boundaries and spatial frequency patterns in
parameter space.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional, List
import pickle
from pathlib import Path

from .parameter_sweeper import BifurcationAnalysisResult
from src.analysis.utils import load_with_version
from .validation import ValidationReport
from .config import NUMERICAL_TOLERANCES


class BifurcationPlotter:
    """Creates Figure 5A-style bifurcation diagrams."""
    
    def __init__(self, dpi: int = 300, style: str = 'default'):
        """Initialize bifurcation plotter.
        
        Args:
            dpi: Resolution for saved plots
            style: Matplotlib style to use
        """
        self.dpi = dpi
        plt.style.use(style)
        
        # Color settings
        self.stable_color = 'lightgray'
        self.unstable_colormap = 'viridis'
        
    def plot_bifurcation_map(self, analysis_result: BifurcationAnalysisResult,
                           title: Optional[str] = None,
                           save_path: Optional[str] = None,
                           show_plot: bool = False) -> plt.Figure:
        """Create Figure 5A-style bifurcation diagram.
        
        Args:
            analysis_result: Complete bifurcation analysis result
            title: Optional plot title
            save_path: Optional path to save the plot
            show_plot: Whether to display the plot
            
        Returns:
            Matplotlib figure object
        """
        fig, ax = plt.subplots(1, 1, figsize=(8, 6), dpi=self.dpi)
        
        # Get data
        param1_range = analysis_result.param1_range
        param2_range = analysis_result.param2_range
        stability_map = analysis_result.stability_map
        color_map = analysis_result.color_map
        
        # Create meshgrid for plotting
        P1, P2 = np.meshgrid(param1_range, param2_range, indexing='ij')
        
        # Plot stable regions in gray
        stable_mask = stability_map
        if np.any(stable_mask):
            ax.contourf(P1, P2, stable_mask.astype(float), 
                       levels=[0.5, 1.5], colors=[self.stable_color], alpha=0.8)
        
        # Plot unstable regions with color coding by spatial frequency
        unstable_mask = ~stability_map & ~np.isnan(color_map)
        if np.any(unstable_mask):
            # Create masked color array
            color_data = np.ma.masked_where(~unstable_mask, color_map)
            
            # Plot colored regions
            im = ax.contourf(P1, P2, color_data, levels=20, cmap=self.unstable_colormap, alpha=0.9)
            
            # Add colorbar for spatial frequency
            cbar = plt.colorbar(im, ax=ax, shrink=0.8)
            cbar.set_label('Spatial Frequency ||n*||', fontsize=12)
        
        # Set axis labels
        ax.set_xlabel(analysis_result.param1_name.replace('_', ' ').title(), fontsize=12)
        ax.set_ylabel(analysis_result.param2_name.replace('_', ' ').title(), fontsize=12)
        
        # Set title
        if title is None:
            title = f'Bifurcation Diagram - {analysis_result.analysis_type.replace("_", " ").title()}'
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        # Add stability region labels
        if np.any(stable_mask):
            # Find a good location for stable region label
            stable_indices = np.where(stable_mask)
            if len(stable_indices[0]) > 0:
                # Use center of stable region
                center_idx = len(stable_indices[0]) // 2
                label_i, label_j = stable_indices[0][center_idx], stable_indices[1][center_idx]
                ax.text(P1[label_i, label_j], P2[label_i, label_j], 'Stable', 
                       ha='center', va='center', fontsize=10, fontweight='bold')
        
        # Format axes
        ax.grid(True, alpha=0.3)
        ax.set_xlim(param1_range[0], param1_range[-1])
        ax.set_ylim(param2_range[0], param2_range[-1])
        
        # Use log scale if parameter ranges span orders of magnitude
        if param1_range[-1] / param1_range[0] > 10:
            ax.set_xscale('log')
        if param2_range[-1] / param2_range[0] > 10:
            ax.set_yscale('log')
        
        plt.tight_layout()
        
        # Save plot if requested
        if save_path:
            plt.savefig(save_path, format='svg', dpi=self.dpi, bbox_inches='tight')
            print(f"Bifurcation diagram saved to {save_path}")
        
        # Show plot if requested
        if show_plot:
            plt.show()
        
        return fig
    
    def plot_eigenvalue_spectrum(self, analysis_result: BifurcationAnalysisResult,
                               param_point_indices: Tuple[int, int],
                               title: Optional[str] = None,
                               save_path: Optional[str] = None,
                               show_plot: bool = False) -> plt.Figure:
        """Plot eigenvalue spectrum vs mode radius for a specific parameter point.
        
        Args:
            analysis_result: Complete bifurcation analysis result
            param_point_indices: (i, j) indices of parameter point to plot
            title: Optional plot title
            save_path: Optional path to save the plot
            show_plot: Whether to display the plot
            
        Returns:
            Matplotlib figure object
        """
        i, j = param_point_indices
        param_point = analysis_result.parameter_points[i][j]
        
        if not param_point.analysis_success or not param_point.eigenvalue_result:
            raise ValueError(f"Parameter point ({i}, {j}) does not have valid analysis results")
        
        eigenvalue_result = param_point.eigenvalue_result
        
        # Extract data
        mode_radii = []
        real_eigenvalues = []
        stability_flags = []
        
        for mode_data in eigenvalue_result.all_mode_data:
            if mode_data.max_eigenvalue is not None:
                mode_radii.append(mode_data.mode_radius)
                real_eigenvalues.append(np.real(mode_data.max_eigenvalue))
                stability_flags.append(np.real(mode_data.max_eigenvalue) < -NUMERICAL_TOLERANCES['stability_threshold'])
        
        mode_radii = np.array(mode_radii)
        real_eigenvalues = np.array(real_eigenvalues)
        stability_flags = np.array(stability_flags)
        
        # Create plot
        fig, ax = plt.subplots(1, 1, figsize=(8, 6), dpi=self.dpi)
        
        # Plot stable and unstable modes with different colors
        stable_mask = stability_flags
        unstable_mask = ~stability_flags
        
        if np.any(stable_mask):
            ax.scatter(mode_radii[stable_mask], real_eigenvalues[stable_mask], 
                      c='blue', alpha=0.6, s=20, label='Stable modes')
        
        if np.any(unstable_mask):
            ax.scatter(mode_radii[unstable_mask], real_eigenvalues[unstable_mask], 
                      c='red', alpha=0.8, s=30, label='Unstable modes')
        
        # Add horizontal line at zero
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5, linewidth=1)
        
        # Highlight most unstable mode
        winning_mode_radius = eigenvalue_result.winning_mode_radius
        max_eigenvalue = eigenvalue_result.max_real_eigenvalue
        ax.scatter([winning_mode_radius], [max_eigenvalue], 
                  c='orange', s=100, marker='*', edgecolors='black', 
                  linewidth=1, label='Most unstable mode', zorder=10)
        
        # Set labels and title
        ax.set_xlabel('Mode Radius ||n||', fontsize=12)
        ax.set_ylabel('Real Part of Max Eigenvalue', fontsize=12)
        
        if title is None:
            param_str = f"({param_point.param1_value:.3f}, {param_point.param2_value:.3f})"
            title = f'Eigenvalue Spectrum at {param_str}'
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        # Add legend
        ax.legend(fontsize=10)
        
        # Format axes
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot if requested
        if save_path:
            plt.savefig(save_path, format='svg', dpi=self.dpi, bbox_inches='tight')
            print(f"Eigenvalue spectrum saved to {save_path}")
        
        # Show plot if requested
        if show_plot:
            plt.show()
        
        return fig
    
    def plot_stability_comparison(self, pv_result: BifurcationAnalysisResult,
                                sst_result: BifurcationAnalysisResult,
                                title: Optional[str] = None,
                                save_path: Optional[str] = None,
                                show_plot: bool = False) -> plt.Figure:
        """Plot side-by-side comparison of PV and SST bifurcation diagrams.
        
        Args:
            pv_result: PV bifurcation analysis result
            sst_result: SST bifurcation analysis result
            title: Optional plot title
            save_path: Optional path to save the plot
            show_plot: Whether to display the plot
            
        Returns:
            Matplotlib figure object
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), dpi=self.dpi)
        
        # Plot PV analysis
        self._plot_single_bifurcation(ax1, pv_result, 'PV Analysis')
        
        # Plot SST analysis  
        self._plot_single_bifurcation(ax2, sst_result, 'SST Analysis')
        
        # Set overall title
        if title is None:
            title = 'PV vs SST Bifurcation Comparison'
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        
        # Save plot if requested
        if save_path:
            plt.savefig(save_path, format='svg', dpi=self.dpi, bbox_inches='tight')
            print(f"Comparison plot saved to {save_path}")
        
        # Show plot if requested
        if show_plot:
            plt.show()
        
        return fig
    
    def _plot_single_bifurcation(self, ax: plt.Axes, analysis_result: BifurcationAnalysisResult,
                               subplot_title: str):
        """Plot a single bifurcation diagram on given axes."""
        # Get data
        param1_range = analysis_result.param1_range
        param2_range = analysis_result.param2_range
        stability_map = analysis_result.stability_map
        color_map = analysis_result.color_map
        
        # Create meshgrid
        P1, P2 = np.meshgrid(param1_range, param2_range, indexing='ij')
        
        # Plot stable regions
        stable_mask = stability_map
        if np.any(stable_mask):
            ax.contourf(P1, P2, stable_mask.astype(float), 
                       levels=[0.5, 1.5], colors=[self.stable_color], alpha=0.8)
        
        # Plot unstable regions
        unstable_mask = ~stability_map & ~np.isnan(color_map)
        if np.any(unstable_mask):
            color_data = np.ma.masked_where(~unstable_mask, color_map)
            im = ax.contourf(P1, P2, color_data, levels=20, cmap=self.unstable_colormap, alpha=0.9)
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax, shrink=0.8)
            cbar.set_label('||n*||', fontsize=10)
        
        # Set labels and title
        ax.set_xlabel(analysis_result.param1_name.replace('_', ' ').title(), fontsize=12)
        ax.set_ylabel(analysis_result.param2_name.replace('_', ' ').title(), fontsize=12)
        ax.set_title(subplot_title, fontsize=14, fontweight='bold')
        
        # Format axes
        ax.grid(True, alpha=0.3)
        
        # Use log scale if needed
        if param1_range[-1] / param1_range[0] > 10:
            ax.set_xscale('log')
        if param2_range[-1] / param2_range[0] > 10:
            ax.set_yscale('log')
    
    def plot_validation_summary(self, validation_report: ValidationReport,
                              analysis_result: BifurcationAnalysisResult,
                              title: Optional[str] = None,
                              save_path: Optional[str] = None,
                              show_plot: bool = False) -> plt.Figure:
        """Plot validation summary with key metrics.
        
        Args:
            validation_report: Validation report
            analysis_result: Analysis result that was validated
            title: Optional plot title
            save_path: Optional path to save the plot
            show_plot: Whether to display the plot
            
        Returns:
            Matplotlib figure object
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10), dpi=self.dpi)
        
        # Plot 1: Overall validation scores
        validation_components = [
            'DC Gain',
            'Math Consistency', 
            'Continuity',
            'Cross-Validation'
        ]
        scores = [
            validation_report.dc_gain_validation.get('validation_rate', 0.0),
            validation_report.mathematical_consistency['overall_score'],
            validation_report.parameter_continuity['overall_score'],
            validation_report.cross_validation_results['overall_score']
        ]
        
        colors = ['green' if score > 0.8 else 'orange' if score > 0.6 else 'red' for score in scores]
        bars = ax1.bar(validation_components, scores, color=colors, alpha=0.7)
        ax1.set_ylabel('Validation Score', fontsize=12)
        ax1.set_title('Validation Component Scores', fontsize=12, fontweight='bold')
        ax1.set_ylim(0, 1)
        ax1.grid(True, alpha=0.3)
        
        # Add score labels on bars
        for bar, score in zip(bars, scores):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{score:.3f}', ha='center', va='bottom', fontsize=10)
        
        # Plot 2: Success rate breakdown
        success_data = [
            analysis_result.successful_points,
            analysis_result.total_points - analysis_result.successful_points
        ]
        success_labels = ['Successful', 'Failed']
        ax2.pie(success_data, labels=success_labels, autopct='%1.1f%%', 
               colors=['lightgreen', 'lightcoral'])
        ax2.set_title('Analysis Success Rate', fontsize=12, fontweight='bold')
        
        # Plot 3: Stability breakdown
        if analysis_result.successful_points > 0:
            stability_data = [
                analysis_result.stable_points,
                analysis_result.unstable_points
            ]
            stability_labels = ['Stable', 'Unstable']
            ax3.pie(stability_data, labels=stability_labels, autopct='%1.1f%%',
                   colors=['lightblue', 'lightyellow'])
        ax3.set_title('Stability Classification', fontsize=12, fontweight='bold')
        
        # Plot 4: Eigenvalue distribution
        eigenvalue_map = analysis_result.eigenvalue_map
        valid_eigenvalues = eigenvalue_map[~np.isnan(eigenvalue_map)]
        
        if len(valid_eigenvalues) > 0:
            ax4.hist(valid_eigenvalues, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            ax4.axvline(x=0, color='red', linestyle='--', alpha=0.7, linewidth=2, label='Stability threshold')
            ax4.set_xlabel('Max Real Eigenvalue', fontsize=12)
            ax4.set_ylabel('Count', fontsize=12)
            ax4.set_title('Eigenvalue Distribution', fontsize=12, fontweight='bold')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        # Set overall title
        if title is None:
            title = f'Validation Summary - {analysis_result.analysis_type.replace("_", " ").title()}'
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        
        # Save plot if requested
        if save_path:
            plt.savefig(save_path, format='svg', dpi=self.dpi, bbox_inches='tight')
            print(f"Validation summary saved to {save_path}")
        
        # Show plot if requested
        if show_plot:
            plt.show()
        
        return fig
    
    def plot_parameter_space_overview(self, analysis_result: BifurcationAnalysisResult,
                                    title: Optional[str] = None,
                                    save_path: Optional[str] = None,
                                    show_plot: bool = False) -> plt.Figure:
        """Plot overview of parameter space with multiple visualizations.
        
        Args:
            analysis_result: Complete bifurcation analysis result
            title: Optional plot title
            save_path: Optional path to save the plot
            show_plot: Whether to display the plot
            
        Returns:
            Matplotlib figure object
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10), dpi=self.dpi)
        
        # Get common data
        param1_range = analysis_result.param1_range
        param2_range = analysis_result.param2_range
        P1, P2 = np.meshgrid(param1_range, param2_range, indexing='ij')
        
        # Plot 1: Stability map
        stability_map = analysis_result.stability_map.astype(float)
        ax1.contourf(P1, P2, stability_map, levels=[0.5, 1.5], 
                    colors=['red', 'green'], alpha=0.7)
        ax1.set_title('Stability Map', fontsize=12, fontweight='bold')
        ax1.set_xlabel(analysis_result.param1_name.replace('_', ' ').title())
        ax1.set_ylabel(analysis_result.param2_name.replace('_', ' ').title())
        
        # Plot 2: Eigenvalue map
        eigenvalue_map = analysis_result.eigenvalue_map
        im2 = ax2.contourf(P1, P2, eigenvalue_map, levels=20, cmap='RdBu_r')
        cbar2 = plt.colorbar(im2, ax=ax2, shrink=0.8)
        cbar2.set_label('Max Real Eigenvalue')
        ax2.set_title('Eigenvalue Map', fontsize=12, fontweight='bold')
        ax2.set_xlabel(analysis_result.param1_name.replace('_', ' ').title())
        ax2.set_ylabel(analysis_result.param2_name.replace('_', ' ').title())
        
        # Plot 3: Spatial frequency map (color map)
        color_map = analysis_result.color_map
        if not np.all(np.isnan(color_map)):
            im3 = ax3.contourf(P1, P2, color_map, levels=20, cmap=self.unstable_colormap)
            cbar3 = plt.colorbar(im3, ax=ax3, shrink=0.8)
            cbar3.set_label('Spatial Frequency ||n*||')
        else:
            ax3.text(0.5, 0.5, 'All Stable\n(No Spatial Frequencies)', 
                    ha='center', va='center', transform=ax3.transAxes, fontsize=14)
        ax3.set_title('Spatial Frequency Map', fontsize=12, fontweight='bold')
        ax3.set_xlabel(analysis_result.param1_name.replace('_', ' ').title())
        ax3.set_ylabel(analysis_result.param2_name.replace('_', ' ').title())
        
        # Plot 4: Analysis statistics
        stats_text = f"""Analysis Statistics:
        
Grid Shape: {analysis_result.grid_shape[0]}×{analysis_result.grid_shape[1]}
Total Points: {analysis_result.total_points}
Successful: {analysis_result.successful_points} ({analysis_result.successful_points/analysis_result.total_points:.1%})
Stable: {analysis_result.stable_points} ({analysis_result.stable_points/analysis_result.successful_points:.1%})
Unstable: {analysis_result.unstable_points} ({analysis_result.unstable_points/analysis_result.successful_points:.1%})

Analysis Time: {analysis_result.analysis_time:.1f} seconds
Time per Point: {analysis_result.analysis_time/analysis_result.total_points:.2f} sec

Parameter Ranges:
{analysis_result.param1_name}: [{analysis_result.param1_range[0]:.3f}, {analysis_result.param1_range[-1]:.3f}]
{analysis_result.param2_name}: [{analysis_result.param2_range[0]:.3f}, {analysis_result.param2_range[-1]:.3f}]
        """
        
        ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace')
        ax4.set_title('Analysis Statistics', fontsize=12, fontweight='bold')
        ax4.axis('off')
        
        # Set overall title
        if title is None:
            title = f'Parameter Space Overview - {analysis_result.analysis_type.replace("_", " ").title()}'
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        
        # Save plot if requested
        if save_path:
            plt.savefig(save_path, format='svg', dpi=self.dpi, bbox_inches='tight')
            print(f"Parameter space overview saved to {save_path}")
        
        # Show plot if requested
        if show_plot:
            plt.show()
        
        return fig


class BifurcationVisualizationSuite:
    """Complete visualization suite for bifurcation analysis results."""
    
    def __init__(self, output_dir: Optional[str] = None, dpi: int = 300):
        """Initialize visualization suite.
        
        Args:
            output_dir: Output directory for plots
            dpi: Resolution for saved plots
        """
        self.output_dir = Path(output_dir) if output_dir else Path('outputs') / 'bifurcation'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.plotter = BifurcationPlotter(dpi=dpi)
        
    def create_complete_visualization(self, analysis_result: BifurcationAnalysisResult,
                                    validation_report: Optional[ValidationReport] = None,
                                    show_plots: bool = False) -> Dict[str, str]:
        """Create complete visualization suite for analysis results.
        
        Args:
            analysis_result: Complete bifurcation analysis result
            validation_report: Optional validation report
            show_plots: Whether to display plots
            
        Returns:
            Dictionary mapping plot types to saved file paths
        """
        print(f"Creating visualization suite for {analysis_result.analysis_type}...")
        
        # Create timestamp for file naming
        timestamp = analysis_result.timestamp.replace(' ', '_').replace(':', '-')
        analysis_name = analysis_result.analysis_type
        
        saved_plots = {}
        
        # 1. Main bifurcation diagram
        print("  Creating bifurcation diagram...")
        bifurcation_path = self.output_dir / f'bifurcation_diagram_{analysis_name}_{timestamp}.svg'
        fig1 = self.plotter.plot_bifurcation_map(
            analysis_result, 
            save_path=str(bifurcation_path),
            show_plot=show_plots
        )
        saved_plots['bifurcation_diagram'] = str(bifurcation_path)
        plt.close(fig1)
        
        # 2. Parameter space overview
        print("  Creating parameter space overview...")
        overview_path = self.output_dir / f'parameter_overview_{analysis_name}_{timestamp}.svg'
        fig2 = self.plotter.plot_parameter_space_overview(
            analysis_result,
            save_path=str(overview_path),
            show_plot=show_plots
        )
        saved_plots['parameter_overview'] = str(overview_path)
        plt.close(fig2)
        
        # 3. Eigenvalue spectrum for interesting points
        print("  Creating eigenvalue spectra...")
        spectrum_plots = self._create_spectrum_plots(analysis_result, timestamp, show_plots)
        saved_plots.update(spectrum_plots)
        
        # 4. Validation summary (if available)
        if validation_report:
            print("  Creating validation summary...")
            validation_path = self.output_dir / f'validation_summary_{analysis_name}_{timestamp}.svg'
            fig4 = self.plotter.plot_validation_summary(
                validation_report,
                analysis_result,
                save_path=str(validation_path),
                show_plot=show_plots
            )
            saved_plots['validation_summary'] = str(validation_path)
            plt.close(fig4)
        
        print(f"Created {len(saved_plots)} plots in {self.output_dir}")
        return saved_plots
    
    def _create_spectrum_plots(self, analysis_result: BifurcationAnalysisResult,
                             timestamp: str, show_plots: bool) -> Dict[str, str]:
        """Create eigenvalue spectrum plots for interesting parameter points."""
        saved_plots = {}
        
        # Find interesting points to plot
        interesting_points = self._find_interesting_points(analysis_result)
        
        for point_name, (i, j) in interesting_points.items():
            try:
                spectrum_path = self.output_dir / f'eigenvalue_spectrum_{point_name}_{analysis_result.analysis_type}_{timestamp}.svg'
                fig = self.plotter.plot_eigenvalue_spectrum(
                    analysis_result,
                    (i, j),
                    title=f'Eigenvalue Spectrum - {point_name.replace("_", " ").title()}',
                    save_path=str(spectrum_path),
                    show_plot=show_plots
                )
                saved_plots[f'spectrum_{point_name}'] = str(spectrum_path)
                plt.close(fig)
                
            except (ValueError, IndexError) as e:
                print(f"    Warning: Could not create spectrum plot for {point_name}: {e}")
        
        return saved_plots
    
    def _find_interesting_points(self, analysis_result: BifurcationAnalysisResult) -> Dict[str, Tuple[int, int]]:
        """Find interesting parameter points for detailed analysis."""
        interesting_points = {}
        grid_shape = analysis_result.grid_shape
        
        # Center point
        center_i, center_j = grid_shape[0] // 2, grid_shape[1] // 2
        interesting_points['center'] = (center_i, center_j)
        
        # Corner points
        interesting_points['bottom_left'] = (0, 0)
        interesting_points['top_right'] = (grid_shape[0] - 1, grid_shape[1] - 1)
        
        # Find most unstable point (if any)
        eigenvalue_map = analysis_result.eigenvalue_map
        if not np.all(np.isnan(eigenvalue_map)):
            valid_eigenvalues = eigenvalue_map[~np.isnan(eigenvalue_map)]
            if len(valid_eigenvalues) > 0:
                max_eigenvalue = np.max(valid_eigenvalues)
                max_indices = np.where(eigenvalue_map == max_eigenvalue)
                if len(max_indices[0]) > 0:
                    interesting_points['most_unstable'] = (max_indices[0][0], max_indices[1][0])
        
        return interesting_points


def create_publication_plots(analysis_result: BifurcationAnalysisResult,
                           validation_report: Optional[ValidationReport] = None,
                           output_dir: Optional[str] = None,
                           show_plots: bool = False) -> Dict[str, str]:
    """Create publication-quality plots for bifurcation analysis.
    
    Args:
        analysis_result: Complete bifurcation analysis result
        validation_report: Optional validation report
        output_dir: Output directory for plots
        show_plots: Whether to display plots
        
    Returns:
        Dictionary mapping plot types to saved file paths
    """
    # Initialize visualization suite
    viz_suite = BifurcationVisualizationSuite(output_dir=output_dir, dpi=300)
    
    # Create complete visualization
    return viz_suite.create_complete_visualization(
        analysis_result, validation_report, show_plots
    )


def plot_multiple_analyses(analysis_results: List[BifurcationAnalysisResult],
                         output_dir: Optional[str] = None,
                         show_plots: bool = False) -> Dict[str, str]:
    """Create comparison plots for multiple bifurcation analyses.
    
    Args:
        analysis_results: List of bifurcation analysis results
        output_dir: Output directory for plots
        show_plots: Whether to display plots
        
    Returns:
        Dictionary mapping plot types to saved file paths
    """
    if len(analysis_results) < 2:
        raise ValueError("Need at least 2 analysis results for comparison")
    
    # Initialize plotter
    output_path = Path(output_dir) if output_dir else Path('outputs') / 'bifurcation'
    output_path.mkdir(parents=True, exist_ok=True)
    
    plotter = BifurcationPlotter(dpi=300)
    saved_plots = {}
    
    # Create pairwise comparisons
    for i in range(len(analysis_results)):
        for j in range(i + 1, len(analysis_results)):
            result1, result2 = analysis_results[i], analysis_results[j]
            
            # Create comparison plot
            comparison_name = f"{result1.analysis_type}_vs_{result2.analysis_type}"
            timestamp = result1.timestamp.replace(' ', '_').replace(':', '-')
            comparison_path = output_path / f'comparison_{comparison_name}_{timestamp}.svg'
            
            fig = plotter.plot_stability_comparison(
                result1, result2,
                save_path=str(comparison_path),
                show_plot=show_plots
            )
            saved_plots[comparison_name] = str(comparison_path)
            plt.close(fig)
    
    return saved_plots


def load_and_visualize_results(results_file: str, 
                             output_dir: Optional[str] = None,
                             show_plots: bool = False) -> Dict[str, str]:
    """Load bifurcation results from file and create visualizations.
    
    Args:
        results_file: Path to saved bifurcation analysis results
        output_dir: Output directory for plots
        show_plots: Whether to display plots
        
    Returns:
        Dictionary mapping plot types to saved file paths
    """
    print(f"Loading and visualizing results from {results_file}...")
    
    # Load results with version checking
    versioned_data = load_with_version(results_file)
    analysis_result = versioned_data['data']
    
    if not isinstance(analysis_result, BifurcationAnalysisResult):
        raise ValueError("File does not contain BifurcationAnalysisResult")
    
    print(f"Loaded {analysis_result.analysis_type} analysis with {analysis_result.total_points} points")
    
    # Create visualizations
    return create_publication_plots(analysis_result, output_dir=output_dir, show_plots=show_plots)
