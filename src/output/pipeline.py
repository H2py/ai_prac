"""
Clean output pipeline orchestrating transformation and serialization.
"""

from typing import Dict, List, Union, Optional
from pathlib import Path
import logging
from datetime import datetime

from src.models import AnalysisResults
from src.utils.logger import PerformanceLogger
from .transformer import DataTransformer
from .serializers import create_serializer, FormatSerializer
from .formats import OutputFormat, ExportData

logger = logging.getLogger(__name__)


class OutputPipeline:
    """Orchestrates data transformation and format serialization with clean separation of concerns."""
    
    def __init__(self, 
                 transformer: Optional[DataTransformer] = None,
                 output_dir: Optional[Path] = None):
        """Initialize output pipeline.
        
        Args:
            transformer: Data transformer instance (creates default if None)
            output_dir: Base output directory
        """
        self.transformer = transformer or DataTransformer()
        self.output_dir = Path(output_dir) if output_dir else Path("output")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Performance logging
        self.perf_logger = PerformanceLogger(logger)
        
        # Cache serializers to avoid recreation
        self._serializer_cache: Dict[OutputFormat, FormatSerializer] = {}
        
        # Statistics tracking
        self._stats = {
            'exports': 0,
            'successful_exports': 0,
            'failed_exports': 0,
            'total_export_time': 0.0,
            'formats_used': set(),
            'files_created': []
        }
        
        logger.info(f"OutputPipeline initialized with output directory: {self.output_dir}")
    
    def export_results(self,
                      results: AnalysisResults,
                      formats: List[Union[str, OutputFormat]],
                      base_filename: Optional[str] = None,
                      output_dir: Optional[Path] = None) -> Dict[OutputFormat, Path]:
        """Export results in multiple formats with optimized processing.
        
        Args:
            results: Analysis results to export
            formats: List of output formats to generate
            base_filename: Base filename (without extension)
            output_dir: Output directory (overrides instance default)
            
        Returns:
            Dictionary mapping formats to output file paths
        """
        self.perf_logger.start_timer("export_all_formats")
        
        # Use specified output directory or default
        export_dir = Path(output_dir) if output_dir else self.output_dir
        export_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate base filename if not provided
        if not base_filename:
            base_filename = f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Convert string formats to OutputFormat enums
        output_formats = self._parse_formats(formats)
        
        if not output_formats:
            logger.warning("No valid output formats specified")
            return {}
        
        logger.info(f"Exporting to {len(output_formats)} formats: {[f.value for f in output_formats]}")
        
        # Prepare export data once (shared across all formats)
        export_data = self._prepare_export_data(results, base_filename)
        
        # Export each format
        exported_files = {}
        successful_exports = 0
        
        for output_format in output_formats:
            try:
                file_path = self._export_single_format(
                    export_data, output_format, export_dir
                )
                exported_files[output_format] = file_path
                successful_exports += 1
                
                logger.info(f"Successfully exported {output_format.value}: {file_path}")
                
            except Exception as e:
                logger.error(f"Failed to export {output_format.value}: {e}")
                self._stats['failed_exports'] += 1
                continue
        
        # Update statistics
        duration = self.perf_logger.stop_timer("export_all_formats")
        self._update_stats(output_formats, successful_exports, duration, list(exported_files.values()))
        
        logger.info(f"Export completed: {successful_exports}/{len(output_formats)} formats "
                   f"in {duration:.2f}s")
        
        return exported_files
    
    def export_single_format(self,
                            results: AnalysisResults,
                            output_format: Union[str, OutputFormat],
                            base_filename: Optional[str] = None,
                            output_dir: Optional[Path] = None) -> Path:
        """Export results in a single format.
        
        Args:
            results: Analysis results to export
            output_format: Output format
            base_filename: Base filename (without extension)
            output_dir: Output directory (overrides instance default)
            
        Returns:
            Path to exported file
        """
        # Parse format
        formats = self._parse_formats([output_format])
        if not formats:
            raise ValueError(f"Invalid output format: {output_format}")
        
        format_enum = formats[0]
        
        # Export using main method
        exported_files = self.export_results(
            results, [format_enum], base_filename, output_dir
        )
        
        if format_enum not in exported_files:
            raise RuntimeError(f"Failed to export {format_enum.value}")
        
        return exported_files[format_enum]
    
    def _prepare_export_data(self, results: AnalysisResults, base_filename: str) -> ExportData:
        """Prepare export data using transformer."""
        self.perf_logger.start_timer("data_transformation")
        
        try:
            export_data = self.transformer.prepare_export_data(results, base_filename)
            duration = self.perf_logger.stop_timer("data_transformation")
            
            logger.debug(f"Data transformation completed in {duration:.3f}s")
            return export_data
            
        except Exception as e:
            self.perf_logger.stop_timer("data_transformation")
            logger.error(f"Data transformation failed: {e}")
            raise
    
    def _export_single_format(self,
                             export_data: ExportData,
                             output_format: OutputFormat,
                             output_dir: Path) -> Path:
        """Export single format using cached serializer."""
        self.perf_logger.start_timer(f"serialize_{output_format.value}")
        
        try:
            # Get or create serializer
            serializer = self._get_serializer(output_format)
            
            # Serialize data
            serialized_content = serializer.serialize(export_data)
            
            # Write to file
            filename = serializer.get_filename(export_data.base_filename or "analysis")
            file_path = output_dir / filename
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(serialized_content)
            
            duration = self.perf_logger.stop_timer(f"serialize_{output_format.value}")
            logger.debug(f"Serialization for {output_format.value} completed in {duration:.3f}s")
            
            return file_path
            
        except Exception as e:
            self.perf_logger.stop_timer(f"serialize_{output_format.value}")
            logger.error(f"Serialization for {output_format.value} failed: {e}")
            raise
    
    def _get_serializer(self, output_format: OutputFormat) -> FormatSerializer:
        """Get or create serializer with caching."""
        if output_format not in self._serializer_cache:
            self._serializer_cache[output_format] = create_serializer(output_format)
            logger.debug(f"Created serializer for {output_format.value}")
        
        return self._serializer_cache[output_format]
    
    def _parse_formats(self, formats: List[Union[str, OutputFormat]]) -> List[OutputFormat]:
        """Parse and validate output formats."""
        output_formats = []
        
        for fmt in formats:
            if isinstance(fmt, str):
                try:
                    output_formats.append(OutputFormat(fmt.lower()))
                except ValueError:
                    logger.warning(f"Unknown output format: {fmt}")
                    continue
            elif isinstance(fmt, OutputFormat):
                output_formats.append(fmt)
            else:
                logger.warning(f"Invalid format type: {type(fmt)}")
        
        return output_formats
    
    def _update_stats(self,
                     formats: List[OutputFormat],
                     successful_exports: int,
                     duration: float,
                     created_files: List[Path]) -> None:
        """Update pipeline statistics."""
        self._stats['exports'] += 1
        self._stats['successful_exports'] += successful_exports
        self._stats['failed_exports'] += len(formats) - successful_exports
        self._stats['total_export_time'] += duration
        self._stats['formats_used'].update(f.value for f in formats)
        self._stats['files_created'].extend(str(f) for f in created_files)
    
    def get_supported_formats(self) -> List[str]:
        """Get list of supported output formats."""
        return [fmt.value for fmt in OutputFormat]
    
    def get_stats(self) -> Dict[str, any]:
        """Get pipeline statistics."""
        avg_export_time = (
            self._stats['total_export_time'] / self._stats['exports']
            if self._stats['exports'] > 0 else 0.0
        )
        
        success_rate = (
            self._stats['successful_exports'] / 
            (self._stats['successful_exports'] + self._stats['failed_exports'])
            if (self._stats['successful_exports'] + self._stats['failed_exports']) > 0 else 0.0
        )
        
        return {
            'total_exports': self._stats['exports'],
            'successful_exports': self._stats['successful_exports'],
            'failed_exports': self._stats['failed_exports'],
            'success_rate': success_rate,
            'average_export_time': avg_export_time,
            'total_export_time': self._stats['total_export_time'],
            'formats_used': list(self._stats['formats_used']),
            'files_created': len(self._stats['files_created']),
            'cached_serializers': len(self._serializer_cache)
        }
    
    def reset_stats(self) -> None:
        """Reset pipeline statistics."""
        self._stats = {
            'exports': 0,
            'successful_exports': 0,
            'failed_exports': 0,
            'total_export_time': 0.0,
            'formats_used': set(),
            'files_created': []
        }
        logger.debug("Pipeline statistics reset")
    
    def clear_cache(self) -> None:
        """Clear serializer cache."""
        self._serializer_cache.clear()
        logger.debug("Serializer cache cleared")
    
    def cleanup_temp_files(self, pattern: str = "temp_*") -> int:
        """Clean up temporary files in output directory.
        
        Args:
            pattern: Glob pattern for temporary files
            
        Returns:
            Number of files cleaned up
        """
        temp_files = list(self.output_dir.glob(pattern))
        cleaned_count = 0
        
        for temp_file in temp_files:
            try:
                temp_file.unlink()
                cleaned_count += 1
                logger.debug(f"Cleaned up temporary file: {temp_file}")
            except Exception as e:
                logger.warning(f"Failed to clean up {temp_file}: {e}")
        
        if cleaned_count > 0:
            logger.info(f"Cleaned up {cleaned_count} temporary files")
        
        return cleaned_count
    
    def validate_results(self, results: AnalysisResults) -> List[str]:
        """Validate analysis results for export.
        
        Args:
            results: Analysis results to validate
            
        Returns:
            List of validation warnings (empty if all good)
        """
        warnings = []
        
        # Check if results have any segments
        if not results.segments:
            warnings.append("No analysis segments found")
        
        # Check media info
        if not results.media_info:
            warnings.append("Missing media information")
        elif results.media_info.duration <= 0:
            warnings.append("Invalid media duration")
        
        # Check metadata
        if not results.metadata or not results.metadata.timestamp:
            warnings.append("Missing processing metadata")
        
        # Check for transcription data (needed for subtitles)
        from src.models.segments import TranscriptionSegment
        transcription_segments = results.get_segments_by_type(TranscriptionSegment)
        if not transcription_segments:
            warnings.append("No transcription segments found (subtitle formats may be empty)")
        
        return warnings
    
    def __str__(self) -> str:
        """String representation of pipeline."""
        stats = self.get_stats()
        return (f"OutputPipeline(exports={stats['total_exports']}, "
                f"success_rate={stats['success_rate']:.1%}, "
                f"cached_serializers={stats['cached_serializers']})")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        try:
            self.cleanup_temp_files()
            self.clear_cache()
        except Exception as e:
            logger.warning(f"Cleanup failed: {e}")