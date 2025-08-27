"""Report generation for duplicate code findings."""

import re
from pathlib import Path
from typing import List, Dict, Any, Optional, TextIO, Tuple
from datetime import datetime
from dataclasses import dataclass, asdict
from collections import defaultdict
import difflib
import logging

from .storage import DuplicateFinding, EchoDatabase, BlockRecord, FileRecord
from .scan import ScanResult, ScanStatistics
from .utils import JsonHandler, PathHandler, handle_errors

logger = logging.getLogger(__name__)


@dataclass
class ReportConfig:
    """Configuration for report generation."""
    include_code_frames: bool = True
    max_code_lines: int = 20
    group_by_folder: bool = True
    sort_by_score: bool = True
    show_side_by_side: bool = True
    include_statistics: bool = True
    include_file_paths: bool = True
    highlight_syntax: bool = True
    min_refactor_score: float = 100.0
    context_lines: int = 2


class MarkdownReporter:
    """Generate Markdown reports for duplicate findings."""
    
    def __init__(self, config: ReportConfig, database: Optional[EchoDatabase] = None):
        self.config = config
        self.database = database
        
    def generate_report(self, scan_result: ScanResult, output_path: Path) -> None:
        """Generate a Markdown report."""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                self._write_header(f, scan_result)
                self._write_summary(f, scan_result)
                self._write_findings(f, scan_result.findings)
                self._write_footer(f)
        except Exception as e:
            logger.error(f"Failed to generate markdown report: {e}")
            raise
            
    def _write_header(self, f: TextIO, scan_result: ScanResult) -> None:
        """Write report header."""
        f.write("# 🔍 Duplicate Code Detection Report\n\n")
        f.write(f"**Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  \n")
        f.write(f"**Scan time:** {scan_result.statistics.scan_time_ms:,}ms  \n")
        f.write(f"**Total findings:** {len(scan_result.findings):,}  \n\n")
        
        if scan_result.findings:
            high_score_findings = [f for f in scan_result.findings 
                                 if f.scores.get('R', 0) >= self.config.min_refactor_score]
            f.write(f"**High-priority findings:** {len(high_score_findings):,}  \n\n")

    def _write_summary(self, f: TextIO, scan_result: ScanResult) -> None:
        """Write summary statistics."""
        if not self.config.include_statistics:
            return
            
        stats = scan_result.statistics
        findings = scan_result.findings
        
        f.write("## 📊 Summary\n\n")
        
        # Processing statistics
        f.write("### Processing Statistics\n\n")
        f.write(f"| Metric | Count |\n")
        f.write(f"|--------|-------|\n")
        f.write(f"| Files processed | {stats.files_processed:,} |\n")
        f.write(f"| Code blocks extracted | {stats.blocks_extracted:,} |\n")
        f.write(f"| LSH queries | {stats.lsh_queries:,} |\n")
        f.write(f"| Candidates generated | {stats.candidates_generated:,} |\n")
        f.write(f"| Semantic comparisons | {stats.semantic_comparisons:,} |\n")
        f.write(f"| Duplicate findings | {len(findings):,} |\n\n")
        
        # Quality distribution
        if findings:
            f.write("### Finding Quality Distribution\n\n")
            
            # Categorize by refactor score
            high_quality = len([f for f in findings if f.scores.get('R', 0) >= 500])
            medium_quality = len([f for f in findings if 200 <= f.scores.get('R', 0) < 500])
            low_quality = len([f for f in findings if f.scores.get('R', 0) < 200])
            
            f.write(f"- **High Priority** (R ≥ 500): {high_quality} findings\n")
            f.write(f"- **Medium Priority** (200 ≤ R < 500): {medium_quality} findings\n")
            f.write(f"- **Low Priority** (R < 200): {low_quality} findings\n\n")
            
            # Semantic similarity distribution
            semantic_scores = [f.scores.get('semantic', 0) for f in findings]
            if semantic_scores:
                avg_semantic = sum(semantic_scores) / len(semantic_scores)
                f.write(f"**Average semantic similarity:** {avg_semantic:.3f}\n\n")
        
        # Performance breakdown
        if stats.stage_times_ms:
            f.write("### Performance Breakdown\n\n")
            f.write(f"| Stage | Time (ms) | Percentage |\n")
            f.write(f"|-------|-----------|------------|\n")
            
            total_time = stats.scan_time_ms or sum(stats.stage_times_ms.values())
            for stage, time_ms in sorted(stats.stage_times_ms.items()):
                percentage = (time_ms / total_time * 100) if total_time > 0 else 0
                f.write(f"| {stage.replace('_', ' ').title()} | {time_ms:,} | {percentage:.1f}% |\n")
            f.write("\n")

    def _write_findings(self, f: TextIO, findings: List[DuplicateFinding]) -> None:
        """Write detailed findings."""
        if not findings:
            f.write("## 🔍 Findings\n\n✅ No duplicate code found.\n")
            return
        
        # Filter by minimum refactor score if configured
        filtered_findings = [
            f for f in findings 
            if f.scores.get('R', 0) >= self.config.min_refactor_score
        ]
        
        if not filtered_findings:
            f.write(f"## 🔍 Findings\n\n")
            f.write(f"ℹ️  {len(findings)} findings found, but none meet the minimum refactor score threshold of {self.config.min_refactor_score}.\n\n")
            return
            
        f.write(f"## 🔍 Findings ({len(filtered_findings)} of {len(findings)} shown)\n\n")
        
        # Group by folder if configured
        if self.config.group_by_folder:
            grouped = self._group_by_folder(filtered_findings)
            for folder, folder_findings in sorted(grouped.items()):
                if folder_findings:
                    f.write(f"### 📁 {folder}\n\n")
                    self._write_finding_group(f, folder_findings)
        else:
            self._write_finding_group(f, filtered_findings)

    def _group_by_folder(self, findings: List[DuplicateFinding]) -> Dict[str, List[DuplicateFinding]]:
        """Group findings by folder."""
        grouped = defaultdict(list)
        
        for finding in findings:
            # Extract folder from block IDs if we have database access
            folder = "Unknown"
            
            if self.database:
                try:
                    # Get file paths for both blocks
                    block1 = self.database.session.query(BlockRecord).filter(
                        BlockRecord.id == finding.block_id
                    ).first()
                    
                    if block1 and block1.file_id:
                        file_record = self.database.session.query(FileRecord).filter(
                            FileRecord.id == block1.file_id
                        ).first()
                        
                        if file_record:
                            file_path = Path(file_record.path)
                            folder = str(file_path.parent) if file_path.parent != Path('.') else "Root"
                        
                except Exception as e:
                    logger.debug(f"Could not determine folder for finding {finding.id}: {e}")
            
            grouped[folder].append(finding)
        
        return dict(grouped)

    def _write_finding_group(self, f: TextIO, findings: List[DuplicateFinding]) -> None:
        """Write a group of findings."""
        # Sort by refactor score if configured
        if self.config.sort_by_score:
            findings = sorted(findings, key=lambda x: x.scores.get('R', 0), reverse=True)
        
        for i, finding in enumerate(findings, 1):
            self._write_finding(f, finding, i)

    def _write_finding(self, f: TextIO, finding: DuplicateFinding, index: int) -> None:
        """Write a single finding with detailed information."""
        refactor_score = finding.scores.get('R', 0)
        semantic_score = finding.scores.get('semantic', 0)
        
        # Priority emoji based on refactor score
        if refactor_score >= 500:
            priority_emoji = "🔴"
            priority_text = "HIGH"
        elif refactor_score >= 200:
            priority_emoji = "🟡"
            priority_text = "MEDIUM"
        else:
            priority_emoji = "🟢"
            priority_text = "LOW"
        
        f.write(f"#### {priority_emoji} Finding #{index}: {priority_text} Priority\n\n")
        
        # Core metrics
        f.write(f"**Refactor Score:** {refactor_score:.1f} | ")
        f.write(f"**Semantic Similarity:** {semantic_score:.3f} | ")
        f.write(f"**Type:** {finding.type}\n\n")
        
        # Additional scores if available
        other_scores = {k: v for k, v in finding.scores.items() 
                       if k not in ['R', 'semantic']}
        if other_scores:
            f.write("**Additional Metrics:** ")
            score_parts = [f"{k}: {v:.3f}" for k, v in other_scores.items()]
            f.write(" | ".join(score_parts))
            f.write("\n\n")
        
        # Code blocks comparison
        if self.config.include_code_frames:
            self._write_code_comparison(f, finding)
        
        f.write("---\n\n")
    
    def _write_code_comparison(self, f: TextIO, finding: DuplicateFinding) -> None:
        """Write code comparison for a finding."""
        if not self.database:
            f.write("**Code Blocks:** _Database connection required to display code_\n\n")
            return
        
        try:
            # Get code blocks from database
            block1 = self._get_block_content(finding.block_id)
            block2 = self._get_block_content(finding.match_block_id)
            
            if not block1 or not block2:
                f.write("**Code Blocks:** _Could not retrieve code content_\n\n")
                return
            
            f.write("**Code Comparison:**\n\n")
            
            if self.config.show_side_by_side:
                self._write_side_by_side_comparison(f, block1, block2)
            else:
                self._write_sequential_blocks(f, block1, block2)
                
        except Exception as e:
            logger.debug(f"Failed to write code comparison: {e}")
            f.write("**Code Blocks:** _Error retrieving code content_\n\n")
    
    def _get_block_content(self, block_id: str) -> Optional[Dict[str, Any]]:
        """Get block content and metadata from database."""
        try:
            block = self.database.session.query(BlockRecord).filter(
                BlockRecord.id == block_id
            ).first()
            
            if not block:
                return None
            
            # Get file information
            file_record = None
            if block.file_id:
                file_record = self.database.session.query(FileRecord).filter(
                    FileRecord.id == block.file_id
                ).first()
            
            # Read file content if available
            content_lines = []
            language = "text"
            
            if file_record and Path(file_record.path).exists():
                try:
                    with open(file_record.path, 'r', encoding='utf-8') as file:
                        all_lines = file.readlines()
                        
                        # Extract lines around the block
                        start_line = max(0, block.start_line - self.config.context_lines - 1)
                        end_line = min(len(all_lines), block.end_line + self.config.context_lines)
                        
                        content_lines = all_lines[start_line:end_line]
                        
                        # Determine language from file extension
                        file_path = Path(file_record.path)
                        language = self._get_language_from_extension(file_path.suffix)
                        
                except Exception as e:
                    logger.debug(f"Could not read file {file_record.path}: {e}")
                    content_lines = []
            
            return {
                'id': block_id,
                'file_path': file_record.path if file_record else 'Unknown',
                'start_line': block.start_line,
                'end_line': block.end_line,
                'content_lines': content_lines,
                'language': language,
                'content': block.content or ''
            }
            
        except Exception as e:
            logger.debug(f"Error getting block content for {block_id}: {e}")
            return None
    
    def _get_language_from_extension(self, extension: str) -> str:
        """Map file extension to language identifier."""
        extension_map = {
            '.py': 'python',
            '.js': 'javascript',
            '.ts': 'typescript',
            '.tsx': 'typescript',
            '.jsx': 'javascript',
            '.java': 'java',
            '.c': 'c',
            '.cpp': 'cpp',
            '.cc': 'cpp',
            '.h': 'c',
            '.hpp': 'cpp',
            '.cs': 'csharp',
            '.php': 'php',
            '.rb': 'ruby',
            '.go': 'go',
            '.rs': 'rust',
            '.swift': 'swift',
            '.kt': 'kotlin',
            '.scala': 'scala',
        }
        return extension_map.get(extension.lower(), 'text')
    
    def _write_side_by_side_comparison(self, f: TextIO, block1: Dict, block2: Dict) -> None:
        """Write side-by-side code comparison."""
        f.write('<div style="display: flex; gap: 20px;">\n\n')
        
        # Block 1
        f.write('<div style="flex: 1;">\n\n')
        f.write(f"**📄 {Path(block1['file_path']).name}** (lines {block1['start_line']}-{block1['end_line']})\n\n")
        
        if block1['content_lines']:
            language = block1['language'] if self.config.highlight_syntax else 'text'
            f.write(f"```{language}\n")
            for i, line in enumerate(block1['content_lines']):
                line_num = block1['start_line'] - self.config.context_lines + i
                if line_num >= 1:  # Only show valid line numbers
                    f.write(f"{line_num:4d}: {line.rstrip()}\n")
            f.write("```\n\n")
        else:
            f.write("```\n[Content not available]\n```\n\n")
        
        f.write('</div>\n\n')
        
        # Block 2
        f.write('<div style="flex: 1;">\n\n')
        f.write(f"**📄 {Path(block2['file_path']).name}** (lines {block2['start_line']}-{block2['end_line']})\n\n")
        
        if block2['content_lines']:
            language = block2['language'] if self.config.highlight_syntax else 'text'
            f.write(f"```{language}\n")
            for i, line in enumerate(block2['content_lines']):
                line_num = block2['start_line'] - self.config.context_lines + i
                if line_num >= 1:  # Only show valid line numbers
                    f.write(f"{line_num:4d}: {line.rstrip()}\n")
            f.write("```\n\n")
        else:
            f.write("```\n[Content not available]\n```\n\n")
        
        f.write('</div>\n\n')
        f.write('</div>\n\n')
        
        # Show diff if both blocks have content
        if block1['content_lines'] and block2['content_lines']:
            self._write_unified_diff(f, block1, block2)
    
    def _write_sequential_blocks(self, f: TextIO, block1: Dict, block2: Dict) -> None:
        """Write code blocks sequentially."""
        for i, block in enumerate([block1, block2], 1):
            f.write(f"**Block {i}:** `{Path(block['file_path']).name}` "
                   f"(lines {block['start_line']}-{block['end_line']})\n\n")
            
            if block['content_lines']:
                language = block['language'] if self.config.highlight_syntax else 'text'
                f.write(f"```{language}\n")
                for j, line in enumerate(block['content_lines']):
                    line_num = block['start_line'] - self.config.context_lines + j
                    if line_num >= 1:
                        f.write(f"{line_num:4d}: {line.rstrip()}\n")
                f.write("```\n\n")
            else:
                f.write("```\n[Content not available]\n```\n\n")
    
    def _write_unified_diff(self, f: TextIO, block1: Dict, block2: Dict) -> None:
        """Write unified diff between two blocks."""
        try:
            # Clean content for diffing
            content1 = [line.rstrip() for line in block1['content_lines']]
            content2 = [line.rstrip() for line in block2['content_lines']]
            
            diff = list(difflib.unified_diff(
                content1,
                content2,
                fromfile=f"{Path(block1['file_path']).name} (lines {block1['start_line']}-{block1['end_line']})",
                tofile=f"{Path(block2['file_path']).name} (lines {block2['start_line']}-{block2['end_line']})",
                lineterm=''
            ))
            
            if diff:
                f.write("**Unified Diff:**\n\n")
                f.write("```diff\n")
                for line in diff:
                    f.write(f"{line}\n")
                f.write("```\n\n")
                
        except Exception as e:
            logger.debug(f"Could not generate diff: {e}")
    
    def _write_footer(self, f: TextIO) -> None:
        """Write report footer."""
        f.write("---\n\n")
        f.write("*Generated by Echo Duplicate Code Detection*\n")
        f.write(f"*Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")


class JSONReporter:
    """Generate JSON reports for duplicate findings."""
    
    def __init__(self, database: Optional[EchoDatabase] = None):
        self.database = database
    
    def generate_report(self, scan_result: ScanResult, output_path: Path) -> None:
        """Generate a comprehensive JSON report."""
        try:
            report_data = {
                'metadata': {
                    'generated_at': datetime.now().isoformat(),
                    'echo_version': '0.1.0',  # TODO: Get from __version__
                    'scan_time_ms': scan_result.statistics.scan_time_ms,
                    'total_findings': len(scan_result.findings)
                },
                'statistics': self._serialize_statistics(scan_result.statistics),
                'findings': [self._serialize_finding(f) for f in scan_result.findings],
                'summary': self._generate_summary(scan_result)
            }
            
            JsonHandler.write_file(report_data, output_path)
                
        except Exception as e:
            logger.error(f"Failed to generate JSON report: {e}")
            raise
    
    def _serialize_statistics(self, stats: ScanStatistics) -> Dict[str, Any]:
        """Serialize scan statistics."""
        return {
            'files_processed': stats.files_processed,
            'blocks_extracted': stats.blocks_extracted,
            'blocks_normalized': stats.blocks_normalized,
            'blocks_indexed': stats.blocks_indexed,
            'lsh_queries': stats.lsh_queries,
            'candidates_generated': stats.candidates_generated,
            'near_miss_verifications': stats.near_miss_verifications,
            'semantic_comparisons': stats.semantic_comparisons,
            'findings_generated': stats.findings_generated,
            'findings_filtered': stats.findings_filtered,
            'errors': stats.errors,
            'skipped_files': stats.skipped_files,
            'scan_time_ms': stats.scan_time_ms,
            'stage_times_ms': stats.stage_times_ms
        }
    
    def _generate_summary(self, scan_result: ScanResult) -> Dict[str, Any]:
        """Generate summary statistics."""
        findings = scan_result.findings
        
        if not findings:
            return {
                'total_findings': 0,
                'quality_distribution': {},
                'semantic_similarity': {}
            }
        
        # Quality distribution by refactor score
        high_quality = len([f for f in findings if f.scores.get('R', 0) >= 500])
        medium_quality = len([f for f in findings if 200 <= f.scores.get('R', 0) < 500])
        low_quality = len([f for f in findings if f.scores.get('R', 0) < 200])
        
        # Semantic similarity statistics
        semantic_scores = [f.scores.get('semantic', 0) for f in findings]
        semantic_stats = {}
        if semantic_scores:
            semantic_stats = {
                'average': sum(semantic_scores) / len(semantic_scores),
                'min': min(semantic_scores),
                'max': max(semantic_scores),
                'median': sorted(semantic_scores)[len(semantic_scores) // 2]
            }
        
        return {
            'total_findings': len(findings),
            'quality_distribution': {
                'high_priority': high_quality,
                'medium_priority': medium_quality,
                'low_priority': low_quality
            },
            'semantic_similarity': semantic_stats,
            'top_findings': [
                {
                    'id': f.id,
                    'refactor_score': f.scores.get('R', 0),
                    'semantic_score': f.scores.get('semantic', 0)
                }
                for f in sorted(findings, key=lambda x: x.scores.get('R', 0), reverse=True)[:5]
            ]
        }
            
    def _serialize_finding(self, finding: DuplicateFinding) -> Dict[str, Any]:
        """Serialize a finding for JSON output with enhanced information."""
        base_data = {
            'id': finding.id,
            'block_id': finding.block_id,
            'match_block_id': finding.match_block_id,
            'scores': finding.scores,
            'type': finding.type,
            'confidence': finding.confidence,
            'refactor_score': finding.refactor_score,
            'status': finding.status,
            'created_at': finding.created_at.isoformat() if finding.created_at else None
        }
        
        # Add block information if database is available
        if self.database:
            try:
                block1_info = self._get_block_info(finding.block_id)
                block2_info = self._get_block_info(finding.match_block_id)
                
                if block1_info or block2_info:
                    base_data['blocks'] = {
                        'block1': block1_info,
                        'block2': block2_info
                    }
                    
            except Exception as e:
                logger.debug(f"Could not fetch block info for finding {finding.id}: {e}")
        
        return base_data
    
    def _get_block_info(self, block_id: str) -> Optional[Dict[str, Any]]:
        """Get block information from database."""
        try:
            block = self.database.session.query(BlockRecord).filter(
                BlockRecord.id == block_id
            ).first()
            
            if not block:
                return None
            
            # Get file information
            file_info = None
            if block.file_id:
                file_record = self.database.session.query(FileRecord).filter(
                    FileRecord.id == block.file_id
                ).first()
                
                if file_record:
                    file_info = {
                        'path': file_record.path,
                        'language': file_record.language,
                        'size_bytes': file_record.size_bytes
                    }
            
            return {
                'id': block_id,
                'start_line': block.start_line,
                'end_line': block.end_line,
                'token_count': block.token_count,
                'ast_hash': block.ast_hash,
                'file': file_info
            }
            
        except Exception as e:
            logger.debug(f"Error getting block info for {block_id}: {e}")
            return None
        

def generate_markdown_report(scan_result: ScanResult, output_path: Path,
                           config: Optional[ReportConfig] = None,
                           database: Optional[EchoDatabase] = None) -> None:
    """Generate a comprehensive Markdown report."""
    if config is None:
        config = ReportConfig()
    
    # Initialize database if not provided
    if database is None:
        try:
            from .storage import create_database
            from .config import EchoConfig
            
            echo_config = EchoConfig()
            cache_dir = echo_config.get_cache_dir()
            db_path = cache_dir / "echo.db"
            
            if db_path.exists():
                database = create_database(db_path)
                
        except Exception as e:
            logger.debug(f"Could not initialize database for report: {e}")
    
    reporter = MarkdownReporter(config, database)
    reporter.generate_report(scan_result, output_path)
    
    if database:
        database.session.close()
    
    
def generate_json_report(scan_result: ScanResult, output_path: Path,
                        database: Optional[EchoDatabase] = None) -> None:
    """Generate a comprehensive JSON report."""
    # Initialize database if not provided
    if database is None:
        try:
            from .storage import create_database
            from .config import EchoConfig
            
            echo_config = EchoConfig()
            cache_dir = echo_config.get_cache_dir()
            db_path = cache_dir / "echo.db"
            
            if db_path.exists():
                database = create_database(db_path)
                
        except Exception as e:
            logger.debug(f"Could not initialize database for report: {e}")
    
    reporter = JSONReporter(database)
    reporter.generate_report(scan_result, output_path)
    
    if database:
        database.session.close()


def create_report_config(**kwargs) -> ReportConfig:
    """Create a ReportConfig with custom settings."""
    return ReportConfig(**kwargs)


def format_code_snippet(content: str, language: str = "text", 
                       max_lines: int = 20, show_line_numbers: bool = True) -> str:
    """Format a code snippet for display."""
    lines = content.split('\n')
    
    if len(lines) > max_lines:
        # Truncate and add ellipsis
        lines = lines[:max_lines]
        lines.append("... (truncated)")
    
    if show_line_numbers:
        numbered_lines = [f"{i+1:4d}: {line}" for i, line in enumerate(lines)]
        return '\n'.join(numbered_lines)
    else:
        return '\n'.join(lines)


def calculate_similarity_percentage(scores: Dict[str, float]) -> float:
    """Calculate overall similarity percentage from scores."""
    # Weighted average of available similarity scores
    semantic_weight = 0.6
    structural_weight = 0.4
    
    semantic_score = scores.get('semantic', 0.0)
    structural_score = scores.get('structural', scores.get('jaccard', 0.0))
    
    return (semantic_score * semantic_weight + structural_score * structural_weight) * 100


def get_priority_label(refactor_score: float) -> Tuple[str, str]:
    """Get priority label and emoji for a refactor score."""
    if refactor_score >= 500:
        return ("HIGH", "🔴")
    elif refactor_score >= 200:
        return ("MEDIUM", "🟡")
    else:
        return ("LOW", "🟢")