#!/usr/bin/env python3
"""
CLI entrypoint for echo duplicate code detection tool.

Provides commands for indexing repositories, scanning for duplicates,
and generating reports in various formats.
"""

import sys
import logging
from pathlib import Path
from typing import Optional, List
import subprocess
import os

import click
from tqdm import tqdm

from . import __version__
from .config import EchoConfig
from .scan import scan_repository, scan_changed_files, ScanResult
from .report import generate_markdown_report, generate_json_report, ReportConfig
from .index import index_repository
from .storage import EchoDatabase, create_database

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@click.group()
@click.version_option(version=__version__)
@click.option('--config', type=click.Path(exists=True, path_type=Path), 
              help='Path to configuration file')
@click.option('--cache-dir', type=click.Path(path_type=Path),
              help='Cache directory for database and models')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
@click.option('--quiet', '-q', is_flag=True, help='Suppress non-error output')
@click.pass_context
def cli(ctx: click.Context, config: Optional[Path], cache_dir: Optional[Path],
        verbose: bool, quiet: bool) -> None:
    """Echo: Duplicate code detection for polyglot repositories."""
    ctx.ensure_object(dict)
    
    # Configure logging level
    if quiet:
        logging.getLogger('echo').setLevel(logging.ERROR)
    elif verbose:
        logging.getLogger('echo').setLevel(logging.DEBUG)
    else:
        logging.getLogger('echo').setLevel(logging.INFO)
    
    # Load configuration
    if config and config.exists():
        echo_config = EchoConfig.from_file(config)
        ctx.obj['config'] = echo_config
    else:
        echo_config = EchoConfig()
        ctx.obj['config'] = echo_config
    
    # Override cache directory if specified
    if cache_dir:
        echo_config.cache_dir = cache_dir
        
    ctx.obj['verbose'] = verbose
    ctx.obj['quiet'] = quiet


@cli.command()
@click.argument('paths', nargs=-1, type=click.Path(exists=True, path_type=Path))
@click.option('--reindex', is_flag=True, help='Force full reindex')
@click.pass_context
def index(ctx: click.Context, paths: tuple[Path, ...], reindex: bool) -> None:
    """Index repository files for duplicate detection."""
    config: EchoConfig = ctx.obj['config']
    verbose: bool = ctx.obj['verbose']
    
    # Use current directory if no paths specified
    if not paths:
        paths = (Path.cwd(),)
    
    with tqdm(desc="Indexing files", disable=ctx.obj['quiet']) as pbar:
        def progress_callback(message: str, progress: Optional[float] = None) -> None:
            if progress is not None:
                pbar.n = int(progress * 100)
                pbar.refresh()
            if verbose:
                tqdm.write(f"[INDEX] {message}")
        
        try:
            for path in paths:
                if verbose:
                    click.echo(f"Indexing: {path} (reindex={reindex})")
                
                # Initialize database
                cache_dir = config.get_cache_dir()
                cache_dir.mkdir(parents=True, exist_ok=True)
                db_path = cache_dir / "echo.db"
                
                database = create_database(db_path)
                
                try:
                    # Run indexing
                    result = index_repository(
                        path, 
                        config=config, 
                        reindex=reindex,
                        progress_callback=progress_callback
                    )
                    
                    click.echo(f"✓ Indexed {path}: {result.get('files_indexed', 0)} files, "
                              f"{result.get('blocks_indexed', 0)} blocks")
                    
                except Exception as e:
                    click.echo(f"✗ Failed to index {path}: {e}", err=True)
                    if verbose:
                        import traceback
                        click.echo(traceback.format_exc(), err=True)
                finally:
                    database.session.close()
                    
        except KeyboardInterrupt:
            click.echo("\nIndexing interrupted by user", err=True)
            sys.exit(1)
        except Exception as e:
            click.echo(f"Indexing failed: {e}", err=True)
            if verbose:
                import traceback
                click.echo(traceback.format_exc(), err=True)
            sys.exit(1)
    

@cli.command()
@click.argument('path', type=click.Path(exists=True, path_type=Path), required=False)
@click.option('--changed', is_flag=True, help='Only scan changed files')
@click.option('--repo', is_flag=True, help='Scan entire repository')
@click.option('--json', 'json_output', type=click.Path(path_type=Path), help='JSON output file')
@click.option('--markdown', type=click.Path(path_type=Path), help='Markdown output file')
@click.option('--budget-ms', type=int, default=20000, help='Time budget in milliseconds')
@click.option('--min-tokens', type=int, help='Minimum tokens per block (overrides config)')
@click.option('--tau-sem', type=float, help='Semantic similarity threshold (overrides config)')
@click.option('--exclude', multiple=True, help='Additional file patterns to exclude')
@click.option('--only-changed', is_flag=True, help='Alias for --changed')
@click.pass_context
def scan(ctx: click.Context, path: Optional[Path], changed: bool, repo: bool,
         json_output: Optional[Path], markdown: Optional[Path], budget_ms: int,
         min_tokens: Optional[int], tau_sem: Optional[float], exclude: tuple[str, ...],
         only_changed: bool) -> None:
    """Scan for duplicate code blocks."""
    config: EchoConfig = ctx.obj['config']
    verbose: bool = ctx.obj['verbose']
    
    # Handle flag aliases
    if only_changed:
        changed = True
    
    # Use current directory if no path specified
    if not path:
        path = Path.cwd()
    
    # Apply CLI overrides to config
    if min_tokens is not None:
        config.min_tokens = min_tokens
    if tau_sem is not None:
        config.tau_semantic = tau_sem
    if exclude:
        config.ignore_patterns.extend(exclude)
    
    # Determine default output files if none specified
    if not json_output and not markdown:
        markdown = path / "dupes.md"
    
    with tqdm(desc="Scanning for duplicates", disable=ctx.obj['quiet']) as pbar:
        def progress_callback(message: str, progress: Optional[float] = None) -> None:
            if progress is not None:
                pbar.n = int(progress * 100)
                pbar.refresh()
            if verbose:
                tqdm.write(f"[SCAN] {message}")
        
        try:
            # Run the appropriate scan
            if changed:
                # Get changed files from git
                changed_files = _get_changed_files(path)
                if not changed_files:
                    click.echo("No changed files detected")
                    return
                
                if verbose:
                    click.echo(f"Scanning {len(changed_files)} changed files")
                
                result = scan_changed_files(
                    changed_files,
                    config=config,
                    progress_callback=progress_callback
                )
            else:
                # Full repository scan
                if verbose:
                    click.echo(f"Scanning repository: {path}")
                
                result = scan_repository(
                    path,
                    config=config,
                    budget_ms=budget_ms,
                    progress_callback=progress_callback
                )
            
            # Generate reports
            _generate_reports(result, json_output, markdown, config, verbose)
            
            # Print summary
            _print_scan_summary(result, verbose)
            
        except KeyboardInterrupt:
            click.echo("\nScan interrupted by user", err=True)
            sys.exit(1)
        except Exception as e:
            click.echo(f"Scan failed: {e}", err=True)
            if verbose:
                import traceback
                click.echo(traceback.format_exc(), err=True)
            sys.exit(1)
    

@cli.command()
@click.pass_context
def status(ctx: click.Context) -> None:
    """Show indexing status and statistics."""
    config: EchoConfig = ctx.obj['config']
    verbose: bool = ctx.obj['verbose']
    
    try:
        # Initialize database connection
        cache_dir = config.get_cache_dir()
        db_path = cache_dir / "echo.db"
        
        if not db_path.exists():
            click.echo("No index found. Run 'echo-cli index' to create one.")
            return
        
        database = create_database(db_path)
        
        try:
            session = database.session
            
            # Get database statistics
            from .storage import FileRecord, BlockRecord, FindingRecord
            
            file_count = session.query(FileRecord).count()
            block_count = session.query(BlockRecord).count()
            finding_count = session.query(FindingRecord).count()
            
            # Get recent activity
            from sqlalchemy import func
            recent_files = session.query(
                FileRecord.path, FileRecord.indexed_at
            ).order_by(FileRecord.indexed_at.desc()).limit(5).all()
            
            # Print status
            click.echo("Echo Duplicate Detection Status")
            click.echo("=" * 35)
            click.echo(f"Cache Directory: {cache_dir}")
            click.echo(f"Database: {db_path}")
            click.echo(f"Database Size: {_format_file_size(db_path.stat().st_size)}")
            click.echo()
            click.echo("Index Statistics:")
            click.echo(f"  Files indexed: {file_count:,}")
            click.echo(f"  Code blocks: {block_count:,}")
            click.echo(f"  Duplicate findings: {finding_count:,}")
            
            if recent_files and verbose:
                click.echo()
                click.echo("Recently indexed files:")
                for file_path, indexed_at in recent_files:
                    click.echo(f"  {file_path} ({indexed_at})")
            
            # Configuration summary
            if verbose:
                click.echo()
                click.echo("Configuration:")
                click.echo(f"  Min tokens: {config.min_tokens}")
                click.echo(f"  Semantic threshold: {config.tau_semantic}")
                click.echo(f"  Supported languages: {', '.join(config.supported_languages)}")
                
        finally:
            database.session.close()
            
    except Exception as e:
        click.echo(f"Failed to get status: {e}", err=True)
        if verbose:
            import traceback
            click.echo(traceback.format_exc(), err=True)
        sys.exit(1)
    

def _get_changed_files(repo_path: Path) -> List[Path]:
    """Get list of changed files from git."""
    try:
        # Get changed files from git
        result = subprocess.run(
            ['git', 'diff', '--name-only', 'HEAD'],
            cwd=repo_path,
            capture_output=True,
            text=True,
            check=True
        )
        
        changed_files = []
        for line in result.stdout.strip().split('\n'):
            if line.strip():
                file_path = repo_path / line.strip()
                if file_path.exists():
                    changed_files.append(file_path)
        
        return changed_files
        
    except subprocess.CalledProcessError:
        # Fallback: return all files if not in git repo
        logger.warning("Not in git repository, scanning all files")
        return []
    except FileNotFoundError:
        logger.warning("Git not found, scanning all files")
        return []


def _generate_reports(result: ScanResult, json_output: Optional[Path],
                     markdown: Optional[Path], config: EchoConfig, verbose: bool) -> None:
    """Generate output reports."""
    if json_output:
        if verbose:
            click.echo(f"Generating JSON report: {json_output}")
        generate_json_report(result, json_output)
        click.echo(f"✓ JSON report saved to: {json_output}")
    
    if markdown:
        if verbose:
            click.echo(f"Generating Markdown report: {markdown}")
        
        report_config = ReportConfig(
            include_code_frames=True,
            max_code_lines=20,
            group_by_folder=True,
            sort_by_score=True
        )
        
        generate_markdown_report(result, markdown, report_config)
        click.echo(f"✓ Markdown report saved to: {markdown}")


def _print_scan_summary(result: ScanResult, verbose: bool) -> None:
    """Print scan summary to console."""
    stats = result.statistics
    findings = result.findings
    
    click.echo()
    click.echo("Scan Summary")
    click.echo("=" * 20)
    click.echo(f"Files processed: {stats.files_processed:,}")
    click.echo(f"Code blocks extracted: {stats.blocks_extracted:,}")
    click.echo(f"Duplicate findings: {len(findings):,}")
    click.echo(f"Scan time: {stats.scan_time_ms:,}ms")
    
    if findings:
        # Show top findings by refactor score
        top_findings = sorted(
            findings, 
            key=lambda f: f.scores.get('R', 0), 
            reverse=True
        )[:3]
        
        click.echo()
        click.echo("Top duplicate findings:")
        for i, finding in enumerate(top_findings, 1):
            refactor_score = finding.scores.get('R', 0)
            semantic_score = finding.scores.get('semantic', 0)
            click.echo(f"  {i}. Refactor Score: {refactor_score:.1f}, "
                      f"Semantic: {semantic_score:.3f}")
    
    if verbose and stats.stage_times_ms:
        click.echo()
        click.echo("Stage timings:")
        for stage, time_ms in stats.stage_times_ms.items():
            click.echo(f"  {stage}: {time_ms:,}ms")


def _format_file_size(size_bytes: int) -> str:
    """Format file size in human readable format."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"


def main() -> None:
    """Main entrypoint for the CLI."""
    try:
        cli()
    except KeyboardInterrupt:
        click.echo("\nOperation cancelled by user", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"Unexpected error: {e}", err=True)
        sys.exit(1)
    
    
if __name__ == '__main__':
    main()