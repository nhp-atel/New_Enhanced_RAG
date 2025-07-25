"""Command Line Interface for the RAG system."""

import asyncio
import json
import sys
from pathlib import Path
from typing import Optional, Dict, Any, List
import click
import yaml
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
from rich.syntax import Syntax

from ..utils.config import ConfigManager, RAGConfig
from ..utils.logging import setup_logging
from ..utils.metrics import setup_metrics
from ..pipeline import RAGPipeline

console = Console()


@click.group()
@click.option('--config-dir', default='./config', help='Configuration directory path')
@click.option('--environment', '-e', default=None, help='Environment (local, staging, production)')
@click.option('--log-level', default='INFO', help='Logging level')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
@click.pass_context
def cli(ctx, config_dir, environment, log_level, verbose):
    """Enhanced RAG System CLI."""
    
    # Ensure context object exists
    ctx.ensure_object(dict)
    
    # Setup logging
    setup_logging(
        level=log_level,
        format_type="text" if verbose else "json"
    )
    
    # Load configuration
    config_manager = ConfigManager(config_dir)
    config = config_manager.load_config(environment)
    
    ctx.obj['config'] = config
    ctx.obj['config_manager'] = config_manager
    ctx.obj['verbose'] = verbose


@cli.command()
@click.argument('document_paths', nargs=-1, required=True)
@click.option('--batch-size', default=10, help='Batch size for processing')
@click.option('--overwrite', is_flag=True, help='Overwrite existing documents')
@click.pass_context
def ingest(ctx, document_paths: tuple, batch_size: int, overwrite: bool):
    """Ingest documents into the RAG system."""
    
    config: RAGConfig = ctx.obj['config']
    
    with console.status("Initializing RAG pipeline..."):
        pipeline = _create_pipeline(config)
    
    # Validate document paths
    valid_paths = []
    for path in document_paths:
        path_obj = Path(path)
        if path_obj.exists():
            if path_obj.is_file():
                valid_paths.append(str(path_obj))
            elif path_obj.is_dir():
                # Add all supported files in directory
                for file_path in path_obj.rglob('*'):
                    if file_path.is_file() and file_path.suffix.lower() in ['.txt', '.pdf', '.md']:
                        valid_paths.append(str(file_path))
        else:
            console.print(f"[red]Warning: Path not found: {path}[/red]")
    
    if not valid_paths:
        console.print("[red]Error: No valid document paths found[/red]")
        sys.exit(1)
    
    console.print(f"[green]Found {len(valid_paths)} documents to ingest[/green]")
    
    if ctx.obj['verbose']:
        for path in valid_paths:
            console.print(f"  • {path}")
    
    # Run ingestion
    async def run_ingestion():
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Ingesting documents...", total=None)
            
            try:
                stats = await pipeline.ingest_documents(valid_paths, batch_size)
                progress.update(task, completed=True)
                return stats
            except Exception as e:
                progress.update(task, completed=True)
                raise e
    
    try:
        stats = asyncio.run(run_ingestion())
        
        # Display results
        _display_ingestion_stats(stats)
        
    except Exception as e:
        console.print(f"[red]Error during ingestion: {e}[/red]")
        sys.exit(1)


@cli.command()
@click.argument('question')
@click.option('--top-k', default=None, type=int, help='Number of chunks to retrieve')
@click.option('--include-sources', is_flag=True, help='Include source information')
@click.option('--json-output', is_flag=True, help='Output results as JSON')
@click.pass_context
def query(ctx, question: str, top_k: Optional[int], include_sources: bool, json_output: bool):
    """Query the RAG system."""
    
    config: RAGConfig = ctx.obj['config']
    
    with console.status("Initializing RAG pipeline..."):
        pipeline = _create_pipeline(config)
    
    async def run_query():
        return await pipeline.query(
            question=question,
            top_k=top_k,
            include_metadata=include_sources
        )
    
    try:
        response = asyncio.run(run_query())
        
        if json_output:
            console.print(json.dumps(response, indent=2))
        else:
            _display_query_response(response, include_sources)
            
    except Exception as e:
        console.print(f"[red]Error during query: {e}[/red]")
        sys.exit(1)


@cli.command()
@click.option('--json-output', is_flag=True, help='Output as JSON')
@click.pass_context
def stats(ctx, json_output: bool):
    """Show system statistics."""
    
    config: RAGConfig = ctx.obj['config']
    
    with console.status("Gathering statistics..."):
        pipeline = _create_pipeline(config)
    
    async def get_stats():
        return await pipeline.get_stats()
    
    try:
        stats = asyncio.run(get_stats())
        
        if json_output:
            console.print(json.dumps(stats, indent=2))
        else:
            _display_stats(stats)
            
    except Exception as e:
        console.print(f"[red]Error retrieving stats: {e}[/red]")
        sys.exit(1)


@cli.command()
@click.option('--json-output', is_flag=True, help='Output as JSON')
@click.pass_context
def health(ctx, json_output: bool):
    """Check system health."""
    
    config: RAGConfig = ctx.obj['config']
    
    with console.status("Checking system health..."):
        pipeline = _create_pipeline(config)
    
    async def check_health():
        return await pipeline.health_check()
    
    try:
        health_status = asyncio.run(check_health())
        
        if json_output:
            console.print(json.dumps(health_status, indent=2))
        else:
            _display_health_status(health_status)
            
    except Exception as e:
        console.print(f"[red]Error during health check: {e}[/red]")
        sys.exit(1)


@cli.command()
@click.option('--port', default=8000, help='Port to run metrics server on')
@click.option('--duration', default=60, help='Duration to run server (seconds)')
@click.pass_context
def serve_metrics(ctx, port: int, duration: int):
    """Start metrics server."""
    
    try:
        metrics = setup_metrics(port)
        console.print(f"[green]Metrics server started on port {port}[/green]")
        console.print(f"[blue]Visit http://localhost:{port}/metrics[/blue]")
        
        import time
        time.sleep(duration)
        
    except Exception as e:
        console.print(f"[red]Error starting metrics server: {e}[/red]")
        sys.exit(1)


@cli.command()
@click.option('--output-file', help='Output configuration to file')
@click.pass_context
def show_config(ctx, output_file: Optional[str]):
    """Show current configuration."""
    
    config: RAGConfig = ctx.obj['config']
    config_dict = config.model_dump()
    
    if output_file:
        with open(output_file, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
        console.print(f"[green]Configuration saved to {output_file}[/green]")
    else:
        # Display as YAML
        yaml_content = yaml.dump(config_dict, default_flow_style=False, indent=2)
        syntax = Syntax(yaml_content, "yaml", theme="monokai", line_numbers=True)
        console.print(Panel(syntax, title="Current Configuration"))


@cli.command()
@click.option('--interactive', '-i', is_flag=True, help='Interactive query mode')
@click.pass_context
def chat(ctx, interactive: bool):
    """Interactive chat mode."""
    
    config: RAGConfig = ctx.obj['config']
    
    with console.status("Initializing RAG pipeline..."):
        pipeline = _create_pipeline(config)
    
    console.print("[green]RAG Chat Mode - Type 'exit' to quit[/green]")
    console.print()
    
    async def process_question(question: str):
        return await pipeline.query(question, include_metadata=True)
    
    while True:
        try:
            question = console.input("[bold blue]You: [/bold blue]")
            
            if question.lower() in ['exit', 'quit', 'bye']:
                console.print("[yellow]Goodbye![/yellow]")
                break
            
            if not question.strip():
                continue
            
            with console.status("Thinking..."):
                response = asyncio.run(process_question(question))
            
            console.print(f"[bold green]Assistant:[/bold green] {response['answer']}")
            
            if ctx.obj['verbose'] and response.get('sources'):
                console.print(f"\n[dim]Sources: {response['retrieved_chunks']} chunks[/dim]")
            
            console.print()
            
        except KeyboardInterrupt:
            console.print("\n[yellow]Goodbye![/yellow]")
            break
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")


def _create_pipeline(config: RAGConfig) -> RAGPipeline:
    """Create RAG pipeline from configuration."""
    # This would normally create actual implementations
    # For now, return a mock pipeline for demonstration
    raise NotImplementedError("Pipeline creation requires actual implementations")


def _display_ingestion_stats(stats: Dict[str, Any]):
    """Display ingestion statistics."""
    
    table = Table(title="Ingestion Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Documents Processed", str(stats['documents_processed']))
    table.add_row("Chunks Created", str(stats['chunks_created']))
    table.add_row("Embeddings Generated", str(stats['embeddings_generated']))
    table.add_row("Processing Time", f"{stats['processing_time_seconds']:.2f}s")
    
    console.print(table)
    
    if stats['errors']:
        console.print(f"\n[red]Errors ({len(stats['errors'])}):[/red]")
        for error in stats['errors']:
            console.print(f"  • {error}")


def _display_query_response(response: Dict[str, Any], include_sources: bool):
    """Display query response."""
    
    console.print(Panel(response['answer'], title="Answer", border_style="green"))
    
    # Metadata
    metadata_table = Table(show_header=False, box=None)
    metadata_table.add_column("Key", style="dim")
    metadata_table.add_column("Value", style="cyan")
    
    metadata_table.add_row("Processing Time", f"{response['processing_time_ms']}ms")
    metadata_table.add_row("Retrieved Chunks", str(response['retrieved_chunks']))
    metadata_table.add_row("Correlation ID", response['correlation_id'])
    
    console.print(metadata_table)
    
    # Sources
    if include_sources and response.get('sources'):
        sources_table = Table(title="Sources")
        sources_table.add_column("Rank", width=4)
        sources_table.add_column("Score", width=6)
        sources_table.add_column("Content", style="dim")
        
        for i, source in enumerate(response['sources'][:5], 1):
            sources_table.add_row(
                str(i),
                f"{source['score']:.3f}",
                source['content'][:100] + "..."
            )
        
        console.print(sources_table)


def _display_stats(stats: Dict[str, Any]):
    """Display system statistics."""
    
    # Pipeline config
    config_table = Table(title="Pipeline Configuration")
    config_table.add_column("Setting", style="cyan")
    config_table.add_column("Value", style="green")
    
    for key, value in stats['pipeline_config'].items():
        config_table.add_row(key.replace('_', ' ').title(), str(value))
    
    console.print(config_table)
    
    # Vector store stats
    if stats.get('vector_store'):
        vs_table = Table(title="Vector Store")
        vs_table.add_column("Metric", style="cyan")
        vs_table.add_column("Value", style="green")
        
        for key, value in stats['vector_store'].items():
            vs_table.add_row(key.replace('_', ' ').title(), str(value))
        
        console.print(vs_table)


def _display_health_status(health: Dict[str, Any]):
    """Display health check results."""
    
    status_color = {
        "healthy": "green",
        "degraded": "yellow", 
        "unhealthy": "red"
    }.get(health['status'], "white")
    
    console.print(f"[{status_color}]System Status: {health['status'].upper()}[/{status_color}]")
    
    components_table = Table(title="Component Health")
    components_table.add_column("Component", style="cyan")
    components_table.add_column("Status", style="white")
    
    for component, status in health['components'].items():
        if status == "healthy":
            status_text = f"[green]{status}[/green]"
        else:
            status_text = f"[red]{status}[/red]"
        
        components_table.add_row(component, status_text)
    
    console.print(components_table)


if __name__ == '__main__':
    cli()