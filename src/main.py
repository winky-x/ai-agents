#!/usr/bin/env python3
"""
Consiglio Agent - Main CLI Interface
A secure, agentic AI assistant with policy-based tool control
"""

import os
import sys
import json
import click
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt, Confirm
from rich.text import Text

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from core.orchestrator import Orchestrator
from core.policy import PolicyEngine

console = Console()


@click.group()
@click.option('--config', '-c', default='.env', help='Configuration file path')
@click.option('--policy', '-p', default='policy.yaml', help='Policy file path')
@click.pass_context
def cli(ctx, config, policy):
    """Consiglio Agent - Secure AI Assistant CLI"""
    ctx.ensure_object(dict)
    ctx.obj['config'] = config
    ctx.obj['policy'] = policy
    
    # Initialize orchestrator
    try:
        ctx.obj['orchestrator'] = Orchestrator(config, policy)
    except Exception as e:
        console.print(f"[red]Failed to initialize orchestrator: {e}[/red]")
        sys.exit(1)


@cli.command()
@click.pass_context
def status(ctx):
    """Show system status"""
    orchestrator = ctx.obj['orchestrator']
    
    status_data = orchestrator.get_system_status()
    
    # Create status table
    table = Table(title="Consiglio Agent Status")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Agent Name", status_data["agent_name"])
    table.add_row("Version", status_data["agent_version"])
    table.add_row("Status", status_data["status"])
    table.add_row("Active Tasks", str(status_data["active_tasks"]))
    table.add_row("Pending Tasks", str(status_data["pending_tasks"]))
    table.add_row("Pending Tool Calls", str(status_data["pending_tool_calls"]))
    
    console.print(table)
    
    # Security profile info
    profile = status_data["security_profile"]
    
    # Format allowed tools for display
    allowed_tools = []
    for tool in profile.get('allowed_tools', []):
        if isinstance(tool, dict):
            # Extract tool names from dictionary format
            for tool_name in tool.keys():
                allowed_tools.append(tool_name)
        else:
            allowed_tools.append(str(tool))
    
    profile_panel = Panel(
        f"[bold]Current Profile:[/bold] {profile['name']}\n"
        f"[bold]Description:[/bold] {profile['description']}\n"
        f"[bold]Allowed Tools:[/bold] {', '.join(allowed_tools) if allowed_tools else 'None'}\n"
        f"[bold]Denied Tools:[/bold] {', '.join(profile['denied_tools']) if profile['denied_tools'] else 'None'}",
        title="Security Profile",
        border_style="blue"
    )
    console.print(profile_panel)


@cli.command()
@click.argument('goal')
@click.option('--profile', '-p', help='Security profile to use')
@click.option('--execute', '-e', is_flag=True, help='Execute immediately after creation')
@click.pass_context
def task(ctx, goal, profile, execute):
    """Create and optionally execute a task"""
    orchestrator = ctx.obj['orchestrator']
    
    # Set profile if specified
    if profile:
        if orchestrator.set_security_profile(profile):
            console.print(f"[green]Security profile set to: {profile}[/green]")
        else:
            console.print(f"[red]Failed to set profile: {profile}[/red]")
            return
    
    # Create task
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("Creating task...", total=None)
        task_id = orchestrator.create_task(goal)
        progress.update(task, completed=True)
    
    console.print(f"[green]Task created: {task_id}[/green]")
    console.print(f"[blue]Goal:[/blue] {goal}")
    
    if execute:
        console.print("\n[yellow]Executing task...[/yellow]")
        result = orchestrator.execute_task(task_id)
        
        if result["success"]:
            console.print(f"[green]Task completed successfully![/green]")
            if result.get("results"):
                console.print("\n[bold]Results:[/bold]")
                for step_result in result["results"]:
                    console.print(f"  • {step_result['description']}: {step_result['status']}")
        else:
            console.print(f"[red]Task failed: {result['error']}[/red]")
    else:
        console.print("\n[blue]Use 'consiglio execute <task_id>' to run this task[/blue]")


@cli.command()
@click.pass_context
def tasks(ctx):
    """List all tasks"""
    orchestrator = ctx.obj['orchestrator']
    
    pending_tasks = orchestrator.get_pending_tasks()
    active_tasks = orchestrator.get_active_tasks()
    
    if not pending_tasks and not active_tasks:
        console.print("[yellow]No tasks found[/yellow]")
        return
    
    # Pending tasks
    if pending_tasks:
        table = Table(title="Pending Tasks")
        table.add_column("ID", style="cyan")
        table.add_column("Goal", style="green")
        table.add_column("Created", style="blue")
        
        for task in pending_tasks:
            table.add_row(
                task["id"],
                task["goal"][:50] + "..." if len(task["goal"]) > 50 else task["goal"],
                task["created_at"]
            )
        
        console.print(table)
    
    # Active tasks
    if active_tasks:
        table = Table(title="Active Tasks")
        table.add_column("ID", style="cyan")
        table.add_column("Goal", style="green")
        table.add_column("Status", style="yellow")
        table.add_column("Started", style="blue")
        
        for task in active_tasks:
            table.add_row(
                task["id"],
                task["goal"][:50] + "..." if len(task["goal"]) > 50 else task["goal"],
                task["status"],
                task["started_at"]
            )
        
        console.print(table)


@cli.command()
@click.argument('task_id')
@click.pass_context
def execute(ctx, task_id):
    """Execute a specific task"""
    orchestrator = ctx.obj['orchestrator']
    
    console.print(f"[yellow]Executing task: {task_id}[/yellow]")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("Executing task...", total=None)
        result = orchestrator.execute_task(task_id)
        progress.update(task, completed=True)
    
    if result["success"]:
        console.print(f"[green]Task completed successfully![/green]")
        if result.get("results"):
            console.print("\n[bold]Results:[/bold]")
            for step_result in result["results"]:
                status_color = "green" if step_result["status"] == "completed" else "red"
                console.print(f"  • {step_result['description']}: [{status_color}]{step_result['status']}[/{status_color}]")
    else:
        console.print(f"[red]Task failed: {result['error']}[/red]")


@cli.command()
@click.pass_context
def approve(ctx):
    """Approve pending tool calls"""
    orchestrator = ctx.obj['orchestrator']
    
    pending_calls = orchestrator.get_pending_tool_calls()
    
    if not pending_calls:
        console.print("[yellow]No pending tool calls[/yellow]")
        return
    
    console.print(f"[blue]Found {len(pending_calls)} pending tool calls[/blue]\n")
    
    for call in pending_calls:
        # Display tool call info
        panel = Panel(
            f"[bold]Tool:[/bold] {call['tool']}\n"
            f"[bold]Reason:[/bold] {call['reason']}\n"
            f"[bold]Arguments:[/bold] {json.dumps(call['args'], indent=2)}\n"
            f"[bold]Risk Level:[/bold] {call.get('risk_level', 'unknown')}",
            title=f"Tool Call {call['id'][:8]}...",
            border_style="yellow"
        )
        console.print(panel)
        
        # Ask for approval
        if Confirm.ask(f"Approve this {call['tool']} call?"):
            result = orchestrator.approve_tool_call(call['id'])
            if result["success"]:
                console.print(f"[green]Tool call approved and executed![/green]")
                if result.get("data"):
                    console.print(f"[blue]Result:[/blue] {json.dumps(result['data'], indent=2)}")
            else:
                console.print(f"[red]Failed to execute tool call: {result['error']}[/red]")
        else:
            reason = Prompt.ask("Reason for rejection")
            result = orchestrator.reject_tool_call(call['id'], reason)
            if result["success"]:
                console.print(f"[yellow]Tool call rejected[/yellow]")
        
        console.print()  # Empty line for readability


@cli.command()
@click.pass_context
def audit(ctx):
    """Show audit log"""
    orchestrator = ctx.obj['orchestrator']
    
    audit_log = orchestrator.get_audit_log(limit=50)
    
    if not audit_log:
        console.print("[yellow]No audit log entries found[/yellow]")
        return
    
    table = Table(title="Audit Log (Last 50 Entries)")
    table.add_column("Timestamp", style="cyan")
    table.add_column("Event", style="green")
    table.add_column("Tool", style="blue")
    table.add_column("Risk Level", style="yellow")
    table.add_column("Details", style="white")
    
    for entry in audit_log:
        # Truncate long details
        details = str(entry.get("additional_info", ""))[:30]
        if len(str(entry.get("additional_info", ""))) > 30:
            details += "..."
        
        table.add_row(
            entry["timestamp"],
            entry["event_type"],
            entry["tool"],
            entry.get("risk_level", "unknown"),
            details
        )
    
    console.print(table)


@cli.command()
@click.argument('profile_name')
@click.pass_context
def profile(ctx, profile_name):
    """Set security profile"""
    orchestrator = ctx.obj['orchestrator']
    
    if orchestrator.set_security_profile(profile_name):
        console.print(f"[green]Security profile set to: {profile_name}[/green]")
        
        # Show profile info
        profile_info = orchestrator.get_security_profile()
        
        # Format allowed tools for display
        allowed_tools = []
        for tool in profile_info.get('allowed_tools', []):
            if isinstance(tool, dict):
                # Extract tool names from dictionary format
                for tool_name in tool.keys():
                    allowed_tools.append(tool_name)
            else:
                allowed_tools.append(str(tool))
        
        panel = Panel(
            f"[bold]Profile:[/bold] {profile_info['name']}\n"
            f"[bold]Description:[/bold] {profile_info['description']}\n"
            f"[bold]Allowed Tools:[/bold] {', '.join(allowed_tools) if allowed_tools else 'None'}\n"
            f"[bold]Denied Tools:[/bold] {', '.join(profile_info['denied_tools']) if profile_info['denied_tools'] else 'None'}",
            title="Current Security Profile",
            border_style="green"
        )
        console.print(panel)
    else:
        console.print(f"[red]Failed to set profile: {profile_name}[/red]")


@cli.command()
@click.pass_context
def reload(ctx):
    """Reload security policy"""
    orchestrator = ctx.obj['orchestrator']
    
    console.print("[yellow]Reloading security policy...[/yellow]")
    orchestrator.reload_policy()
    console.print("[green]Policy reloaded successfully![/green]")


@cli.command()
@click.pass_context
def demo(ctx):
    """Run a demo task to test the system"""
    orchestrator = ctx.obj['orchestrator']
    
    console.print("[bold blue]Consiglio Agent Demo[/bold blue]")
    console.print("This will create and execute a simple demo task.\n")
    
    # Create demo task
    goal = "Search for information about AI agents and summarize the findings"
    task_id = orchestrator.create_task(goal)
    
    console.print(f"[green]Demo task created: {task_id}[/green]")
    console.print(f"[blue]Goal:[/blue] {goal}\n")
    
    # Execute demo task
    console.print("[yellow]Executing demo task...[/yellow]")
    result = orchestrator.execute_task(task_id)
    
    if result["success"]:
        console.print(f"[green]Demo completed successfully![/green]")
        console.print("\n[bold]What happened:[/bold]")
        console.print("1. ✅ Task created and planned")
        console.print("2. ✅ Tool calls validated against security policy")
        console.print("3. ✅ Safe tool handlers executed (no-ops for demo)")
        console.print("4. ✅ Task completed with results")
        
        console.print("\n[bold]Next steps:[/bold]")
        console.print("• Use 'consiglio status' to see system status")
        console.print("• Use 'consiglio profile admin' to enable more tools")
        console.print("• Use 'consiglio approve' to approve pending tool calls")
        console.print("• Check 'consiglio audit' for activity logs")
    else:
        console.print(f"[red]Demo failed: {result['error']}[/red]")


@cli.command()
@click.pass_context
def chat(ctx):
    """Interactive chat to talk to the agent and control tasks"""
    orchestrator = ctx.obj['orchestrator']

    console.print(Panel.fit(
        "[bold blue]Consiglio Chat[/bold blue]\n"
        "Type your message to create a task from it.\n"
        "Use slash commands to control tasks:\n"
        "- /task <goal>\n"
        "- /exec <task_id>\n"
        "- /tasks\n"
        "- /approve\n"
        "- /status\n"
        "- /profile <name>\n"
        "- /audit\n"
        "- /help\n"
        "- /quit",
        title="Interactive Mode",
        border_style="magenta"
    ))

    while True:
        try:
            user_input = Prompt.ask("[bold green]You[/bold green]")
        except (EOFError, KeyboardInterrupt):
            console.print("\n[yellow]Exiting chat...[/yellow]")
            break

        if not user_input:
            continue

        if user_input.startswith('/'):
            parts = user_input.strip().split(maxsplit=1)
            cmd = parts[0].lower()
            arg = parts[1] if len(parts) > 1 else ""

            if cmd == '/quit' or cmd == '/exit':
                console.print("[yellow]Goodbye![/yellow]")
                break

            if cmd == '/help':
                console.print("""
Commands:
/task <goal>        Create a task
/exec <task_id>     Execute a task
/tasks              List tasks
/approve            Review and approve pending tool calls
/status             Show system status
/profile <name>     Set security profile
/audit              Show recent audit log
/help               Show this help
/quit               Exit chat
""")
                continue

            if cmd == '/status':
                ctx.invoke(status)
                continue

            if cmd == '/tasks':
                ctx.invoke(tasks)
                continue

            if cmd == '/audit':
                ctx.invoke(audit)
                continue

            if cmd == '/approve':
                ctx.invoke(approve)
                continue

            if cmd == '/profile':
                if not arg:
                    console.print("[red]Usage: /profile <name>[/red]")
                else:
                    ctx.invoke(profile, profile_name=arg)
                continue

            if cmd == '/task':
                if not arg:
                    console.print("[red]Usage: /task <goal>[/red]")
                    continue
                goal = arg
                with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
                    p = progress.add_task("Creating task...", total=None)
                    task_id = orchestrator.create_task(goal)
                    progress.update(p, completed=True)
                console.print(f"[green]Task created:[/green] {task_id}")
                if Confirm.ask("Execute now?", default=False):
                    ctx.invoke(execute, task_id=task_id)
                continue

            if cmd == '/exec':
                if not arg:
                    console.print("[red]Usage: /exec <task_id>[/red]")
                    continue
                ctx.invoke(execute, task_id=arg)
                continue

            console.print(f"[red]Unknown command:[/red] {cmd}. Type /help")
            continue

        # Natural language: create a task from the message
        goal = user_input.strip()
        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
            p = progress.add_task("Creating task from message...", total=None)
            task_id = orchestrator.create_task(goal)
            progress.update(p, completed=True)
        console.print(f"[green]Created task[/green] {task_id} for: [blue]{goal}[/blue]")
        if Confirm.ask("Execute now?", default=True):
            ctx.invoke(execute, task_id=task_id)


if __name__ == '__main__':
    cli()