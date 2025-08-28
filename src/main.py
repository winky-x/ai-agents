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
    """Interactive natural language chat with smart action confirmations"""
    orchestrator = ctx.obj['orchestrator']

    winky = """
 __        __        _         _         _        _         
 \\ \\      / /__  ___| | ____ _| |_ _   _| | __ _| |_ ___  
  \\ \\ /\\ / / _ \\/ __| |/ / _` | __| | | | |/ _` | __/ _ \\ 
   \\ V  V /  __/ (__|   < (_| | |_| |_| | | (_| | || (_) |
    \\_/\\_/ \\___|\\___|_|\\_\\__,_|\\__|\\__,_|_|\\__,_|\\__\\___/ 
                         [bold cyan]Winky AI Agent[/bold cyan]
    """
    console.print(Panel.fit(winky, title="Welcome", border_style="magenta"))
    console.print(Text("Type what you want. I'll detect if it's a command, web search, or a question.\nI'll ask before running any command or external action.", style="bright_black"))

    while True:
        try:
            user_input = Prompt.ask("[bold green]You[/bold green]")
        except (EOFError, KeyboardInterrupt):
            console.print("\n[yellow]Exiting chat...[/yellow]")
            break

        if not user_input:
            continue

        text = user_input.strip()
        lower = text.lower()
        is_shell = lower.startswith("run ") or lower.startswith("execute ") or lower.startswith("! ") or lower.startswith("bash ") or lower.startswith("sh ") or lower.startswith("sudo ") or lower.startswith("apt ") or lower.startswith("pip ") or lower.startswith("git ")
        is_search = any(k in lower for k in ["search ", "google ", "look up", "find info", "web search"]) and not is_shell

        if is_shell:
            cmd = text.lstrip("! ")
            if not Confirm.ask(f"Run shell command?\n[bold]{cmd}[/bold]", default=False):
                console.print("[bright_black]Cancelled.[/bright_black]")
                continue
            result = orchestrator.tool_router.route_tool_call({
                "tool": "shell.exec",
                "args": {"command": cmd},
                "reason": "User requested shell execution from chat",
            })
            if result.get("requires_confirmation"):
                if Confirm.ask("Approve this tool call now?", default=False):
                    approve_result = orchestrator.tool_router.approve_tool_call(result.get("tool_call_id"))
                    console.print(approve_result)
                else:
                    console.print("[yellow]Pending approval. Use 'approve' command later.[/yellow]")
            else:
                console.print(result.get("data", {}))
            continue

        if is_search:
            if not Confirm.ask("Perform a web action/search?", default=True):
                console.print("[bright_black]Skipped web action.[/bright_black]")
            else:
                result = orchestrator.tool_router.route_tool_call({
                    "tool": "web.get",
                    "args": {"url": "https://example.com"},
                    "reason": f"Search intent from: {text[:80]}",
                })
                console.print(result)
            continue

        # Get conversation history for context
        conversation_history = []
        memories = orchestrator.persistent_memory.get_memories("conversation", limit=10)
        for memory in memories:
            conversation_history.append({
                "user_input": memory.content.get("user_input", ""),
                "agent_response": memory.content.get("agent_response", "")
            })
        
        # Understand the request with full context
        understanding = orchestrator.context_understanding.understand_request(text, conversation_history)
        
        if understanding.get("understanding") == "incomplete":
            # Need clarification
            questions = understanding.get("questions", [])
            response_text = "I need some clarification:\n" + "\n".join([f"• {q}" for q in questions])
            console.print(Panel.fit(response_text, title="🤔 Clarification Needed", border_style="yellow"))
            continue
        
        # Get best approach based on learning
        task_type = understanding.get("intent", {}).type.value
        input_data = understanding.get("intent", {}).entities or {}
        best_approach, confidence = orchestrator.adaptive_learning.get_best_approach(task_type, input_data)
        
        # Check if we should avoid this approach
        if orchestrator.adaptive_learning.should_avoid_approach(task_type, input_data):
            best_approach = "alternative_approach"
            confidence = 0.3
        
        # Use intelligent problem-solving with learned approach
        intelligent_result = orchestrator.intelligence_engine.get_intelligent_response(
            understanding.get("resolved_input", text), 
            user_context={"approach": best_approach, "confidence": confidence}
        )
        
        if intelligent_result.get("intelligence_used"):
            analysis = intelligent_result.get("analysis", {})
            execution_result = intelligent_result.get("execution_result", {})
            
            if execution_result.get("success"):
                # Format the response
                response_parts = []
                
                # Add analysis summary
                if analysis:
                    complexity = analysis.get("complexity", "unknown")
                    intent = analysis.get("intent", "general")
                    response_parts.append(f"[bold]Analysis:[/bold] {intent} ({complexity} complexity)")
                
                # Add execution summary
                solution = intelligent_result.get("solution")
                if solution:
                    steps_completed = len(solution.steps)
                    response_parts.append(f"[bold]Solution:[/bold] Completed {steps_completed} steps")
                
                # Add results
                results = execution_result.get("results", [])
                if results:
                    response_parts.append(f"[bold]Results:[/bold]")
                    for i, step_result in enumerate(results[:3], 1):  # Show first 3 results
                        step_id = step_result.get("step_id", f"step_{i}")
                        success = "✅" if step_result.get("success") else "❌"
                        response_parts.append(f"  {success} {step_id}")
                
                # Add final response
                if results and results[-1].get("result", {}).get("text"):
                    response_parts.append(f"\n{results[-1]['result']['text']}")
                
                response_text = "\n".join(response_parts)
                
                # Store in memory and record learning
                orchestrator.persistent_memory.learn_from_interaction(text, response_text)
                
                # Record learning event
                orchestrator.adaptive_learning.record_learning_event(
                    task_type=task_type,
                    input_data=input_data,
                    approach_used=best_approach,
                    result=execution_result,
                    performance_metrics={
                        "execution_time": execution_result.get("total_time", 0),
                        "success_rate": 1.0 if execution_result.get("success") else 0.0,
                        "user_satisfaction": 1.0  # Assume satisfaction if no error
                    },
                    context={"user_input": text, "response": response_text}
                )
                
                # Update context
                orchestrator.context_understanding.update_context(
                    level=orchestrator.context_understanding.ContextLevel.SESSION,
                    key="last_task",
                    value={"type": task_type, "result": "success"},
                    ttl_hours=2
                )
                
                console.print(Panel.fit(response_text, title="🤖 Intelligent Agent", border_style="green"))
            else:
                error_msg = execution_result.get("error", "Unknown error")
                console.print(f"[red]Intelligent processing failed: {error_msg}[/red]")
        else:
            # Fallback to simple LLM call
            result = orchestrator.tool_router.route_tool_call({
                "tool": "llm.call",
                "args": {"prompt": text},
                "reason": "Fallback chat response",
            })
            
            if result.get("success"):
                data = result.get("data", {})
                response_text = data.get("text", "")
                
                # Store in memory and record learning
                orchestrator.persistent_memory.learn_from_interaction(text, response_text)
                
                # Record learning event for fallback
                orchestrator.adaptive_learning.record_learning_event(
                    task_type="fallback_llm",
                    input_data={"original_input": text},
                    approach_used="simple_llm_call",
                    result={"success": True, "method": "fallback"},
                    performance_metrics={
                        "execution_time": 0,
                        "success_rate": 1.0,
                        "user_satisfaction": 0.5  # Lower satisfaction for fallback
                    },
                    context={"user_input": text, "response": response_text}
                )
                
                console.print(Panel.fit(response_text, title=f"{data.get('model', 'LLM')} ({data.get('mode', 'fast')})", border_style="blue"))
            else:
                console.print(f"[red]{result.get('error')}[/red]")


if __name__ == '__main__':
    cli()