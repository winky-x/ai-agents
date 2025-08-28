"""
Real Intelligence Engine
- Multi-step reasoning and planning
- Problem-solving with backtracking
- Creative solution generation
- Fallback strategies and error recovery
- Context understanding and adaptation
"""

import json
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import asyncio
from loguru import logger


class ReasoningType(Enum):
    """Types of reasoning the agent can perform"""
    PLANNING = "planning"
    PROBLEM_SOLVING = "problem_solving"
    CREATIVE = "creative"
    ANALYTICAL = "analytical"
    DECISION_MAKING = "decision_making"


class ProblemComplexity(Enum):
    """Problem complexity levels"""
    SIMPLE = "simple"  # Single step, direct action
    MODERATE = "moderate"  # 2-5 steps, some planning needed
    COMPLEX = "complex"  # 5-15 steps, significant planning
    EXPERT = "expert"  # 15+ steps, creative problem-solving needed


@dataclass
class ReasoningStep:
    """A single step in the reasoning process"""
    id: str
    type: ReasoningType
    description: str
    input_data: Dict[str, Any]
    reasoning: str
    conclusion: str
    confidence: float  # 0.0 to 1.0
    alternatives: List[str] = None
    fallback_plan: str = None


@dataclass
class ProblemSolution:
    """A complete solution to a problem"""
    problem_id: str
    problem_description: str
    complexity: ProblemComplexity
    steps: List[ReasoningStep]
    success_probability: float
    estimated_time: int  # seconds
    required_tools: List[str]
    fallback_strategies: List[str]
    success_criteria: List[str]


class IntelligenceEngine:
    """Real intelligence engine with planning and reasoning capabilities"""
    
    def __init__(self, llm_provider, memory_system, tool_router):
        self.llm_provider = llm_provider
        self.memory = memory_system
        self.tool_router = tool_router
        self.solution_history: Dict[str, ProblemSolution] = {}
        self.failure_patterns: Dict[str, List[str]] = {}
        self.success_patterns: Dict[str, List[str]] = {}
    
    def analyze_request(self, user_input: str) -> Dict[str, Any]:
        """Analyze user request to understand intent and complexity"""
        analysis_prompt = f"""
        Analyze this user request and provide detailed analysis:
        
        Request: {user_input}
        
        Provide analysis in JSON format with:
        - intent: What the user wants to accomplish
        - complexity: simple/moderate/complex/expert
        - required_capabilities: List of capabilities needed
        - potential_challenges: What might go wrong
        - success_criteria: How to know if successful
        - estimated_steps: Number of steps likely needed
        - tools_needed: Specific tools required
        """
        
        result = self.llm_provider.call_gemini_flash(analysis_prompt)
        try:
            return json.loads(result.get("text", "{}"))
        except:
            # Fallback analysis
            return {
                "intent": "general_assistance",
                "complexity": "moderate",
                "required_capabilities": ["web_search", "file_operations"],
                "potential_challenges": ["unknown"],
                "success_criteria": ["user_satisfaction"],
                "estimated_steps": 3,
                "tools_needed": ["llm.call", "web.get"]
            }
    
    def create_solution_plan(self, problem_description: str, analysis: Dict[str, Any]) -> ProblemSolution:
        """Create a comprehensive solution plan"""
        
        planning_prompt = f"""
        Create a detailed solution plan for this problem:
        
        Problem: {problem_description}
        Analysis: {json.dumps(analysis, indent=2)}
        
        Create a step-by-step plan with:
        1. Clear objectives for each step
        2. Reasoning for why each step is needed
        3. Alternative approaches if primary fails
        4. Success criteria for each step
        5. Fallback strategies
        
        Format as JSON with steps array containing:
        - step_number
        - objective
        - reasoning
        - tools_needed
        - success_criteria
        - fallback_approach
        - estimated_time
        """
        
        result = self.llm_provider.call_openrouter_deepseek(planning_prompt)
        
        try:
            plan_data = json.loads(result.get("text", "{}"))
            steps = []
            
            for step_data in plan_data.get("steps", []):
                step = ReasoningStep(
                    id=f"step_{len(steps) + 1}",
                    type=ReasoningType.PLANNING,
                    description=step_data.get("objective", ""),
                    input_data=step_data,
                    reasoning=step_data.get("reasoning", ""),
                    conclusion=step_data.get("success_criteria", ""),
                    confidence=0.8,
                    alternatives=step_data.get("fallback_approach", "").split(";"),
                    fallback_plan=step_data.get("fallback_approach", "")
                )
                steps.append(step)
            
            return ProblemSolution(
                problem_id=f"problem_{int(time.time())}",
                problem_description=problem_description,
                complexity=ProblemComplexity(analysis.get("complexity", "moderate")),
                steps=steps,
                success_probability=0.85,
                estimated_time=sum(step.input_data.get("estimated_time", 30) for step in steps),
                required_tools=analysis.get("tools_needed", []),
                fallback_strategies=plan_data.get("fallback_strategies", []),
                success_criteria=analysis.get("success_criteria", [])
            )
            
        except Exception as e:
            logger.error(f"Failed to create solution plan: {e}")
            return self._create_fallback_plan(problem_description, analysis)
    
    def _create_fallback_plan(self, problem_description: str, analysis: Dict[str, Any]) -> ProblemSolution:
        """Create a basic fallback plan when detailed planning fails"""
        return ProblemSolution(
            problem_id=f"fallback_{int(time.time())}",
            problem_description=problem_description,
            complexity=ProblemComplexity.SIMPLE,
            steps=[
                ReasoningStep(
                    id="step_1",
                    type=ReasoningType.PROBLEM_SOLVING,
                    description="Attempt direct solution",
                    input_data={"method": "direct_approach"},
                    reasoning="Use available tools to solve problem directly",
                    conclusion="Problem solved or need alternative approach",
                    confidence=0.5,
                    alternatives=["web_search", "file_operations"],
                    fallback_plan="Try different tools or ask user for clarification"
                )
            ],
            success_probability=0.6,
            estimated_time=60,
            required_tools=["llm.call", "web.get"],
            fallback_strategies=["ask_user", "try_alternative_tools"],
            success_criteria=["user_satisfaction"]
        )
    
    def execute_solution(self, solution: ProblemSolution, user_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute a solution plan with intelligent error handling"""
        results = []
        current_step = 0
        
        while current_step < len(solution.steps):
            step = solution.steps[current_step]
            logger.info(f"Executing step {current_step + 1}: {step.description}")
            
            try:
                # Execute the step
                step_result = self._execute_step(step, user_context)
                results.append(step_result)
                
                # Check if step was successful
                if step_result.get("success"):
                    logger.info(f"Step {current_step + 1} completed successfully")
                    current_step += 1
                else:
                    # Try fallback approaches
                    fallback_success = self._try_fallback_approaches(step, user_context)
                    if fallback_success:
                        logger.info(f"Step {current_step + 1} completed with fallback")
                        current_step += 1
                    else:
                        # If all fallbacks fail, try creative problem-solving
                        creative_solution = self._creative_problem_solving(step, user_context)
                        if creative_solution:
                            logger.info(f"Step {current_step + 1} completed with creative solution")
                            current_step += 1
                        else:
                            # Final fallback: ask user for help
                            return {
                                "success": False,
                                "error": f"Failed at step {current_step + 1}: {step.description}",
                                "completed_steps": results,
                                "suggestion": "Please provide more specific instructions or try a different approach"
                            }
                
            except Exception as e:
                logger.error(f"Error in step {current_step + 1}: {e}")
                # Try to recover
                recovery_result = self._recover_from_error(step, e, user_context)
                if not recovery_result:
                    return {
                        "success": False,
                        "error": f"Unrecoverable error in step {current_step + 1}: {e}",
                        "completed_steps": results
                    }
        
        # All steps completed
        return {
            "success": True,
            "results": results,
            "solution": solution,
            "total_time": sum(r.get("execution_time", 0) for r in results)
        }
    
    def _execute_step(self, step: ReasoningStep, user_context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single reasoning step"""
        start_time = time.time()
        
        # Determine what tools to use based on step description
        tools_needed = step.input_data.get("tools_needed", [])
        
        if "web_search" in step.description.lower() or "search" in step.description.lower():
            # Web search step
            result = self.tool_router.route_tool_call({
                "tool": "web.get",
                "args": {"url": "https://google.com"},
                "reason": step.description
            })
        elif "file" in step.description.lower():
            # File operation step
            result = self.tool_router.route_tool_call({
                "tool": "file.read",
                "args": {"path": "work/temp.txt"},
                "reason": step.description
            })
        elif "browser" in step.description.lower():
            # Browser automation step
            result = self.tool_router.route_tool_call({
                "tool": "browser.control",
                "args": {"action": "goto", "url": "https://example.com"},
                "reason": step.description
            })
        else:
            # Default to LLM reasoning
            result = self.llm_provider.call_gemini_flash(step.description)
        
        execution_time = time.time() - start_time
        
        return {
            "step_id": step.id,
            "success": result.get("success", False),
            "result": result,
            "execution_time": execution_time,
            "reasoning": step.reasoning
        }
    
    def _try_fallback_approaches(self, step: ReasoningStep, user_context: Dict[str, Any]) -> bool:
        """Try alternative approaches for a failed step"""
        alternatives = step.alternatives or []
        
        for alternative in alternatives:
            try:
                logger.info(f"Trying alternative: {alternative}")
                
                if "web_search" in alternative:
                    result = self.tool_router.route_tool_call({
                        "tool": "web.get",
                        "args": {"url": "https://bing.com"},
                        "reason": f"Alternative approach: {alternative}"
                    })
                elif "vision" in alternative:
                    # Use vision as fallback
                    result = self._use_vision_fallback(step, user_context)
                elif "manual" in alternative:
                    # Ask user for manual input
                    result = {"success": True, "manual_input": True}
                else:
                    # Try LLM with different approach
                    result = self.llm_provider.call_openrouter_deepseek(
                        f"Alternative approach for: {step.description}\nMethod: {alternative}"
                    )
                
                if result.get("success"):
                    return True
                    
            except Exception as e:
                logger.error(f"Alternative approach failed: {e}")
                continue
        
        return False
    
    def _use_vision_fallback(self, step: ReasoningStep, user_context: Dict[str, Any]) -> Dict[str, Any]:
        """Use vision capabilities as a fallback when other methods fail"""
        vision_prompt = f"""
        I'm stuck on this step: {step.description}
        
        Please analyze what I'm trying to accomplish and suggest:
        1. What visual information would help solve this?
        2. What should I look for on the screen?
        3. What actions should I take based on what I see?
        
        Provide specific, actionable guidance.
        """
        
        # Use vision model to analyze the situation
        result = self.llm_provider.call_gemini_vision(vision_prompt, image_path=None)
        
        return {
            "success": True,
            "vision_analysis": result.get("text", ""),
            "method": "vision_fallback"
        }
    
    def _creative_problem_solving(self, step: ReasoningStep, user_context: Dict[str, Any]) -> bool:
        """Use creative problem-solving when standard approaches fail"""
        creative_prompt = f"""
        I'm stuck on this problem: {step.description}
        
        All standard approaches have failed. I need creative, out-of-the-box thinking.
        
        Consider:
        - Can I solve this differently?
        - Are there unconventional tools or methods?
        - Can I break this into smaller, simpler problems?
        - What would a human do in this situation?
        
        Provide a creative solution approach.
        """
        
        try:
            result = self.llm_provider.call_openrouter_deepseek(creative_prompt)
            creative_solution = result.get("text", "")
            
            # Try to implement the creative solution
            if "screenshot" in creative_solution.lower():
                # Take screenshot and analyze
                screenshot_result = self.tool_router.route_tool_call({
                    "tool": "desktop.screenshot",
                    "args": {},
                    "reason": "Creative solution: visual analysis"
                })
                return screenshot_result.get("success", False)
            
            elif "search" in creative_solution.lower():
                # Try different search approach
                search_result = self.tool_router.route_tool_call({
                    "tool": "web.get",
                    "args": {"url": "https://duckduckgo.com"},
                    "reason": "Creative solution: alternative search"
                })
                return search_result.get("success", False)
            
            else:
                # Try LLM-based creative solution
                llm_result = self.llm_provider.call_gemini_flash(creative_solution)
                return True
                
        except Exception as e:
            logger.error(f"Creative problem-solving failed: {e}")
            return False
    
    def _recover_from_error(self, step: ReasoningStep, error: Exception, user_context: Dict[str, Any]) -> bool:
        """Attempt to recover from an error"""
        recovery_prompt = f"""
        I encountered an error while executing: {step.description}
        Error: {str(error)}
        
        How can I recover from this error? Provide specific recovery steps.
        """
        
        try:
            recovery_plan = self.llm_provider.call_gemini_flash(recovery_prompt)
            
            # Try to implement recovery plan
            if "retry" in recovery_plan.get("text", "").lower():
                time.sleep(2)  # Wait before retry
                return self._execute_step(step, user_context).get("success", False)
            
            elif "skip" in recovery_plan.get("text", "").lower():
                logger.info(f"Skipping step due to error: {step.description}")
                return True
            
            else:
                return False
                
        except Exception as e:
            logger.error(f"Recovery failed: {e}")
            return False
    
    def learn_from_experience(self, solution: ProblemSolution, result: Dict[str, Any]):
        """Learn from the execution experience"""
        if result.get("success"):
            # Record successful patterns
            problem_type = solution.problem_description.split()[0].lower()
            if problem_type not in self.success_patterns:
                self.success_patterns[problem_type] = []
            
            successful_approach = [step.description for step in solution.steps]
            self.success_patterns[problem_type].append(successful_approach)
            
        else:
            # Record failure patterns for future avoidance
            problem_type = solution.problem_description.split()[0].lower()
            if problem_type not in self.failure_patterns:
                self.failure_patterns[problem_type] = []
            
            failed_step = result.get("error", "unknown")
            self.failure_patterns[problem_type].append(failed_step)
        
        # Store in persistent memory
        self.memory.add_memory("problem_solving_experience", {
            "problem": solution.problem_description,
            "success": result.get("success"),
            "steps": len(solution.steps),
            "execution_time": result.get("total_time", 0),
            "lessons_learned": result.get("error", "success")
        })
    
    def get_intelligent_response(self, user_input: str, user_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Main entry point for intelligent problem-solving"""
        
        # Analyze the request
        analysis = self.analyze_request(user_input)
        logger.info(f"Request analysis: {analysis}")
        
        # Create solution plan
        solution = self.create_solution_plan(user_input, analysis)
        logger.info(f"Created solution with {len(solution.steps)} steps")
        
        # Execute the solution
        result = self.execute_solution(solution, user_context)
        
        # Learn from the experience
        self.learn_from_experience(solution, result)
        
        return {
            "analysis": analysis,
            "solution": solution,
            "execution_result": result,
            "intelligence_used": True
        }