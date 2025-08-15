"""
Consiglio Policy Engine
Handles security policies, permissions, and tool validation
"""

import os
import yaml
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from urllib.parse import urlparse
from dataclasses import dataclass
from loguru import logger


@dataclass
class ToolPermission:
    """Represents a tool permission with its constraints"""
    tool_name: str
    allowed: bool
    require_manual_confirmation: bool
    constraints: Dict[str, Any]
    rate_limit: Optional[int] = None


@dataclass
class PolicyValidationResult:
    """Result of policy validation"""
    allowed: bool
    reason: str
    requires_confirmation: bool
    constraints: Dict[str, Any]
    risk_level: str  # low, medium, high


class PolicyEngine:
    """Main policy engine for Consiglio agent"""
    
    def __init__(self, policy_path: str = "policy.yaml"):
        self.policy_path = policy_path
        self.policy_data = {}
        self.current_profile = "dev"  # default profile
        self.rate_limiters = {}
        self.load_policy()
    
    def load_policy(self) -> None:
        """Load policy from YAML file"""
        try:
            if os.path.exists(self.policy_path):
                with open(self.policy_path, 'r') as f:
                    self.policy_data = yaml.safe_load(f)
                logger.info(f"Policy loaded from {self.policy_path}")
            else:
                logger.warning(f"Policy file {self.policy_path} not found, using defaults")
                self.policy_data = self._get_default_policy()
        except Exception as e:
            logger.error(f"Failed to load policy: {e}")
            self.policy_data = self._get_default_policy()
    
    def _get_default_policy(self) -> Dict:
        """Return a safe default policy"""
        return {
            "defaults": {"require_manual_confirmation": True},
            "profiles": {
                "dev": {
                    "allow": ["file.read"],
                    "deny": ["shell.exec", "file.write", "browser.control", "system.control"]
                }
            }
        }
    
    def set_profile(self, profile_name: str) -> bool:
        """Set the current security profile"""
        if profile_name in self.policy_data.get("profiles", {}):
            self.current_profile = profile_name
            logger.info(f"Security profile set to: {profile_name}")
            return True
        logger.warning(f"Profile {profile_name} not found, keeping current profile")
        return False
    
    def get_profile_info(self) -> Dict:
        """Get information about the current profile"""
        profile = self.policy_data.get("profiles", {}).get(self.current_profile, {})
        return {
            "name": self.current_profile,
            "description": profile.get("description", "No description"),
            "allowed_tools": profile.get("allow", []),
            "denied_tools": profile.get("deny", [])
        }
    
    def validate_tool_call(self, tool_name: str, args: Dict, user_context: Dict = None) -> PolicyValidationResult:
        """Validate a tool call against current policy"""
        try:
            # Check if tool is explicitly denied
            if self._is_tool_denied(tool_name):
                return PolicyValidationResult(
                    allowed=False,
                    reason=f"Tool {tool_name} is denied in profile {self.current_profile}",
                    requires_confirmation=False,
                    constraints={},
                    risk_level="high"
                )
            
            # Check if tool is allowed
            tool_permission = self._get_tool_permission(tool_name)
            if not tool_permission:
                return PolicyValidationResult(
                    allowed=False,
                    reason=f"Tool {tool_name} not permitted in profile {self.current_profile}",
                    requires_confirmation=False,
                    constraints={},
                    risk_level="high"
                )
            
            # Validate tool-specific constraints
            validation_result = self._validate_tool_constraints(tool_name, args, tool_permission)
            if not validation_result["valid"]:
                return PolicyValidationResult(
                    allowed=False,
                    reason=validation_result["reason"],
                    requires_confirmation=False,
                    constraints=tool_permission.constraints,
                    risk_level="high"
                )
            
            # Check rate limits
            if not self._check_rate_limit(tool_name, tool_permission):
                return PolicyValidationResult(
                    allowed=False,
                    reason=f"Rate limit exceeded for {tool_name}",
                    requires_confirmation=False,
                    constraints=tool_permission.constraints,
                    risk_level="medium"
                )
            
            # Determine risk level and confirmation requirements
            risk_level = self._assess_risk_level(tool_name, args, tool_permission)
            requires_confirmation = tool_permission.require_manual_confirmation or risk_level == "high"
            
            return PolicyValidationResult(
                allowed=True,
                reason="Tool call validated successfully",
                requires_confirmation=requires_confirmation,
                constraints=tool_permission.constraints,
                risk_level=risk_level
            )
            
        except Exception as e:
            logger.error(f"Policy validation error: {e}")
            return PolicyValidationResult(
                allowed=False,
                reason=f"Policy validation error: {e}",
                requires_confirmation=True,
                constraints={},
                risk_level="high"
            )
    
    def _is_tool_denied(self, tool_name: str) -> bool:
        """Check if tool is explicitly denied in current profile"""
        profile = self.policy_data.get("profiles", {}).get(self.current_profile, {})
        denied_tools = profile.get("deny", [])
        
        # Check for exact matches
        if tool_name in denied_tools:
            return True
        
        # Check for wildcard denials
        for denied in denied_tools:
            if denied.endswith(".*") and tool_name.startswith(denied[:-1]):
                return True
        
        return False
    
    def _get_tool_permission(self, tool_name: str) -> Optional[ToolPermission]:
        """Get permission configuration for a tool"""
        profile = self.policy_data.get("profiles", {}).get(self.current_profile, {})
        allowed_tools = profile.get("allow", [])
        
        for tool_config in allowed_tools:
            if isinstance(tool_config, dict):
                # Tool with configuration
                for config_tool, config in tool_config.items():
                    if config_tool == tool_name:
                        return ToolPermission(
                            tool_name=tool_name,
                            allowed=True,
                            require_manual_confirmation=config.get("require_manual_confirmation", True),
                            constraints=config,
                            rate_limit=config.get("max_requests_per_minute")
                        )
            elif isinstance(tool_config, str) and tool_config == tool_name:
                # Simple tool name
                return ToolPermission(
                    tool_name=tool_name,
                    allowed=True,
                    require_manual_confirmation=True,
                    constraints={},
                    rate_limit=None
                )
        
        return None
    
    def _validate_tool_constraints(self, tool_name: str, args: Dict, permission: ToolPermission) -> Dict:
        """Validate tool-specific constraints"""
        if tool_name == "web.get":
            return self._validate_web_get(args, permission.constraints)
        elif tool_name == "file.read":
            return self._validate_file_read(args, permission.constraints)
        elif tool_name == "file.write":
            return self._validate_file_write(args, permission.constraints)
        elif tool_name == "shell.exec":
            return self._validate_shell_exec(args, permission.constraints)
        elif tool_name == "browser.control":
            return self._validate_browser_control(args, permission.constraints)
        
        return {"valid": True, "reason": "No specific validation required"}
    
    def _validate_web_get(self, args: Dict, constraints: Dict) -> Dict:
        """Validate web.get tool constraints"""
        url = args.get("url", "")
        if not url:
            return {"valid": False, "reason": "URL is required"}
        
        try:
            parsed_url = urlparse(url)
            domain = parsed_url.netloc
            
            # Check domain allowlist
            allowed_domains = constraints.get("domains", [])
            if allowed_domains and "*" not in allowed_domains:
                if domain not in allowed_domains:
                    return {"valid": False, "reason": f"Domain {domain} not in allowlist"}
            
            # Check blocked domains
            blocked_domains = self.policy_data.get("tools", {}).get("web.get", {}).get("blocked_domains", [])
            if domain in blocked_domains:
                return {"valid": False, "reason": f"Domain {domain} is blocked"}
            
            return {"valid": True, "reason": "Web request validated"}
            
        except Exception as e:
            return {"valid": False, "reason": f"Invalid URL format: {e}"}
    
    def _validate_file_read(self, args: Dict, constraints: Dict) -> Dict:
        """Validate file.read tool constraints"""
        file_path = args.get("path", "")
        if not file_path:
            return {"valid": False, "reason": "File path is required"}
        
        try:
            abs_path = os.path.abspath(file_path)
            
            # Check allowed paths
            allowed_paths = constraints.get("paths", [])
            if allowed_paths:
                path_allowed = False
                for allowed_path in allowed_paths:
                    if allowed_path == "*" or abs_path.startswith(os.path.abspath(allowed_path)):
                        path_allowed = True
                        break
                
                if not path_allowed:
                    return {"valid": False, "reason": f"Path {abs_path} not in allowed paths"}
            
            # Check blocked paths
            blocked_paths = self.policy_data.get("tools", {}).get("file.read", {}).get("blocked_paths", [])
            for blocked_path in blocked_paths:
                if abs_path.startswith(os.path.abspath(blocked_path)):
                    return {"valid": False, "reason": f"Path {abs_path} is blocked"}
            
            # Check file size
            if os.path.exists(abs_path):
                file_size = os.path.getsize(abs_path)
                max_size = constraints.get("max_file_size", 10485760)  # 10MB default
                if file_size > max_size:
                    return {"valid": False, "reason": f"File size {file_size} exceeds limit {max_size}"}
            
            return {"valid": True, "reason": "File read validated"}
            
        except Exception as e:
            return {"valid": False, "reason": f"File validation error: {e}"}
    
    def _validate_file_write(self, args: Dict, constraints: Dict) -> Dict:
        """Validate file.write tool constraints"""
        file_path = args.get("path", "")
        if not file_path:
            return {"valid": False, "reason": "File path is required"}
        
        try:
            abs_path = os.path.abspath(file_path)
            
            # Check allowed paths
            allowed_paths = constraints.get("paths", [])
            if allowed_paths:
                path_allowed = False
                for allowed_path in allowed_paths:
                    if allowed_path == "*" or abs_path.startswith(os.path.abspath(allowed_path)):
                        path_allowed = True
                        break
                
                if not path_allowed:
                    return {"valid": False, "reason": f"Path {abs_path} not in allowed paths"}
            
            # Check file extension
            allowed_extensions = constraints.get("allowed_extensions", [])
            if allowed_extensions:
                file_ext = Path(file_path).suffix
                if file_ext not in allowed_extensions:
                    return {"valid": False, "reason": f"File extension {file_ext} not allowed"}
            
            return {"valid": True, "reason": "File write validated"}
            
        except Exception as e:
            return {"valid": False, "reason": f"File validation error: {e}"}
    
    def _validate_shell_exec(self, args: Dict, constraints: Dict) -> Dict:
        """Validate shell.exec tool constraints"""
        command = args.get("command", "")
        if not command:
            return {"valid": False, "reason": "Command is required"}
        
        # Check allowed commands
        allowed_commands = constraints.get("allowed_commands", [])
        if allowed_commands:
            command_base = command.split()[0] if command else ""
            if command_base not in allowed_commands:
                return {"valid": False, "reason": f"Command {command_base} not in allowed list"}
        
        # Check blocked commands
        blocked_commands = constraints.get("blocked_commands", [])
        for blocked_cmd in blocked_commands:
            if blocked_cmd in command:
                return {"valid": False, "reason": f"Command contains blocked pattern: {blocked_cmd}"}
        
        return {"valid": True, "reason": "Shell command validated"}
    
    def _validate_browser_control(self, args: Dict, constraints: Dict) -> Dict:
        """Validate browser.control tool constraints"""
        url = args.get("url", "")
        if url:
            # Validate URL like web.get
            url_validation = self._validate_web_get({"url": url}, constraints)
            if not url_validation["valid"]:
                return url_validation
        
        # Check headless requirement
        if constraints.get("headless_only", True):
            headless = args.get("headless", True)
            if not headless:
                return {"valid": False, "reason": "Headless mode required"}
        
        return {"valid": True, "reason": "Browser control validated"}
    
    def _check_rate_limit(self, tool_name: str, permission: ToolPermission) -> bool:
        """Check rate limiting for tool"""
        if not permission.rate_limit:
            return True
        
        # Simple in-memory rate limiting (in production, use Redis)
        current_time = int(time.time())
        window_start = current_time - 60  # 1 minute window
        
        if tool_name not in self.rate_limiters:
            self.rate_limiters[tool_name] = []
        
        # Clean old entries
        self.rate_limiters[tool_name] = [
            t for t in self.rate_limiters[tool_name] if t > window_start
        ]
        
        # Check if limit exceeded
        if len(self.rate_limiters[tool_name]) >= permission.rate_limit:
            return False
        
        # Add current request
        self.rate_limiters[tool_name].append(current_time)
        return True
    
    def _assess_risk_level(self, tool_name: str, args: Dict, permission: ToolPermission) -> str:
        """Assess risk level of tool call"""
        high_risk_tools = ["shell.exec", "system.control", "browser.control"]
        medium_risk_tools = ["file.write", "email.send"]
        
        if tool_name in high_risk_tools:
            return "high"
        elif tool_name in medium_risk_tools:
            return "medium"
        else:
            return "low"
    
    def get_emergency_override(self, tool_name: str, activation_phrase: str) -> Optional[Dict]:
        """Check for emergency override"""
        overrides = self.policy_data.get("emergency_overrides", {}).get(tool_name, {})
        
        if (overrides.get("enabled", False) and 
            overrides.get("activation_phrase") == activation_phrase):
            return overrides
        
        return None
    
    def reload_policy(self) -> None:
        """Reload policy from file"""
        self.load_policy()
        logger.info("Policy reloaded")


# Import time module for rate limiting
import time