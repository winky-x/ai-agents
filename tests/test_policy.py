"""
Tests for Consiglio Policy Engine
"""

import pytest
import os
import tempfile
from pathlib import Path

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.policy import PolicyEngine, PolicyValidationResult


class TestPolicyEngine:
    """Test cases for PolicyEngine"""
    
    def setup_method(self):
        """Setup test environment"""
        # Create temporary policy file
        self.temp_dir = tempfile.mkdtemp()
        self.policy_path = os.path.join(self.temp_dir, "test_policy.yaml")
        
        # Create test policy
        test_policy = """
profiles:
  test:
    description: "Test profile"
    allow:
      - web.get:
          domains: ["example.com", "test.org"]
          require_manual_confirmation: false
      - file.read:
          paths: ["./work", "./data"]
          require_manual_confirmation: false
      - rag.search:
          require_manual_confirmation: false
      - llm.call:
          require_manual_confirmation: false
    deny:
      - shell.exec
      - system.control
        """
        
        with open(self.policy_path, 'w') as f:
            f.write(test_policy)
        
        self.policy_engine = PolicyEngine(self.policy_path)
    
    def teardown_method(self):
        """Cleanup test environment"""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_load_policy(self):
        """Test policy loading"""
        assert self.policy_engine.policy_data is not None
        assert "profiles" in self.policy_engine.policy_data
        assert "test" in self.policy_engine.policy_data["profiles"]
    
    def test_set_profile(self):
        """Test profile setting"""
        assert self.policy_engine.set_profile("test")
        assert self.policy_engine.current_profile == "test"
        
        # Test invalid profile
        assert not self.policy_engine.set_profile("invalid_profile")
    
    def test_web_get_validation(self):
        """Test web.get tool validation"""
        # Set the test profile
        self.policy_engine.set_profile("test")
        
        # Valid domain
        result = self.policy_engine.validate_tool_call(
            "web.get", 
            {"url": "https://example.com/page"}
        )
        assert result.allowed
        assert not result.requires_confirmation
        
        # Invalid domain
        result = self.policy_engine.validate_tool_call(
            "web.get", 
            {"url": "https://malicious.com/page"}
        )
        assert not result.allowed
        
        # Missing URL
        result = self.policy_engine.validate_tool_call(
            "web.get", 
            {}
        )
        assert not result.allowed
    
    def test_file_read_validation(self):
        """Test file.read tool validation"""
        # Set the test profile
        self.policy_engine.set_profile("test")
        
        # Valid path
        result = self.policy_engine.validate_tool_call(
            "file.read", 
            {"path": "./work/file.txt"}
        )
        assert result.allowed
        assert not result.requires_confirmation
        
        # Invalid path
        result = self.policy_engine.validate_tool_call(
            "file.read", 
            {"path": "/etc/passwd"}
        )
        assert not result.allowed
        
        # Missing path
        result = self.policy_engine.validate_tool_call(
            "file.read", 
            {}
        )
        assert not result.allowed
    
    def test_denied_tools(self):
        """Test denied tools"""
        # Shell exec should be denied
        result = self.policy_engine.validate_tool_call(
            "shell.exec", 
            {"command": "ls -la"}
        )
        assert not result.allowed
        
        # System control should be denied
        result = self.policy_engine.validate_tool_call(
            "system.control", 
            {"action": "restart"}
        )
        assert not result.allowed
    
    def test_get_profile_info(self):
        """Test profile information retrieval"""
        self.policy_engine.set_profile("test")
        profile_info = self.policy_engine.get_profile_info()
        
        assert profile_info["name"] == "test"
        assert profile_info["description"] == "Test profile"
        assert "web.get" in str(profile_info["allowed_tools"])
        assert "shell.exec" in profile_info["denied_tools"]


if __name__ == "__main__":
    pytest.main([__file__])