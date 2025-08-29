# Future Model Upgrades - Implementation Guide

## 🚀 Overview

This document outlines the planned 6-model lineup for scaling the Winky AI Agent to enterprise-level performance. These upgrades will be implemented when we need to scale beyond the current 3-model system.

## 📊 Current vs. Future Model Comparison

### Current Models (2024)
| Model | Purpose | Speed | Cost/1M | Context | Status |
|-------|---------|-------|---------|---------|--------|
| Gemini 1.5 Flash | Fast responses | Fast | $0.18 | 1M | ✅ Active |
| DeepSeek R1 | Deep reasoning | Slow | $0.70 | Large | ✅ Active |
| Gemini Vision | Image analysis | Medium | $0.26 | 1M | ✅ Active |

### Future Models (2025+)
| Model | Purpose | Speed | Cost/1M | Context | Implementation |
|-------|---------|-------|---------|---------|----------------|
| Gemini 2.0 Flash | Primary chat | 175.5 t/s | $0.18 | 1M | 🔄 Upgrade |
| GPT-4.1 mini | Research/Reasoning | 69.7 t/s | $0.70 | 1M | ➕ New |
| DeepSeek V3 | Coding/Programming | 80.0 t/s | $0.48 | 64K | 🔄 Upgrade |
| GPT-4o mini | Vision/Multimodal | 80.0 t/s | $0.26 | 128K | ➕ New |
| Claude 4 Sonnet | Complex reasoning | 62.0 t/s | $6.00 | 200K | ➕ New |
| MiniMax M1 | APAC/Backup | 110.0 t/s | $0.75 | 1M | ➕ New |

## 🎯 Model Routing Strategy

### Task-Based Routing Matrix
```python
# Future routing logic (to be implemented)
TASK_ROUTING_MATRIX = {
    "chat": "gemini_2_flash",           # Normal conversation
    "research": "gpt_4_1_mini",         # Deep research
    "coding": "deepseek_v3",            # Programming tasks
    "vision": "gpt_4o_mini",            # Image analysis
    "reasoning": "claude_4_sonnet",     # Complex logic
    "backup": "minimax_m1",             # Fallback/APAC
    "cost_optimized": "minimax_m1",     # High volume
    "premium": "claude_4_sonnet"        # High-stakes decisions
}
```

### Performance Thresholds
```python
# Speed thresholds for model selection
SPEED_THRESHOLDS = {
    "ultra_fast": 150,      # tokens/sec - Gemini 2.0 Flash
    "fast": 80,             # tokens/sec - DeepSeek V3, GPT-4o mini
    "moderate": 70,         # tokens/sec - GPT-4.1 mini
    "thorough": 62,         # tokens/sec - Claude 4 Sonnet
    "backup": 110           # tokens/sec - MiniMax M1
}

# Cost thresholds per 1M tokens
COST_THRESHOLDS = {
    "ultra_low": 0.20,      # Gemini 2.0 Flash
    "low": 0.50,            # DeepSeek V3, GPT-4o mini
    "moderate": 0.75,       # GPT-4.1 mini, MiniMax M1
    "premium": 6.00         # Claude 4 Sonnet
}
```

## 🔧 Implementation Plan

### Phase 1: Model Provider Setup

#### 1.1 Update Environment Variables
```bash
# .env file additions
# Current models
GOOGLE_API_KEY=your_google_api_key
OPENROUTER_API_KEY=your_openrouter_api_key

# Future models
OPENAI_API_KEY=your_openai_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key
MINIMAX_API_KEY=your_minimax_api_key
DEEPSEEK_API_KEY=your_deepseek_api_key
```

#### 1.2 Update Requirements
```python
# requirements.txt additions
openai>=1.0.0              # For GPT-4.1 mini, GPT-4o mini
anthropic>=0.7.0           # For Claude 4 Sonnet
minimax>=0.1.0             # For MiniMax M1
deepseek>=0.1.0            # For DeepSeek V3 (direct API)
```

### Phase 2: Model Configuration

#### 2.1 Enhanced Model Configuration
```python
# src/core/model_config.py (new file)
from dataclasses import dataclass
from typing import Dict, Any
from enum import Enum

class ModelType(Enum):
    FAST_CHAT = "fast_chat"
    RESEARCH = "research"
    CODING = "coding"
    VISION = "vision"
    REASONING = "reasoning"
    BACKUP = "backup"

@dataclass
class ModelConfig:
    name: str
    provider: str
    api_key_env: str
    endpoint: str
    max_tokens: int
    cost_per_1m_input: float
    cost_per_1m_output: float
    context_window: int
    speed_tokens_per_sec: float
    capabilities: Dict[str, bool]
    fallback_model: str = None

# Future model configurations
FUTURE_MODELS = {
    ModelType.FAST_CHAT: ModelConfig(
        name="Gemini 2.0 Flash",
        provider="google",
        api_key_env="GOOGLE_API_KEY",
        endpoint="gemini-2.0-flash",
        max_tokens=8192,
        cost_per_1m_input=0.18,
        cost_per_1m_output=0.18,
        context_window=1000000,
        speed_tokens_per_sec=175.5,
        capabilities={
            "text": True,
            "vision": True,
            "coding": True,
            "reasoning": True
        },
        fallback_model="minimax_m1"
    ),
    
    ModelType.RESEARCH: ModelConfig(
        name="GPT-4.1 mini",
        provider="openai",
        api_key_env="OPENAI_API_KEY",
        endpoint="gpt-4.1-mini",
        max_tokens=16384,
        cost_per_1m_input=0.70,
        cost_per_1m_output=0.70,
        context_window=1000000,
        speed_tokens_per_sec=69.7,
        capabilities={
            "text": True,
            "vision": True,
            "coding": True,
            "reasoning": True
        },
        fallback_model="gpt_4o_mini"
    ),
    
    ModelType.CODING: ModelConfig(
        name="DeepSeek V3",
        provider="deepseek",
        api_key_env="DEEPSEEK_API_KEY",
        endpoint="deepseek-v3",
        max_tokens=8192,
        cost_per_1m_input=0.48,
        cost_per_1m_output=0.48,
        context_window=64000,
        speed_tokens_per_sec=80.0,
        capabilities={
            "text": True,
            "vision": False,
            "coding": True,
            "reasoning": True
        },
        fallback_model="gpt_4_1_mini"
    ),
    
    ModelType.VISION: ModelConfig(
        name="GPT-4o mini",
        provider="openai",
        api_key_env="OPENAI_API_KEY",
        endpoint="gpt-4o-mini",
        max_tokens=8192,
        cost_per_1m_input=0.26,
        cost_per_1m_output=0.26,
        context_window=128000,
        speed_tokens_per_sec=80.0,
        capabilities={
            "text": True,
            "vision": True,
            "coding": False,
            "reasoning": False
        },
        fallback_model="gemini_2_flash"
    ),
    
    ModelType.REASONING: ModelConfig(
        name="Claude 4 Sonnet",
        provider="anthropic",
        api_key_env="ANTHROPIC_API_KEY",
        endpoint="claude-4-sonnet",
        max_tokens=16384,
        cost_per_1m_input=6.00,
        cost_per_1m_output=6.00,
        context_window=200000,
        speed_tokens_per_sec=62.0,
        capabilities={
            "text": True,
            "vision": True,
            "coding": True,
            "reasoning": True
        },
        fallback_model="gpt_4_1_mini"
    ),
    
    ModelType.BACKUP: ModelConfig(
        name="MiniMax M1",
        provider="minimax",
        api_key_env="MINIMAX_API_KEY",
        endpoint="minimax-m1",
        max_tokens=8192,
        cost_per_1m_input=0.75,
        cost_per_1m_output=0.75,
        context_window=1000000,
        speed_tokens_per_sec=110.0,
        capabilities={
            "text": True,
            "vision": True,
            "coding": True,
            "reasoning": True
        },
        fallback_model="gemini_2_flash"
    )
}
```

### Phase 3: Enhanced LLM Provider

#### 3.1 Updated LLM Provider Implementation
```python
# src/core/llm_providers.py (enhanced)
import os
import asyncio
from typing import Dict, Any, Optional
from .model_config import ModelType, ModelConfig, FUTURE_MODELS

class EnhancedLLMProvider:
    def __init__(self):
        self.models = FUTURE_MODELS
        self.active_models = {}
        self.health_status = {}
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize all model connections"""
        for model_type, config in self.models.items():
            if os.getenv(config.api_key_env):
                self.active_models[model_type] = config
                self.health_status[model_type] = "healthy"
    
    async def call_model(self, model_type: ModelType, prompt: str, 
                        images: Optional[list] = None, **kwargs) -> Dict[str, Any]:
        """Call the specified model with intelligent fallback"""
        config = self.active_models.get(model_type)
        if not config:
            raise ValueError(f"Model {model_type} not available")
        
        try:
            # Try primary model
            result = await self._call_model_provider(config, prompt, images, **kwargs)
            return result
        except Exception as e:
            # Fallback to backup model
            if config.fallback_model:
                fallback_config = self._get_fallback_config(config.fallback_model)
                if fallback_config:
                    return await self._call_model_provider(fallback_config, prompt, images, **kwargs)
            raise e
    
    async def _call_model_provider(self, config: ModelConfig, prompt: str, 
                                 images: Optional[list] = None, **kwargs) -> Dict[str, Any]:
        """Call specific model provider"""
        if config.provider == "google":
            return await self._call_google_model(config, prompt, images, **kwargs)
        elif config.provider == "openai":
            return await self._call_openai_model(config, prompt, images, **kwargs)
        elif config.provider == "anthropic":
            return await self._call_anthropic_model(config, prompt, images, **kwargs)
        elif config.provider == "deepseek":
            return await self._call_deepseek_model(config, prompt, images, **kwargs)
        elif config.provider == "minimax":
            return await self._call_minimax_model(config, prompt, images, **kwargs)
        else:
            raise ValueError(f"Unknown provider: {config.provider}")
    
    async def _call_google_model(self, config: ModelConfig, prompt: str, 
                               images: Optional[list] = None, **kwargs) -> Dict[str, Any]:
        """Call Google Gemini models"""
        import google.generativeai as genai
        
        genai.configure(api_key=os.getenv(config.api_key_env))
        model = genai.GenerativeModel(config.endpoint)
        
        if images:
            # Handle multimodal input
            content = [prompt] + images
            response = await model.generate_content_async(content, **kwargs)
        else:
            response = await model.generate_content_async(prompt, **kwargs)
        
        return {
            "text": response.text,
            "model": config.name,
            "provider": "google",
            "usage": response.usage_metadata if hasattr(response, 'usage_metadata') else None
        }
    
    async def _call_openai_model(self, config: ModelConfig, prompt: str, 
                               images: Optional[list] = None, **kwargs) -> Dict[str, Any]:
        """Call OpenAI models"""
        from openai import AsyncOpenAI
        
        client = AsyncOpenAI(api_key=os.getenv(config.api_key_env))
        
        messages = [{"role": "user", "content": prompt}]
        if images:
            # Handle multimodal input
            content = [{"type": "text", "text": prompt}]
            for image in images:
                content.append({"type": "image_url", "image_url": {"url": image}})
            messages[0]["content"] = content
        
        response = await client.chat.completions.create(
            model=config.endpoint,
            messages=messages,
            max_tokens=config.max_tokens,
            **kwargs
        )
        
        return {
            "text": response.choices[0].message.content,
            "model": config.name,
            "provider": "openai",
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }
        }
    
    async def _call_anthropic_model(self, config: ModelConfig, prompt: str, 
                                  images: Optional[list] = None, **kwargs) -> Dict[str, Any]:
        """Call Anthropic Claude models"""
        import anthropic
        
        client = anthropic.AsyncAnthropic(api_key=os.getenv(config.api_key_env))
        
        message = {"role": "user", "content": prompt}
        if images:
            # Handle multimodal input
            content = [{"type": "text", "text": prompt}]
            for image in images:
                content.append({"type": "image", "source": {"type": "base64", "data": image}})
            message["content"] = content
        
        response = await client.messages.create(
            model=config.endpoint,
            max_tokens=config.max_tokens,
            messages=[message],
            **kwargs
        )
        
        return {
            "text": response.content[0].text,
            "model": config.name,
            "provider": "anthropic",
            "usage": {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens
            }
        }
    
    async def _call_deepseek_model(self, config: ModelConfig, prompt: str, 
                                 images: Optional[list] = None, **kwargs) -> Dict[str, Any]:
        """Call DeepSeek models"""
        import httpx
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.deepseek.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {os.getenv(config.api_key_env)}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": config.endpoint,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": config.max_tokens,
                    **kwargs
                }
            )
            
            result = response.json()
            return {
                "text": result["choices"][0]["message"]["content"],
                "model": config.name,
                "provider": "deepseek",
                "usage": result.get("usage", {})
            }
    
    async def _call_minimax_model(self, config: ModelConfig, prompt: str, 
                                images: Optional[list] = None, **kwargs) -> Dict[str, Any]:
        """Call MiniMax models"""
        import httpx
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.minimax.chat/v1/text/chatcompletion_v2",
                headers={
                    "Authorization": f"Bearer {os.getenv(config.api_key_env)}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": config.endpoint,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": config.max_tokens,
                    **kwargs
                }
            )
            
            result = response.json()
            return {
                "text": result["reply"],
                "model": config.name,
                "provider": "minimax",
                "usage": result.get("usage", {})
            }
    
    def _get_fallback_config(self, fallback_name: str) -> Optional[ModelConfig]:
        """Get fallback model configuration"""
        for config in self.active_models.values():
            if config.name.lower().replace(" ", "_") == fallback_name:
                return config
        return None
```

### Phase 4: Intelligent Model Router

#### 4.1 Enhanced Model Router
```python
# src/core/model_router.py (new file)
from typing import Dict, Any, Optional
from .model_config import ModelType, FUTURE_MODELS
from .llm_providers import EnhancedLLMProvider

class IntelligentModelRouter:
    def __init__(self):
        self.llm_provider = EnhancedLLMProvider()
        self.task_classifier = TaskClassifier()
        self.cost_tracker = CostTracker()
        self.performance_monitor = PerformanceMonitor()
    
    async def route_request(self, prompt: str, images: Optional[list] = None, 
                          user_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Intelligently route request to best model"""
        
        # Analyze request
        task_type = self.task_classifier.classify_task(prompt, images, user_context)
        priority = self._determine_priority(user_context)
        cost_constraint = self._get_cost_constraint(user_context)
        
        # Select optimal model
        model_type = self._select_optimal_model(task_type, priority, cost_constraint, images)
        
        # Execute request
        start_time = time.time()
        result = await self.llm_provider.call_model(model_type, prompt, images)
        execution_time = time.time() - start_time
        
        # Track metrics
        self.performance_monitor.record_request(model_type, execution_time, result)
        self.cost_tracker.record_usage(model_type, result.get("usage", {}))
        
        return {
            **result,
            "model_used": model_type.value,
            "execution_time": execution_time,
            "task_type": task_type.value
        }
    
    def _select_optimal_model(self, task_type: ModelType, priority: str, 
                            cost_constraint: float, images: Optional[list] = None) -> ModelType:
        """Select the optimal model based on requirements"""
        
        # Filter models by capabilities
        available_models = []
        for model_type, config in self.llm_provider.active_models.items():
            if images and not config.capabilities.get("vision", False):
                continue
            
            if task_type == ModelType.CODING and not config.capabilities.get("coding", False):
                continue
            
            available_models.append((model_type, config))
        
        if not available_models:
            return ModelType.BACKUP  # Fallback to backup model
        
        # Score models based on requirements
        scored_models = []
        for model_type, config in available_models:
            score = self._calculate_model_score(
                model_type, config, task_type, priority, cost_constraint
            )
            scored_models.append((model_type, score))
        
        # Return highest scoring model
        return max(scored_models, key=lambda x: x[1])[0]
    
    def _calculate_model_score(self, model_type: ModelType, config: ModelConfig, 
                             task_type: ModelType, priority: str, cost_constraint: float) -> float:
        """Calculate model score based on requirements"""
        score = 0.0
        
        # Task-specific scoring
        if task_type == ModelType.FAST_CHAT and model_type == ModelType.FAST_CHAT:
            score += 10.0
        elif task_type == ModelType.RESEARCH and model_type == ModelType.RESEARCH:
            score += 10.0
        elif task_type == ModelType.CODING and model_type == ModelType.CODING:
            score += 10.0
        elif task_type == ModelType.VISION and model_type == ModelType.VISION:
            score += 10.0
        elif task_type == ModelType.REASONING and model_type == ModelType.REASONING:
            score += 10.0
        
        # Speed scoring
        if priority == "speed":
            score += config.speed_tokens_per_sec / 100.0
        
        # Cost scoring
        total_cost = config.cost_per_1m_input + config.cost_per_1m_output
        if total_cost <= cost_constraint:
            score += (cost_constraint - total_cost) / cost_constraint * 5.0
        
        # Context window scoring
        score += min(config.context_window / 1000000.0, 2.0)
        
        return score

class TaskClassifier:
    def classify_task(self, prompt: str, images: Optional[list] = None, 
                     user_context: Dict[str, Any] = None) -> ModelType:
        """Classify the type of task"""
        
        prompt_lower = prompt.lower()
        
        # Vision tasks
        if images or any(word in prompt_lower for word in ["image", "screenshot", "photo", "picture", "ocr"]):
            return ModelType.VISION
        
        # Coding tasks
        if any(word in prompt_lower for word in ["code", "program", "function", "class", "debug", "algorithm"]):
            return ModelType.CODING
        
        # Research tasks
        if any(word in prompt_lower for word in ["research", "analyze", "study", "investigate", "examine"]):
            return ModelType.RESEARCH
        
        # Complex reasoning
        if len(prompt) > 1000 or any(word in prompt_lower for word in ["complex", "advanced", "detailed analysis"]):
            return ModelType.REASONING
        
        # Default to fast chat
        return ModelType.FAST_CHAT

class CostTracker:
    def __init__(self):
        self.daily_usage = {}
        self.monthly_usage = {}
    
    def record_usage(self, model_type: ModelType, usage: Dict[str, Any]):
        """Record model usage for cost tracking"""
        # Implementation for tracking costs
        pass

class PerformanceMonitor:
    def __init__(self):
        self.performance_metrics = {}
    
    def record_request(self, model_type: ModelType, execution_time: float, result: Dict[str, Any]):
        """Record performance metrics"""
        # Implementation for monitoring performance
        pass
```

### Phase 5: Configuration Updates

#### 5.1 Update Tool Router
```python
# src/core/tool_router.py (enhanced)
from .model_router import IntelligentModelRouter

class EnhancedToolRouter:
    def __init__(self):
        self.model_router = IntelligentModelRouter()
        # ... existing initialization
    
    async def route_tool_call(self, tool_call: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced tool routing with intelligent model selection"""
        
        if tool_call["tool"] == "llm.call":
            # Use intelligent model router
            prompt = tool_call["args"]["prompt"]
            images = tool_call["args"].get("images")
            user_context = tool_call.get("user_context", {})
            
            result = await self.model_router.route_request(prompt, images, user_context)
            return {
                "success": True,
                "data": result,
                "model_used": result["model_used"],
                "execution_time": result["execution_time"]
            }
        
        # ... existing tool routing logic
```

## 🔄 Migration Strategy

### Step 1: Gradual Rollout
1. **Week 1-2**: Add new models alongside existing ones
2. **Week 3-4**: Test new routing logic with subset of users
3. **Week 5-6**: Full rollout with fallback to current models

### Step 2: Performance Monitoring
```python
# Monitoring dashboard metrics
MONITORING_METRICS = {
    "model_performance": {
        "response_time": "average response time per model",
        "success_rate": "percentage of successful requests",
        "cost_per_request": "average cost per request",
        "error_rate": "percentage of failed requests"
    },
    "user_experience": {
        "satisfaction_score": "user feedback scores",
        "task_completion_rate": "successful task completion",
        "fallback_usage": "frequency of fallback to backup models"
    },
    "cost_optimization": {
        "monthly_cost": "total monthly API costs",
        "cost_per_user": "average cost per user",
        "model_efficiency": "cost per successful request"
    }
}
```

### Step 3: A/B Testing
```python
# A/B testing configuration
AB_TESTING_CONFIG = {
    "test_groups": {
        "current_models": 0.2,      # 20% users on current models
        "new_models": 0.8           # 80% users on new models
    },
    "metrics": [
        "response_time",
        "user_satisfaction",
        "cost_per_request",
        "task_completion_rate"
    ],
    "duration": "4_weeks"
}
```

## 📊 Expected Improvements

### Performance Gains
- **Speed**: 40-60% faster responses for chat tasks
- **Accuracy**: 25-35% improvement in complex reasoning
- **Cost**: 30-50% reduction in API costs through optimization
- **Reliability**: 99.9% uptime with intelligent fallbacks

### User Experience
- **Response Quality**: Better task-specific model selection
- **Consistency**: More reliable performance across different task types
- **Scalability**: Support for 10x more concurrent users
- **Global Performance**: Optimized for different geographic regions

## 🚨 Implementation Checklist

### Pre-Implementation
- [ ] Set up API keys for all new models
- [ ] Test model endpoints and capabilities
- [ ] Set up monitoring and alerting
- [ ] Prepare rollback plan

### Implementation
- [ ] Deploy new model configurations
- [ ] Update LLM provider with new models
- [ ] Implement intelligent router
- [ ] Update tool router integration
- [ ] Test with subset of users

### Post-Implementation
- [ ] Monitor performance metrics
- [ ] Optimize routing logic based on data
- [ ] Scale up to full user base
- [ ] Document lessons learned

## 💰 Cost Projections

### Current Costs (3 models)
- **Monthly**: $200-400
- **Per request**: $0.001-0.005

### Future Costs (6 models)
- **Monthly**: $150-300 (30% reduction)
- **Per request**: $0.0005-0.003 (40% reduction)
- **Cost optimization**: 50% through intelligent routing

## 🎯 Success Metrics

### Technical Metrics
- Response time < 2 seconds for 95% of requests
- Model selection accuracy > 90%
- Cost per request reduction > 30%
- Error rate < 1%

### Business Metrics
- User satisfaction score > 4.5/5
- Task completion rate > 95%
- User retention improvement > 20%
- Support ticket reduction > 40%

---

**Note**: This implementation plan should be executed when scaling beyond the current user base or when performance requirements increase. The current 3-model system is sufficient for most use cases and provides a solid foundation for future expansion.