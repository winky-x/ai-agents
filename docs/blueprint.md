# Winky AI Agent - Technical Blueprint

## Overview
Winky AI Agent is a secure, autonomous terminal-based AI assistant with policy-based tool control, real browser automation, and intelligent task execution.

## Core Architecture

### 1. LLM Routing System
- **Fast Mode**: Gemini 1.5 Flash for quick responses (< 2s)
- **Deep Mode**: DeepSeek R1 via OpenRouter for complex reasoning
- **Vision Mode**: Gemini 2.0 Flash with image analysis capabilities
- **Auto-routing**: Heuristic detection based on prompt content and length

### 2. Tool Execution Engine
- **Policy Engine**: YAML-based security profiles with granular permissions
- **Tool Router**: Centralized routing with validation and audit logging
- **Confirmation System**: Y/N prompts for risky actions with risk assessment

### 3. Browser Automation (Playwright)
- **Headless Mode**: Default for security and speed
- **Actions**: open, goto, search, type, click, screenshot
- **Downloads**: Click-to-download with file interception
- **Multi-step**: Sequential action execution with error handling

### 4. File Operations
- **Read/Write**: UTF-8 text files with size limits
- **Copy/Move**: Cross-directory operations with path validation
- **Download**: HTTP streaming with timeout and error handling
- **Permissions**: Policy-gated with confirmation requirements

### 5. RAG System
- **TF-IDF**: Lightweight local search (scikit-learn)
- **Embeddings**: OpenAI embeddings with chunking (optional)
- **Chunking**: 800-token chunks with 200-token overlap
- **Sources**: Attribution and metadata tracking

### 6. Task Management
- **Persistence**: JSON-based task storage in data/tasks/
- **Memory**: Short-term session memory for context
- **Retry Logic**: Single retry for failed tool steps
- **Resume**: Task inspection and continuation

## Security Features

### Policy Profiles
- **dev**: Minimal permissions, safe defaults
- **research**: Web access, file reading, controlled writes
- **admin**: Full access with confirmations

### Tool Restrictions
- **Shell**: Allowlisted commands only (ls, cat, grep, etc.)
- **Browser**: Headless by default, domain restrictions
- **Files**: Path validation, size limits, extension filtering
- **Web**: Domain allowlists, content type blocking

### Audit System
- **Logging**: Comprehensive audit trail with timestamps
- **Risk Assessment**: Low/medium/high risk categorization
- **User Tracking**: Approval/rejection with reasons

## Key Features

### Natural Language Interface
- **Intent Detection**: Automatic command/web/search classification
- **Smart Confirmations**: Context-aware Y/N prompts
- **No Slash Commands**: Pure natural language interaction

### Vision Capabilities
- **Image Analysis**: Gemini vision for image understanding
- **Multi-modal**: Text + image prompts
- **Round-trip**: Image processing and generation workflows

### Autonomous Workflows
- **Logo Finding**: Multi-step web search → download → organize
- **Image Processing**: Vision analysis → generation → validation
- **Research Tasks**: Web search → RAG → summarization

### Error Handling
- **Graceful Degradation**: Fallback to safe defaults
- **Retry Logic**: Automatic retry for transient failures
- **User Feedback**: Clear error messages and recovery options

## Technical Stack

### Dependencies
- **Core**: click, rich, loguru, pyyaml, python-dotenv
- **LLMs**: google-generativeai, httpx (OpenRouter)
- **Browser**: playwright, chromium
- **ML**: scikit-learn (TF-IDF), numpy (embeddings)
- **HTTP**: httpx for web requests and downloads

### File Structure
```
src/
├── main.py              # CLI entry point
├── core/
│   ├── orchestrator.py  # Task management
│   ├── tool_router.py   # Tool execution
│   ├── policy.py        # Security engine
│   ├── browser.py       # Playwright controller
│   ├── rag.py          # Search and embeddings
│   └── llm_providers.py # LLM integrations
data/
├── tasks/              # Task persistence
├── vectorstore/        # RAG storage
└── work/              # Working directory
```

## Usage Examples

### Basic Chat
```bash
python src/main.py chat
# "What's the weather like?"
# "Find a logo for my startup"
# "Analyze this image: desktop/logo.png"
```

### Task Management
```bash
python src/main.py task "Research AI agents"
python src/main.py execute task_20250828_123456_0
python src/main.py status
```

### Policy Control
```bash
python src/main.py profile research
python src/main.py approve
python src/main.py audit
```

## Future Enhancements

### Planned Features
- **Custom Browser UI**: Embedded agent panel with Chromium
- **Advanced Planning**: Multi-step reasoning with memory
- **Background Mode**: Daemon with scheduled tasks
- **Plugin System**: Extensible tool ecosystem
- **Multi-modal RAG**: Image + text search capabilities

### Scalability
- **Distributed Tasks**: Multi-agent coordination
- **Persistent Memory**: Long-term context storage
- **API Server**: RESTful interface for integrations
- **Web Dashboard**: Real-time monitoring and control

## Security Considerations

### Best Practices
- **Principle of Least Privilege**: Minimal permissions by default
- **Confirmation Gates**: User approval for risky actions
- **Audit Trails**: Complete activity logging
- **Rate Limiting**: Prevent abuse and resource exhaustion
- **Input Validation**: Sanitize all user inputs

### Risk Mitigation
- **Sandboxed Execution**: Isolated tool environments
- **Content Filtering**: Block malicious file types
- **Network Security**: Domain allowlists and HTTPS enforcement
- **Resource Limits**: Timeout and size restrictions