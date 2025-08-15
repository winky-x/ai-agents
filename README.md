# Consiglio Agent 🤖

> *"An honest, efficient, and risk-aware AI agent"*

Consiglio is a secure, agentic AI assistant that prioritizes safety, user privacy, and explicit source citations. Built with a hardened security architecture, it provides policy-based tool control with human-in-the-loop approval for sensitive operations.

## 🏗️ Architecture

Consiglio follows a layered security architecture:

- **Policy Engine**: YAML-based security policies with safe-by-default settings
- **Tool Router**: Validates and routes tool calls through security checks
- **Orchestrator**: Manages task execution and agent lifecycle
- **Audit System**: Comprehensive logging of all operations
- **Human-in-the-Loop**: Manual approval for high-risk operations

### Security Profiles

- **`dev`**: Minimal permissions, safe for development
- **`research`**: Web access and file reading for information gathering
- **`admin`**: Full access with confirmation requirements

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd consiglio-agent
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys and preferences
   ```

4. **Run the demo**
   ```bash
   python src/main.py demo
   ```

## 📖 Usage

### Basic Commands

```bash
# Show system status
python src/main.py status

# Create a task
python src/main.py task "Search for information about AI agents"

# Execute a task
python src/main.py execute <task_id>

# List all tasks
python src/main.py tasks

# Approve pending tool calls
python src/main.py approve

# View audit log
python src/main.py audit

# Change security profile
python src/main.py profile admin

# Reload security policy
python src/main.py reload
```

### Security Profile Management

```bash
# Switch to research profile for web access
python src/main.py profile research

# Switch to admin profile for full access
python src/main.py profile admin

# Check current profile
python src/main.py status
```

## 🔒 Security Features

### Safe-by-Default Design

- **No shell access** by default
- **Domain allowlisting** for web requests
- **Path restrictions** for file operations
- **Rate limiting** on all external operations
- **Manual confirmation** for high-risk actions

### Policy-Based Control

```yaml
# Example policy configuration
profiles:
  research:
    allow:
      - web.get:
          domains: ["*"]
          max_requests_per_minute: 20
      - file.read:
          paths: ["./work", "./data"]
    deny:
      - shell.exec
      - system.control
```

### Audit & Compliance

- **Complete audit trail** of all operations
- **User decision logging** for approvals/rejections
- **Risk assessment** for each tool call
- **Compliance reporting** capabilities

## 🛠️ Tool Ecosystem

### Available Tools

- **`web.get`**: HTTP requests with domain validation
- **`file.read`/`file.write`**: File operations with path restrictions
- **`rag.search`**: Vector database search
- **`llm.call`**: LLM provider integration
- **`browser.control`**: Automated browser control (sandboxed)
- **`shell.exec`**: Shell command execution (requires approval)

### Tool Validation

Every tool call goes through:
1. **Policy validation** against current security profile
2. **Constraint checking** (domains, paths, commands)
3. **Rate limiting** verification
4. **Risk assessment** and confirmation requirements

## 🔧 Configuration

### Environment Variables

```bash
# Agent Configuration
AGENT_NAME=Consiglio
AGENT_VERSION=1.0.0
LOG_LEVEL=INFO

# Security Settings
ALLOW_SHELL=false
ALLOW_BROWSER_CONTROL=false
ALLOW_FILE_WRITE=false
USER_APPROVAL_REQUIRED=true

# LLM Providers
GOOGLE_API_KEY=your_key_here
OPENAI_API_KEY=your_key_here
OPENROUTER_API_KEY=your_key_here

# Vector Store
VECTOR_DB_PATH=./data/vectorstore
EMBEDDING_MODEL=all-MiniLM-L6-v2
```

### Policy Configuration

The `policy.yaml` file controls:
- Security profiles and permissions
- Tool-specific constraints
- Rate limiting rules
- Emergency override procedures

## 🧪 Testing

### Run Tests

```bash
# Install test dependencies
pip install pytest pytest-asyncio

# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_policy.py

# Run with coverage
pytest --cov=src tests/
```

### Test Coverage

- **Policy Engine**: Security validation and profile management
- **Tool Router**: Tool call routing and validation
- **Orchestrator**: Task execution and lifecycle management
- **Integration**: End-to-end workflow testing

## 🚨 Security Considerations

### Production Deployment

1. **Environment Isolation**: Run in dedicated user context
2. **Network Security**: Restrict external network access
3. **File Permissions**: Limit file system access
4. **API Key Management**: Use secure secret management
5. **Monitoring**: Implement comprehensive logging and alerting

### Risk Mitigation

- **Default Deny**: All operations denied unless explicitly allowed
- **Principle of Least Privilege**: Minimal permissions for each profile
- **Human Oversight**: Manual approval for sensitive operations
- **Audit Trail**: Complete logging of all activities
- **Rate Limiting**: Prevent abuse and resource exhaustion

## 🔮 Roadmap

### Phase 1: Core Security ✅
- [x] Policy engine with YAML configuration
- [x] Tool router with validation
- [x] Basic orchestrator
- [x] CLI interface

### Phase 2: Tool Implementation 🚧
- [ ] Web request handler with domain validation
- [ ] File operations with path restrictions
- [ ] RAG system with vector store
- [ ] LLM provider integration

### Phase 3: Advanced Features 📋
- [ ] Browser automation (Playwright)
- [ ] Shell execution with approval flow
- [ ] Multi-agent orchestration
- [ ] Web UI for management

### Phase 4: Production Features 📋
- [ ] Kubernetes deployment
- [ ] Prometheus metrics
- [ ] Advanced RAG capabilities
- [ ] Offline LLM support

## 🤝 Contributing

### Development Setup

1. **Fork the repository**
2. **Create feature branch**: `git checkout -b feature/amazing-feature`
3. **Make changes** with tests
4. **Run tests**: `pytest tests/`
5. **Submit pull request**

### Code Standards

- **Type hints** for all functions
- **Docstrings** for all classes and methods
- **Unit tests** for all new functionality
- **Security review** for all tool implementations

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

Consiglio is designed for security and safety, but no system is completely secure. Always:
- Review and understand the security policies
- Monitor system activity
- Keep dependencies updated
- Test thoroughly in your environment
- Never run with elevated privileges unless necessary

## 🆘 Support

- **Issues**: [GitHub Issues](https://github.com/your-repo/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-repo/discussions)
- **Security**: [Security Policy](SECURITY.md)

---

*"In the world of AI agents, trust is earned through transparency, security through design, and safety through human oversight."* - Consiglio Team 
