# Winky AI Agent - User Instructions

## Quick Start

### 1. Prerequisites
- Python 3.8+ installed
- Git for cloning the repository
- API keys for LLM services (optional but recommended)

### 2. Installation

#### Clone the repository:
```bash
git clone https://github.com/winky-x/ai-agents.git
cd ai-agents
```

#### Create virtual environment:
```bash
# Windows
python -m venv .venv
.\.venv\Scripts\activate

# macOS/Linux
python3 -m venv .venv
source .venv/bin/activate
```

#### Install dependencies:
```bash
pip install -r requirements.txt
pip install playwright
playwright install chromium
```

### 3. Configuration

#### Create `.env` file in the project root:
```env
# Required for full functionality
OPENAI_API_KEY=your_openai_api_key_here
OPENROUTER_API_KEY=your_openrouter_api_key_here
GOOGLE_API_KEY=your_google_api_key_here

# Optional settings
AGENT_NAME=Winky
LOG_LEVEL=INFO
USER_APPROVAL_REQUIRED=true
```

#### API Key Setup:
- **OpenAI**: Get from https://platform.openai.com/api-keys
- **OpenRouter**: Get from https://openrouter.ai/keys
- **Google**: Get from https://makersuite.google.com/app/apikey

### 4. Running the Agent

#### Interactive Chat Mode (Recommended):
```bash
python src/main.py chat
```

#### Command Examples:
```bash
# Basic questions
"What's the weather like today?"
"Explain quantum computing in simple terms"

# Web tasks
"Find a logo for my startup and save it to work/logos/"
"Search for the latest AI news and summarize"

# File operations
"Read the file at data/example.txt"
"Create a summary of my project in work/summary.md"

# Image analysis
"Analyze this image: desktop/logo.png"
"Look at this screenshot and tell me what's wrong"
```

## Available Commands

### Core Commands
```bash
python src/main.py chat          # Interactive chat mode
python src/main.py demo          # Run demo task
python src/main.py status        # Show system status
python src/main.py tasks         # List all tasks
```

### Task Management
```bash
python src/main.py task "Your goal here"           # Create task
python src/main.py execute task_id                 # Execute specific task
python src/main.py approve                         # Review pending actions
```

### Security & Policy
```bash
python src/main.py profile dev                     # Set dev profile (minimal)
python src/main.py profile research                # Set research profile
python src/main.py profile admin                   # Set admin profile (full access)
python src/main.py audit                           # View audit log
python src/main.py reload                          # Reload security policy
```

## Security Profiles

### Dev Profile (Default)
- Minimal permissions
- Safe for development
- Requires confirmation for most actions

### Research Profile
- Web access enabled
- File reading allowed
- Controlled file writing
- Good for research tasks

### Admin Profile
- Full access with confirmations
- Shell commands allowed (restricted)
- All file operations permitted
- Use with caution

## Features Guide

### Natural Language Interface
- No slash commands needed
- Type naturally like talking to a person
- Agent detects intent automatically
- Confirms before risky actions

### Browser Automation
- Headless browsing for security
- Web search and navigation
- File downloads with confirmation
- Screenshot capture

### File Operations
- Read/write text files
- Copy/move files between locations
- HTTP downloads with progress
- Path validation and security

### Vision Capabilities
- Image analysis with Gemini
- Multi-modal prompts (text + image)
- Image processing workflows
- Visual content understanding

### RAG (Retrieval-Augmented Generation)
- Local TF-IDF search
- Optional OpenAI embeddings
- Document chunking and indexing
- Source attribution

## Troubleshooting

### Common Issues

#### "No such command 'chat'"
- Make sure you're in the correct directory
- Update to latest version: `git pull origin main`
- Check if virtual environment is activated

#### "Missing API key" errors
- Create `.env` file with your API keys
- Restart the application after adding keys
- Check key format and validity

#### Browser automation fails
- Install Playwright: `pip install playwright`
- Install Chromium: `playwright install chromium`
- Check internet connection

#### Permission errors
- Use appropriate security profile
- Check file/directory permissions
- Review audit log for details

### Getting Help

#### Check Status:
```bash
python src/main.py status
```

#### View Logs:
```bash
python src/main.py audit
```

#### Test Installation:
```bash
python src/main.py demo
```

## Advanced Usage

### Custom Tasks
```bash
# Create complex multi-step task
python src/main.py task "Research AI agents, download logos, create summary report"
```

### Batch Operations
```bash
# Execute multiple tasks
python src/main.py task "Task 1"
python src/main.py task "Task 2"
python src/main.py tasks
```

### Policy Customization
Edit `policy.yaml` to customize:
- Allowed domains for web access
- File operation permissions
- Shell command allowlists
- Risk assessment rules

### Development
```bash
# Install in development mode
pip install -e .

# Use as console command
consiglio chat
consiglio --help
```

## Security Notes

### Best Practices
- Start with `dev` profile for safety
- Review pending actions before approval
- Monitor audit logs regularly
- Keep API keys secure

### Risk Management
- Agent asks for confirmation before risky actions
- All actions are logged with timestamps
- File operations are path-validated
- Web access uses domain restrictions

### Privacy
- No data is sent to external services without API keys
- Local file operations stay on your machine
- Audit logs are stored locally
- Browser automation runs in isolated environment

## Support

### Documentation
- Technical details: `docs/blueprint.md`
- Source code: `src/` directory
- Configuration: `policy.yaml` and `.env`

### Issues
- Check existing issues on GitHub
- Create new issue with error details
- Include logs and configuration

### Contributing
- Fork the repository
- Create feature branch
- Submit pull request with tests