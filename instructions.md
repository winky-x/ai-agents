# Winky AI Agent - User Instructions

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- Git
- Internet connection for API access

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/winky-x/ai-agents.git
   cd ai-agents
   ```

2. **Set up Python environment**
   ```bash
   # Create virtual environment
   python3 -m venv venv
   
   # Activate virtual environment
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up API keys**
   Create a `.env` file in the root directory:
   ```bash
   # Required API keys
   GOOGLE_API_KEY=your_google_api_key_here
   OPENAI_API_KEY=your_openai_api_key_here
   OPENROUTER_API_KEY=your_openrouter_api_key_here
   
   # Optional API keys for enhanced features
   STRIPE_SECRET_KEY=your_stripe_secret_key
   OPENWEATHER_API_KEY=your_openweather_api_key
   GOOGLE_MAPS_API_KEY=your_google_maps_api_key
   GOOGLE_CLIENT_ID=your_google_client_id
   GOOGLE_CLIENT_SECRET=your_google_client_secret
   ```

5. **Run the agent**
   ```bash
   python src/main.py chat
   ```

## 🎯 Basic Usage

### Starting a Chat Session
```bash
python src/main.py chat
```

You'll see the Winky AI Agent banner and can start chatting naturally. The agent will:
- Understand your intent automatically
- Ask for confirmation before performing actions
- Learn from your interactions
- Remember context across conversations

### Example Interactions

**Simple Questions:**
```
You: What's the weather like in New York?
Agent: I'll check the weather for you. [Shows weather information]
```

**File Operations:**
```
You: Create a file called notes.txt with some meeting notes
Agent: I'll create a file with meeting notes for you. Would you like me to proceed? (Y/N)
```

**Web Searches:**
```
You: Find information about AI agents
Agent: I'll search the web for information about AI agents. Would you like me to proceed? (Y/N)
```

**Complex Tasks:**
```
You: Find a logo for my startup called "TechFlow"
Agent: I'll help you find a logo for TechFlow. This involves:
1. Searching for logo designs
2. Downloading suitable options
3. Organizing them in a folder
Would you like me to proceed? (Y/N)
```

## 🛠️ Advanced Features

### Task Management

**Create a Task:**
```bash
python src/main.py task "Research the latest AI developments"
```

**Execute a Task:**
```bash
python src/main.py execute task_20241201_143022_0
```

**List Tasks:**
```bash
python src/main.py tasks
```

### Security Profiles

**Set Security Profile:**
```bash
# Development profile (minimal permissions)
python src/main.py profile dev

# Research profile (web access, file reading)
python src/main.py profile research

# Admin profile (full access with confirmations)
python src/main.py profile admin
```

**Approve Pending Actions:**
```bash
python src/main.py approve
```

**View Audit Log:**
```bash
python src/main.py audit
```

### System Status

**Check System Status:**
```bash
python src/main.py status
```

**View System Health:**
The agent automatically monitors its own health and performance.

## 🧠 Understanding the Agent's Capabilities

### What the Agent Can Do

1. **Natural Language Understanding**
   - Understands context and intent
   - Resolves ambiguous references ("it", "that", "my project")
   - Remembers conversation history

2. **Real-World Actions**
   - Send emails via Gmail
   - Create calendar events
   - Process payments via Stripe
   - Order food delivery
   - Get weather and directions
   - Search and shop online

3. **Desktop Automation**
   - Launch applications
   - Control mouse and keyboard
   - Take screenshots
   - Read text from images (OCR)
   - Find elements on screen

4. **File Operations**
   - Create, read, write files
   - Copy and move files
   - Download files from the web
   - Organize file structures

5. **Web Automation**
   - Browse websites
   - Fill out forms
   - Extract data
   - Download files
   - Perform searches

6. **Learning and Adaptation**
   - Learns from every interaction
   - Improves performance over time
   - Adapts to your preferences
   - Prevents repeated errors

### What Makes It "Real"

Unlike simple chatbots, this agent:

- **Actually understands** what you're asking for
- **Remembers context** across conversations
- **Learns from experience** and improves
- **Performs real actions** (not just talks about them)
- **Handles errors intelligently** and recovers automatically
- **Integrates with real services** (Gmail, Stripe, etc.)
- **Adapts to your preferences** and work style

## 🔧 Configuration

### Environment Variables

**Required:**
- `GOOGLE_API_KEY`: For Gemini AI access
- `OPENAI_API_KEY`: For OpenAI services
- `OPENROUTER_API_KEY`: For DeepSeek and other models

**Optional (for enhanced features):**
- `STRIPE_SECRET_KEY`: For payment processing
- `OPENWEATHER_API_KEY`: For weather information
- `GOOGLE_MAPS_API_KEY`: For maps and directions
- `GOOGLE_CLIENT_ID` & `GOOGLE_CLIENT_SECRET`: For Gmail and Calendar

### Policy Configuration

The `policy.yaml` file controls what the agent can do:

```yaml
profiles:
  dev:
    description: "Development profile - minimal permissions"
    allow:
      - llm.call
      - file.read
      - file.write
    
  research:
    description: "Research profile - web access and file operations"
    allow:
      - llm.call
      - web.get
      - file.read
      - file.write
      - file.download
    
  admin:
    description: "Admin profile - full access with confirmations"
    allow:
      - "*"  # All tools
```

## 🚨 Troubleshooting

### Common Issues

**"Module not found" errors:**
```bash
# Make sure you're in the virtual environment
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Reinstall dependencies
pip install -r requirements.txt
```

**API key errors:**
- Check that your `.env` file exists and contains valid API keys
- Ensure API keys have the necessary permissions
- Verify API quotas haven't been exceeded

**Permission errors:**
- Check the security profile settings
- Use `python src/main.py approve` to approve pending actions
- Switch to a more permissive profile if needed

**Browser automation issues:**
- Ensure Playwright browsers are installed: `playwright install`
- Check that the system has display access (for non-headless mode)

### Getting Help

1. **Check the logs**: The agent provides detailed logging
2. **Review audit trail**: `python src/main.py audit`
3. **Check system status**: `python src/main.py status`
4. **Reset to defaults**: Delete the `data/` directory to reset

## 🎯 Best Practices

### Effective Communication

1. **Be specific**: "Find a logo for my startup" vs "Find a logo"
2. **Provide context**: "Move the file I just downloaded" (agent remembers)
3. **Use natural language**: No need for special commands
4. **Confirm actions**: The agent will ask before doing anything risky

### Security

1. **Start with 'dev' profile**: Minimal permissions for safety
2. **Review audit logs**: Regularly check what the agent has done
3. **Use confirmations**: Always confirm before allowing sensitive actions
4. **Keep API keys secure**: Don't share your `.env` file

### Performance

1. **Let it learn**: The agent improves with use
2. **Provide feedback**: Tell it when it does something well or poorly
3. **Use context**: Reference previous conversations
4. **Be patient**: Complex tasks may take time

## 🔮 Advanced Usage

### Custom Workflows

The agent can create and execute complex workflows:

```
You: Create a workflow that checks my email, summarizes important messages, and creates calendar events for meetings
Agent: I'll create a workflow for email processing and calendar management. This involves:
1. Connecting to your email
2. Analyzing message content
3. Extracting meeting information
4. Creating calendar events
Would you like me to set this up? (Y/N)
```

### Background Tasks

The agent can run tasks in the background:

```
You: Monitor my inbox every hour and alert me to urgent messages
Agent: I'll set up a background task to monitor your email. This will run every hour and notify you of urgent messages.
```

### Integration Examples

**Email Management:**
```
You: Send an email to john@example.com about our meeting tomorrow
Agent: I'll compose and send an email about tomorrow's meeting. Would you like me to proceed? (Y/N)
```

**Calendar Management:**
```
You: Schedule a meeting with the team for next Tuesday at 2 PM
Agent: I'll create a calendar event for the team meeting. Would you like me to proceed? (Y/N)
```

**Payment Processing:**
```
You: Process a $50 payment for the subscription
Agent: I'll process the payment through Stripe. Would you like me to proceed? (Y/N)
```

## 📚 Learning Resources

### Understanding AI Agents

- **Context Understanding**: How the agent maintains conversation context
- **Adaptive Learning**: How it improves from experience
- **Error Recovery**: How it handles and learns from failures
- **Real-World Integration**: How it connects to actual services

### Advanced Features

- **Vision Understanding**: Using image analysis capabilities
- **Workflow Automation**: Creating complex multi-step processes
- **Background Processing**: Running tasks automatically
- **Performance Optimization**: Monitoring and improving efficiency

## 🎉 Getting the Most Out of Your Agent

1. **Start Simple**: Begin with basic tasks and gradually increase complexity
2. **Provide Feedback**: Tell the agent when it does well or needs improvement
3. **Explore Features**: Try different types of tasks to discover capabilities
4. **Customize**: Let it learn your preferences and work style
5. **Stay Updated**: Check for new features and improvements

The Winky AI Agent is designed to be your intelligent, autonomous assistant that actually understands and helps with real-world tasks. The more you use it, the better it becomes at helping you!