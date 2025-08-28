# Quick Start Guide

Get Winky AI Agent up and running in **5 minutes**! 🚀

## ⚡ 5-Minute Setup

### Step 1: Clone the Repository (1 minute)
```bash
git clone https://github.com/winky-x/ai-agents.git
cd ai-agents
```

### Step 2: Set Up Python Environment (1 minute)
```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies (2 minutes)
```bash
pip install -r requirements.txt
```

### Step 4: Configure API Keys (1 minute)
Create a `.env` file in the root directory:
```bash
# Required for basic functionality
GOOGLE_API_KEY=your_google_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
OPENROUTER_API_KEY=your_openrouter_api_key_here
```

### Step 5: Start Chatting! (0 minutes)
```bash
python src/main.py chat
```

You'll see the Winky AI Agent banner and can start chatting immediately!

## 🎯 Your First Conversation

Try these examples to see what Winky can do:

### Basic Questions
```
You: What's the weather like in New York?
Agent: I'll check the weather for you. [Shows weather information]
```

### File Operations
```
You: Create a file called notes.txt with some meeting notes
Agent: I'll create a file with meeting notes for you. Would you like me to proceed? (Y/N)
```

### Web Searches
```
You: Find information about AI agents
Agent: I'll search the web for information about AI agents. Would you like me to proceed? (Y/N)
```

### Complex Tasks
```
You: Find a logo for my startup called "TechFlow"
Agent: I'll help you find a logo for TechFlow. This involves:
1. Searching for logo designs
2. Downloading suitable options
3. Organizing them in a folder
Would you like me to proceed? (Y/N)
```

## 🔑 Getting API Keys

### Google API Key (Required)
1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create a new API key
3. Copy the key to your `.env` file

### OpenAI API Key (Required)
1. Go to [OpenAI Platform](https://platform.openai.com/api-keys)
2. Create a new API key
3. Copy the key to your `.env` file

### OpenRouter API Key (Required)
1. Go to [OpenRouter](https://openrouter.ai/keys)
2. Create a new API key
3. Copy the key to your `.env` file

## 🚨 Troubleshooting

### "Module not found" errors
```bash
# Make sure you're in the virtual environment
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Reinstall dependencies
pip install -r requirements.txt
```

### "No such command 'chat'" error
```bash
# Make sure you're in the correct directory
cd ai-agents

# Check if the file exists
ls src/main.py
```

### API key errors
- Check that your `.env` file exists and contains valid API keys
- Ensure API keys have the necessary permissions
- Verify API quotas haven't been exceeded

## 🎉 What's Next?

Now that you're up and running, explore:

- **[User Instructions](instructions.md)** - Complete guide to all features
- **[Feature Guide](features.md)** - Learn about all capabilities
- **[Use Cases](use-cases.md)** - Real-world examples
- **[Troubleshooting](troubleshooting.md)** - Common issues and solutions

## 💡 Pro Tips

1. **Start Simple**: Begin with basic questions and gradually try more complex tasks
2. **Be Specific**: "Find a logo for my startup" is better than "Find a logo"
3. **Use Context**: The agent remembers conversations, so you can say "move that file" and it knows what you mean
4. **Confirm Actions**: The agent will ask before doing anything risky
5. **Let It Learn**: The more you use it, the better it becomes

## 🆘 Need Help?

- **Documentation**: Check the [User Instructions](instructions.md)
- **Issues**: Report bugs on [GitHub](https://github.com/winky-x/ai-agents/issues)
- **Discussions**: Ask questions in [GitHub Discussions](https://github.com/winky-x/ai-agents/discussions)

---

**Ready to experience the future of AI assistance?** Start chatting with Winky now! 🤖✨