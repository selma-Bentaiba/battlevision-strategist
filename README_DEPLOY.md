# BattleVision Strategist - Deployment Guide

Complete guide for deploying BattleVision Strategist to Streamlit Cloud or running it locally.

---

## Table of Contents

1. [Quick Deploy to Streamlit Cloud](#quick-deploy-to-streamlit-cloud)
2. [Local Installation](#local-installation)
3. [Project Structure](#project-structure)
4. [Configuration](#configuration)
5. [Troubleshooting](#troubleshooting)
6. [Advanced Deployment](#advanced-deployment)

---

## Quick Deploy to Streamlit Cloud

### Prerequisites
- GitHub account
- All project files ready

### Step 1: Prepare Your GitHub Repository

1. **Create a new GitHub repository**
   - Go to https://github.com/new
   - Repository name: `battlevision-strategist` (or your preferred name)
   - Set visibility to Public (required for free Streamlit Cloud)
   - Do NOT initialize with README (you'll upload files directly)
   - Click "Create repository"

2. **Upload project files**
   
   Option A: Using GitHub web interface
   - Click "uploading an existing file"
   - Drag and drop all files maintaining folder structure
   - Commit changes
   
   Option B: Using Git command line
   ```bash
   cd battlevision_app
   git init
   git add .
   git commit -m "Initial commit"
   git branch -M main
   git remote add origin https://github.com/YOUR-USERNAME/battlevision-strategist.git
   git push -u origin main
   ```

3. **Verify repository structure**
   
   Ensure your repository contains:
   ```
   battlevision-strategist/
   ├── app.py
   ├── requirements.txt
   ├── packages.txt (if needed)
   └── utils/
       ├── __init__.py
       ├── cv_functions.py
       └── game_theory.py
   ```

### Step 2: Deploy on Streamlit Cloud

1. **Sign in to Streamlit Cloud**
   - Go to https://share.streamlit.io/
   - Sign in with your GitHub account
   - Authorize Streamlit to access your repositories

2. **Create new app**
   - Click "New app" button
   - Repository: Select `YOUR-USERNAME/battlevision-strategist`
   - Branch: `main` (or your default branch)
   - Main file path: `app.py`
   - App URL (optional): Choose custom URL like `your-name-battlevision`

3. **Configure advanced settings (optional)**
   - Python version: 3.9 or 3.10 (recommended)
   - Secrets: Not needed for this app
   - Click "Deploy!"

4. **Wait for deployment**
   - Initial deployment takes 2-5 minutes
   - You'll see build logs in real-time
   - App will automatically open when ready

5. **Your app is live!**
   - URL format: `https://[your-app-name].streamlit.app`
   - Share this URL with anyone
   - App will auto-wake when accessed

### Step 3: Manage Your Deployed App

**Access app settings:**
- Click the menu (⋮) on your app in Streamlit Cloud dashboard
- Options: Reboot, Delete, Settings, Logs, Analytics

**Update your app:**
- Simply push changes to your GitHub repository
- Streamlit Cloud automatically detects and redeploys
- Changes appear within 1-2 minutes

**View logs:**
- Useful for debugging deployment issues
- Access via app settings menu
- Shows import errors, runtime errors, etc.

---

## Local Installation

### System Requirements

- **Operating System**: Windows, macOS, or Linux
- **Python**: 3.8 or higher (3.9 or 3.10 recommended)
- **RAM**: 4GB minimum, 8GB recommended
- **Storage**: 500MB for dependencies

### Installation Steps

1. **Install Python**
   
   Check if Python is installed:
   ```bash
   python --version
   # or
   python3 --version
   ```
   
   If not installed, download from https://www.python.org/downloads/

2. **Download project files**
   
   Option A: Clone repository
   ```bash
   git clone https://github.com/YOUR-USERNAME/battlevision-strategist.git
   cd battlevision-strategist
   ```
   
   Option B: Download ZIP
   - Download ZIP from GitHub
   - Extract to desired location
   - Open terminal in extracted folder

3. **Create virtual environment (recommended)**
   
   ```bash
   # Create virtual environment
   python -m venv venv
   
   # Activate virtual environment
   # On Windows:
   venv\Scripts\activate
   
   # On macOS/Linux:
   source venv/bin/activate
   ```

4. **Install dependencies**
   
   ```bash
   pip install -r requirements.txt
   ```
   
   This installs:
   - streamlit
   - numpy
   - opencv-python-headless
   - Pillow
   - matplotlib
   - nashpy
   - scipy

5. **Install system dependencies (Linux only)**
   
   If you get OpenCV errors on Linux:
   ```bash
   sudo apt-get update
   sudo apt-get install -y libgl1-mesa-glx libglib2.0-0
   ```

6. **Run the application**
   
   ```bash
   streamlit run app.py
   ```
   
   Or use provided scripts:
   ```bash
   # Linux/Mac
   ./start.sh
   
   # Windows
   start.bat
   ```

7. **Access the application**
   
   - Browser opens automatically at http://localhost:8501
   - If not, manually navigate to that URL
   - Press Ctrl+C in terminal to stop

### Updating Local Installation

```bash
# Pull latest changes (if using git)
git pull

# Update dependencies
pip install -r requirements.txt --upgrade

# Restart the app
streamlit run app.py
```

---

## Project Structure

### Required Files

```
battlevision-strategist/
│
├── app.py                      # Main Streamlit application (REQUIRED)
│   └── Contains UI, tabs, and main logic
│
├── requirements.txt            # Python dependencies (REQUIRED)
│   └── Lists all pip packages needed
│
├── packages.txt                # System dependencies (OPTIONAL)
│   └── Only needed if OpenCV has issues
│
├── .streamlit/                 # Configuration folder (OPTIONAL)
│   └── config.toml            # Streamlit settings
│
└── utils/                      # Utility modules (REQUIRED)
    ├── __init__.py            # Makes utils a package (REQUIRED)
    ├── cv_functions.py        # Computer vision functions (REQUIRED)
    └── game_theory.py         # Game theory calculations (REQUIRED)
```

### Optional Files

```
├── README.md                   # Project documentation
├── DEPLOYMENT_GUIDE.md        # This file
├── start.sh                   # Linux/Mac launcher
├── start.bat                  # Windows launcher
├── test_components.py         # Testing suite
└── .gitignore                 # Git ignore file
```

### File Descriptions

**app.py**
- Main application entry point
- Defines UI layout and tabs
- Handles user interactions
- Calls utility functions
- ~730 lines of code

**requirements.txt**
- Lists Python package dependencies
- Format: `package==version`
- Used by pip and Streamlit Cloud
- Critical for deployment

**packages.txt** (create if needed)
- System-level dependencies
- One package per line
- Example content:
  ```
  libgl1-mesa-glx
  libglib2.0-0
  ```

**utils/__init__.py**
- Can be empty file
- Makes `utils` folder a Python package
- Allows imports like `from utils.cv_functions import ...`
- MUST exist for imports to work

**utils/cv_functions.py**
- Computer vision utilities
- Object detection simulation
- Adversarial patch generation
- Defense mechanisms
- ~250 lines of code

**utils/game_theory.py**
- Game theory calculations
- Payoff matrix generation
- Nash equilibrium computation
- Strategy evolution
- ~300 lines of code

---

## Configuration

### Streamlit Configuration (.streamlit/config.toml)

Create `.streamlit/config.toml` for custom settings:

```toml
[theme]
primaryColor = "#00b894"
backgroundColor = "#1a1a1a"
secondaryBackgroundColor = "#2d3436"
textColor = "#ffffff"

[server]
headless = true
enableCORS = false
port = 8501

[browser]
gatherUsageStats = false
```

### Environment Variables

If you need secrets or API keys (not currently used):

Create `.streamlit/secrets.toml`:
```toml
# Example (not used in current version)
# API_KEY = "your-api-key-here"
```

Access in app:
```python
import streamlit as st
api_key = st.secrets["API_KEY"]
```

### Resource Limits

**Streamlit Cloud Free Tier:**
- 1GB RAM per app
- 1 CPU core
- Apps sleep after inactivity (auto-wake on access)
- Up to 3 apps per account

**Optimization tips:**
- Use `@st.cache_data` for expensive computations
- Minimize image sizes
- Lazy load heavy operations
- Use session state wisely

---

## Troubleshooting

### Common Deployment Issues

#### Issue 1: Import Errors

**Symptom:**
```
ModuleNotFoundError: No module named 'utils'
ModuleNotFoundError: No module named 'cv2'
```

**Solutions:**
1. Ensure `utils/__init__.py` exists (can be empty)
2. Verify `requirements.txt` includes all dependencies
3. Check file structure matches required layout
4. Try clearing Streamlit cache and redeploying

#### Issue 2: OpenCV Not Working

**Symptom:**
```
ImportError: libGL.so.1: cannot open shared object file
```

**Solutions:**

On Streamlit Cloud:
1. Create `packages.txt` in repository root:
   ```
   libgl1-mesa-glx
   libglib2.0-0
   ```
2. Commit and push changes
3. Streamlit Cloud will auto-redeploy

On local Linux:
```bash
sudo apt-get update
sudo apt-get install -y libgl1-mesa-glx libglib2.0-0
```

On local macOS/Windows:
- Usually works out of the box with opencv-python-headless
- If issues persist, try `opencv-python` instead in requirements.txt

#### Issue 3: App Deployment Fails

**Symptom:**
- Deployment never completes
- Build logs show errors
- App shows "Oh no" error page

**Solutions:**
1. Check build logs in Streamlit Cloud dashboard
2. Verify all files are present in repository
3. Test locally first: `streamlit run app.py`
4. Check Python version compatibility (use 3.9 or 3.10)
5. Ensure requirements.txt has correct package versions
6. Try deleting and recreating the app

#### Issue 4: App is Slow or Crashes

**Symptoms:**
- App takes long to load
- Operations timeout
- Out of memory errors

**Solutions:**
1. Optimize image sizes (resize before processing)
2. Use caching for expensive operations:
   ```python
   @st.cache_data
   def expensive_function():
       # Your code here
   ```
3. Reduce number of replicator dynamics iterations
4. Use simpler patch types for demos
5. Consider upgrading to Streamlit Cloud paid tier

#### Issue 5: Nash Equilibrium Calculation Fails

**Symptom:**
```
No equilibrium found
Solver did not converge
```

**Solutions:**
1. Adjust slider values to avoid degenerate cases
2. Ensure payoff matrix is not all zeros
3. Check that probabilities sum to 1.0
4. Try fallback LP solver if support enumeration fails
5. Verify nashpy and scipy are installed correctly

#### Issue 6: Images Not Displaying

**Symptoms:**
- Placeholder images show broken links
- Uploaded images fail to load

**Solutions:**
1. Check file size limits (Streamlit: 200MB max)
2. Use supported formats: JPG, PNG, BMP
3. Verify PIL/Pillow is installed
4. Test with sample images first
5. Check browser console for errors

### Testing Locally Before Deploy

Always test locally before deploying:

```bash
# Run test suite
python test_components.py

# Expected output:
# ✅ All imports working
# ✅ CV functions operational
# ✅ Game theory calculations correct
# ✅ Nash equilibrium valid

# Run the app
streamlit run app.py

# Test all tabs
# Try all features
# Check for errors in terminal
```

### Getting Help

**Streamlit Cloud Issues:**
- Documentation: https://docs.streamlit.io/streamlit-community-cloud
- Community Forum: https://discuss.streamlit.io/
- GitHub Issues: Check for similar problems

**Application Issues:**
- Review code comments in app.py
- Check test_components.py output
- Verify all dependencies are installed
- Test individual functions in Python console

**General Python Issues:**
- Stack Overflow: Search error messages
- Python documentation: https://docs.python.org
- Package-specific docs (OpenCV, NumPy, etc.)

---

## Advanced Deployment

### Custom Domain

**Streamlit Cloud Pro:**
1. Upgrade to Pro tier
2. Add custom domain in app settings
3. Configure DNS records as instructed
4. SSL/HTTPS handled automatically

**Alternative (Free):**
- Use URL shortener (bit.ly, tinyurl.com)
- Create redirect from your domain
- Free tier apps get `*.streamlit.app` URL

### Analytics and Monitoring

**Built-in Analytics:**
- Streamlit Cloud provides basic analytics
- View in app dashboard
- Tracks visitors, usage patterns

**Custom Analytics:**
Add to app.py:
```python
import streamlit as st

# Track page views
if 'views' not in st.session_state:
    st.session_state.views = 0
st.session_state.views += 1

# Log to console (visible in Streamlit Cloud logs)
print(f"App views: {st.session_state.views}")
```

### Performance Optimization

**1. Caching Strategy**

```python
import streamlit as st

@st.cache_data
def load_model():
    # Expensive model loading
    return model

@st.cache_data
def compute_nash_equilibrium(payoff_matrix):
    # Cache Nash equilibrium results
    return strategies
```

**2. Lazy Loading**

```python
# Load heavy resources only when needed
if st.button("Run Analysis"):
    with st.spinner("Computing..."):
        results = expensive_operation()
```

**3. Session State Management**

```python
# Store results in session state
if 'nash_results' not in st.session_state:
    st.session_state.nash_results = None

# Only recompute if needed
if st.button("Calculate") or st.session_state.nash_results is None:
    st.session_state.nash_results = calculate_nash()
```

### Multi-Page Apps

To add more pages:

```
battlevision-strategist/
├── app.py                  # Main entry
└── pages/
    ├── 1_Tutorial.py      # Auto-numbered pages
    ├── 2_Advanced.py
    └── 3_About.py
```

Streamlit automatically creates navigation sidebar.

### CI/CD Integration

**Automatic Testing Before Deploy:**

Create `.github/workflows/test.yml`:
```yaml
name: Test

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: '3.9'
      - run: pip install -r requirements.txt
      - run: python test_components.py
```

### Scaling to Production

**For High Traffic:**
1. Consider Streamlit Cloud Pro/Enterprise
2. Self-host on AWS/GCP/Azure
3. Use Docker for containerization
4. Implement load balancing
5. Add database for persistence

**Docker Deployment:**

Create `Dockerfile`:
```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py"]
```

Build and run:
```bash
docker build -t battlevision .
docker run -p 8501:8501 battlevision
```

---

## Security Considerations

### For Public Deployment

**Do NOT include:**
- API keys in code
- Passwords or credentials
- Private data in repository

**Use Streamlit Secrets for sensitive data:**
```python
import streamlit as st
api_key = st.secrets["API_KEY"]
```

**Input Validation:**
```python
# Validate user uploads
if uploaded_file is not None:
    if uploaded_file.size > 10 * 1024 * 1024:  # 10MB limit
        st.error("File too large")
        return
```

### Access Control

**Password Protection (Simple):**
```python
import streamlit as st

def check_password():
    if 'password_correct' not in st.session_state:
        st.text_input("Password", type="password", key="password")
        if st.button("Login"):
            if st.session_state.password == "your-password":
                st.session_state.password_correct = True
            else:
                st.error("Incorrect password")
    return st.session_state.get('password_correct', False)

if check_password():
    # Show app
    st.title("BattleVision Strategist")
    # ... rest of app
```

---

## Maintenance

### Regular Updates

**Monthly:**
- Check for package updates
- Test with latest Streamlit version
- Review and merge dependency updates

**Update dependencies:**
```bash
pip install -r requirements.txt --upgrade
pip freeze > requirements.txt  # Save new versions
```

### Monitoring

**Check regularly:**
- Deployment status in Streamlit Cloud
- Error logs for runtime issues
- User feedback and bug reports
- GitHub issues and pull requests

### Backup Strategy

**Version Control:**
- Commit regularly to Git
- Use meaningful commit messages
- Tag releases: `git tag v1.0`

**Export Important Data:**
- Save configuration files
- Document custom changes
- Keep local backup of repository

