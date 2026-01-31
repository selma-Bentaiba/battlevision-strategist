# 🎯 BattleVision Strategist - Streamlit Cloud Deployment

## Quick Deploy to Streamlit Cloud

### Step 1: Prepare Repository
1. Create a new GitHub repository (e.g., `battlevision-strategist`)
2. Upload all files from this folder to the repository

### Step 2: Deploy on Streamlit Cloud
1. Go to https://share.streamlit.io/
2. Click "New app"
3. Select your GitHub repository
4. Main file path: `app.py`
5. Click "Deploy"

That's it! Your app will be live in ~5 minutes.

## Files Structure
```
battlevision_deploy/
├── app.py                    # Main application
├── requirements.txt          # Python dependencies
├── packages.txt             # System dependencies
├── .streamlit/
│   └── config.toml          # Streamlit configuration
└── utils/
    ├── __init__.py
    ├── cv_functions.py
    ├── game_theory.py
    └── report_generator.py
```

## Local Testing (Optional)

If you want to test locally before deploying:

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

## Troubleshooting

### If you get import errors:
- Make sure all files are in the correct folders
- Ensure `utils/__init__.py` exists

### If OpenCV doesn't work:
- The `packages.txt` file should handle this automatically on Streamlit Cloud
- For local testing, you may need: `sudo apt-get install libgl1-mesa-glx`

### If the app is slow:
- Streamlit Cloud provides better performance than local machines
- Consider caching expensive operations with `@st.cache_data`

## Support

For issues with Streamlit Cloud deployment:
- Check https://docs.streamlit.io/streamlit-community-cloud
- Streamlit Community Forum: https://discuss.streamlit.io/

## License
Educational and Research Use Only
