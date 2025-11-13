# Deploying to Streamlit Cloud

## Step 1: Create a GitHub Repository

1. Go to [GitHub](https://github.com) and sign in
2. Click the "+" icon in the top right corner
3. Select "New repository"
4. Name it (e.g., `re-pack` or `repack-optimisation`)
5. **Do NOT** initialize with README, .gitignore, or license (we already have these)
6. Click "Create repository"

## Step 2: Push Your Code to GitHub

After creating the repository, GitHub will show you commands. Use these commands in your terminal:

```bash
cd "/Users/chandrus/Library/Mobile Documents/com~apple~CloudDocs/Developer/CookieJarAI/re-pack"

# Add the remote (replace YOUR_USERNAME and REPO_NAME with your actual values)
git remote add origin https://github.com/YOUR_USERNAME/REPO_NAME.git

# Push your code
git branch -M main
git push -u origin main
```

**OR** if you prefer SSH:

```bash
git remote add origin git@github.com:YOUR_USERNAME/REPO_NAME.git
git branch -M main
git push -u origin main
```

## Step 3: Deploy on Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with your GitHub account
3. Click "New app"
4. Select your repository from the dropdown
5. Set the **Main file path** to: `multipack_poc/main.py`
6. Click "Deploy"

Streamlit Cloud will automatically:
- Detect `requirements.txt` in the root directory
- Install all dependencies
- Deploy your app

## Important Notes

- The `requirements.txt` file is now in the root directory (as required by Streamlit Cloud)
- The main file path for Streamlit Cloud should be: `multipack_poc/main.py`
- Make sure all your code is committed and pushed to GitHub before deploying

