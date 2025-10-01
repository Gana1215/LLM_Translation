import streamlit as st
import subprocess
import sys
import shutil

st.title("Git LFS Test in Streamlit")

# ----------------- Install Python wrapper (for testing only) -----------------
try:
    import git_lfs
    st.success("Python package git_lfs already installed ✅")
except ImportError:
    st.info("Installing git_lfs Python package...")
    subprocess.run([sys.executable, "-m", "pip", "install", "git_lfs"], check=True)
    st.success("Python package git_lfs installed ✅")

# ----------------- Check if Git CLI exists -----------------
if shutil.which("git"):
    st.info("Git CLI is available ✅")
    
    # Check Git LFS CLI
    try:
        result = subprocess.run(["git", "lfs", "version"], capture_output=True, text=True, check=True)
        st.success(f"Git LFS CLI detected: {result.stdout.strip()}")
        
        # ----------------- Run test LFS commands -----------------
        for cmd in [
            ["git", "lfs", "ls-files"],
            ["git", "lfs", "fetch", "--all"],
            ["git", "lfs", "checkout"]
        ]:
            st.markdown(f"**Running:** `{' '.join(cmd)}`")
            try:
                proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
                st.text(proc.stdout if proc.stdout else "(No output)")
            except subprocess.CalledProcessError as e:
                st.warning(f"Command failed:\n{e.stderr.strip()}")
    except subprocess.CalledProcessError:
        st.warning("Git LFS CLI is NOT installed!")
else:
    st.error("Git CLI not found. Git LFS cannot run!")
