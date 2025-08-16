#!/usr/bin/env python3
"""
Robust script to run Streamlit with ngrok tunneling
"""

import subprocess
import time
import requests
from pyngrok import ngrok
import sys
import os
import signal
import threading
import psutil

def kill_process_on_port(port):
    """Kill any process using the specified port"""
    try:
        for proc in psutil.process_iter(['pid', 'name', 'connections']):
            try:
                for conn in proc.connections():
                    if conn.laddr.port == port:
                        print(f"🔄 Killing process {proc.name()} (PID: {proc.pid}) on port {port}")
                        proc.terminate()
                        proc.wait(timeout=5)
                        return True
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.TimeoutExpired):
                pass
    except Exception as e:
        print(f"⚠️ Warning: Could not check port {port}: {e}")
    return False

def check_streamlit_ready(port=8502, max_attempts=30):
    """Check if Streamlit is ready to accept connections"""
    for i in range(max_attempts):
        try:
            response = requests.get(f"http://localhost:{port}", timeout=2)
            if response.status_code == 200:
                return True
        except requests.exceptions.RequestException:
            pass
        time.sleep(2)
        print(f"⏳ Waiting for Streamlit to start... ({i+1}/{max_attempts})")
    return False

def run_streamlit_with_ngrok():
    """Run Streamlit app with ngrok tunneling"""
    
    port = 8502
    streamlit_process = None
    tunnel = None
    
    try:
        print("🧹 Cleaning up any existing processes...")
        kill_process_on_port(port)
        time.sleep(2)  # Give time for cleanup
        
        print("🚀 Starting Streamlit app...")
        
        # Start Streamlit in a subprocess with better error handling
        streamlit_process = subprocess.Popen([
            sys.executable, "-m", "streamlit", "run", "app.py",
            "--server.port", str(port),
            "--server.address", "localhost",
            "--server.headless", "true",
            "--server.enableCORS", "false",
            "--server.enableXsrfProtection", "false"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait for Streamlit to be ready
        if not check_streamlit_ready(port):
            print("❌ Streamlit failed to start properly")
            return
        
        print("✅ Streamlit is running on http://localhost:8502")
        
        # Create ngrok tunnel
        print("🔗 Setting up ngrok tunnel...")
        tunnel = ngrok.connect(port, "http")
        public_url = tunnel.public_url
        
        print(f"✅ Ngrok tunnel created: {public_url}")
        print(f"🌐 Local URL: http://localhost:{port}")
        
        print("\n" + "="*60)
        print("🎉 Your GenAI Playground is now live!")
        print(f"📱 Share this URL: {public_url}")
        print("="*60)
        print("\n💡 Tips:")
        print("- The app may take a moment to fully load")
        print("- If you see ERR_NGROK_3200, wait 30 seconds and refresh")
        print("- Press Ctrl+C to stop the server")
        print("="*60 + "\n")
        
        # Keep the script running and monitor processes
        try:
            while True:
                time.sleep(1)
                # Check if processes are still running
                if streamlit_process.poll() is not None:
                    print("❌ Streamlit process stopped unexpectedly")
                    break
        except KeyboardInterrupt:
            print("\n🛑 Shutting down...")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        
    finally:
        # Clean up
        print("🧹 Cleaning up...")
        if tunnel:
            try:
                ngrok.disconnect(tunnel.public_url)
            except:
                pass
        if streamlit_process:
            try:
                streamlit_process.terminate()
                streamlit_process.wait(timeout=10)
            except:
                pass
        print("✅ Done!")

if __name__ == "__main__":
    run_streamlit_with_ngrok() 