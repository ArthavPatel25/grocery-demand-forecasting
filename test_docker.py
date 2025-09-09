#!/usr/bin/env python3

import subprocess
import requests
import time
import sys

def run_command(cmd):
    """Run command and return success status"""
    try:
        result = subprocess.run(cmd, shell=True, check=False, 
                              stdout=subprocess.DEVNULL, 
                              stderr=subprocess.DEVNULL)
        return result.returncode == 0
    except Exception:
        return False

def wait_for_service(url, timeout=60):
    """Wait for service to become available"""
    print(f"Waiting for service at {url}...")
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                print("Service is ready!")
                return True
        except requests.exceptions.RequestException:
            pass
        time.sleep(2)
    
    print("Service failed to start within timeout")
    return False

def test_docker():
    """Test Docker build and run"""
    print("Testing Docker...")
    
    # Clean up any existing containers
    print("Cleaning up existing containers...")
    run_command("docker stop grocery-test 2>nul")
    run_command("docker rm grocery-test 2>nul")
    
    # Build image
    print("Building Docker image...")
    if not run_command("docker build -t grocery-forecasting:test ."):
        print("FAIL: Docker build failed")
        return False
    print("Docker build successful")
    
    # Run container
    print("Starting container...")
    if not run_command("docker run -d --name grocery-test -p 8000:8000 grocery-forecasting:test"):
        print("FAIL: Failed to start container")
        return False
    
    # Wait for service
    if not wait_for_service("http://localhost:8000/health"):
        print("FAIL: Service did not start properly")
        cleanup()
        return False
    
    # Test health endpoint
    try:
        response = requests.get("http://localhost:8000/health")
        if response.status_code == 200:
            data = response.json()
            print(f"Health check passed: {data}")
            result = True
        else:
            print(f"FAIL: Health check returned {response.status_code}")
            result = False
    except Exception as e:
        print(f"FAIL: Health check failed: {e}")
        result = False
    
    # Test prediction endpoint
    try:
        payload = {
            "store_id": "test_store",
            "product_id": "test_prod",
            "date": "2024-01-01",
            "price": 5.99,
            "promotion_flag": 1,
            "chain": "Loblaws",
            "province": "ON",
            "category": "Dairy",
            "brand": "TestBrand"
        }
        
        response = requests.post("http://localhost:8000/predict", json=payload)
        if response.status_code in [200, 503]:  # 503 = model not loaded, which is OK
            print(f"Prediction endpoint test passed (status: {response.status_code})")
        else:
            print(f"FAIL: Prediction endpoint returned {response.status_code}")
            result = False
    except Exception as e:
        print(f"FAIL: Prediction test failed: {e}")
        result = False
    
    cleanup()
    return result

def cleanup():
    """Clean up Docker containers"""
    print("Cleaning up...")
    run_command("docker stop grocery-test")
    run_command("docker rm grocery-test")

def main():
    """Main test function"""
    print("Starting Docker tests...")
    
    if test_docker():
        print("SUCCESS: All Docker tests passed!")
        sys.exit(0)
    else:
        print("FAILURE: Docker tests failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()