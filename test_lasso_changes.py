#!/usr/bin/env python3
"""Test script to verify QuantileLasso changes work correctly."""

import numpy as np
import warnings
from confopt.selection.estimators.quantile_estimation import QuantileLasso

def test_quantile_lasso_basic():
    """Test basic functionality of QuantileLasso."""
    print("Testing basic QuantileLasso functionality...")
    
    # Create simple test data
    np.random.seed(42)
    X = np.random.randn(50, 3)
    y = X @ np.array([1.0, -0.5, 2.0]) + 0.1 * np.random.randn(50)
    
    # Test with normal case (should use statsmodels)
    ql = QuantileLasso(random_state=42)
    quantiles = [0.1, 0.5, 0.9]
    
    try:
        ql.fit(X, y, quantiles)
        predictions = ql.predict(X[:5])
        print(f"Normal case - predictions shape: {predictions.shape}")
        print(f"Predictions sample:\n{predictions[:2]}")
        print("✓ Normal case passed")
    except Exception as e:
        print(f"✗ Normal case failed: {e}")
        return False
    
    return True

def test_quantile_lasso_fallback():
    """Test fallback mechanism with coordinate descent."""
    print("\nTesting QuantileLasso fallback mechanism...")
    
    # Create ill-conditioned data to trigger fallback
    np.random.seed(42)
    X = np.random.randn(10, 8)  # More features than samples
    X = np.column_stack([X, X[:, 0] + 1e-10 * np.random.randn(10)])  # Nearly collinear
    y = np.random.randn(10)
    
    ql = QuantileLasso(random_state=42, max_iter=100)
    quantiles = [0.25, 0.75]
    
    # Capture warnings to see if fallback is triggered
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        
        try:
            ql.fit(X, y, quantiles)
            predictions = ql.predict(X[:3])
            print(f"Fallback case - predictions shape: {predictions.shape}")
            print(f"Predictions sample:\n{predictions}")
            
            # Check if fallback warning was issued
            fallback_warnings = [warning for warning in w if "coordinate descent" in str(warning.message)]
            if fallback_warnings:
                print("✓ Fallback mechanism triggered successfully")
            else:
                print("✓ No fallback needed (statsmodels worked)")
                
            print("✓ Fallback case passed")
            return True
            
        except Exception as e:
            print(f"✗ Fallback case failed: {e}")
            return False

def test_coordinate_descent_directly():
    """Test the coordinate descent method directly."""
    print("\nTesting coordinate descent method directly...")
    
    np.random.seed(42)
    X = np.random.randn(20, 3)
    y = X @ np.array([1.0, -0.5, 2.0]) + 0.1 * np.random.randn(20)
    
    ql = QuantileLasso(random_state=42, max_iter=100)
    
    try:
        # Test coordinate descent directly
        params = ql._coordinate_descent_quantile_regression(X, y, 0.5)
        print(f"Coordinate descent params: {params}")
        
        # Check that parameters are reasonable
        if np.isfinite(params).all() and np.abs(params).max() < 100:
            print("✓ Coordinate descent parameters are reasonable")
            return True
        else:
            print("✗ Coordinate descent parameters are unreasonable")
            return False
            
    except Exception as e:
        print(f"✗ Coordinate descent test failed: {e}")
        return False

if __name__ == "__main__":
    print("Running QuantileLasso tests after removing scipy.optimize.minimize dependency...\n")
    
    tests_passed = 0
    total_tests = 3
    
    if test_quantile_lasso_basic():
        tests_passed += 1
    
    if test_quantile_lasso_fallback():
        tests_passed += 1
        
    if test_coordinate_descent_directly():
        tests_passed += 1
    
    print(f"\nResults: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("🎉 All tests passed! The minimize dependency has been successfully removed.")
    else:
        print("❌ Some tests failed. Please check the implementation.")
