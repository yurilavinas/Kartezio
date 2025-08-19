#!/usr/bin/env python3
"""
Test validation script for Kartezio PyPI preparation.
This script validates that our new comprehensive test suite is working properly.
"""

import unittest
import sys
import time

def run_security_tests():
    """Run security-focused tests."""
    print("🔒 Running Security Tests...")
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromName('tests.test_security')
    runner = unittest.TextTestRunner(verbosity=0)
    result = runner.run(suite)
    return result.wasSuccessful(), result.testsRun

def run_core_component_tests():
    """Run core component tests."""
    print("⚙️  Running Core Component Tests...")
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromName('tests.test_core_components')
    runner = unittest.TextTestRunner(verbosity=0)
    result = runner.run(suite)
    return result.wasSuccessful(), result.testsRun

def run_integration_tests():
    """Run integration tests (those that don't depend on external data)."""
    print("🔗 Running Basic Integration Tests...")
    try:
        loader = unittest.TestLoader()
        suite = loader.loadTestsFromName('tests.test_integration')
        runner = unittest.TextTestRunner(verbosity=0)
        result = runner.run(suite)
        return result.wasSuccessful(), result.testsRun
    except ImportError:
        print("   ⚠️  Integration tests skipped (dependencies unavailable)")
        return True, 0

def validate_package_integrity():
    """Quick package integrity check."""
    print("📦 Validating Package Integrity...")
    try:
        import kartezio
        from kartezio.core.components import Components
        from kartezio.primitives.matrix import default_matrix_lib
        lib = default_matrix_lib()
        print(f"   ✅ Package loaded successfully ({lib.size} primitives available)")
        return True
    except Exception as e:
        print(f"   ❌ Package integrity check failed: {e}")
        return False

def main():
    """Main validation function."""
    print("Kartezio Test Validation")
    print("=" * 50)
    
    start_time = time.time()
    all_passed = True
    total_tests = 0
    
    # Package integrity first
    if not validate_package_integrity():
        print("\n❌ Package integrity failed. Aborting.")
        sys.exit(1)
    
    # Run test suites
    test_suites = [
        ("Security", run_security_tests),
        ("Core Components", run_core_component_tests),
        ("Integration", run_integration_tests),
    ]
    
    results = []
    
    for suite_name, test_func in test_suites:
        try:
            success, test_count = test_func()
            results.append((suite_name, success, test_count))
            total_tests += test_count
            
            if success:
                print(f"   ✅ {suite_name}: {test_count} tests passed")
            else:
                print(f"   ❌ {suite_name}: Failed")
                all_passed = False
                
        except Exception as e:
            print(f"   ❌ {suite_name}: Error - {e}")
            results.append((suite_name, False, 0))
            all_passed = False
    
    elapsed = time.time() - start_time
    
    # Final summary
    print("\n" + "=" * 50)
    print("VALIDATION SUMMARY")
    print("=" * 50)
    
    for suite_name, success, test_count in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {suite_name}: {test_count} tests")
    
    print(f"\nTotal tests run: {total_tests}")
    print(f"Execution time: {elapsed:.2f}s")
    
    if all_passed and total_tests > 0:
        print("\n🎉 ALL VALIDATION TESTS PASSED! 🎉")
        print("\nThe package is ready for safe refactoring!")
        print("✅ Security vulnerabilities are properly tested")
        print("✅ Core components are thoroughly validated")
        print("✅ Package integrity is confirmed")
        return 0
    else:
        print("\n❌ VALIDATION FAILED")
        if total_tests == 0:
            print("No tests were run successfully")
        else:
            print("Some test suites failed")
        return 1

if __name__ == '__main__':
    sys.exit(main())