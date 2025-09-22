#!/usr/bin/env python3
"""
Quick test script to verify the epoch runner setup works
"""

import os
import sys
import importlib.util

def test_traditional_simulation():
    """Test if traditional simulation can be loaded and run"""
    print("Testing Traditional RLBEEP simulation...")
    
    try:
        # Load simulation module
        current_dir = os.path.dirname(os.path.abspath(__file__))
        simulation_path = os.path.join(current_dir, 'simulation.py')
        
        spec = importlib.util.spec_from_file_location("simulation", simulation_path)
        simulation_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(simulation_module)
        
        print("✓ Simulation module loaded successfully")
        
        # Create simulation instance
        dataset_path = os.path.join(current_dir, "Dataset")
        simulation = simulation_module.RLBEEPSimulation(
            dataset_path=dataset_path,
            num_nodes=10,
            num_clusters=4,
            max_longitude=60.0,
            max_latitude=60.0,
            send_range=10,
            alpha=0.5,
            dfr_min=5.0,
            dfr_max=55.0,
            total_time=100,  # Short test run
            sleep_threshold=10,
            change_threshold=10.0,
            ch_rotation_interval=200
        )
        
        print("✓ Simulation instance created successfully")
        
        # Run short simulation
        results = simulation.run_simulation()
        
        print("✓ Simulation ran successfully")
        print(f"  First death time: {results.get('first_death_time', 'N/A')}")
        print(f"  Final live percentage: {results.get('final_live_percentage', 'N/A')}")
        
        return True
        
    except Exception as e:
        print(f"✗ Traditional simulation test failed: {e}")
        return False

def test_dql_enhanced_simulation():
    """Test if DQL-enhanced simulation can be loaded and run"""
    print("\nTesting DQL-Enhanced RLBEEP simulation...")
    
    try:
        # Load main module
        current_dir = os.path.dirname(os.path.abspath(__file__))
        main_path = os.path.join(current_dir, 'main.py')
        
        spec = importlib.util.spec_from_file_location("main", main_path)
        main_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(main_module)
        
        print("✓ Main module loaded successfully")
        
        # Create simulation instance with reduced time
        dataset_path = os.path.join(current_dir, "Dataset")
        
        # Temporarily modify the total simulation time
        original_time = main_module.TOTAL_SIMULATION_TIME
        main_module.TOTAL_SIMULATION_TIME = 100  # Short test run
        
        simulation = main_module.RLBEEPSimulation(
            num_nodes=10,
            num_clusters=4,
            max_longitude=60.0,
            max_latitude=60.0,
            dataset_path=dataset_path,
            total_time=100
        )
        
        print("✓ Simulation instance created successfully")
        
        # Run short simulation
        results = simulation.run_simulation()
        
        # Restore original time
        main_module.TOTAL_SIMULATION_TIME = original_time
        
        print("✓ Simulation ran successfully")
        print(f"  First death time: {results.get('first_death_time', 'N/A')}")
        print(f"  Final live percentage: {results.get('final_live_percentage', 'N/A')}")
        
        return True
        
    except Exception as e:
        print(f"✗ DQL-enhanced simulation test failed: {e}")
        return False

def main():
    """Run tests for both simulation types"""
    print("="*50)
    print("   EPOCH RUNNER SETUP TEST")
    print("="*50)
    
    # Test traditional simulation
    traditional_ok = test_traditional_simulation()
    
    # Test DQL-enhanced simulation  
    dql_ok = test_dql_enhanced_simulation()
    
    print("\n" + "="*50)
    print("   TEST RESULTS")
    print("="*50)
    print(f"Traditional RLBEEP: {'✓ PASS' if traditional_ok else '✗ FAIL'}")
    print(f"DQL-Enhanced RLBEEP: {'✓ PASS' if dql_ok else '✗ FAIL'}")
    
    if traditional_ok and dql_ok:
        print("\n🎉 All tests passed! Ready to run 300-epoch analysis.")
        print("Run the following scripts:")
        print("  - python3 run_traditional_epochs.py")
        print("  - python3 run_dql_enhanced_epochs.py")
    else:
        print("\n⚠️  Some tests failed. Please check the setup before running full epochs.")
    
    print("="*50)

if __name__ == "__main__":
    main()
