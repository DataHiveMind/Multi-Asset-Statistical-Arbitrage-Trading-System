#!/usr/bin/env python3
"""
C++ Test Runner

Simple script to build and run C++ Order Book tests from the main project directory.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def run_command(cmd, cwd=None, shell=False):
    """Run a command and return success status."""
    print(f"Running: {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    try:
        result = subprocess.run(cmd, cwd=cwd, shell=shell, check=True, 
                              capture_output=False, text=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error: Command failed with exit code {e.returncode}")
        return False
    except FileNotFoundError:
        print(f"Error: Command not found: {cmd[0] if isinstance(cmd, list) else cmd}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Build and run C++ Order Book tests')
    parser.add_argument('--build-type', choices=['Debug', 'Release'], default='Debug',
                       help='CMake build type')
    parser.add_argument('--clean', action='store_true',
                       help='Clean build directory before building')
    parser.add_argument('--cmake-only', action='store_true',
                       help='Only run CMake configuration, do not build')
    parser.add_argument('--test-only', action='store_true',
                       help='Only run tests, do not build')
    parser.add_argument('--coverage', action='store_true',
                       help='Enable code coverage reporting')
    parser.add_argument('--filter', type=str,
                       help='Google Test filter pattern')
    parser.add_argument('--xml-output', action='store_true',
                       help='Generate XML test output')
    
    args = parser.parse_args()
    
    # Determine project root and test directory
    script_dir = Path(__file__).parent
    test_dir = script_dir / 'src' / 'tests' / 'cpp_tests'
    build_dir = test_dir / 'build'
    
    if not test_dir.exists():
        print(f"Error: Test directory not found: {test_dir}")
        return 1
    
    print(f"C++ Order Book Test Runner")
    print(f"Project root: {script_dir}")
    print(f"Test directory: {test_dir}")
    print(f"Build type: {args.build_type}")
    print()
    
    # Clean build directory if requested
    if args.clean and build_dir.exists():
        print("Cleaning build directory...")
        import shutil
        shutil.rmtree(build_dir)
    
    # Create build directory
    build_dir.mkdir(exist_ok=True)
    
    if not args.test_only:
        # Configure with CMake
        cmake_args = [
            'cmake', '..',
            f'-DCMAKE_BUILD_TYPE={args.build_type}'
        ]
        
        if args.coverage:
            cmake_args.append('-DENABLE_COVERAGE=ON')
        
        print("Configuring with CMake...")
        if not run_command(cmake_args, cwd=build_dir):
            return 1
        
        if args.cmake_only:
            print("CMake configuration completed.")
            return 0
        
        # Build the project
        print("\nBuilding project...")
        build_args = ['cmake', '--build', '.', '--config', args.build_type]
        if not run_command(build_args, cwd=build_dir):
            return 1
    
    # Run tests
    print("\nRunning tests...")
    
    # Find test executable
    test_exe = None
    possible_locations = [
        build_dir / 'OrderBookTests',
        build_dir / 'OrderBookTests.exe', 
        build_dir / 'Debug' / 'OrderBookTests.exe',
        build_dir / 'Release' / 'OrderBookTests.exe'
    ]
    
    for location in possible_locations:
        if location.exists():
            test_exe = location
            break
    
    if not test_exe:
        print("Error: Test executable not found. Build may have failed.")
        return 1
    
    # Prepare test command
    test_args = [str(test_exe)]
    
    if args.filter:
        test_args.extend(['--gtest_filter', args.filter])
    
    if args.xml_output:
        xml_file = build_dir / 'test_results.xml'
        test_args.extend(['--gtest_output', f'xml:{xml_file}'])
    
    # Run the tests
    success = run_command(test_args, cwd=build_dir)
    
    if args.xml_output:
        xml_file = build_dir / 'test_results.xml'
        if xml_file.exists():
            print(f"\nTest results saved to: {xml_file}")
    
    # Generate coverage report if enabled
    if args.coverage and success:
        print("\nGenerating coverage report...")
        coverage_commands = [
            ['lcov', '--directory', '.', '--capture', '--output-file', 'coverage.info'],
            ['lcov', '--remove', 'coverage.info', '/usr/*', '--output-file', 'coverage.info'],
            ['genhtml', '-o', 'coverage_report', 'coverage.info']
        ]
        
        for cmd in coverage_commands:
            if not run_command(cmd, cwd=build_dir):
                print("Warning: Coverage report generation failed")
                break
        else:
            coverage_report = build_dir / 'coverage_report' / 'index.html'
            if coverage_report.exists():
                print(f"Coverage report generated: {coverage_report}")
    
    return 0 if success else 1

if __name__ == '__main__':
    sys.exit(main())
