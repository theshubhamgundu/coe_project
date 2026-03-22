#!/usr/bin/env python
"""
Quick Start Script for Grievance Classification Project
Runs the complete ML pipeline in one command
"""

import os
import sys
import subprocess

def run_command(description, command):
    """Run a command and report results"""
    print(f"\n{'='*70}")
    print(f"📋 {description}")
    print(f"{'='*70}")
    
    try:
        result = subprocess.run(command, shell=True, capture_output=False)
        if result.returncode == 0:
            print(f"✅ {description} - COMPLETE")
            return True
        else:
            print(f"❌ {description} - FAILED (Exit code: {result.returncode})")
            return False
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False

def main():
    """Run complete pipeline"""
    print("\n" + "="*70)
    print("🎯 PUBLIC GRIEVANCE CLASSIFICATION - QUICK START")
    print("="*70)
    
    steps = [
        ("Generating Sample Data", "python scripts/generate_sample_data.py"),
        ("Preprocessing Data", "python scripts/data_preprocessing.py"),
        ("Training Models", "python scripts/model_training.py"),
        ("Evaluating & Visualizing", "python scripts/evaluation.py"),
    ]
    
    success_count = 0
    failed_steps = []
    
    for description, command in steps:
        if run_command(description, command):
            success_count += 1
        else:
            failed_steps.append(description)
    
    # Final summary
    print("\n" + "="*70)
    print("📊 PIPELINE EXECUTION SUMMARY")
    print("="*70)
    print(f"✅ Completed: {success_count}/{len(steps)}")
    
    if failed_steps:
        print(f"❌ Failed Steps:")
        for step in failed_steps:
            print(f"   - {step}")
    else:
        print("\n🎉 ALL STEPS COMPLETED SUCCESSFULLY!")
        print("\n📁 Output Files Generated:")
        print("   ✓ data/grievances_dataset.csv - Raw data")
        print("   ✓ data/grievances_processed.csv - Processed features")
        print("   ✓ models/*.pkl - Trained models")
        print("   ✓ visualizations/*.html - Interactive dashboards")
        print("   ✓ visualizations/*.png - Static charts")
        
        print("\n🚀 Next Steps:")
        print("   1. Review visualizations in visualizations/")
        print("   2. Analyze model performance in models/")
        print("   3. Run notebooks/EDA.ipynb for interactive exploration")
        print("   4. Deploy models for real-world predictions")
    
    print("\n" + "="*70 + "\n")

if __name__ == "__main__":
    main()
