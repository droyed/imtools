# demos/__main__.py
import sys
import os
import runpy
import glob

def run_demos():
    # Find all demo scripts in the current directory
    current_dir = os.path.dirname(__file__)
    demo_scripts = glob.glob(os.path.join(current_dir, "demo_*.py"))
    
    # Track results
    passed = []
    failed = []  # List of dicts with name, path, error
    
    print(f"Found {len(demo_scripts)} demos to run...")

    for i, script in enumerate(demo_scripts, 1):
        script_name = os.path.basename(script)
        print(f"\n{'=' * 60}\n>>> 🚀 Running demo {i}/{len(demo_scripts)}: {script_name}...\n{'=' * 60}")
        
        # Execute the script
        try:
            # Add demos directory to path so demo scripts can import helper modules
            sys.path.insert(0, current_dir)
            runpy.run_path(script, run_name="__main__")
            print(f"✅ {script_name} completed successfully")
            passed.append(script_name)
        except Exception as e:
            print(f"❌ Failed to run {script_name}: {e}")
            failed.append({"name": script_name, "path": script, "error": str(e)})
    
    # Print summary
    print(f"\n{'=' * 60}")
    print(f"📊 DEMO SUMMARY")
    print(f"{'=' * 60}")
    print(f"✅ PASSED: {len(passed)}")
    print(f"❌ FAILED: {len(failed)}")
    print(f"📁 TOTAL:  {len(demo_scripts)}")
    if failed:
        print(f"\n❌ FAILED DEMOS:")
        for f in failed:
            print(f"   Demo Name: {f['name']}")
            print(f"   Script:    {f['path']}")
            print(f"   Error:     {f['error']}")
            print()

if __name__ == "__main__":
    run_demos()
