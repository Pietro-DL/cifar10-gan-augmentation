import os
from pathlib import Path

def generate_tree(dir_path: Path, prefix: str = "", ignore_hidden: bool = True):
    """
    Recursively prints a tree diagram of the directory structure.
    """
    try:
        # Get items, ignoring hidden files (like .git) AND __pycache__ folders
        items = [
            p for p in dir_path.iterdir() 
            if not (ignore_hidden and (p.name.startswith('.') or p.name == '__pycache__' or p.name.startswith('report') or p.name.startswith('tectonic') or p.name.startswith('cifar-10-batches-py') or p.name.startswith('data')))
        ]
        
        # Sort so directories appear first, then files alphabetically
        items.sort(key=lambda x: (x.is_file(), x.name.lower()))
        
        count = len(items)
        for i, path in enumerate(items):
            is_last = (i == count - 1)
            
            # Determine the correct drawing characters for the tree branch
            connector = "└── " if is_last else "├── "
            
            # Print the current item
            icon = "📁 " if path.is_dir() else "📄 "
            print(f"{prefix}{connector}{icon}{path.name}")
            
            # If it's a directory, dive deeper (recursively)
            if path.is_dir():
                # Extend the prefix line down for nested items
                extension = "    " if is_last else "│   "
                generate_tree(path, prefix=prefix + extension, ignore_hidden=ignore_hidden)
                
    except PermissionError:
        print(f"{prefix}└── 🔒 [Permission Denied]")

if __name__ == "__main__":
    # Get the exact directory where THIS script file is saved
    script_directory = Path(__file__).resolve().parent
    
    print(f"📁 {script_directory.name}/")
    generate_tree(script_directory)