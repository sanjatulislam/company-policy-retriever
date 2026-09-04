from pathlib import Path

def get_data_dir_path(data_dir="data"):
    dir_path = ""
    curr_dir = Path(__file__).resolve()

    for parent in curr_dir.parents:
        if (parent / data_dir).exists():
            dir_path = parent / data_dir
            break

    return dir_path