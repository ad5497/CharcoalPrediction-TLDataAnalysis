import shutil

def progress_bar(progress, total, bar_length="terminal_width"):
    """Print a progress bar as a script runs for my sanity"""
    if bar_length == "terminal_width":
        bar_length = shutil.get_terminal_size().columns - 7
    percent = int((progress / total) * 100)
    bar = "#" * int((progress / total) * bar_length)
    space = " " * (bar_length - len(bar))
    print(f"\r[{bar}{space}] {percent}%", end="", flush=True)
    if progress == total:
        print()

def main():
    import time
    total = 100
    for i in range(total):
        progress_bar(i + 1, total)
        time.sleep(0.1)

if __name__ == "__main__":
    main()