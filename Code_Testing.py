import platform
import ctypes
import tkinter as tk
from AppKit import NSScreen

def get_screen_width_inches():
    # Initialize tkinter to get screen resolution
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    screen_width_pixels = root.winfo_screenwidth()
    screen_height_pixels = root.winfo_screenheight()

    # Get DPI based on the operating system
    system = platform.system()
    if system == "Windows":
        # For Windows, use ctypes to get DPI
        user32 = ctypes.windll.user32
        user32.SetProcessDPIAware()
        screen_dpi = user32.GetDpiForSystem()
    elif system == "Darwin":  # macOS
        screen = NSScreen.mainScreen()
        screen_dpi = screen.backingScaleFactor() * 72  # Default macOS DPI is 72\
        print(screen_dpi)
    else:
        # For other systems, assume a default DPI of 96 (common for Linux)
        screen_dpi = 96

    # Calculate screen width in inches
    screen_width_inches = screen_width_pixels / screen_dpi
    screen_height_inches = screen_height_pixels / screen_dpi
    return screen_width_inches, screen_height_inches, screen_width_pixels, screen_height_pixels

if __name__ == "__main__":
    width_inches, height_inches, width_pixels, height_pixels = get_screen_width_inches()
    print(f"Screen width: {width_inches:.2f} inches, height: {height_inches:.2f} inches")
    print(f"Screen resolution: {width_pixels}x{height_pixels} pixels")


