import pyautogui
print("Move your mouse to the desired position and press Enter...")
input()
x, y = pyautogui.position()
print(f"Mouse position: ({x}, {y})")