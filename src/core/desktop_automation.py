"""
Desktop Automation System
- Real PC control with pyautogui
- Keyboard and mouse automation
- Application launching and control
- Screen capture and OCR
"""

import os
import time
import subprocess
from typing import Dict, Any, List, Optional
from pathlib import Path
import pyautogui
import cv2
import numpy as np
from PIL import Image
import pytesseract
from loguru import logger

# Configure pyautogui safety
pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.1


class DesktopAutomation:
    """Desktop automation using pyautogui and keyboard"""
    
    def __init__(self):
        self.screen_width, self.screen_height = pyautogui.size()
        self.current_app = None
        self.screenshot_dir = Path("work/screenshots")
        self.screenshot_dir.mkdir(parents=True, exist_ok=True)
    
    def launch_app(self, app_name: str) -> Dict[str, Any]:
        """Launch an application"""
        try:
            # Common app mappings
            app_mappings = {
                "chrome": "chrome.exe",
                "firefox": "firefox.exe",
                "notepad": "notepad.exe",
                "calculator": "calc.exe",
                "explorer": "explorer.exe",
                "cursor": "Cursor.exe",
                "vscode": "code.exe",
                "word": "WINWORD.EXE",
                "excel": "EXCEL.EXE",
                "powerpoint": "POWERPNT.EXE"
            }
            
            app_path = app_mappings.get(app_name.lower(), app_name)
            
            # Launch the application
            subprocess.Popen(app_path, shell=True)
            time.sleep(2)  # Wait for app to load
            
            # Bring to front
            pyautogui.hotkey('alt', 'tab')
            time.sleep(0.5)
            
            self.current_app = app_name
            return {"success": True, "app": app_name, "message": f"Launched {app_name}"}
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def click_element(self, x: int = None, y: int = None, 
                     image_path: str = None, text: str = None) -> Dict[str, Any]:
        """Click on screen element"""
        try:
            if x is not None and y is not None:
                # Direct coordinates
                pyautogui.click(x, y)
                return {"success": True, "method": "coordinates", "x": x, "y": y}
            
            elif image_path and os.path.exists(image_path):
                # Image recognition
                location = pyautogui.locateOnScreen(image_path, confidence=0.8)
                if location:
                    center = pyautogui.center(location)
                    pyautogui.click(center)
                    return {"success": True, "method": "image", "location": location}
                else:
                    return {"success": False, "error": "Image not found on screen"}
            
            elif text:
                # OCR-based text search
                screenshot = pyautogui.screenshot()
                screenshot.save("temp_screenshot.png")
                
                # Use OCR to find text
                img = cv2.imread("temp_screenshot.png")
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                
                # Simple text detection (basic implementation)
                # In production, use more sophisticated OCR
                return {"success": False, "error": "OCR text search not implemented yet"}
            
            else:
                return {"success": False, "error": "No valid click method specified"}
                
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def type_text(self, text: str, delay: float = 0.1) -> Dict[str, Any]:
        """Type text with optional delay"""
        try:
            pyautogui.write(text, interval=delay)
            return {"success": True, "text": text}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def press_key(self, key: str) -> Dict[str, Any]:
        """Press a single key or key combination"""
        try:
            pyautogui.press(key)
            return {"success": True, "key": key}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def hotkey(self, *keys) -> Dict[str, Any]:
        """Press key combination"""
        try:
            pyautogui.hotkey(*keys)
            return {"success": True, "keys": keys}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def take_screenshot(self, region: tuple = None, save_path: str = None) -> Dict[str, Any]:
        """Take screenshot of screen or region"""
        try:
            if save_path is None:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                save_path = str(self.screenshot_dir / f"screenshot_{timestamp}.png")
            
            screenshot = pyautogui.screenshot(region=region)
            screenshot.save(save_path)
            
            return {
                "success": True, 
                "path": save_path,
                "size": screenshot.size,
                "region": region
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def find_image_on_screen(self, image_path: str, confidence: float = 0.8) -> Dict[str, Any]:
        """Find image on screen"""
        try:
            location = pyautogui.locateOnScreen(image_path, confidence=confidence)
            if location:
                center = pyautogui.center(location)
                return {
                    "success": True,
                    "found": True,
                    "location": location,
                    "center": center
                }
            else:
                return {"success": True, "found": False}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def scroll(self, x: int = None, y: int = None, clicks: int = 3) -> Dict[str, Any]:
        """Scroll at position or center"""
        try:
            if x is None or y is None:
                x, y = self.screen_width // 2, self.screen_height // 2
            
            pyautogui.scroll(clicks, x=x, y=y)
            return {"success": True, "x": x, "y": y, "clicks": clicks}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def drag(self, start_x: int, start_y: int, end_x: int, end_y: int, 
             duration: float = 1.0) -> Dict[str, Any]:
        """Drag from start to end position"""
        try:
            pyautogui.drag(end_x - start_x, end_y - start_y, duration=duration, 
                          button='left', x=start_x, y=start_y)
            return {"success": True, "start": (start_x, start_y), "end": (end_x, end_y)}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def get_screen_info(self) -> Dict[str, Any]:
        """Get screen information"""
        return {
            "width": self.screen_width,
            "height": self.screen_height,
            "current_app": self.current_app
        }
    
    def wait_for_image(self, image_path: str, timeout: int = 10, 
                      confidence: float = 0.8) -> Dict[str, Any]:
        """Wait for image to appear on screen"""
        try:
            start_time = time.time()
            while time.time() - start_time < timeout:
                location = pyautogui.locateOnScreen(image_path, confidence=confidence)
                if location:
                    center = pyautogui.center(location)
                    return {
                        "success": True,
                        "found": True,
                        "location": location,
                        "center": center,
                        "wait_time": time.time() - start_time
                    }
                time.sleep(0.5)
            
            return {"success": True, "found": False, "timeout": timeout}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def execute_action_sequence(self, actions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Execute a sequence of actions"""
        results = []
        
        for action in actions:
            action_type = action.get("type")
            
            if action_type == "click":
                result = self.click_element(**action.get("params", {}))
            elif action_type == "type":
                result = self.type_text(**action.get("params", {}))
            elif action_type == "press":
                result = self.press_key(**action.get("params", {}))
            elif action_type == "hotkey":
                result = self.hotkey(**action.get("params", {}))
            elif action_type == "wait":
                time.sleep(action.get("params", {}).get("seconds", 1))
                result = {"success": True, "action": "wait"}
            elif action_type == "screenshot":
                result = self.take_screenshot(**action.get("params", {}))
            else:
                result = {"success": False, "error": f"Unknown action type: {action_type}"}
            
            results.append(result)
            
            # Stop on failure
            if not result.get("success"):
                break
        
        return {
            "success": all(r.get("success") for r in results),
            "results": results
        }