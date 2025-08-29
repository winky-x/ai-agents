"""
Advanced Vision Understanding System
- Screen analysis and UI element detection
- Visual reasoning and context understanding
- Image processing and OCR
- Visual problem-solving
- Fallback vision capabilities
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pytesseract
from typing import Dict, List, Any, Optional, Tuple
import json
import time
from dataclasses import dataclass
from loguru import logger
from .llm_providers import call_ui_tars_vision, call_openrouter_generic


@dataclass
class UIElement:
    """Detected UI element"""
    type: str  # button, text, input, image, link
    text: str = None
    confidence: float = 0.0
    bbox: Tuple[int, int, int, int] = None  # x, y, width, height
    attributes: Dict[str, Any] = None


@dataclass
class ScreenAnalysis:
    """Complete screen analysis"""
    elements: List[UIElement]
    layout: Dict[str, Any]
    text_content: str
    interactive_elements: List[UIElement]
    visual_context: str
    suggested_actions: List[str]


class VisionUnderstanding:
    """Advanced vision understanding and screen analysis"""
    
    def __init__(self, llm_provider):
        self.llm_provider = llm_provider
        self.element_templates = self._load_element_templates()
        self.ocr_config = '--oem 3 --psm 6'  # Tesseract config
    
    def _load_element_templates(self) -> Dict[str, Any]:
        """Load UI element templates for detection"""
        return {
            "button": {
                "colors": [(0, 120, 215), (0, 102, 184), (0, 84, 153)],  # Blue buttons
                "shapes": ["rectangle", "rounded_rectangle"],
                "min_size": (50, 25)
            },
            "input": {
                "colors": [(255, 255, 255), (240, 240, 240)],  # White/light inputs
                "shapes": ["rectangle"],
                "min_size": (100, 20)
            },
            "link": {
                "colors": [(0, 102, 204), (0, 0, 255)],  # Blue links
                "shapes": ["text"],
                "min_size": (20, 10)
            }
        }
    
    def analyze_screen(self, image_path: str) -> ScreenAnalysis:
        """Complete screen analysis"""
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            # Detect UI elements
            elements = self._detect_ui_elements(image)
            
            # Extract text content
            text_content = self._extract_text(image)
            
            # Analyze layout
            layout = self._analyze_layout(image, elements)
            
            # Identify interactive elements
            interactive_elements = [e for e in elements if e.type in ["button", "input", "link"]]
            
            # Generate visual context
            visual_context = self._generate_visual_context(image, elements, text_content)
            
            # Suggest actions
            suggested_actions = self._suggest_actions(elements, text_content, visual_context)
            
            return ScreenAnalysis(
                elements=elements,
                layout=layout,
                text_content=text_content,
                interactive_elements=interactive_elements,
                visual_context=visual_context,
                suggested_actions=suggested_actions
            )
            
        except Exception as e:
            logger.error(f"Screen analysis failed: {e}")
            return self._create_fallback_analysis(image_path)
    
    def _detect_ui_elements(self, image: np.ndarray) -> List[UIElement]:
        """Detect UI elements in the image"""
        elements = []
        
        # Convert to different color spaces
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Detect buttons
        buttons = self._detect_buttons(image, gray)
        elements.extend(buttons)
        
        # Detect input fields
        inputs = self._detect_input_fields(image, gray)
        elements.extend(inputs)
        
        # Detect text areas
        text_areas = self._detect_text_areas(gray)
        elements.extend(text_areas)
        
        # Detect links
        links = self._detect_links(image, hsv)
        elements.extend(links)
        
        return elements
    
    def _detect_buttons(self, image: np.ndarray, gray: np.ndarray) -> List[UIElement]:
        """Detect button elements"""
        buttons = []
        
        # Edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter by size
            if w < 50 or h < 25:
                continue
            
            # Check aspect ratio (buttons are usually wider than tall)
            aspect_ratio = w / h
            if aspect_ratio < 0.5 or aspect_ratio > 5:
                continue
            
            # Extract region and analyze
            region = image[y:y+h, x:x+w]
            
            # Check if it looks like a button
            if self._is_button_like(region):
                # Extract text from button
                text = self._extract_text_from_region(region)
                
                buttons.append(UIElement(
                    type="button",
                    text=text,
                    confidence=0.8,
                    bbox=(x, y, w, h),
                    attributes={"aspect_ratio": aspect_ratio}
                ))
        
        return buttons
    
    def _detect_input_fields(self, image: np.ndarray, gray: np.ndarray) -> List[UIElement]:
        """Detect input field elements"""
        inputs = []
        
        # Look for rectangular regions with light backgrounds
        # and potential placeholder text
        
        # Threshold to find light regions
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter by size
            if w < 100 or h < 20:
                continue
            
            # Check aspect ratio (inputs are usually wider than tall)
            aspect_ratio = w / h
            if aspect_ratio < 2 or aspect_ratio > 20:
                continue
            
            # Extract region
            region = image[y:y+h, x:x+w]
            
            # Check if it looks like an input field
            if self._is_input_like(region):
                # Extract placeholder text
                text = self._extract_text_from_region(region)
                
                inputs.append(UIElement(
                    type="input",
                    text=text,
                    confidence=0.7,
                    bbox=(x, y, w, h),
                    attributes={"aspect_ratio": aspect_ratio}
                ))
        
        return inputs
    
    def _detect_text_areas(self, gray: np.ndarray) -> List[UIElement]:
        """Detect text areas"""
        text_areas = []
        
        # Use OCR to find text regions
        try:
            # Get OCR data
            ocr_data = pytesseract.image_to_data(gray, config=self.ocr_config, output_type=pytesseract.Output.DICT)
            
            for i in range(len(ocr_data['text'])):
                text = ocr_data['text'][i].strip()
                if text and len(text) > 2:  # Filter out short text
                    x = ocr_data['left'][i]
                    y = ocr_data['top'][i]
                    w = ocr_data['width'][i]
                    h = ocr_data['height'][i]
                    conf = ocr_data['conf'][i]
                    
                    if conf > 30:  # Confidence threshold
                        text_areas.append(UIElement(
                            type="text",
                            text=text,
                            confidence=conf / 100.0,
                            bbox=(x, y, w, h),
                            attributes={"font_size": h}
                        ))
        except Exception as e:
            logger.error(f"OCR failed: {e}")
        
        return text_areas
    
    def _detect_links(self, image: np.ndarray, hsv: np.ndarray) -> List[UIElement]:
        """Detect link elements"""
        links = []
        
        # Look for blue text (common for links)
        lower_blue = np.array([100, 50, 50])
        upper_blue = np.array([130, 255, 255])
        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
        
        # Find contours in blue regions
        contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            if w < 20 or h < 10:
                continue
            
            # Extract region and check for text
            region = image[y:y+h, x:x+w]
            text = self._extract_text_from_region(region)
            
            if text and len(text) > 3:
                links.append(UIElement(
                    type="link",
                    text=text,
                    confidence=0.6,
                    bbox=(x, y, w, h),
                    attributes={"color": "blue"}
                ))
        
        return links
    
    def _is_button_like(self, region: np.ndarray) -> bool:
        """Check if a region looks like a button"""
        # Convert to grayscale
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        
        # Check for consistent color (button background)
        std_dev = np.std(gray)
        if std_dev < 30:  # Low variance indicates solid color
            return True
        
        # Check for rounded corners or borders
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (region.shape[0] * region.shape[1])
        
        if edge_density > 0.1:  # Has some edges (borders)
            return True
        
        return False
    
    def _is_input_like(self, region: np.ndarray) -> bool:
        """Check if a region looks like an input field"""
        # Convert to grayscale
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        
        # Check for light background
        mean_brightness = np.mean(gray)
        if mean_brightness > 200:  # Light background
            return True
        
        # Check for borders
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (region.shape[0] * region.shape[1])
        
        if edge_density > 0.05:  # Has some edges (borders)
            return True
        
        return False
    
    def _extract_text_from_region(self, region: np.ndarray) -> str:
        """Extract text from a specific region"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
            
            # Apply preprocessing
            # Resize if too small
            if gray.shape[0] < 20 or gray.shape[1] < 20:
                gray = cv2.resize(gray, (gray.shape[1] * 2, gray.shape[0] * 2))
            
            # Apply threshold
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Extract text
            text = pytesseract.image_to_string(thresh, config=self.ocr_config)
            return text.strip()
            
        except Exception as e:
            logger.error(f"Text extraction failed: {e}")
            return ""
    
    def _extract_text(self, image: np.ndarray) -> str:
        """Extract all text from the image"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply preprocessing
            # Denoise
            denoised = cv2.fastNlMeansDenoising(gray)
            
            # Apply adaptive threshold
            thresh = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            
            # Extract text
            text = pytesseract.image_to_string(thresh, config=self.ocr_config)
            return text.strip()
            
        except Exception as e:
            logger.error(f"Text extraction failed: {e}")
            return ""
    
    def _analyze_layout(self, image: np.ndarray, elements: List[UIElement]) -> Dict[str, Any]:
        """Analyze the layout of the screen"""
        height, width = image.shape[:2]
        
        # Group elements by regions
        top_elements = [e for e in elements if e.bbox and e.bbox[1] < height // 3]
        middle_elements = [e for e in elements if e.bbox and height // 3 <= e.bbox[1] < 2 * height // 3]
        bottom_elements = [e for e in elements if e.bbox and e.bbox[1] >= 2 * height // 3]
        
        # Analyze element distribution
        button_count = len([e for e in elements if e.type == "button"])
        input_count = len([e for e in elements if e.type == "input"])
        text_count = len([e for e in elements if e.type == "text"])
        
        return {
            "dimensions": {"width": width, "height": height},
            "regions": {
                "top": len(top_elements),
                "middle": len(middle_elements),
                "bottom": len(bottom_elements)
            },
            "element_counts": {
                "buttons": button_count,
                "inputs": input_count,
                "text": text_count
            },
            "layout_type": self._determine_layout_type(elements, width, height)
        }
    
    def _determine_layout_type(self, elements: List[UIElement], width: int, height: int) -> str:
        """Determine the type of layout"""
        button_count = len([e for e in elements if e.type == "button"])
        input_count = len([e for e in elements if e.type == "input"])
        
        if input_count > 2:
            return "form"
        elif button_count > 3:
            return "navigation"
        elif input_count > 0 and button_count > 0:
            return "interactive"
        else:
            return "content"
    
    def _generate_visual_context(self, image: np.ndarray, elements: List[UIElement], text_content: str) -> str:
        """Generate visual context description"""
        context_parts = []
        
        # Analyze overall layout
        height, width = image.shape[:2]
        context_parts.append(f"Screen size: {width}x{height} pixels")
        
        # Count elements
        button_count = len([e for e in elements if e.type == "button"])
        input_count = len([e for e in elements if e.type == "input"])
        text_count = len([e for e in elements if e.type == "text"])
        
        context_parts.append(f"Found {button_count} buttons, {input_count} input fields, {text_count} text elements")
        
        # Describe prominent elements
        if elements:
            prominent_elements = sorted(elements, key=lambda e: e.confidence, reverse=True)[:3]
            for elem in prominent_elements:
                if elem.text:
                    context_parts.append(f"Prominent {elem.type}: '{elem.text}'")
        
        # Add text content summary
        if text_content:
            lines = text_content.split('\n')[:5]  # First 5 lines
            context_parts.append(f"Text content: {'; '.join(lines)}")
        
        return ". ".join(context_parts)
    
    def _suggest_actions(self, elements: List[UIElement], text_content: str, visual_context: str) -> List[str]:
        """Suggest possible actions based on screen analysis"""
        suggestions = []
        
        # Check for buttons
        buttons = [e for e in elements if e.type == "button"]
        for button in buttons:
            if button.text:
                suggestions.append(f"Click '{button.text}' button")
        
        # Check for input fields
        inputs = [e for e in elements if e.type == "input"]
        for input_field in inputs:
            if input_field.text:
                suggestions.append(f"Fill '{input_field.text}' input field")
            else:
                suggestions.append("Fill input field")
        
        # Check for links
        links = [e for e in elements if e.type == "link"]
        for link in links:
            if link.text:
                suggestions.append(f"Click '{link.text}' link")
        
        # Analyze text content for actions
        if "login" in text_content.lower() or "sign in" in text_content.lower():
            suggestions.append("Enter login credentials")
        
        if "search" in text_content.lower():
            suggestions.append("Enter search query")
        
        if "submit" in text_content.lower() or "send" in text_content.lower():
            suggestions.append("Submit form")
        
        return suggestions[:5]  # Limit to 5 suggestions
    
    def _create_fallback_analysis(self, image_path: str) -> ScreenAnalysis:
        """Create a fallback analysis when main analysis fails"""
        return ScreenAnalysis(
            elements=[],
            layout={"dimensions": {"width": 0, "height": 0}, "layout_type": "unknown"},
            text_content="",
            interactive_elements=[],
            visual_context="Screen analysis failed",
            suggested_actions=["Take screenshot", "Try different approach"]
        )
    
    def get_visual_guidance(self, task_description: str, image_path: str) -> Dict[str, Any]:
        """Get visual guidance for a specific task with LLM-first (UI-TARS) strategy and fallbacks."""
        try:
            # 1) Try UI-TARS (agentic GUI expert)
            prompt = (
                f"You are a multimodal GUI assistant for desktops/browsers. Task: {task_description}. "
                "Given the screenshot, provide concise, step-by-step actions (1-5), mention key UI elements, "
                "explain likely errors if visible, and suggest the next best click or input."
            )
            ui_tars = call_ui_tars_vision(prompt, [image_path])
            if ui_tars.get("success") and ui_tars.get("text"):
                return {"success": True, "guidance": ui_tars.get("text", ""), "model": "bytedance/ui-tars-1.5-7b"}

            # 2) Fallback to Gemini 2.0 flash exp via OpenRouter
            gem_fallback = call_openrouter_generic(prompt, model="google/gemini-2.0-flash-exp:free", image_paths=[image_path])
            if gem_fallback.get("success") and gem_fallback.get("text"):
                return {"success": True, "guidance": gem_fallback.get("text", ""), "model": "google/gemini-2.0-flash-exp:free"}

            # 3) Final fallback: run local CV analysis and produce heuristic guidance
            analysis = self.analyze_screen(image_path)
            steps = [
                f"Task: {task_description}",
                f"Detected layout: {analysis.layout.get('layout_type')}",
                "Suggested next actions:",
            ] + [f"- {s}" for s in analysis.suggested_actions]
            return {"success": True, "guidance": "\n".join(steps), "model": "heuristic"}

        except Exception as e:
            logger.error(f"Visual guidance failed: {e}")
            return {"success": False, "error": str(e), "guidance": "Unable to provide visual guidance"}
    
    def _format_elements_for_guidance(self, elements: List[UIElement]) -> str:
        """Format elements for guidance prompt"""
        formatted = []
        for elem in elements[:10]:  # Limit to 10 elements
            if elem.text:
                formatted.append(f"- {elem.type}: '{elem.text}' (confidence: {elem.confidence:.2f})")
            else:
                formatted.append(f"- {elem.type} (confidence: {elem.confidence:.2f})")
        
        return "\n".join(formatted)
    
    def find_element_by_text(self, image_path: str, target_text: str) -> Optional[UIElement]:
        """Find a specific element by text content"""
        try:
            analysis = self.analyze_screen(image_path)
            
            # Search for exact match
            for element in analysis.elements:
                if element.text and target_text.lower() in element.text.lower():
                    return element
            
            # Search for partial match
            for element in analysis.elements:
                if element.text:
                    words = target_text.lower().split()
                    element_words = element.text.lower().split()
                    if any(word in element_words for word in words):
                        return element
            
            return None
            
        except Exception as e:
            logger.error(f"Element search failed: {e}")
            return None
    
    def get_screen_summary(self, image_path: str) -> str:
        """Get a human-readable summary of the screen"""
        try:
            analysis = self.analyze_screen(image_path)
            
            summary_parts = []
            summary_parts.append(f"Screen Analysis Summary:")
            summary_parts.append(f"- Layout: {analysis.layout['layout_type']}")
            summary_parts.append(f"- Interactive elements: {len(analysis.interactive_elements)}")
            
            if analysis.text_content:
                summary_parts.append(f"- Main text: {analysis.text_content[:100]}...")
            
            if analysis.suggested_actions:
                summary_parts.append(f"- Suggested actions: {', '.join(analysis.suggested_actions[:3])}")
            
            return "\n".join(summary_parts)
            
        except Exception as e:
            logger.error(f"Screen summary failed: {e}")
            return "Unable to analyze screen"