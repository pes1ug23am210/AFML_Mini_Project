# ocr.py
import easyocr
import re
import numpy as np
import cv2
from PIL import Image
from config import class_names

class OCRProcessor:
    def __init__(self):
        print("🚀 Initializing EasyOCR...")
        # Initialize EasyOCR reader (English language)
        self.reader = easyocr.Reader(['en'])
        print("✅ EasyOCR Loaded")
        
        # Create search patterns for each class
        self.patterns = {}
        for class_id, class_name in class_names.items():
            pattern = class_name.lower().replace('-', ' ').replace('_', ' ')
            self.patterns[class_name] = re.compile(r'\b' + re.escape(pattern) + r'\b', re.IGNORECASE)
        
        print(f"🎯 OCR configured for {len(self.patterns)} classes")
    
    def extract_text_from_object(self, object_image):
        """Extract text from isolated object image using EasyOCR.

        Ensures the crop is in the right type/format for EasyOCR and
        handles different orientations by testing multiple rotation
        angles. Returns the text from the angle with the highest
        average confidence.
        """
        # Guard against empty crops
        if object_image is None or getattr(object_image, "size", 0) == 0:
            return "", 0.0

        # Convert PIL -> numpy if needed
        if isinstance(object_image, Image.Image):
            object_image = np.array(object_image)

        # Ensure we have a proper HxWxC array
        if not isinstance(object_image, np.ndarray) or object_image.ndim not in (2, 3):
            return "", 0.0

        # If grayscale (HxW), stack to 3 channels
        if object_image.ndim == 2:
            object_image = np.stack([object_image] * 3, axis=-1)

        # Ensure uint8 range for EasyOCR
        if object_image.dtype != np.uint8:
            object_image = np.clip(object_image, 0, 255).astype(np.uint8)

        # EasyOCR expects RGB; many OpenCV images are BGR
        # Our pipeline already uses RGB crops, but we normalize defensively.
        try:
            base_image = cv2.cvtColor(object_image, cv2.COLOR_BGR2RGB)
        except Exception:
            # If conversion fails (already RGB-like), continue with original
            base_image = object_image

        # Very tiny crops are almost impossible for OCR; skip them
        h, w = base_image.shape[:2]
        if h < 20 or w < 20:
            return "", 0.0

        # Basic enhancement: upscale small crops and boost contrast
        scale_factor = 1.0
        if max(h, w) < 128:
            scale_factor = 2.0
        elif max(h, w) < 256:
            scale_factor = 1.5

        if scale_factor != 1.0:
            new_size = (int(w * scale_factor), int(h * scale_factor))
            base_image = cv2.resize(base_image, new_size, interpolation=cv2.INTER_CUBIC)
            h, w = base_image.shape[:2]

        # Convert to grayscale and apply CLAHE to improve text contrast,
        # but keep a 3-channel image for EasyOCR
        gray = cv2.cvtColor(base_image, cv2.COLOR_RGB2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        gray_eq = clahe.apply(gray)
        base_image = cv2.cvtColor(gray_eq, cv2.COLOR_GRAY2RGB)

        # Helper: rotate image by arbitrary angle around center
        def rotate_image(img, angle):
            (hh, ww) = img.shape[:2]
            center = (ww // 2, hh // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(
                img,
                M,
                (ww, hh),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(255, 255, 255),  # white background
            )
            return rotated

        # Angles to try (in degrees). Includes upright, upside-down,
        # and several tilted orientations.
        angles = [0, 90, 180, 270, -45, 45, -30, 30, -60, 60]

        best_text = ""
        best_conf = 0.0

        try:
            for angle in angles:
                if angle in (0, 90, 180, 270):
                    # Use fast 90° rotations when possible
                    k = (angle // 90) % 4
                    img_rot = np.rot90(base_image, k=k) if angle != 0 else base_image
                else:
                    img_rot = rotate_image(base_image, angle)

                rh, rw = img_rot.shape[:2]
                if rh < 20 or rw < 20:
                    continue

                results = self.reader.readtext(img_rot, detail=1)

                if not results:
                    continue

                texts = [res[1] for res in results if len(res) >= 3]
                confs = [float(res[2]) for res in results if len(res) >= 3]

                if not texts or not confs:
                    continue

                all_text = " ".join(texts).strip()
                avg_conf = float(np.mean(confs))

                if avg_conf > best_conf:
                    best_conf = avg_conf
                    best_text = all_text

            if not best_text:
                return "", 0.0

            return best_text, best_conf

        except Exception as e:
            print(f"OCR Error: {e}")
            return "", 0.0
    
    def match_text_to_class(self, text, min_confidence: float = 0.6):
        """Match extracted text to class names.

        Very conservative: only accepts matches when the brand or full
        class name is clearly present in the OCR text. This avoids
        confusing unrelated brands (e.g. 'himalaya' vs 'sprite').
        Returns (best_class_name, confidence) or (None, 0.0).
        """
        if not text or len(text.strip()) < 2:
            return None, 0.0

        text_lower = text.lower()
        best_class = None
        best_confidence = 0.0

        text_tokens = text_lower.split()

        for class_name, pattern in self.patterns.items():
            confidence = 0.0
            class_name_clean = class_name.lower().replace('-', ' ').replace('_', ' ')
            class_tokens = class_name_clean.split()
            brand_token = class_tokens[0] if class_tokens else ""

            # 1) Exact pattern match (very strong)
            if pattern.search(text_lower):
                confidence = 0.99

            # 2) Full class name as substring
            elif class_name_clean and class_name_clean in text_lower:
                confidence = 0.95

            # 3) Brand token exact match in text tokens
            elif brand_token and brand_token in text_tokens:
                confidence = 0.9

            # 4) Other tokens from class name present (e.g. 'face', 'wash')
            else:
                common_tokens = set(class_tokens).intersection(text_tokens)
                if common_tokens:
                    ratio = len(common_tokens) / max(len(class_tokens), 1)
                    # Keep contribution modest so we don't flip brands
                    confidence = max(confidence, 0.5 + 0.2 * ratio)

            if confidence > best_confidence:
                best_confidence = confidence
                best_class = class_name

        # Enforce a minimum confidence threshold to avoid random matches
        if best_class is None or best_confidence < min_confidence:
            return None, 0.0

        return best_class, best_confidence