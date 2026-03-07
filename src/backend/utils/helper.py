from __future__ import annotations

import math
import os
import sys
import tempfile
from typing import Optional, Tuple

from dotenv import load_dotenv
from groq import Groq
from google.cloud import vision
from PIL import Image                   # used for image validation / conversion

from backend.logger import get_logger
from backend.exceptions import Agent_Exception

load_dotenv()
logger = get_logger(__name__)

class MediaProcessor:
    
    def __init__(self):
        self.vision_client: Optional[vision.ImageAnnotatorClient] = None
        self.groq_client:   Optional[Groq]                        = None
        self._initialize_clients()

    def _initialize_clients(self) -> None:
        try:
            import os
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r"J:\MathTutor\secrets\mathtutorocr-7144a11193df.json"
            self.vision_client = vision.ImageAnnotatorClient()
            logger.info("[MediaProcessor] Google Cloud Vision OCR initialised")
        except Exception as exc:
            logger.error(f"[MediaProcessor] Google Vision init failed: {exc}")

        try:
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                raise ValueError("GROQ_API_KEY not set")
            self.groq_client = Groq(api_key=api_key)
            logger.info("[MediaProcessor] Groq Whisper client initialised")
        except Exception as exc:
            logger.error(f"[MediaProcessor] Groq Whisper init failed: {exc}")

    def process_image(self, image_input) -> Tuple[str, float]:
        
        if not self.vision_client:
            raise RuntimeError(
                "Google Vision client not initialised. "
                "Ensure GOOGLE_APPLICATION_CREDENTIALS is set."
            )

        try:
            # Normalise to bytes
            if isinstance(image_input, (str, os.PathLike)):
                with open(image_input, "rb") as fh:
                    image_bytes = fh.read()
            else:
                image_bytes = bytes(image_input)

            vision_image = vision.Image(content=image_bytes)
            response     = self.vision_client.text_detection(image=vision_image)

            if response.error.message:
                raise RuntimeError(f"Vision API error: {response.error.message}")

            annotations = response.text_annotations
            if not annotations:
                logger.warning("[OCR] No text detected in image")
                return "", 0.0

            raw_text   = annotations[0].description          # full concatenated text
            confidence = self._estimate_vision_confidence(response)

            cleaned = self.clean_extracted_text(raw_text)
            logger.info(f"[OCR] {len(cleaned)} chars | conf={confidence:.2f}")
            return cleaned, confidence

        except Agent_Exception:
            raise
        except Exception as exc:
            logger.error(f"[OCR] Error: {exc}")
            raise Agent_Exception(exc, sys)

    def process_audio(self, audio_input) -> Tuple[str, float]:
        
        if not self.groq_client:
            raise RuntimeError(
                "Groq client not initialised. Ensure GROQ_API_KEY is set."
            )

        tmp_path: Optional[str] = None
        try:
            # Normalise to a file path (Groq SDK needs an open file handle)
            if isinstance(audio_input, (str, os.PathLike)):
                audio_path = str(audio_input)
                owns_tmp   = False
            else:
                tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                tmp.write(bytes(audio_input))
                tmp.close()
                audio_path = tmp.name
                owns_tmp   = True
                tmp_path   = audio_path

            with open(audio_path, "rb") as audio_fh:
                transcription = self.groq_client.audio.transcriptions.create(
                    file=audio_fh,
                    model="whisper-large-v3",
                )

            transcript = transcription.text.strip()
            confidence = self._estimate_transcription_confidence(transcript)
            logger.info(f"[ASR] {len(transcript)} chars | conf={confidence:.2f}")
            return transcript, confidence

        except Agent_Exception:
            raise
        except Exception as exc:
            logger.error(f"[ASR] Error: {exc}")
            raise Agent_Exception(exc, sys)
        finally:
            if tmp_path:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

    def _estimate_vision_confidence(self, response) -> float:
        """Derive a confidence score from Vision API block confidences."""
        try:
            blocks = [
                block
                for page in response.full_text_annotation.pages
                for block in page.blocks
            ]
            if not blocks:
                return 0.7
            confidences = [b.confidence for b in blocks if b.confidence > 0]
            return round(sum(confidences) / len(confidences), 3) if confidences else 0.7
        except Exception:
            return 0.7

    def _estimate_transcription_confidence(self, transcript: str) -> float:
        """Heuristic confidence for Whisper (no per-word probs in Groq API)."""
        if not transcript:
            return 0.0
        score      = 0.9
        word_count = len(transcript.split())
        if word_count < 3:
            score -= 0.3
        elif word_count < 6:
            score -= 0.1
        for marker in ["?", "(unclear)", "(inaudible)", "[noise]", "[BLANK_AUDIO]"]:
            if marker.lower() in transcript.lower():
                score -= 0.1
        return round(max(0.0, min(1.0, score)), 3)

    def clean_extracted_text(self, text: str) -> str:
        """Remove common OCR noise and normalise whitespace."""
        if not text:
            return ""
        cleaned = text.strip()
        # Strip common OCR artifacts
        for artifact in ["|", "—", "•", "■", "□", "\x0c"]:
            cleaned = cleaned.replace(artifact, " ")
        # Collapse whitespace
        return " ".join(cleaned.split())

try:
    import numpy as _np
    _NUMPY_GLOBALS: dict = {
        "np":     _np,
        # convenience aliases the LLM commonly writes bare (no np. prefix)
        "array":  _np.array,
        "cross":  _np.cross,
        "dot":    _np.dot,
        "norm":   _np.linalg.norm,
        "linalg": _np.linalg,
    }
except ImportError:
    _NUMPY_GLOBALS = {}

_SAFE_GLOBALS: dict = {
    "__builtins__": {},
    # built-in math operations
    "abs": abs, "round": round, "min": min, "max": max,
    "sum": sum, "pow": pow, "divmod": divmod,
    # math module — arithmetic
    "sqrt":    math.sqrt,    "log":     math.log,     "log2":  math.log2,
    "log10":   math.log10,   "exp":     math.exp,
    # trig
    "sin":     math.sin,     "cos":     math.cos,     "tan":   math.tan,
    "asin":    math.asin,    "acos":    math.acos,    "atan":  math.atan,
    "atan2":   math.atan2,   "degrees": math.degrees, "radians": math.radians,
    "sinh":    math.sinh,    "cosh":    math.cosh,    "tanh":  math.tanh,
    # rounding / number theory
    "floor":     math.floor,   "ceil":      math.ceil,
    "factorial": math.factorial, "comb":    math.comb,
    "perm":      math.perm,    "gcd":       math.gcd,
    # constants
    "pi": math.pi, "e": math.e, "inf": math.inf, "nan": math.nan,
    # type conversions safe for math
    "int": int, "float": float,
    # numpy — vectors, cross/dot product, linear algebra
    **_NUMPY_GLOBALS,
}


def python_calculator(expression: str) -> dict:
   
    expression = expression.strip()
    try:
        result = eval(expression, _SAFE_GLOBALS, {})   
        logger.info(f"[Calculator] {expression} = {result}")
        return {"result": result, "expression": expression}
    except ZeroDivisionError:
        return {"error": "Division by zero", "expression": expression}
    except NameError as exc:
        return {"error": f"Unknown name: {exc}", "expression": expression}
    except SyntaxError as exc:
        return {"error": f"Syntax error: {exc}", "expression": expression}
    except Exception as exc:
        return {"error": str(exc), "expression": expression}