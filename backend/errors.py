# -*- coding: utf-8 -*-
"""
============================================================
backend/errors.py
============================================================
Typed application exceptions.

Handlers in backend/main.py translate these into HTTP
responses. The rule everywhere else: raise a specific
exception, never return a plausible-looking fake result.
============================================================
"""


class CropServiceError(RuntimeError):
    """Base class for all application errors."""


class ModelsUnavailable(CropServiceError):
    """
    The trained artifacts needed to serve a request are missing.

    Raised instead of degrading to a placeholder prediction --
    a wrong crop is worse than no crop.
    """


class ExplanationFailed(CropServiceError):
    """SHAP explanation could not be produced for a valid prediction."""
