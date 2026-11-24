"""Shared plumbing: loads the JavaScript and CSS that draw each demo."""
import asyncio
from pathlib import Path

ASSETS = Path(__file__).parent / "assets"
CHECKPOINTS = Path(__file__).parent.parent / "checkpoints"


def esm(name):
    """JavaScript for one demo, with the shared drawing helpers in front."""
    return (ASSETS / "common.js").read_text("utf-8") + "\n" + (ASSETS / name).read_text("utf-8")


def css():
    return (ASSETS / "theme.css").read_text("utf-8")


async def breathe():
    """Give the notebook a moment to handle button clicks during training."""
    await asyncio.sleep(0)
