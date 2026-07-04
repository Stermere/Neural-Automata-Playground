"""Filesystem locations shared by the training scripts."""
import os

_here = os.path.dirname(os.path.abspath(__file__))
default_image = os.path.join(_here, "..", "trainingImages", "Emoji.png")
checkpoints_root = os.path.join(_here, "..", "checkpoints")
