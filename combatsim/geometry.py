"""Shared geometry without environment or policy dependencies."""
import math


def distance(a, b):
    return math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2 + (a.z-b.z)**2)


def wrap(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi
