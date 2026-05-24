"""Allow ``python -m keppy`` to invoke the CLI.

Usage::

    python -m keppy solar --bodies sun,earth --duration 1y --plot
    python -m keppy list-bodies
    python -m keppy run my_sim.toml --plot
"""
from keppy.cli import main

main()
