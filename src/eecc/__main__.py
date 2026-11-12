"""``python -m eecc``: the same as the ``eecc`` command.

Useful where pip's script folder is not on PATH (common on Windows).
"""

from eecc.cli import main

if __name__ == "__main__":
    main()
