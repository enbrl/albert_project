"""Helpers shared by the command lines of the project.

A problem that the user can fix (a file that does not exist, a CSV with no car, a wrong
option...) must be shown as one clear line, not as a page of technical error. A real bug must
still show its full error, so that it can be understood and fixed.

Each script ends with:

    if __name__ == '__main__':
        run(main)
"""

import argparse
import sys

import pandas as pd
import requests


class UserError(ValueError):
    """A problem that the user can fix (wrong file, not enough data...), not a bug."""


def positive_int(text):
    """Type of a command line option: a whole number of at least 1 (for argparse)."""
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"'{text}' is not a whole number") from None
    if value < 1:
        raise argparse.ArgumentTypeError(f'{value} is not valid: it must be at least 1')
    return value


def friendly_message(error):
    """
    The message to show for an error that the user can fix, or None if the error looks like a bug.
    """
    if isinstance(error, UserError):
        return str(error)
    if isinstance(error, FileNotFoundError):
        return f'File not found: {error.filename}' if error.filename else 'File not found'
    if isinstance(error, IsADirectoryError):
        return f'{error.filename} is a folder, a file is needed'
    if isinstance(error, PermissionError):
        return f'Not allowed to read or write {error.filename}'
    if isinstance(error, pd.errors.EmptyDataError):
        return 'The CSV file is empty'
    if isinstance(error, pd.errors.ParserError):
        return f'The file is not a valid CSV file ({error})'
    if isinstance(error, requests.RequestException):
        return f'Network problem: {error}'
    return None


def run(main, print_result=False):
    """
    Run the main function of a script; show the errors that the user can fix as a message.

    Parameters:
    - main: the main function of the script (it reads the command line itself)
    - print_result: print what main returns (used by the scripts that return a table)

    The script stops with the exit code 1 after a message. Any other error is not caught.
    """
    try:
        result = main()
    except Exception as error:
        message = friendly_message(error)
        if message is None:
            raise  # probably a bug: show the whole error
        print(f'Error: {message}', file=sys.stderr)
        sys.exit(1)
    if print_result:
        print(result)
    return result
