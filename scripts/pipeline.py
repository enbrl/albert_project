"""The whole project in one command: scrape, clean, make the report, train the model.

Usage (from the root of the project):

    python -m scripts.pipeline                  # the first 5 pages (120 cars), a quick try
    python -m scripts.pipeline --all            # the whole listing (about 10 minutes)
    python -m scripts.pipeline --skip-scrape    # start again from the file already scraped

It runs the four steps one after the other, exactly like the commands of the README:

    1. python -m scripts.aramis_scraper   ->  data/cars_raw.csv
    2. python -m scripts.datacleaning     ->  data/cars_clean.csv
    3. python -m scripts.analysis         ->  reports/report.html and the charts
    4. python -m scripts.models           ->  models/price_model.joblib

If a step fails, the pipeline stops and tells which one: the files of the steps before are kept.
"""

import argparse
import os
import sys
import time

from scripts import analysis, aramis_scraper, datacleaning, models


def build_steps(args):
    """
    The steps to run, as a list of (title, function, arguments of the function).

    The functions are the main functions of the scripts: the arguments are the ones of their command lines.
    """
    raw = os.path.join(args.data_dir, 'cars_raw.csv')
    clean = os.path.join(args.data_dir, 'cars_clean.csv')

    steps = []
    if not args.skip_scrape:
        pages = ['--all'] if args.all else ['--pages', str(args.pages)]
        steps.append(('Scrape the listings of aramisauto.com', aramis_scraper.main, [*pages, '--output', raw]))
    steps.append(('Clean the data', datacleaning.main, [raw, '--output', clean]))
    if not args.no_report:
        steps.append(('Make the report and the charts', analysis.main, [clean, '--output', args.reports_dir]))
    steps.append(('Train and save the price model', models.main, [clean, '--kind', args.kind, '--model', args.model]))
    return steps


def run_step(title, function, arguments):
    """
    Run one step.

    Returns:
    - None if the step went well, else the sentence that tells what went wrong
    """
    try:
        function(arguments)
    except SystemExit as stop:  # the command lines stop with an exit code: 0 is fine
        if stop.code in (None, 0):
            return None
        return f'the step stopped with the code {stop.code} (see the message above)'
    except Exception as error:  # whatever goes wrong, say which step it was
        if os.environ.get('PIPELINE_DEBUG'):
            raise
        return f'{type(error).__name__}: {error}'
    return None


def main(argv=None):
    parser = argparse.ArgumentParser(description='Scrape, clean, analyse and train, in one command')
    how_many = parser.add_mutually_exclusive_group()
    how_many.add_argument('--pages', type=int, default=5, help='number of pages to scrape, 24 cars each (default: 5)')
    how_many.add_argument('--all', action='store_true', help='scrape the whole listing (about 10 minutes)')
    parser.add_argument('--skip-scrape', action='store_true',
                        help='do not scrape again: use the file already in the data folder')
    parser.add_argument('--no-report', action='store_true', help='do not make the report and the charts')
    parser.add_argument('--kind', default='best', choices=('best', *models.MODEL_KINDS),
                        help="model to train (default: 'best', chosen by the comparison)")
    parser.add_argument('--data-dir', default='data', help='folder of the CSV files (default: data)')
    parser.add_argument('--reports-dir', default='reports', help='folder of the report (default: reports)')
    parser.add_argument('--model', default=models.DEFAULT_MODEL_PATH,
                        help=f'file where the model is saved (default: {models.DEFAULT_MODEL_PATH})')
    args = parser.parse_args(argv)

    if args.pages < 1:
        parser.error('--pages must be at least 1')
    raw = os.path.join(args.data_dir, 'cars_raw.csv')
    if args.skip_scrape and not os.path.exists(raw):
        parser.error(f'--skip-scrape needs {raw}, which does not exist: run the pipeline without it first')

    steps = build_steps(args)
    started = time.time()
    for number, (title, function, arguments) in enumerate(steps, start=1):
        print(f'\n=== Step {number}/{len(steps)}: {title} ===', flush=True)
        begin = time.time()
        problem = run_step(title, function, arguments)
        if problem:
            print(f'\nStopped at step {number}/{len(steps)} ({title}): {problem}', file=sys.stderr)
            if number > 1 or args.skip_scrape:
                print('The files of the steps before were kept.', file=sys.stderr)
            if number > 1 and not args.skip_scrape:
                print('After fixing the problem, add --skip-scrape to not scrape again.', file=sys.stderr)
            raise SystemExit(1)
        print(f'(step done in {time.time() - begin:.0f} s)', flush=True)

    print(f'\nDone in {time.time() - started:.0f} s. Made:')
    made = [os.path.join(args.data_dir, 'cars_clean.csv'), args.model]
    if not args.skip_scrape:
        made.insert(0, raw)
    if not args.no_report:
        made.append(os.path.join(args.reports_dir, 'report.html'))
    for path in made:
        print(f'  {path}')
    return made


if __name__ == '__main__':
    main()
