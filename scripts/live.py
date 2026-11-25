"""Open the simulator live evaluation screen in a local browser."""
from combatsim.config_defaults import APPLICATION_DEFAULTS

import argparse
from pathlib import Path

import torch

from combatsim.evaluation import EvaluationSession
from combatsim.live import serve_live



def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', default=str(root/'models'))
    parser.add_argument('--checkpoint', action='append', default=None,
                        help='Own RLlib checkpoint folder; may be repeated')
    parser.add_argument('--output', default=str(root/'artifacts/live'))
    parser.add_argument('--port', type=int, default=APPLICATION_DEFAULTS['live']['port'])
    parser.add_argument('--no-browser', action='store_true')
    args = parser.parse_args()
    if not 0 <= args.port <= 65535:
        parser.error('port must be 0..65535')
    torch.set_num_threads(1)
    checkpoints = (args.checkpoint if args.checkpoint is not None else
                   [str(root/'artifacts/checkpoint')])
    session = EvaluationSession(root, args.output, args.models, checkpoints)
    try:
        serve_live(session, args.port, not args.no_browser)
    except OSError as error:
        parser.error(str(error))


if __name__ == '__main__':
    main()
