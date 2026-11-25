from .config_defaults import APPLICATION_DEFAULTS
LIVE_DEFAULTS = APPLICATION_DEFAULTS["live"]
"""Model selection, paired schedules and a background live-evaluation session."""
from collections import defaultdict
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
from itertools import combinations
import json
from pathlib import Path
import re
import pickle
import threading
import time
import uuid

import torch


from .checkpoints import atomic_bytes, load_checkpoint_network
from .env import Config, ToyEnv
from .factory import load_settings
from .observation_schema import OBSERVATION_VERSION
from .replay import write_replay
from .rllib_logging import append_record
from .hud import HudRanges
from .tournament import (MatchRunner, load_model, settings_for_team_sizes,
                         standings_from_matches, team_standings_from_matches, _write_table)


SIDES = ('blue', 'red')


def aircraft_display_labels(assignments, models, aircraft):
    """Model names with a per-side suffix only when a display name repeats."""
    groups = defaultdict(list)
    for ident, model in assignments.items():
        groups[(aircraft[ident].team, models[model]['name'])].append(ident)
    return {ident: name if len(ids) == 1 else f'{name} #{index}'
            for (_, name), ids in groups.items() for index, ident in enumerate(ids, 1)}


RAW_STATS = ('played', 'wins', 'draws', 'losses', 'points', 'launches', 'kills',
             'deaths', 'return', 'survivals')


def integer(value, name, minimum, maximum):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f'{name}は{minimum}〜{maximum}の整数で指定してください。')
    return value


def json_file(path, value):
    atomic_bytes(path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False).encode('utf-8'))


class EvaluationModels:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.entries, self.networks = {}, {}
        self.entries['rule:fix'] = {
            'id': 'rule:fix', 'name': 'ルール / fix', 'kind': 'rule', 'profile': 'fix'}
        self.entries['rule:random'] = {
            'id': 'rule:random', 'name': 'ルール / random', 'kind': 'rule', 'profile': 'random'}

    def add(self, kind, path, policy_id='main', name=None):
        if kind not in ('model', 'checkpoint'):
            raise ValueError('モデルまたは学習チェックポイントを指定してください。')
        if not isinstance(path, str) or not path.strip() or len(path) > 4096:
            raise ValueError('モデルのフォルダーを指定してください。')
        folder = (self.root/path).resolve()
        if not folder.is_dir():
            raise ValueError('モデルのフォルダーが見つかりません。')
        if kind == 'model':
            model, network, metadata = load_model(folder)
            digest, iteration = metadata['sha256'], metadata.get('source_iteration')
            label = model
            policy_id = None
        else:
            if not isinstance(policy_id, str) or not re.fullmatch(r'initial|main|snapshot_[0-9]+', policy_id):
                raise ValueError('policy IDはmain・initial・snapshot_番号から指定してください。')
            network, _, digest = load_checkpoint_network(folder, policy_id)
            management = folder/'training-state.json'
            saved = json.loads(management.read_text(encoding='utf-8')) if management.exists() else {}
            iteration = (saved.get('initial_iteration', 0) if policy_id == 'initial'
                         else saved.get('training_iteration') if policy_id == 'main' else None)
            label = f'{folder.parent.name} / {folder.name} · {policy_id}'
        if kind == 'checkpoint' and not all(torch.isfinite(value).all().item() for value in network.state_dict().values()):
            raise ValueError('モデルの重みにNaNまたは無限大が含まれています。')
        if name is not None and (not isinstance(name, str) or not 1 <= len(name.strip()) <= 100):
            raise ValueError('表示名は1〜100文字で指定してください。')
        ident = 'model-'+hashlib.sha256(f'{kind}:{folder}:{policy_id}'.encode()).hexdigest()[:16]
        entry = {'id': ident, 'kind': kind, 'name': name.strip() if name else label,
                 'path': str(folder), 'policy_id': policy_id, 'sha256': digest, 'iteration': iteration}
        self.entries[ident], self.networks[ident] = entry, network
        return entry

    def reload(self, ident):
        entry = self.entries[ident]
        if entry['kind'] == 'rule':
            return entry
        return self.add(entry['kind'], entry['path'], entry.get('policy_id'), entry['name'])

    def checkpoint(self, folder):
        folder = Path(folder)
        if not folder.is_dir():
            return
        policies = sorted(p.name for p in (folder/'policies').glob('*')
                          if p.is_dir() and re.fullmatch(r'main|snapshot_[0-9]+', p.name))
        if (folder/'run-initial-weights.pth').is_file():
            policies.insert(0, 'initial')
        for policy in policies:
            self.add('checkpoint', str(folder), policy)

    def catalog(self):
        return list(self.entries.values())


def build_schedule(request, catalog):
    """Validate a request completely before creating any job or output files."""
    if not isinstance(request, dict):
        raise ValueError('対戦設定はJSONオブジェクトで指定してください。')
    mode = request.get('mode', 'duel')
    if mode not in ('duel', 'mixed', 'round_robin'):
        raise ValueError('対戦方式が不正です。')
    if request.get('action_mode', LIVE_DEFAULTS['action_mode']) not in ('greedy', 'sample'):
        raise ValueError('行動選択はgreedyまたはsampleで指定してください。')
    pace = request.get('pace', LIVE_DEFAULTS['pace'])
    if pace != 'realtime':
        integer(pace, '観戦速度', 0, 1000)
    for key in ('swap_sides', 'save_replays'):
        if type(request.get(key, True)) is not bool:
            raise ValueError(f'{key}はtrueまたはfalseで指定してください。')
    seeds = request.get('seeds')
    if seeds is None:
        start = integer(request.get('seed_start', LIVE_DEFAULTS['seed_start']), '開始seed', 0, 2**31-1)
        count = integer(request.get('seed_count', LIVE_DEFAULTS['seed_count']), 'seed数', 1, 1000)
        if start+count-1 > 2**31-1:
            raise ValueError('seedの上限を超えています。')
        seeds = list(range(start, start+count))
    if not isinstance(seeds, list) or not 1 <= len(seeds) <= 1000:
        raise ValueError('seedは1〜1000個指定してください。')
    for seed in seeds:
        integer(seed, 'seed', 0, 2**31-1)
    if len(set(seeds)) != len(seeds):
        raise ValueError('seedが重複しています。')

    def model(ident):
        if not isinstance(ident, str) or ident not in catalog:
            raise ValueError('一覧から対戦モデルを選択してください。')
        return ident

    names = request.get('team_names', ['チームA', 'チームB'])
    if (not isinstance(names, list) or len(names) != 2 or
            any(not isinstance(name, str) or not 1 <= len(name.strip()) <= 64 for name in names) or
            names[0].strip() == names[1].strip()):
        raise ValueError('異なる2つのチーム名を1〜64文字で指定してください。')
    names = [name.strip() for name in names]
    if mode == 'round_robin':
        selected = request.get('models', [])
        if not isinstance(selected, list) or len(selected) < 2 or len(set(selected)) != len(selected):
            raise ValueError('総当たりには異なるモデルを2つ以上選択してください。')
        selected = [model(ident) for ident in selected]
        count = integer(request.get('aircraft_per_team', LIVE_DEFAULTS['aircraft_per_team']), '各チームの機数', 1, 16)
        pairings = [([left]*count, [right]*count,
                     [catalog[ident]['name']+' ['+ident[-4:]+']' for ident in (left, right)])
                    for left, right in combinations(selected, 2)]
        if not pairings:
            raise ValueError('学習モデルを少なくとも1つ選択してください。')
    else:
        if mode == 'duel':
            left = [model(request.get('blue_model'))]*integer(request.get('blue_count', LIVE_DEFAULTS['blue_count']), 'Blue機数', 1, 31)
            right = [model(request.get('red_model'))]*integer(request.get('red_count', LIVE_DEFAULTS['red_count']), 'Red機数', 1, 31)
        else:
            lineups = [request.get('blue_lineup'), request.get('red_lineup')]
            if any(not isinstance(lineup, list) or not 1 <= len(lineup) <= 31 for lineup in lineups):
                raise ValueError('両チームに担当モデルを1〜31機分指定してください。')
            left, right = [[model(ident) for ident in lineup] for lineup in lineups]
        pairings = [(left, right, names)]
    plan = []
    for pair, (left, right, team_names) in enumerate(pairings, 1):
        if len(left)+len(right) > 32:
            raise ValueError('1試合は合計32機までです。')
        rule_sides = 0
        for lineup in (left, right):
            rules = {ident for ident in lineup if catalog[ident]['kind'] == 'rule'}
            if rules:
                if len(set(lineup)) != 1:
                    raise ValueError('ルール相手はチーム全体に指定してください。')
                rule_sides += 1
        for seed in seeds:
            for swapped in range(2 if request.get('swap_sides', True) else 1):
                blue, red = (right, left) if swapped else (left, right)
                actual_names = list(reversed(team_names)) if swapped else team_names
                assignments = {**{f'blue_wing{i}': ident for i, ident in enumerate(blue)},
                               **{f'red_wing{i}': ident for i, ident in enumerate(red)}}
                plan.append({'seed': seed, 'pair': pair, 'swapped': bool(swapped),
                             'blue_count': len(blue), 'red_count': len(red),
                             'assignments': assignments, 'team_names': dict(zip(SIDES, actual_names))})
                if len(plan) > 5000:
                    raise ValueError('1回の実行は5000試合までです。モデル数またはseed数を減らしてください。')
    single = request.get('single_match', False)
    if type(single) is not bool:
        raise ValueError('single_matchはtrueまたはfalseで指定してください。')
    if single:
        if mode != 'duel':
            raise ValueError('1試合評価は単一モデル同士で選択してください。')
        return plan[:1]
    return plan


class EvaluationSession:
    def __init__(self, root, output, models=None, checkpoints=()):
        self.root, self.output = Path(root).resolve(), Path(output).resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        self.models = EvaluationModels(self.root)
        self.scenarios = {'simulator': self.root/'configs/scenario-default.json'}
        self.condition = threading.Condition(threading.RLock())
        self.stop_event = threading.Event()
        self.worker, self.job, self.matches = None, None, []
        self.frames, self.current, self.current_config = [], None, None
        self.team_totals, self.model_totals, self.model_labels = {}, {}, {}
        self.errors = []
        if models and Path(models).is_dir():
            for folder in sorted(Path(models).iterdir()):
                if folder.is_dir() and (folder/'metadata.json').is_file():
                    self._try(lambda folder=folder: self.models.add('model', str(folder)))
        for folder in checkpoints:
            self._try(lambda folder=folder: self.models.checkpoint(folder))
        saved = self.output/'catalog.json'
        if saved.exists():
            try:
                for entry in json.loads(saved.read_text(encoding='utf-8')):
                    if entry['id'] not in self.models.entries:
                        self._try(lambda entry=entry: self.models.add(entry['kind'], entry['path'], entry.get('policy_id'), entry['name']))
            except (OSError, ValueError, KeyError, TypeError) as error:
                self.errors.append(str(error))
        preview = ToyEnv(Config(**load_settings(self.scenarios['simulator'])))
        preview.reset(seed=LIVE_DEFAULTS['seed_start'])
        self.frames, self.current_config = [preview.snapshot()], asdict(preview.config)

    def _try(self, operation):
        try:
            operation()
        except (OSError, ValueError, RuntimeError, KeyError, TypeError, EOFError, pickle.UnpicklingError) as error:
            self.errors.append(str(error))

    def active(self):
        return self.worker is not None and self.worker.is_alive()

    def add_model(self, request):
        with self.condition:
            if self.active():
                raise ValueError('モデルの追加は対戦を停止してから行ってください。')
            entry = self.models.add(request.get('kind'), request.get('path'),
                                    request.get('policy_id', 'main'), request.get('name') or None)
            json_file(self.output/'catalog.json', [m for m in self.models.catalog() if m['kind'] != 'rule'])
            return entry

    def preview(self, request):
        if request.get('scenario', 'simulator') not in self.scenarios:
            raise ValueError('シナリオが不正です。')
        plan = build_schedule(request, self.models.entries)
        return {'total': len(plan), 'seeds': sorted({item['seed'] for item in plan}),
                'pairs': len({item['pair'] for item in plan})}

    def start(self, request):
        with self.condition:
            if self.active():
                raise ValueError('実行中の対戦があります。停止または完了後に開始してください。')
            self.preview(request)
            plan = build_schedule(request, self.models.entries)
            selected = sorted({ident for item in plan for ident in item['assignments'].values()})
            metadata = {ident: deepcopy(self.models.reload(ident)) for ident in selected}
            networks = {ident: self.models.networks[ident] for ident in selected if metadata[ident]['kind'] != 'rule'}
            rules = {ident: metadata[ident]['profile'] for ident in selected if metadata[ident]['kind'] == 'rule'}
            settings = load_settings(self.scenarios[request.get('scenario', 'simulator')])
            Config(**settings)
            ident = datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')+'-'+uuid.uuid4().hex[:8]
            directory = self.output/ident
            directory.mkdir(exist_ok=False)
            self.job = {'id': ident, 'status': 'running', 'completed': 0, 'total': len(plan),
                        'error': None, 'output': str(directory), 'models': metadata,
                        'created_at': datetime.now(timezone.utc).isoformat(), 'pace': request.get('pace', LIVE_DEFAULTS['pace'])}
            self.matches, self.team_totals, self.model_totals, self.model_labels = [], {}, {}, {}
            self.current = None
            self.stop_event = threading.Event()
            manifest = {'job': deepcopy(self.job), 'request': request, 'settings': settings,
                        'schedule': plan, 'observation_version': OBSERVATION_VERSION,
                        'torch_version': torch.__version__}
            json_file(directory/'manifest.json', manifest)
            self.worker = threading.Thread(target=self._run, args=(plan, settings, networks, rules, request), daemon=True)
            self.worker.start()
            return self.state()

    @staticmethod
    def _merge(table, rows, key):
        for row in rows:
            current = table.setdefault(row[key], {key: row[key], **dict.fromkeys(RAW_STATS, 0)})
            for field in RAW_STATS:
                current[field] += row[field]

    @staticmethod
    def _rows(table, key):
        rows = [{**row, 'kill_difference': row['kills']-row['deaths'],
                 'mean_launches': row['launches']/max(1, row['played']),
                 'mean_return': row['return']/max(1, row['played'])} for row in table.values()]
        rows.sort(key=lambda row: (-row['points'], -row['kill_difference'], -row['mean_return'], row[key]))
        return [{'rank': rank, **row} for rank, row in enumerate(rows, 1)]

    def _aggregate(self, result):
        self._merge(self.team_totals, team_standings_from_matches([result]), 'team')
        converted = deepcopy(result)
        for item in converted['personal'].values():
            model_id, team = item['model_id'], result['team_names'][item['team']]
            key = json.dumps([team, model_id], ensure_ascii=False)
            item['model_id'] = key
            self.model_labels[key] = {'model_id': model_id, 'model_name': self.job['models'][model_id]['name'], 'team': team}
        self._merge(self.model_totals, standings_from_matches([converted]), 'model_id')

    def state(self):
        with self.condition:
            models = [{**row, **self.model_labels[row['model_id']]} for row in self._rows(self.model_totals, 'model_id')]
            return {'defaults': deepcopy(LIVE_DEFAULTS), 'catalog': self.models.catalog(), 'catalog_errors': list(self.errors),
                    'job': deepcopy(self.job), 'current': deepcopy(self.current),
                    'teams': self._rows(self.team_totals, 'team'), 'models': models,
                    'matches': deepcopy(self.matches[-50:])}

    def watch(self, match_id='', after=-1):
        with self.condition:
            ident = self.current['id'] if self.current else 'preview'
            reset = ident != match_id
            frames = self.frames if reset else [frame for frame in self.frames if frame['tick'] > after]
            return {'match_id': ident, 'reset': reset, 'new_frames': list(frames),
                    'status': self.job['status'] if self.job else 'idle',
                    'pace': self.job['pace'] if self.job else LIVE_DEFAULTS['pace'],
                    'frame': self.frames[-1], 'done': bool(self.current and self.current.get('done')),
                    'winner': self.current.get('winner') if self.current else None,
                    'seed': self.current.get('seed', LIVE_DEFAULTS['seed_start']) if self.current else LIVE_DEFAULTS['seed_start'],
                    'config': self.current_config, 'models': self.current.get('models', {}) if self.current else {}}

    def set_pace(self, pace):
        if pace != 'realtime':
            integer(pace, '観戦速度', 0, 1000)
        with self.condition:
            if not self.active() or self.job['status'] not in ('running', 'paused'):
                raise ValueError('実行中の対戦がありません。')
            self.job['pace'] = pace
            self.condition.notify_all()
            return self.state()

    def pause(self, paused):
        if type(paused) is not bool:
            raise ValueError('一時停止の指定が不正です。')
        with self.condition:
            if not self.active() or self.job['status'] not in ('running', 'paused'):
                raise ValueError('実行中の対戦がありません。')
            self.job['status'] = 'paused' if paused else 'running'
            self.condition.notify_all()
            return self.state()

    def stop(self):
        with self.condition:
            if self.active():
                self.stop_event.set()
                self.job['status'] = 'stopping'
                self.condition.notify_all()
            return self.state()

    def _save_summary(self, directory):
        state = self.state()
        json_file(directory/'summary.json', {key: state[key] for key in ('job', 'teams', 'models')})
        _write_table(directory, 'team-standings', state['teams'])
        _write_table(directory, 'model-standings', state['models'])

    def _run(self, plan, settings, networks, rules, request):
        directory = Path(self.job['output'])
        runner = None
        try:
            for number, item in enumerate(plan, 1):
                if self.stop_event.is_set():
                    break
                resized = settings_for_team_sizes(settings, item['blue_count'], item['red_count'])
                runner = MatchRunner(networks, item['assignments'], resized, item['seed'],
                                     request.get('action_mode', LIVE_DEFAULTS['action_mode']), rules)
                hud_ranges = HudRanges(runner.env.env.aircraft_settings)
                labels = aircraft_display_labels(item['assignments'], self.job['models'], runner.env.env.aircraft)

                model_colors = {}
                for side in SIDES:
                    side_models = dict.fromkeys(item['assignments'][a.id]
                                                for a in runner.env.env.aircraft.values() if a.team == side)
                    model_colors[side] = {model: index for index, model in enumerate(side_models)}

                def labeled_frame(frame):
                    hud_ranges.annotate(frame)
                    for aircraft in frame['aircraft']:
                        aircraft['label'] = labels[aircraft['id']]
                        aircraft['model_color'] = model_colors[aircraft['team']][item['assignments'][aircraft['id']]]
                    return frame
                models = {side: {'policy': ' / '.join(dict.fromkeys(self.job['models'][model]['name']
                          for aircraft, model in item['assignments'].items() if aircraft.startswith(side+'_')))} for side in SIDES}
                initial_frame = labeled_frame(runner.snapshot())
                with self.condition:
                    self.current = {**item, 'id': f'{self.job["id"]}:{number}', 'number': number,
                                    'done': False, 'winner': None, 'models': models}
                    self.frames, self.current_config = [initial_frame], asdict(runner.env.config)
                tick_started = time.monotonic()

                def publish_tick(frame):
                    nonlocal tick_started
                    frame = labeled_frame(frame)
                    with self.condition:
                        # Pace individual physics ticks without changing when policies decide.
                        while not self.stop_event.is_set():
                            if self.job['status'] == 'paused':
                                self.condition.wait()
                                tick_started = time.monotonic()
                                continue
                            pace = self.job['pace']
                            if not pace:
                                break
                            seconds = runner.env.config.aero_tick_seconds if pace == 'realtime' else 1/pace
                            remaining = seconds - (time.monotonic() - tick_started)
                            if remaining <= 0:
                                break
                            self.condition.wait(timeout=remaining)
                        self.frames.append(frame)
                        tick_started = time.monotonic()

                while not runner.done and not self.stop_event.is_set():
                    with self.condition:
                        while self.job['status'] == 'paused' and not self.stop_event.is_set():
                            self.condition.wait()
                    if self.stop_event.is_set():
                        break
                    tick_started = time.monotonic()
                    runner.step(record_tick=publish_tick)
                if not runner.done:
                    if request.get('save_replays', True):
                        write_replay(directory/'interrupted.html', self.frames, self.current_config)
                    break
                replay_name = f'match-{number:05d}-seed-{item["seed"]}.html'
                replay = directory/'replays'/replay_name if request.get('save_replays', True) else None
                if replay:
                    write_replay(replay, self.frames, {**self.current_config, 'models': models})
                result = runner.result(replay)
                lineups = {side: [model for aircraft, model in item['assignments'].items()
                                  if aircraft.startswith(side+'_')] for side in SIDES}
                uniform = all(len(set(lineup)) == 1 for lineup in lineups.values())
                result['retry_request'] = {
                    'mode': 'duel' if uniform else 'mixed',
                    'scenario': request.get('scenario', 'simulator'),
                    'blue_model': lineups['blue'][0], 'red_model': lineups['red'][0],
                    'blue_lineup': lineups['blue'], 'red_lineup': lineups['red'],
                    'blue_count': len(lineups['blue']), 'red_count': len(lineups['red']),
                    'seeds': [item['seed']], 'swap_sides': False,
                    'team_names': [item['team_names'][side] for side in SIDES],
                    'action_mode': request.get('action_mode', LIVE_DEFAULTS['action_mode']),
                    'save_replays': True, 'pace': self.job['pace'],
                }
                result['model_names'] = {side: models[side]['policy'] for side in SIDES}
                rule_only = {side: all(model in rules for model in lineups[side]) for side in SIDES}
                result['reference_side'] = 'red' if rule_only['blue'] and not rule_only['red'] else 'blue'
                result.update(match=number, mode=request.get('mode', 'duel'), action_mode=request.get('action_mode', LIVE_DEFAULTS['action_mode']),
                              team_names=item['team_names'], pair=item['pair'], swapped=item['swapped'],
                              replay_url=f'/files/{self.job["id"]}/replays/{replay_name}' if replay else None)
                with self.condition:
                    append_record(directory/'matches.jsonl', result)
                    self.matches.append(result)
                    self._aggregate(result)
                    self.job['completed'] = len(self.matches)
                    self.current.update(done=True, winner=result['winner'])
                    self._save_summary(directory)
            with self.condition:
                self.job['status'] = 'stopped' if self.stop_event.is_set() else 'completed'
        except Exception as error:
            with self.condition:
                self.job.update(status='failed', error=str(error))
        finally:
            with self.condition:
                self.job['finished_at'] = datetime.now(timezone.utc).isoformat()
                if self.current and self.job['status'] != 'completed':
                    self.current['done'] = True
                self._save_summary(directory)
