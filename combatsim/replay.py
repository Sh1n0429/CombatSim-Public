import json
from pathlib import Path


def render_page(frames, config, live=False):
    assets = Path(__file__).parent/'web'
    template = (assets/'replay.html').read_text(encoding='utf-8')
    script = (assets/'viewer.js').read_text(encoding='utf-8')
    data = json.dumps({"frames": frames, "config": config, "live": live}).replace("<", "\\u003c")
    return template.replace('__BOOTSTRAP__', data).replace('__SCRIPT__', script)


def write_replay(path, frames, config):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_page(frames, config), encoding='utf-8')
    output.with_suffix('.json').write_text(json.dumps({'format': 'combatsim-replay-v4-3d', 'arena_x': config['arena_x'], 'arena_y': config['arena_y'], 'ceiling': config['ceiling'], 'config': config, 'frames': frames}), encoding='utf-8')
