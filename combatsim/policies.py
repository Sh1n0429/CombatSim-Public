import math
import random

from .config_defaults import RULE_DEFAULTS, ENVIRONMENT_DEFAULTS
from .env import encode_action, wrap
from .entities import selected_entities, shot_choices


class RulePolicy:
    """Observable-state arcade pilot; each aircraft retains its own mode."""
    DEFAULT_PARAMETERS = RULE_DEFAULTS
    INTEGER_PARAMETERS = {"egress_ticks", "egress_max_ticks", "fire_interval_ticks",
                          "max_simultaneous_own_missiles",
                          "warning_response_delay_ticks"}

    def __init__(self, settings=None, seed=None):
        self.parameters = self.sample_parameters(settings, seed)
        self.reset()

    @classmethod
    def sample_parameters(cls, settings=None, seed=None):
        settings = {} if settings is None else settings
        if not isinstance(settings, dict):
            raise ValueError("rule_policy must be an object")
        unknown = set(settings) - set(cls.DEFAULT_PARAMETERS) - {"randomize"}
        randomize = settings.get("randomize", {})
        if unknown or not isinstance(randomize, dict) or set(randomize) - (set(cls.DEFAULT_PARAMETERS) | {'enabled'}):
            raise ValueError(f"Invalid rule_policy parameters: {sorted(unknown or set(randomize) - set(cls.DEFAULT_PARAMETERS))}")
        enabled = randomize.get('enabled', True)
        if type(enabled) is not bool:
            raise ValueError('rule_policy.randomize.enabled must be boolean')
        values = dict(cls.DEFAULT_PARAMETERS)
        values.update({key: value for key, value in settings.items() if key != "randomize"})
        rng = random.Random(seed)
        for key, bounds in randomize.items():
            if key == 'enabled':
                continue
            if (not isinstance(bounds, (list, tuple)) or len(bounds) != 2
                    or any(type(v) not in (int, float) or not math.isfinite(v) for v in bounds)
                    or bounds[0] > bounds[1]):
                raise ValueError(f"rule_policy.randomize.{key} must be [min, max]")
            if key in cls.INTEGER_PARAMETERS and any(type(v) is not int for v in bounds):
                raise ValueError(f"rule_policy.randomize.{key} must use integer bounds")
            if enabled:
                values[key] = (rng.randint(*bounds) if key in cls.INTEGER_PARAMETERS
                               else rng.uniform(*bounds))
        for key, value in values.items():
            if key in cls.INTEGER_PARAMETERS:
                valid = type(value) is int and value >= 0
            else:
                valid = type(value) in (int, float) and math.isfinite(value) and value >= 0
            if not valid:
                raise ValueError(f"Invalid rule_policy.{key}: {value}")
        for key in ("attack_altitude", "attack_altitude_spread", "evade_altitude",
                    "evade_altitude_spread", "upper_recovery_altitude"):
            if values[key] > 1:
                raise ValueError(f"rule_policy.{key} must be at most 1")
        if values["attack_altitude"] > values["upper_recovery_altitude"]:
            raise ValueError("rule_policy attack_altitude must not exceed upper_recovery_altitude")
        if values["attack_altitude"] < values["attack_altitude_spread"]:
            raise ValueError("rule_policy attack altitude spread leaves the airspace")
        if not (values["evade_altitude_spread"] <= values["evade_altitude"]
                <= 1-values["evade_altitude_spread"]):
            raise ValueError("rule_policy evade altitude spread leaves the airspace")
        for key in ("crank_angle_degrees", "support_vertical_limit_degrees", "egress_heading_tolerance_degrees"):
            if values[key] > 90:
                raise ValueError(f"rule_policy.{key} must be at most 90")
        if values["nonclosing_fire_ratio"] > 1:
            raise ValueError("rule_policy.nonclosing_fire_ratio must be at most 1")
        if values["evade_turn_scale"] > 1:
            raise ValueError("rule_policy.evade_turn_scale must be at most 1")
        if values["evade_vertical_degrees"] > 45:
            raise ValueError("rule_policy.evade_vertical_degrees must be at most 45")
        if not 0 < values['boundary_enter_range'] < values['boundary_exit_range']:
            raise ValueError('boundary ranges must satisfy 0 < enter < exit')
        if values['waypoint_radius'] <= 0 or values['egress_max_ticks'] <= 0:
            raise ValueError('waypoint_radius and egress_max_ticks must be positive')
        return values

    def reset(self):
        self.memory = {}

    def mode(self, ident):
        return self.memory.get(ident, {}).get("mode", "search")

    @staticmethod
    def _turn(error):
        return max(-1.0, min(1.0, wrap(error)/math.radians(45)))

    @staticmethod
    def _pitch(angle):
        degrees = math.degrees(angle)
        return max(-1.0, min(1.0, degrees/(45 if degrees < 0 else 15)))

    @staticmethod
    def _altitude_pitch(z, target):
        # Gradually level out near the desired normalized altitude.
        return math.radians(max(-15.0, min(15.0, (target-z)*100)))

    def act(self, observation):
        own = observation["self"]
        ident = own["id"]
        suffix = ident.rsplit('_', 1)[-1]
        index = own.get('slot', int(suffix) if suffix.isdigit() else 0)
        memory = self.memory.setdefault(ident, {
            "mode": "search", "waypoint": 0, "target_id": None,
            "target_heading": None,
            "crank_side": 1 if index % 2 == 0 else -1,
            "support_missile_id": None, "pending_launch": False,
            "missiles_before_launch": (), "mode_ticks": 0,
            "egress_active": False, "target_position": None,
        })
        tick = observation.get('tick')
        elapsed_ticks = (1 if tick is None or 'last_tick' not in memory else
                         max(0, tick-memory['last_tick']))
        memory['last_tick'] = tick
        contacts = observation["contacts"]
        # Spatial margins and hysteresis keep the game pilot away from clipping.
        x, y, z = own["x"], own["y"], own["z"]
        airspace = observation.get("airspace", {"x": 100, "y": 100})
        distance_scale = max(airspace['x'], airspace['y'])
        performance = observation.get('performance', {})
        params = self.parameters
        radar_range = performance.get('radar_range', ENVIRONMENT_DEFAULTS['radar_range'])
        # Game-unit distances: independent of the size of the arena.
        # Adapt the close-range thresholds for short-range aircraft too.
        fire_range = min(params["fire_range"], radar_range)
        separate_range = min(params["separate_range"], fire_range*.4)
        reengage_range = min(params["reengage_range"], fire_range*.7)
        margin = min(x*airspace['x'], (1-x)*airspace['x'],
                     y*airspace['y'], (1-y)*airspace['y'])
        # Preserve physical distances on rectangular maps. On tiny maps cap
        # both thresholds proportionally so recovery can still finish.
        boundary_scale = min(1.0, .45*min(airspace['x'], airspace['y'])/params['boundary_exit_range'])
        recovering = (margin < params['boundary_enter_range']*boundary_scale
                      or (memory['mode'] == 'boundary' and margin < params['boundary_exit_range']*boundary_scale))
        vertical = 0
        missiles = observation.get("missiles", [])
        threat = min(missiles, key=lambda m: m["distance"]) if missiles else None
        friendly_missiles = observation.get("friendly_missiles", [])
        own_missiles = [m for m in friendly_missiles if m.get("is_own")]
        if memory.get("pending_launch"):
            previous_ids = set(memory.get("missiles_before_launch", ()))
            launched = [m for m in own_missiles if m.get("id") not in previous_ids]
            memory["pending_launch"] = False
            if launched:
                newest = min(launched, key=lambda m: (m.get("age", 0), m.get("id", "")))
                memory["support_missile_id"] = newest.get("id")
                memory["mode"] = "support"
        if memory["mode"] == "support" and memory.get("support_missile_id") is None:
            newest = min(own_missiles,
                         key=lambda m: (m.get("age", 0), m.get("id", "")),
                         default=None)
            if newest is not None:
                memory["support_missile_id"] = newest.get("id")
        supported = next((m for m in own_missiles
                          if m.get("id") == memory.get("support_missile_id")), None)
        target = next((c for c in contacts if c.get("id") == memory.get("target_id")), None)
        if target is not None:
            memory["target_heading"] = wrap(own["heading"] + target["bearing"])
            horizontal = target['distance']*distance_scale*math.cos(target['elevation'])
            memory['target_position'] = (
                x*airspace['x']+horizontal*math.cos(memory['target_heading']),
                y*airspace['y']+horizontal*math.sin(memory['target_heading']))

        # Support persists independently of maneuver mode. Consume the missile
        # ID once when starting egress, including during interrupts.
        if (memory.get("support_missile_id") is not None
                and (supported is None or supported.get("phase") == "active")):
            memory['egress_active'] = True
            memory['support_missile_id'] = None
            supported = None
            memory["mode_ticks"] = 0

        egress_bearing = memory['crank_side']
        egress_distance = 0.0
        if memory.get('target_position') is not None:
            tx, ty = memory['target_position']
            dx, dy = tx-x*airspace['x'], ty-y*airspace['y']
            egress_bearing = wrap(math.atan2(dy, dx)+math.pi-own['heading'])
            egress_distance = math.hypot(dx, dy)
        elif memory.get('target_heading') is not None:
            egress_bearing = wrap(memory['target_heading']+math.pi-own['heading'])
        if memory.get('egress_active'):
            complete = (memory['mode_ticks'] >= params['egress_ticks']
                        and abs(egress_bearing) <= math.radians(params['egress_heading_tolerance_degrees'])
                        and egress_distance >= params['egress_range'])
            timed_out = memory['mode_ticks'] >= max(params['egress_ticks'], params['egress_max_ticks'])
            if complete or timed_out:
                memory['egress_active'] = False
            else:
                memory['mode_ticks'] += elapsed_ticks

        # Optional game difficulty handicap: require a continuous warning for
        # this many physical ticks. Other controls keep updating while waiting.
        mws_warning = own.get('mws_active', False)
        radar_warning = bool(threat and threat["distance"]*distance_scale < (
            params["evade_hold_range"] if memory["mode"] == "evade"
            else params["emergency_evade_range"]))
        if mws_warning or radar_warning:
            memory['warning_ticks'] = (memory.get('warning_ticks', 0) + elapsed_ticks
                                       if memory.get('warning_pending', False) else 0)
            memory['warning_pending'] = True
        else:
            memory['warning_ticks'] = 0
            memory['warning_pending'] = False
        warning_ready = memory['warning_ticks'] >= params['warning_response_delay_ticks']

        if recovering:
            mode = "boundary"
            bearing = wrap(math.atan2((.5-y)*airspace["y"], (.5-x)*airspace["x"])-own["heading"])
            vertical = self._pitch(self._altitude_pitch(z, .5))
        elif mws_warning and warning_ready:
            # Toy warning is independent of radar tracks and contains direction
            # only. Reuse the existing arcade evade response without inventing
            # range, a target identity, or a contact for firing.
            forward, right, up = own.get('mws_direction', (0., 0., 0.))
            mode = "evade"
            bearing = wrap(math.atan2(right, forward) + math.pi) * params["evade_turn_scale"]
            vertical = self._pitch(math.radians(-params["evade_vertical_degrees"] if up > 0
                                                 else params["evade_vertical_degrees"]))
        elif radar_warning and warning_ready:
            mode = "evade"
            bearing = wrap(threat["bearing"] + math.pi) * params["evade_turn_scale"]
            vertical = self._pitch(math.radians(-params["evade_vertical_degrees"]
                                                 if threat["elevation"] > 0
                                                 else params["evade_vertical_degrees"]))
        elif supported is not None and target is not None:
            mode = "support"
            # Crank rather than turn cold: the target stays well inside the
            # radar cone while horizontal separation and a lower altitude are built.
            crank_angle = math.radians(params["crank_angle_degrees"])
            vertical_limit = math.radians(params["support_vertical_limit_degrees"])
            evade_altitude = params["evade_altitude"] + params["evade_altitude_spread"]*(-1 if index % 2 == 0 else 1)
            bearing = wrap(target["bearing"] + memory["crank_side"]*crank_angle)
            descend = self._altitude_pitch(z, evade_altitude)
            desired_pitch = max(target["elevation"]-vertical_limit,
                                min(target["elevation"]+vertical_limit, descend))
            vertical = self._pitch(desired_pitch)
        elif supported is not None:
            # Do not switch to a different contact while supporting a weapon.
            # Continue toward the last known target bearing to try to reacquire it.
            mode = "support"
            last_heading = memory.get("target_heading")
            bearing = (0 if last_heading is None else wrap(last_heading-own["heading"]))
            evade_altitude = params["evade_altitude"] + params["evade_altitude_spread"]*(-1 if index % 2 == 0 else 1)
            vertical = self._pitch(self._altitude_pitch(z, evade_altitude))
        elif memory.get('egress_active'):
            mode = "egress"
            bearing = egress_bearing
            evade_altitude = params["evade_altitude"] + params["evade_altitude_spread"]*(-1 if index % 2 == 0 else 1)
            vertical = self._pitch(self._altitude_pitch(z, evade_altitude))
        elif contacts:
            # Prefer a local contact over a shared-only track, even on cooldown. The
            # same contact ordering and legality contract is used by learners.
            candidates = selected_entities(observation)['contacts']
            target = next((contact for contact in candidates
                           if contact.get('detected_by_self', False)), contacts[0])
            memory["target_id"] = target.get("id")
            memory["target_heading"] = wrap(own["heading"] + target["bearing"])
            target_distance = target['distance']*distance_scale
            near = target_distance < separate_range or (memory['mode'] == 'separate' and target_distance < reengage_range)
            if near:
                mode = "separate"
                bearing = wrap(target["bearing"] + math.pi)
                # Different aircraft use different vertical directions, within bounds.
                vertical = self._pitch(math.radians(15 if index % 2 == 0 else -15))
            else:
                mode = "follow"
                bearing = target["bearing"]
                vertical = self._pitch(target["elevation"])

        else:
            mode = "search"
            waypoints = ((.7, .3), (.7, .7), (.3, .7), (.3, .3))
            if own.get('team') == 'red':
                waypoints = tuple((1-wx, 1-wy) for wx, wy in waypoints)
            offset = index % 4
            index = (memory["waypoint"]+offset) % 4
            wx, wy = waypoints[index]
            if math.hypot((wx-x)*airspace['x'], (wy-y)*airspace['y']) < params['waypoint_radius']:
                memory["waypoint"] += 1
                wx, wy = waypoints[(index+1) % 4]
            bearing = wrap(math.atan2((wy-y)*airspace["y"], (wx-x)*airspace["x"])-own["heading"])
            # Reposition high before the next commit. Adjacent aircraft use
            # slightly different levels to avoid forming one dense altitude band.
            desired_z = params["attack_altitude"] - params["attack_altitude_spread"]*(offset % 2 == 0)
            vertical = self._pitch(self._altitude_pitch(z, desired_z))
        # Altitude recovery takes priority over ordinary vertical commands.
        if z < .12:
            vertical = 1
        elif z > params["upper_recovery_altitude"]:
            vertical = self._pitch(math.radians(-15))
        memory["mode"] = mode
        # Flight and fire are independent action heads for every game player.
        # Maneuver interrupts must not silently disable an otherwise legal shot.
        candidates = selected_entities(observation)['contacts']
        legal_shots = set(shot_choices(observation)) - {0}
        if (params['max_simultaneous_own_missiles']
                and len(own_missiles) >= params['max_simultaneous_own_missiles']):
            legal_shots.clear()
        # Optional NPC pacing is independent of the environment cooldown.
        # Decisions and flight commands still update every physical tick.
        memory['clock'] = memory.get('clock', 0) + elapsed_ticks
        if memory['clock'] - memory.get('last_shot_tick', -params['fire_interval_ticks']) < params['fire_interval_ticks']:
            legal_shots.clear()
        shot = 0
        shot_target = None
        for i, contact in enumerate(candidates, 1):
            if i not in legal_shots:
                continue
            aspect = (math.cos(contact['heading']-contact['bearing'])
                      if 'heading' in contact else -1.0)
            ratio = params['nonclosing_fire_ratio']
            nonclosing_range = (fire_range * ratio if ratio else
                                min(params['nonclosing_fire_range'] or fire_range, fire_range))
            effective_range = (nonclosing_range
                               + (fire_range-nonclosing_range)*(1.0-aspect)*.5)
            if separate_range <= contact['distance']*distance_scale <= effective_range:
                shot, shot_target = i, contact
                break
        if shot:
            memory['last_shot_tick'] = memory['clock']
        if shot and memory.get('support_missile_id') is None:
            # Retain an existing support binding when another shot is issued.
            # Otherwise bind this launch after it appears in the observation.
            memory["mode_ticks"] = 0
            memory["target_id"] = shot_target.get("id")
            memory["target_heading"] = wrap(own['heading'] + shot_target['bearing'])
            horizontal = shot_target['distance']*distance_scale*math.cos(shot_target['elevation'])
            memory['target_position'] = (
                x*airspace['x']+horizontal*math.cos(memory['target_heading']),
                y*airspace['y']+horizontal*math.sin(memory['target_heading']))
            memory["egress_active"] = False
            memory["pending_launch"] = True
            memory["missiles_before_launch"] = tuple(m.get("id") for m in own_missiles)
        team_offset = 0 if own.get('team', 'blue') == 'blue' else math.pi
        target_heading = wrap(own['heading'] + bearing - team_offset)
        return encode_action(target_heading/math.pi, vertical, shot)
