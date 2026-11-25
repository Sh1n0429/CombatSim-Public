"""Read old checkpoint names while keeping new settings and logs canonical."""
from copy import deepcopy


def normalize_rule_settings(settings):
    settings = deepcopy(settings)
    values = settings.get('rule_policy_profiles', {})
    if 'medium' in values:
        old = values.pop('medium')
        values.setdefault('fix', old)
    return settings


def normalize_rule_statistics(coordinator):
    for attribute in ('rule_profile_results', 'rule_profile_ratings', 'matchup_statistics'):
        values = deepcopy(getattr(coordinator, attribute))
        for key in list(values):
            new_key = 'fix' if key == 'medium' else key.replace('rule:medium:', 'rule:fix:')
            if new_key == key:
                continue
            old = values.pop(key)
            if new_key not in values:
                values[new_key] = old
            elif attribute != 'rule_profile_ratings':
                for field, count in old.items():
                    values[new_key][field] = values[new_key].get(field, 0) + count
        setattr(coordinator, attribute, values)
