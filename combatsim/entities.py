"""Shared observable entity ordering for neural inputs and action resolution."""
import math

TOKEN_COUNT = 32
ENTITY_KINDS = ('allies','contacts','missiles','friendly_missiles')


def entity_distance(entity):
    if 'distance' in entity:
        return entity['distance']
    return math.sqrt(sum(entity.get(k,0)**2 for k in ('dx','dy','dz')))


def selected_entities(observation):
    groups={key:sorted(observation.get(key,[]),key=lambda e:(entity_distance(e),e.get('id','')))
            for key in ENTITY_KINDS}
    selected={key:[] for key in ENTITY_KINDS}
    remaining=TOKEN_COUNT-1
    for index in range(TOKEN_COUNT-1):
        for key in ENTITY_KINDS:
            if index < len(groups[key]):
                selected[key].append(groups[key][index])
                remaining-=1
                if remaining==0:
                    return selected
    return selected


def shot_choices(observation):
    """0 is hold fire, 1..31 refer to selected contact rows, never true IDs."""
    own=observation['self']
    ready=own['ammo']>0 and own['shoot_cooldown']==0 and own.get('missile_capacity_available',True)
    return [0]+[index+1 for index,contact in enumerate(selected_entities(observation)['contacts'])
                if ready and contact.get('detected_by_self',False)]
