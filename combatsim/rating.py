"""Match ratings for the RLlib learner and frozen opponent pool."""


from .config_defaults import APPLICATION_DEFAULTS

RATING_DEFAULTS = APPLICATION_DEFAULTS['rating']


def update_rating(rating, score, reference=RATING_DEFAULTS['reference'], k=RATING_DEFAULTS['k']):
    expected = 1/(1+10**((reference-rating)/400))
    return rating+k*(score-expected)
