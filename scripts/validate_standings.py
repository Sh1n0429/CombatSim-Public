"""Regression checks for victory/draw/loss points in both evaluation tables."""
import unittest

from combatsim.tournament import standings_from_matches, team_standings_from_matches


def match(blue, red, winner):
    return {
        'winner':winner, 'draw':winner is None,
        'team_names':{'blue':blue,'red':red},
        'personal':{
            side+'_wing0':{'model_id':name,'team':side,'launches':0,
                'kills':0,'survived':True,'return':0.}
            for side,name in [('blue',blue),('red',red)]},
    }


class StandingsChecks(unittest.TestCase):
    def test_win_draw_loss_points_agree_for_team_and_model(self):
        for aggregate,key in [(team_standings_from_matches,'team'),
                              (standings_from_matches,'model_id')]:
            with self.subTest(table=key):
                rows=aggregate([match('A','B','blue')])
                by_name={row[key]:row for row in rows}
                self.assertEqual(by_name['A']['points'],3)
                self.assertEqual(by_name['B']['points'],0)
                rows=aggregate([match('A','B',None)])
                self.assertEqual([row['points'] for row in rows],[1,1])

    def test_one_win_outranks_two_draws_in_both_tables(self):
        matches=[match('A','B','blue'),match('B','C',None),match('C','B',None)]
        for aggregate,key in [(team_standings_from_matches,'team'),
                              (standings_from_matches,'model_id')]:
            with self.subTest(table=key):
                rows=aggregate(matches)
                self.assertEqual(rows[0][key],'A')
                self.assertEqual(rows[0]['points'],3)
                self.assertEqual({row[key]:row['points'] for row in rows},
                                 {'A':3,'B':2,'C':2})


if __name__=='__main__':
    unittest.main(verbosity=2)
