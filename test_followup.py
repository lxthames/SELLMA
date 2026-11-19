"""Checks for held-out isolation and the unit of statistical resampling."""
import copy
import unittest

import torch

from followup import load_inputs, paired_interval, select_settings


class FollowupDesignTests(unittest.TestCase):
    def test_test_scores_and_labels_cannot_select_settings(self):
        _, protocol, rows = load_inputs()
        margin = torch.linspace(-3, 3, len(rows))
        shift = torch.linspace(1, -1, len(rows))
        signs = torch.tensor([1 if r['expected'] == 'min' else -1 for r in rows])
        expected = select_settings(margin, shift, signs, rows, protocol)
        altered = copy.deepcopy(rows)
        for i, row in enumerate(altered):
            if row['split'] != 'dev' or row['task'] != 'target':
                margin[i] = 10000
                shift[i] = -10000
                signs[i] *= -1
                row['expected'] = 'arbitrary held-out label'
        self.assertEqual(expected, select_settings(margin, shift, signs, altered, protocol))

    def test_equal_development_scores_select_no_edit(self):
        _, protocol, rows = load_inputs()
        settings = select_settings(torch.zeros(len(rows)), torch.zeros(len(rows)),
                                   torch.ones(len(rows)), rows, protocol)
        self.assertEqual(settings['selected_strength'], 0.0)

    def test_frames_are_resampled_together(self):
        rows = []
        for gender in ('common', 'neuter'):
            for noun in ('a', 'b'):
                for frame in (0, 1):
                    rows.append(dict(noun=gender + noun, gender=gender, frame=frame))
        delta = torch.tensor([1., -1.] * 4)
        interval = paired_interval(delta, rows, torch.ones(8, dtype=torch.bool), 1000, 42)
        self.assertEqual(interval, [0.0, 0.0])


if __name__ == '__main__':
    unittest.main()
