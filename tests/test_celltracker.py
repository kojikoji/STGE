import sys
from stge.cell_tracker import cell_tracker
import unittest
import unittest


class TestCellTracker(unittest.TestCase):
    def test_register_df(self):
        ct = cell_tracker()
        ct.register_df('../data/light_cell_df.csv')
        self.assertEqual(len(ct.all_frame), 4)

        
if __name__ == '__main__':
    unittest.main()



        

