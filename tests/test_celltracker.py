from stge.cell_tracker import cell_tracker
import unittest


class TestCellTracker(unittest.TestCase):
    def test_register_df(self):
        ct = cell_tracker()
        ct.register_df('../data/light_cell_df.csv')
        self.assertEqual(len(ct.all_frame), 4)

    def test_load(self):
        ct = cell_tracker(point_num=1000)
        ct.register_df('../data/light_cell_df.csv')
        ct.refresh_nearest_prev()
        ct.load_all_time_lineage()
        
        
        
if __name__ == '__main__':
    unittest.main()



        

