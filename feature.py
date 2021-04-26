import pandas as pd

import statistics as st

STATS_ROWS = ['Count', 'Mean', 'Std', 'Min', '25%', '50%', '75%', 'Max']

class Feature():
    series = None

    def __init__(self, series):
        self.series = series

    def getStats(self):
        return [
            st.count(self.series),
            st.mean(self.series),
            st.std(self.series),
            st.min(self.series),
            st.percentile(self.series, 25),
            st.percentile(self.series, 50),
            st.percentile(self.series, 75),
            st.max(self.series)
        ]
    