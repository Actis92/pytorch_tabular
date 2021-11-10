import random
import pandas as pd
from scipy import stats
from pytorch_tabular.gauss_rank import GaussRankScaler


def test_fit_transform():
    df = pd.DataFrame({"col": [random.uniform(0, 1) for _ in range(100)]})
    scaler = GaussRankScaler()
    result = scaler.fit_transform(df[["col"]])
    k2, p = stats.normaltest(result)
    assert p > 1e-3

