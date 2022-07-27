"""
Downloads interest rates (euribor12m, sofr)
"""
from typing import Optional, Any

import pandas as pd
import ujson
from ong_utils import create_pool_manager


class Rates:
    """Class to download interest rates (sofr, euribor...)"""

    def __init__(self, date_from=None, date_to=None, index=None):
        """
        Creates object and downloads interest rates for sofr and euribor.
        Uses:
        Rates("2022-06-19") to read dates from June 19th 22 to today
        Rates(index=["2022-06-19", "2022-06-20") to read dates from June 19th 22 to June 20th, 22
        Rates("2020-01-01", index=["2022-06-19", "2022-06-20") to read dates from June 19th 22 to June 20th, 22. Note
        that in case index has values the first and last value of index is used to replace date_from and date_to
        :param date_from: first date to download data (anything that can be turned into a pd.Timestamp)
        :param date_to: last date to download data (anything that can be turned into a pd.Timestamp). If None,
        today's date is used
        :param index: An optional list of timestamp for the index, so if there is any missing data it is filled
        with the last value and no gaps against index are found
        """
        if date_from is None and index is None:
            raise ValueError("Either date_from or index are needed")
        self._date_to = pd.Timestamp(date_to or pd.Timestamp.now().normalize()) if index is None else index[-1]
        self._date_from = pd.Timestamp(date_from) if index is None else index[0]
        self._iso_date_from = self._date_from.isoformat()[:10]
        self._iso_date_to = self._date_to.isoformat()[:10]
        self.__pool = create_pool_manager()

        # Download data
        def reindex_data(data):
            return data.reindex(index=index, method="pad") / 100   # Align with indexes and normalice value

        self.sofr = reindex_data(self.get_sofr())
        self.euribor12m = reindex_data(self.get_euribor(euribor_serie=4))
        self.euribor1w = reindex_data(self.get_euribor(euribor_serie=5))
        self.euribor1m = reindex_data(self.get_euribor(euribor_serie=1))
        self.__pool.clear()

    def download_js(self, url) -> Optional[Any]:
        """Downloads a js from an url. Returns None if failed"""
        req = self.__pool.request("GET", url)
        if req.status != 200:
            return None
        js = ujson.loads(req.data)
        return js

    def get_sofr(self):
        """Downloads sofr rates from newyorkfed"""

        url = f"https://markets.newyorkfed.org/read?startDt={self._iso_date_from}&endDt={self._iso_date_to}" \
              "&eventCodes=520&productCode=50&sort=postDt:1,eventCode:1&format=json"
        js = self.download_js(url)
        if js is None:
            return None
        rates = js["refRates"]
        data = {pd.Timestamp(r["effectiveDate"]): float(r["percentRate"]) for r in rates}
        serie = pd.Series(data, dtype=float)
        return serie

    def get_euribor(self, euribor_serie: int):
        """
        Downloads euribor rates for the considered type from euribor-rates.eu
        :param euribor_serie: 5 for 1week euribor, 4 for 12m euribor, 1 for 1m euribor
        :return:
        """
        min_tick = int(self._date_from.timestamp() * 1e3)  # 915148800000
        max_tick = int(self._date_to.timestamp() * 1e3)

        url = f"https://www.euribor-rates.eu/umbraco/api/euriborpageapi/highchartsdata?minticks={min_tick}" \
              f"&maxticks={max_tick}&series[0]={euribor_serie}"

        js = self.download_js(url)
        if js is None:
            return None
        data = js[0]["Data"]
        data2 = {pd.Timestamp.fromtimestamp(d[0] / 1e3): d[1] for d in data}
        serie = pd.Series(data2, dtype=float)

        return serie


if __name__ == '__main__':
    index = pd.date_range("2022-06-01", "2022-06-30")
    rates = Rates("2000-01-01", index[0], index=index)
    print(rates.sofr)
    print(rates.euribor12m)
