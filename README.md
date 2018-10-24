# energypatterns

Here I developed _energypatterns_, a Python module to analyze energy performance over time. First, I pre-process the energy data to detect and remove erroneous values/outliers and create additional features of interest. Following, I merge and normalize five years of data to create energy performance time series. We can then cluster the normalized time series and identify key attributes of buildings with certain energy performance patterns over time using statistical metrics.

The code was mainly developed for [this research project](https://www.sciencedirect.com/science/article/pii/S0306261918304070). Please cite as: _Papadopoulos, S., Bonczak, B. and Kontokosta, C.E., 2018. Pattern recognition in building energy performance over time using energy benchmarking data. Applied Energy, 221, pp.576-586._ 
