from scipy.interpolate import splprep, splev, UnivariateSpline, CubicSpline, splrep
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import filters
from scipy.signal import gaussian

src_stain = [[298.493753162510, 422.270334151040, 334.872494840701],
             [333.420449398812, 345.065721473677, 229.050290707640],
             [294.300401093729, 359.649408002182, 311.276444902501]]

y = [[292.811025495496, 434.889818155038, 328.977613377431],
     [327.413039827433, 340.235284731561, 224.468892633966],
     [288.929820709103, 357.079242905865, 304.888077781462]]

src_stats = [-100,
             38.9754246402554,
             113.101950038887,
             134.145852985486,
             177.219898680259,
             185.755612970198,
             242.644304294941,
             249.912202210124,
             322.907552394936,
             435.824854368044,
             1000]

dst_stats = [-100,
             6.00854984737949,
             37.5947932208002,
             89.5220037186380,
             162.328698206662,
             193.112416125096,
             234.580255234000,
             267.482450641031,
             292.558375388892,
             459.134669486438,
             1000]

# 19584
spline1 = UnivariateSpline(np.array(src_stats), np.array(dst_stats), s=3400)

plt.plot(np.array(src_stain).flatten(), np.array(y).flatten(), 'o', label='data')

print(spline1)

adj_src_stain1 = spline1(np.array(src_stain))
# adj_src_stain1 = splev(range(1, 1000), spline1)

plt.plot(np.array(src_stain).flatten(), adj_src_stain1.flatten(), label="S")

plt.legend(loc='lower left', ncol=2)
plt.show()
print(np.array(src_stain))
print(adj_src_stain1)
