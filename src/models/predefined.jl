
const SE3B_PROPERTIES = CR3BPSystemProperties{Float64}(
    :Sun,
    :EarthBarycenter,
    3.0404234047636514e-6,
    1.495978706137e+8, # km 
    5.02263525109303e+6, # s, 58.1323 days

    695700.0000, # km
    6380.0000,   # km (rounded Earth radius)
    1.3271244004127942e+11, # km3/s2
    4.0350323562548019e+05  # km3/s2
)

const EM3B_PROPERTIES = CR3BPSystemProperties{Float64}(
    :Earth,
    :Moon,
    1.2004720031932892e-2,
    384405.0000, # km 
    3.729387115155678e5, # s, 4.3164 days
    
    6378.1366,   # km 
    1737.4000, # km
    3.9860043543609598e+05, # km3/s2
    4.9028001184575496e+03  # km3/s2
)