from hifinet.fault.type import DriftFault, ErraticFault, HardoverFault, SpikeFault, StuckFault 
DEFAULT_MAPPING = {
    HardoverFault: 1,
    DriftFault: 2,
    SpikeFault: 3,
    ErraticFault: 4,
    StuckFault: 5
}
