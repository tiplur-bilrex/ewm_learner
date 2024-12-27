from typing import Union, List, Any, Callable
import types

class Name:
    def __init__(self: 'Name', name: str):
        self.name: str = name

class Symbol:
    def __init__(self: 'Symbol', symbol: str):
        self.symbol: str = symbol

class Unit:
    def __init__(self: 'Unit', name: Name, symbols: list[Symbol]):
        self.name: Name = name
        self.symbols: list[Symbol] = symbols

kilogram_unit = Unit(
    name=Name("kilogram"),
    symbols=[Symbol("kg")])

meter_unit = Unit(
    name=Name("meter"),
    symbols=[Symbol("m")])

centimeter_unit = Unit(
    name=Name("centimeter"),
    symbols=[Symbol("cm")])

kilometer_unit = Unit(
    name=Name("kilometer"),
    symbols=[Symbol("km")])

inch_unit = Unit(
    name=Name("inch"),
    symbols=[Symbol("in")])

foot_unit = Unit(
    name=Name("foot"),
    symbols=[Symbol("ft")])

newton_unit = Unit(
    name=Name("Newton"),
    symbols=[Symbol("m³ kg⁻¹ s⁻²"), Symbol("N")])

cubic_meter_unit = Unit(
    name=Name("cubic meter"),
    symbols=[Symbol("m³")])

kilogram_per_cubic_meter_unit = Unit(
    name=Name("kilogram per cubic meter"),
    symbols=[Symbol("kg/m³")])

class Value:
    def __init__(self: 'Value', number: Any, unit = Unit):
        self.number: Any = number
        self.unit: Unit = unit

class Mass:
    def __init__(self: 'Mass', value: Value):
        self.value: Value = value

class Distance:
    distance_units: List[Unit] = [meter_unit, centimeter_unit, kilometer_unit, inch_unit, foot_unit]

    def __init__(self: 'Distance', value: Value):
        if value.unit not in self.distance_units:
            raise ValueError(f"Distance must be measured in one of the following units: {', '.join([unit.name.name for unit in self.acceptable_units])}")
        self.value: Value = value

class Radius(Distance):
    def __init__(self: 'Radius', value: Value):
        super().__init__(value)

class Volume:
    def __init__(self: 'Volume', value: Value):
        self.value: Value = value

class PhysicalConstant:
    def __init__(self: 'PhysicalConstant', name: Name, symbol: Symbol, values: list[Value]):
        self.name: Name = name
        self.symbol: Symbol = symbol
        self.values: list[Value] = values
    
    def get_value(self: 'PhysicalConstant', unit: Unit) -> Any:
        for value in self.values:
            if value.unit == unit:
                return value.number
        raise ValueError(f"No value found for unit {unit.name.name}")

gravitational_constant = PhysicalConstant(
    name=Name("Gravitational Constant"),
    symbol=Symbol("G"),
    values=[Value(number=6.67430e-11, unit=newton_unit)])

class GravitationalForce:
    def __init__(self: 'GravitationalForce', value: Value):
        self.value: Value = value
    # Note: class or variable defining a list of "Forces" or "force_types" can be made.
        # The hint is that this class has two words in the name (PhysicalConstant is a similar example)

def calculate_gravitational_force_in_newtons(distance: Distance, mass: Mass) -> Value:
    if  distance.value.unit == meter_unit and \
        mass.value.unit == kilogram_unit:
        force_value = gravitational_constant.get_value(newton_unit) * mass.value.number / (distance.value.number ** 2)
        return Value(number=force_value, unit=newton_unit)
    else:
        raise AttributeError(f"Inputs don't have the correct attributes for gravitational force calculation")

class PhysicalObject:
    def __init__(self: 'PhysicalObject', name: Name, mass: Mass, **kwargs):
        self.name: Name = name
        self.mass: Mass = mass

    def calculate_density(self: 'PhysicalObject') -> Value:
        if hasattr(self, 'calculate_volume_in_cubic_meters'):
            calculated_volume = self.calculate_volume_in_cubic_meters().value.number
            if self.mass.value.unit == kilogram_unit:
                calculated_density = self.mass.value.number / calculated_volume
            else:
                raise AttributeError(f"Cannot calculate density because {self.name.name} doesn't have a mass value with 'kilogram' unit.")
            return Value(number=calculated_density, unit=kilogram_per_cubic_meter_unit)
        else:
            raise AttributeError(f"{self.name.name} doesn't have a method to calculate volume")

statue_of_liberty = PhysicalObject(name=Name(name="Statue of Liberty"), mass=Mass(Value(number=204116, unit=kilogram_unit)))

class Sphere:
    def __init__(self: 'Sphere', radius: Radius, volume: Volume = None, **kwargs):
        self.radius: Radius = radius
        self.volume: Volume = volume

    def calculate_volume_in_cubic_meters(self: 'Sphere') -> Volume:
        if hasattr(self, 'radius') and self.radius.value.unit.name.name == "meter":
            volume_in_cubic_meters = (4/3) * 3.14159 * (self.radius.value.number ** 3)
            return Volume(Value(number=volume_in_cubic_meters, unit=cubic_meter_unit))

class Atmosphere:
    def __init__(self: 'Atmosphere', composition: dict, pressure: float):
        self.composition = composition  # dictionary of gas names and their percentages
        self.pressure = pressure  # in Pascals (Pa)

    def calculate_greenhouse_effect(self: 'Atmosphere') -> float:
        # Simplified calculation based on composition and pressure
        greenhouse_gases = ['CO2', 'CH4', 'H2O']
        effect = sum(self.composition.get(gas, 0) for gas in greenhouse_gases)
        return effect * self.pressure / 101325  # normalized to Earth's pressure

earth_atmosphere = Atmosphere(composition={'N2': 78, 'O2': 21, 'Ar': 0.93, 'CO2': 0.04}, 
                              pressure=101325)

class Planet(PhysicalObject, Sphere):
    def __init__(self: 'Planet', name: Name, mass: Mass, radius: Radius, volume: Volume = None, atmosphere: Atmosphere = None):
        PhysicalObject.__init__(self, name=name, mass=mass)
        Sphere.__init__(self, radius=radius, volume=volume)
        self.atmosphere = atmosphere

    def calculate_surface_gravity(self: 'Planet') -> Value:
        """Calculate the surface gravity of the planet in m/s^2."""
        if self.mass.value.unit == kilogram_unit and self.radius.value.unit == meter_unit:
            G = gravitational_constant.get_value(newton_unit)
            gravity = G * self.mass.value.number / (self.radius.value.number ** 2)
            return Value(number=gravity, unit=Unit(name=Name("meters per second squared"), symbols=[Symbol("m/s²")]))
        else:
            raise AttributeError("Mass must be in kilograms and radius in meters to calculate surface gravity")


earth_planet = Planet(name=Name(name="Earth"), 
                      mass=Mass(value=Value(number=5.97e24, unit=kilogram_unit)), 
                      radius=Radius(value=Value(number=6.37e6, unit=meter_unit)), 
                      atmosphere= earth_atmosphere)

mars_planet = Planet(name=Name(name="Mars"), 
                     mass=Mass(Value(number=6.39e23, unit=kilogram_unit)), 
                     radius=Radius(Value(number=3.39e6,unit=meter_unit)))

venus_planet = Planet(name=Name(name="Venus"), 
                      mass=Mass(Value(number=4.87e24, unit=kilogram_unit)), 
                      radius=Radius(Value(number=6.05e6,unit=meter_unit)),
                      volume=Volume(Value(number=9.275863869983333e+20, unit=cubic_meter_unit)))

class Ball(PhysicalObject, Sphere):
    def __init__(self: 'Ball', name: Name, mass: Mass, radius: Radius, volume: Volume = None):
        PhysicalObject.__init__(self, name=name, mass=mass)
        Sphere.__init__(self, radius=radius, volume=volume)


pool_ball = Ball(name=Name("Pool ball"), 
                 mass=Mass(Value(number=0.17, unit=kilogram_unit)), 
                 radius=Radius(Value(number=0.057, unit=meter_unit)))

def ratio(obj1: Union[int, float], obj2: Union[int, float]) -> Union[int, float]:
    if isinstance(obj1, (int, float)) and isinstance(obj2, (int, float)):
        ratio = obj1 / obj2
        return ratio

def total_value(*objects: Union[int, float]) -> Union[int, float]:
    values = [obj for obj in objects]
    if all(isinstance(v, (int, float)) for v in values):
        return sum(values)


