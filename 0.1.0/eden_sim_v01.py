#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
eden_sim.py

Simulador evolutivo tipo WorldBox:
Adán y Eva empiezan en una cueva dentro de un mundo 256x256.
Tienen cerebros neuronales simples, necesidades, memoria, habilidades,
lesiones, nacimientos, condiciones/discapacidades, clima, estaciones,
animales, plantas buenas/venenosas y primeros pasos de civilización.

Instalación en Mac:

    mkdir eden_sim
    cd eden_sim
    python3 -m venv edensim
    source edensim/bin/activate
    pip install numpy
    python eden_sim.py

Controles:
    El simulador corre solo.
    Pulsa CTRL + C para detenerlo.

Notas:
    Esta versión es una base 0.1.
    La IA usa redes neuronales evolutivas simples:
    - Cada humano tiene un cerebro.
    - Los bebés heredan una mezcla del cerebro de sus padres.
    - Hay mutaciones.
    - Los que sobreviven más tienen más posibilidades de reproducirse.
"""

from __future__ import annotations

import math
import os
import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================

MAP_W = 256
MAP_H = 256

VIEW_RADIUS = 18
TURN_DELAY = 0.03

HOURS_PER_DAY = 24
DAYS_PER_MONTH = 30
MONTHS_PER_YEAR = 12

MAX_DAYS = 365

PROB_CONDICION_NACIMIENTO = 0.04
PROB_REPRODUCCION_DIARIA_BASE = 0.05

INPUT_SIZE = 30
HIDDEN_SIZE = 24

ACTIONS = [
    "explorar",
    "buscar_agua",
    "buscar_comida",
    "comer",
    "beber",
    "descansar",
    "huir",
    "volver_refugio",
    "socializar",
    "cazar",
    "recolectar",
    "construir",
]

OUTPUT_SIZE = len(ACTIONS)


# ============================================================
# TERRENO
# ============================================================

GRASS = "."
FOREST = "T"
WATER = "~"
CAVE = "C"
MOUNTAIN = "M"
SWAMP = "S"
HUT = "h"
HOUSE = "H"
BUILDING = "B"

TERRAIN_NAMES = {
    GRASS: "pradera",
    FOREST: "bosque",
    WATER: "río",
    CAVE: "cueva",
    MOUNTAIN: "montaña",
    SWAMP: "pantano",
    HUT: "choza",
    HOUSE: "casa",
    BUILDING: "edificio",
}


# ============================================================
# COLORES ANSI
# ============================================================

USE_COLOR = True

COLORS = {
    "reset": "\033[0m",
    "green": "\033[32m",
    "lightgreen": "\033[92m",
    "blue": "\033[34m",
    "cyan": "\033[96m",
    "yellow": "\033[93m",
    "red": "\033[91m",
    "magenta": "\033[95m",
    "white": "\033[97m",
    "gray": "\033[90m",
}


def color(text: str, name: str) -> str:
    if not USE_COLOR:
        return text
    return COLORS.get(name, "") + text + COLORS["reset"]


def terrain_symbol(tile: str) -> str:
    if tile == GRASS:
        return color(tile, "green")
    if tile == FOREST:
        return color(tile, "lightgreen")
    if tile == WATER:
        return color(tile, "blue")
    if tile == CAVE:
        return color(tile, "gray")
    if tile == MOUNTAIN:
        return color(tile, "white")
    if tile == SWAMP:
        return color(tile, "yellow")
    if tile in [HUT, HOUSE, BUILDING]:
        return color(tile, "magenta")
    return tile


def clear_screen() -> None:
    os.system("clear" if os.name != "nt" else "cls")


# ============================================================
# UTILIDADES
# ============================================================

def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def distance(a: Tuple[int, int], b: Tuple[int, int]) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def chance(p: float) -> bool:
    return random.random() < p


def random_sex() -> str:
    return "hombre" if random.random() < 0.5 else "mujer"


def step_towards(x: int, y: int, tx: int, ty: int) -> Tuple[int, int]:
    dx = 0 if tx == x else (1 if tx > x else -1)
    dy = 0 if ty == y else (1 if ty > y else -1)
    return x + dx, y + dy


# ============================================================
# CEREBRO NEURONAL EVOLUTIVO
# ============================================================

class Brain:
    """
    Red neuronal pequeña:
    entradas -> capa oculta tanh -> salidas.
    No usa aprendizaje por gradiente todavía.
    Evoluciona por herencia + mutación.
    """

    def __init__(self):
        self.w1 = np.random.randn(INPUT_SIZE, HIDDEN_SIZE) * 0.35
        self.b1 = np.random.randn(HIDDEN_SIZE) * 0.02
        self.w2 = np.random.randn(HIDDEN_SIZE, OUTPUT_SIZE) * 0.35
        self.b2 = np.random.randn(OUTPUT_SIZE) * 0.02

    def decide(self, inputs: List[float]) -> int:
        x = np.array(inputs, dtype=float)

        if x.shape[0] != INPUT_SIZE:
            raise ValueError(f"Brain esperaba {INPUT_SIZE} entradas, recibió {x.shape[0]}.")

        h = np.tanh(x @ self.w1 + self.b1)
        out = h @ self.w2 + self.b2

        # Exploración ocasional, para que prueben cosas nuevas.
        if random.random() < 0.04:
            return random.randint(0, OUTPUT_SIZE - 1)

        return int(np.argmax(out))

    def clone_mutated(self, mutation_rate: float = 0.16, mutation_power: float = 0.10) -> "Brain":
        b = Brain()
        b.w1 = self.w1.copy()
        b.b1 = self.b1.copy()
        b.w2 = self.w2.copy()
        b.b2 = self.b2.copy()

        if chance(mutation_rate):
            b.w1 += np.random.randn(*b.w1.shape) * mutation_power
        if chance(mutation_rate):
            b.b1 += np.random.randn(*b.b1.shape) * mutation_power
        if chance(mutation_rate):
            b.w2 += np.random.randn(*b.w2.shape) * mutation_power
        if chance(mutation_rate):
            b.b2 += np.random.randn(*b.b2.shape) * mutation_power

        return b


def mix_brains(a: Brain, b: Brain) -> Brain:
    child = Brain()

    mask_w1 = np.random.rand(*a.w1.shape) < 0.5
    mask_w2 = np.random.rand(*a.w2.shape) < 0.5

    child.w1 = np.where(mask_w1, a.w1, b.w1)
    child.w2 = np.where(mask_w2, a.w2, b.w2)

    child.b1 = (a.b1 + b.b1) / 2
    child.b2 = (a.b2 + b.b2) / 2

    return child.clone_mutated(mutation_rate=0.24, mutation_power=0.12)


# ============================================================
# DATOS DE SUPERVIVENCIA
# ============================================================

PLANTS = [
    {"id": "manzana_silvestre", "name": "manzana silvestre", "poison": False, "food": 24, "bad": 0.10},
    {"id": "bayas_azules", "name": "bayas azules", "poison": False, "food": 16, "bad": 0.15},
    {"id": "higos", "name": "higos", "poison": False, "food": 22, "bad": 0.08},
    {"id": "nueces", "name": "nueces", "poison": False, "food": 18, "bad": 0.05},
    {"id": "raices_comestibles", "name": "raíces comestibles", "poison": False, "food": 18, "bad": 0.20},
    {"id": "bayas_rojas_venenosas", "name": "bayas rojas venenosas", "poison": True, "food": 4, "bad": 1.00},
    {"id": "hongos_morados", "name": "hongos morados", "poison": True, "food": 4, "bad": 1.00},
    {"id": "fruto_negro_brillante", "name": "fruto negro brillante", "poison": True, "food": 4, "bad": 1.00},
    {"id": "raiz_amarga", "name": "raíz amarga", "poison": True, "food": 4, "bad": 1.00},
]

ANIMALS = [
    {"id": "conejo", "name": "conejo", "aggressive": False, "strength": 8, "meat": 18, "raw_risk": 0.20, "night": False},
    {"id": "ciervo", "name": "ciervo", "aggressive": False, "strength": 22, "meat": 45, "raw_risk": 0.25, "night": False},
    {"id": "pez", "name": "pez", "aggressive": False, "strength": 2, "meat": 14, "raw_risk": 0.30, "night": False},
    {"id": "gallina", "name": "gallina salvaje", "aggressive": False, "strength": 6, "meat": 16, "raw_risk": 0.15, "night": False},
    {"id": "cabra", "name": "cabra salvaje", "aggressive": False, "strength": 18, "meat": 35, "raw_risk": 0.20, "night": False},
    {"id": "lobo", "name": "lobo", "aggressive": True, "strength": 45, "meat": 30, "raw_risk": 0.40, "night": True},
    {"id": "oso", "name": "oso", "aggressive": True, "strength": 85, "meat": 85, "raw_risk": 0.50, "night": True},
    {"id": "serpiente", "name": "serpiente", "aggressive": True, "strength": 35, "meat": 6, "raw_risk": 0.70, "night": True},
    {"id": "jabali", "name": "jabalí", "aggressive": True, "strength": 55, "meat": 50, "raw_risk": 0.35, "night": True},
    {"id": "pantera", "name": "pantera", "aggressive": True, "strength": 70, "meat": 45, "raw_risk": 0.45, "night": True},
]

BIRTH_CONDITIONS = [
    {"id": "movilidad_reducida", "name": "movilidad reducida", "effects": {"speed": -25, "energy_max": -10, "fall_risk": 0.12}},
    {"id": "vision_reducida", "name": "visión reducida", "effects": {"spatial": -18, "accident_risk": 0.08}},
    {"id": "audicion_reducida", "name": "audición reducida", "effects": {"animal_detection": -25, "ambush_risk": 0.12}},
    {"id": "fragilidad_osea", "name": "fragilidad ósea", "effects": {"strength": -10, "fracture_risk": 0.20}},
    {"id": "inmunidad_debil", "name": "sistema inmune débil", "effects": {"immunity": -25, "illness_risk": 0.18}},
    {"id": "dificultad_aprendizaje", "name": "dificultad de aprendizaje", "effects": {"memory": -12, "learning": -18}},
]

INJURIES = {
    "corte": {"life": -5, "energy": -5, "days": 2},
    "torcedura": {"life": -3, "energy": -15, "days": 5},
    "fractura": {"life": -22, "energy": -30, "days": 30, "may_disable": True},
    "herida_grave": {"life": -42, "energy": -35, "days": 15, "may_kill": True},
    "mordedura": {"life": -25, "energy": -20, "days": 10, "infection": 0.25},
    "intoxicacion": {"life": -20, "energy": -25, "hunger": -10, "days": 6},
    "enfermedad": {"life": -15, "energy": -22, "days": 8},
    "ahogamiento": {"life": -80, "energy": -80, "days": 1, "may_kill": True},
}


# ============================================================
# HABILIDADES HUMANAS
# ============================================================

def create_skills(sex: str) -> Dict[str, float]:
    s = {
        "multitask": random.randint(30, 70),
        "empathy": random.randint(30, 70),
        "memory": random.randint(30, 70),
        "immunity": random.randint(30, 70),
        "longevity": random.randint(30, 70),
        "spatial": random.randint(30, 70),
        "strength": random.randint(30, 70),
        "speed": random.randint(30, 70),
        "mechanics": random.randint(30, 70),
        "focus": random.randint(30, 70),
    }

    if sex == "mujer":
        if chance(0.65):
            for k in ["multitask", "empathy", "memory", "immunity", "longevity"]:
                s[k] += random.randint(10, 25)
        if chance(0.35):
            for k in ["strength", "spatial", "mechanics"]:
                s[k] += random.randint(5, 20)

    if sex == "hombre":
        if chance(0.65):
            for k in ["spatial", "strength", "speed", "mechanics", "focus"]:
                s[k] += random.randint(10, 25)
        if chance(0.35):
            for k in ["multitask", "empathy", "memory", "immunity"]:
                s[k] += random.randint(5, 20)

    for k in s:
        s[k] = clamp(s[k], 0, 100)

    return s


def create_personality() -> Dict[str, float]:
    return {
        "curiosity": random.randint(20, 90),
        "caution": random.randint(20, 90),
        "aggression": random.randint(10, 80),
        "social": random.randint(20, 90),
    }


def random_birth_condition() -> Optional[Dict]:
    if chance(PROB_CONDICION_NACIMIENTO):
        return random.choice(BIRTH_CONDITIONS).copy()
    return None


# ============================================================
# ENTIDADES
# ============================================================

@dataclass
class Plant:
    x: int
    y: int
    data: Dict


@dataclass
class Animal:
    x: int
    y: int
    data: Dict
    alive: bool = True

    @property
    def aggressive(self) -> bool:
        return bool(self.data["aggressive"])

    @property
    def strength(self) -> float:
        return float(self.data["strength"])

    def move(self, world: "World") -> None:
        if not self.alive:
            return

        # Los agresivos están más activos de noche.
        steps = 2 if self.aggressive and world.is_night() else 1

        for _ in range(steps):
            nx = self.x + random.randint(-1, 1)
            ny = self.y + random.randint(-1, 1)
            if world.walkable(nx, ny):
                self.x, self.y = nx, ny


@dataclass
class ActiveInjury:
    kind: str
    days_left: int


@dataclass
class Human:
    name: str
    sex: str
    x: int
    y: int
    age: float = 18.0
    brain: Brain = field(default_factory=Brain)

    life: float = 100.0
    hunger: float = 100.0
    thirst: float = 100.0
    energy: float = 100.0
    energy_max: float = 100.0
    alive: bool = True

    skills: Dict[str, float] = field(default_factory=dict)
    personality: Dict[str, float] = field(default_factory=create_personality)
    condition: Optional[Dict] = None

    injuries: List[ActiveInjury] = field(default_factory=list)

    inventory: Dict[str, int] = field(default_factory=lambda: {
        "food": 0,
        "raw_meat": 0,
        "wood": 0,
        "stone": 0,
    })

    memory_food: Dict[str, str] = field(default_factory=dict)
    memory_places: Dict[str, List[Tuple[int, int]]] = field(default_factory=lambda: {
        "water": [],
        "food": [],
        "danger": [],
        "shelter": [],
    })

    children: int = 0
    fitness: float = 0.0

    def __post_init__(self) -> None:
        if not self.skills:
            self.skills = create_skills(self.sex)
        if self.condition is None:
            self.condition = random_birth_condition()
        self.apply_condition_effects()

    def apply_condition_effects(self) -> None:
        if not self.condition:
            return

        effects = self.condition.get("effects", {})

        translation = {
            "speed": "speed",
            "strength": "strength",
            "spatial": "spatial",
            "immunity": "immunity",
            "memory": "memory",
        }

        for effect_key, skill_key in translation.items():
            if effect_key in effects:
                self.skills[skill_key] = clamp(self.skills.get(skill_key, 50) + effects[effect_key], 0, 100)

        if "energy_max" in effects:
            self.energy_max = clamp(self.energy_max + effects["energy_max"], 30, 100)
            self.energy = clamp(self.energy, 0, self.energy_max)

    def symbol(self) -> str:
        if self.name == "Adán":
            return color("A", "cyan")
        if self.name == "Eva":
            return color("E", "magenta")
        return color("@", "cyan" if self.sex == "hombre" else "magenta")

    def max_age(self) -> float:
        base = 65 + (self.skills.get("longevity", 50) - 50) * 0.35
        if self.sex == "mujer":
            base += 4
        return clamp(base, 35, 105)

    def process_hour(self, world: "World") -> None:
        if not self.alive:
            return

        self.age += 1 / (HOURS_PER_DAY * DAYS_PER_MONTH * MONTHS_PER_YEAR)

        self.hunger -= 0.62
        self.thirst -= 0.86
        self.energy -= 0.45

        if world.weather in ["ola_calor", "sequía"]:
            self.thirst -= 0.55
            self.energy -= 0.25

        if world.weather in ["frio_extremo", "nieve"] and not world.is_shelter(self.x, self.y):
            self.life -= 0.45
            self.energy -= 0.35

        if world.weather in ["lluvia_fuerte", "tormenta"] and not world.is_shelter(self.x, self.y):
            self.energy -= 0.25

        if self.hunger <= 0:
            self.life -= 1.1
        if self.thirst <= 0:
            self.life -= 1.6
        if self.energy <= 0:
            self.life -= 0.25

        # Lesiones activas.
        if self.injuries:
            self.energy -= 0.08 * len(self.injuries)

        self.life = clamp(self.life, 0, 100)
        self.hunger = clamp(self.hunger, 0, 100)
        self.thirst = clamp(self.thirst, 0, 100)
        self.energy = clamp(self.energy, 0, self.energy_max)

        self.fitness += 0.01
        if self.life > 60 and self.hunger > 50 and self.thirst > 50:
            self.fitness += 0.02

        if self.life <= 0 or self.age > self.max_age():
            self.alive = False
            world.log(f"{self.name} ha muerto.")

    def neural_inputs(self, world: "World") -> List[float]:
        danger_close = 1.0 if world.danger_close(self.x, self.y, 5) else 0.0
        water_close = 1.0 if world.tile_close(self.x, self.y, WATER, 4) else 0.0
        food_close = 1.0 if world.plant_close(self.x, self.y, 5) else 0.0
        shelter_close = 1.0 if world.shelter_close(self.x, self.y, 8) else 0.0
        human_close = 1.0 if world.human_close(self, 3) else 0.0

        tile = world.map[self.y][self.x]

        has_water_memory = 1.0 if self.memory_places["water"] else 0.0
        has_food_memory = 1.0 if self.memory_places["food"] else 0.0
        has_danger_memory = 1.0 if self.memory_places["danger"] else 0.0
        has_shelter_memory = 1.0 if self.memory_places["shelter"] else 0.0

        return [
            self.life / 100,
            self.hunger / 100,
            self.thirst / 100,
            self.energy / 100,
            clamp(self.age / 80, 0, 1),
            1.0 if world.is_night() else 0.0,
            world.temperature_normalized(),
            1.0 if world.weather in ["lluvia", "lluvia_fuerte", "tormenta"] else 0.0,
            1.0 if world.weather in ["frio_extremo", "nieve"] else 0.0,
            danger_close,
            water_close,
            food_close,
            shelter_close,
            human_close,
            has_water_memory,
            has_food_memory,
            has_danger_memory,
            has_shelter_memory,
            self.skills.get("strength", 50) / 100,
            self.skills.get("speed", 50) / 100,
            self.skills.get("memory", 50) / 100,
            self.skills.get("empathy", 50) / 100,
            self.skills.get("spatial", 50) / 100,
            self.skills.get("focus", 50) / 100,
            self.skills.get("mechanics", 50) / 100,
            self.personality.get("curiosity", 50) / 100,
            self.personality.get("caution", 50) / 100,
            self.personality.get("aggression", 50) / 100,
            1.0 if tile == FOREST else 0.0,
            1.0 if tile == WATER else 0.0,
        ]

    def decide_action(self, world: "World") -> str:
        # Instintos mínimos. La red neuronal sigue mandando, pero no dejamos decisiones absurdas extremas.
        if self.thirst < 12:
            return "buscar_agua"
        if self.hunger < 12 and (self.inventory["food"] > 0 or self.inventory["raw_meat"] > 0):
            return "comer"
        if world.danger_close(self.x, self.y, 2) and (self.life < 70 or self.skills.get("strength", 50) < 75):
            return "huir"
        if world.weather in ["tormenta", "tornado", "terremoto", "frio_extremo"] and not world.is_shelter(self.x, self.y):
            return "volver_refugio"
        if self.energy < 12:
            return "descansar"

        action_id = self.brain.decide(self.neural_inputs(world))
        return ACTIONS[action_id]

    def act(self, world: "World") -> None:
        if not self.alive:
            return

        action = self.decide_action(world)

        if action == "explorar":
            self.explore(world)
        elif action == "buscar_agua":
            self.seek_water(world)
        elif action == "buscar_comida":
            self.seek_food(world)
        elif action == "comer":
            self.eat(world)
        elif action == "beber":
            self.drink(world)
        elif action == "descansar":
            self.rest(world)
        elif action == "huir":
            self.flee(world)
        elif action == "volver_refugio":
            self.return_shelter(world)
        elif action == "socializar":
            self.socialize(world)
        elif action == "cazar":
            self.hunt(world)
        elif action == "recolectar":
            self.gather(world)
        elif action == "construir":
            self.build(world)

    def move_to(self, world: "World", target: Tuple[int, int], steps: int = 1) -> None:
        for _ in range(steps):
            nx, ny = step_towards(self.x, self.y, target[0], target[1])
            if world.walkable(nx, ny):
                self.x, self.y = nx, ny
                self.energy -= 0.25
            else:
                return

    def random_move(self, world: "World", radius: int = 1) -> None:
        nx = self.x + random.randint(-radius, radius)
        ny = self.y + random.randint(-radius, radius)
        if world.walkable(nx, ny):
            self.x, self.y = nx, ny
            self.energy -= 0.55

    def explore(self, world: "World") -> None:
        radius = 2 if self.skills.get("spatial", 50) > 70 else 1
        self.random_move(world, radius=radius)
        self.remember_surroundings(world)

        risk = 0.01
        if world.is_night():
            risk += 0.05
        if world.weather in ["niebla", "lluvia_fuerte", "tormenta", "nieve"]:
            risk += 0.04
        if self.energy < 25:
            risk += 0.03

        self.accident_risk(world, risk, ["corte", "torcedura", "fractura"])

    def seek_water(self, world: "World") -> None:
        pos = world.find_nearest_tile(self.x, self.y, WATER, 28)
        if pos:
            self.memory_places["water"].append(pos)
            self.move_to(world, pos, steps=2)
            if distance((self.x, self.y), pos) <= 1.5:
                self.drink(world)
        elif self.memory_places["water"]:
            self.move_to(world, random.choice(self.memory_places["water"]), steps=2)
        else:
            self.explore(world)

    def seek_food(self, world: "World") -> None:
        plant = world.find_nearest_plant(self.x, self.y, 22)
        if plant:
            self.memory_places["food"].append((plant.x, plant.y))
            self.move_to(world, (plant.x, plant.y), steps=2)
            if distance((self.x, self.y), (plant.x, plant.y)) <= 1.5:
                self.gather(world)
        elif self.memory_places["food"]:
            self.move_to(world, random.choice(self.memory_places["food"]), steps=2)
        else:
            self.explore(world)

    def eat(self, world: "World") -> None:
        if self.inventory["food"] > 0:
            self.inventory["food"] -= 1
            self.hunger = clamp(self.hunger + 28, 0, 100)
            self.energy = clamp(self.energy + 3, 0, self.energy_max)
            self.fitness += 0.2
            return

        if self.inventory["raw_meat"] > 0:
            self.inventory["raw_meat"] -= 1
            self.hunger = clamp(self.hunger + 32, 0, 100)
            self.energy = clamp(self.energy + 2, 0, self.energy_max)

            if chance(0.30):
                self.apply_injury(world, "enfermedad")
                self.memory_food["carne_cruda"] = "riesgo"
            else:
                self.memory_food["carne_cruda"] = "comestible_con_riesgo"
            return

        self.seek_food(world)

    def drink(self, world: "World") -> None:
        if world.map[self.y][self.x] == WATER or world.tile_close(self.x, self.y, WATER, 1):
            self.thirst = clamp(self.thirst + 45, 0, 100)
            self.energy = clamp(self.energy + 2, 0, self.energy_max)
            self.memory_places["water"].append((self.x, self.y))
            self.fitness += 0.2
        else:
            self.seek_water(world)

    def rest(self, world: "World") -> None:
        bonus = 1.0 if world.is_shelter(self.x, self.y) else 0.45
        if world.is_night():
            bonus += 0.25

        self.energy = clamp(self.energy + 6.0 * bonus, 0, self.energy_max)
        self.life = clamp(self.life + 0.35 * bonus, 0, 100)

    def flee(self, world: "World") -> None:
        threat = world.nearest_danger(self.x, self.y, 6)
        if not threat:
            self.explore(world)
            return

        dx = self.x - threat.x
        dy = self.y - threat.y

        tx = int(clamp(self.x + (1 if dx >= 0 else -1) * 5, 0, MAP_W - 1))
        ty = int(clamp(self.y + (1 if dy >= 0 else -1) * 5, 0, MAP_H - 1))
        self.move_to(world, (tx, ty), steps=3)
        self.energy -= 1.4

    def return_shelter(self, world: "World") -> None:
        pos = world.find_nearest_shelter(self.x, self.y, 90)
        if pos:
            self.memory_places["shelter"].append(pos)
            self.move_to(world, pos, steps=3)
        elif self.memory_places["shelter"]:
            self.move_to(world, random.choice(self.memory_places["shelter"]), steps=3)
        else:
            self.explore(world)

    def socialize(self, world: "World") -> None:
        others = world.nearby_humans(self, 3)
        if not others:
            self.return_shelter(world)
            return

        other = random.choice(others)
        share_prob = 0.25 + self.skills.get("empathy", 50) / 220

        for k, v in self.memory_food.items():
            if chance(share_prob):
                other.memory_food[k] = v

        for key in self.memory_places:
            if self.memory_places[key] and chance(share_prob):
                other.memory_places[key].append(random.choice(self.memory_places[key]))

        self.energy = clamp(self.energy + 0.5, 0, self.energy_max)
        self.fitness += 0.05

    def hunt(self, world: "World") -> None:
        animal = world.find_nearest_animal(self.x, self.y, 10)
        if not animal:
            self.explore(world)
            return

        self.move_to(world, (animal.x, animal.y), steps=2)

        if distance((self.x, self.y), (animal.x, animal.y)) > 1.8:
            return

        strength = self.skills.get("strength", 50)
        speed = self.skills.get("speed", 50)
        focus = self.skills.get("focus", 50)

        success = 0.20 + (strength + speed + focus) / 430
        success -= animal.strength / 150

        if animal.aggressive:
            success -= 0.15

        success = clamp(success, 0.04, 0.85)
        self.energy -= 5

        if chance(success):
            animal.alive = False
            self.inventory["raw_meat"] += max(1, int(animal.data["meat"] // 20))
            self.fitness += 0.6
            world.log(f"{self.name} cazó un/a {animal.data['name']}.")
        else:
            risk = 0.08 + animal.strength / 180
            if animal.aggressive:
                risk += 0.15
            self.accident_risk(world, risk, ["corte", "mordedura", "herida_grave", "fractura"])

    def gather(self, world: "World") -> None:
        plant = world.plant_at_or_near(self.x, self.y, 1)
        if not plant:
            self.seek_food(world)
            return

        if plant in world.plants:
            world.plants.remove(plant)

        pid = plant.data["id"]

        if plant.data["poison"]:
            self.hunger = clamp(self.hunger + 3, 0, 100)
            self.apply_injury(world, "intoxicacion")
            self.memory_food[pid] = "venenosa"
            self.memory_places["danger"].append((plant.x, plant.y))
            return

        if chance(plant.data["bad"]):
            self.hunger = clamp(self.hunger + plant.data["food"] * 0.4, 0, 100)
            self.apply_injury(world, "enfermedad")
            self.memory_food[pid] = "mala_a_veces"
        else:
            self.inventory["food"] += 1
            self.memory_food[pid] = "segura"
            self.fitness += 0.25

    def build(self, world: "World") -> None:
        # Construcción inicial: chozas. Más adelante añadiremos casas modernas y edificios.
        tile = world.map[self.y][self.x]

        if tile == WATER:
            self.explore(world)
            return

        if world.tile_close(self.x, self.y, FOREST, 4):
            self.inventory["wood"] += 1
            self.energy -= 2

        if self.inventory["wood"] >= 8 and tile in [GRASS, FOREST]:
            world.map[self.y][self.x] = HUT
            self.inventory["wood"] -= 8
            world.tribe_knowledge["choza"] = True
            world.buildings += 1
            world.log(f"{self.name} construyó una choza.")
            self.fitness += 1.0
        else:
            pos = world.find_nearest_tile(self.x, self.y, FOREST, 18)
            if pos:
                self.move_to(world, pos, steps=2)
            else:
                self.explore(world)

    def remember_surroundings(self, world: "World") -> None:
        for dy in range(-4, 5):
            for dx in range(-4, 5):
                nx = self.x + dx
                ny = self.y + dy
                if not world.in_bounds(nx, ny):
                    continue

                tile = world.map[ny][nx]
                if tile == WATER:
                    self.memory_places["water"].append((nx, ny))
                elif tile in [CAVE, HUT, HOUSE, BUILDING]:
                    self.memory_places["shelter"].append((nx, ny))

        for plant in world.plants:
            if distance((self.x, self.y), (plant.x, plant.y)) <= 5:
                self.memory_places["food"].append((plant.x, plant.y))

        for animal in world.animals:
            if animal.alive and animal.aggressive and distance((self.x, self.y), (animal.x, animal.y)) <= 6:
                self.memory_places["danger"].append((animal.x, animal.y))

        # Limitar memoria para que no crezca infinito.
        for key in self.memory_places:
            if len(self.memory_places[key]) > 80:
                self.memory_places[key] = self.memory_places[key][-80:]

    def accident_risk(self, world: "World", base: float, possible: List[str]) -> None:
        risk = base
        risk += max(0, 30 - self.energy) / 400
        risk += max(0, 30 - self.life) / 400
        risk -= self.skills.get("speed", 50) / 1300
        risk -= self.skills.get("spatial", 50) / 1400

        if self.condition:
            effects = self.condition.get("effects", {})
            risk += effects.get("accident_risk", 0)
            risk += effects.get("fall_risk", 0)
            risk += effects.get("fracture_risk", 0) * 0.4

        if world.weather in ["tormenta", "lluvia_fuerte", "niebla", "nieve"]:
            risk += 0.04

        risk = clamp(risk, 0, 0.90)

        if chance(risk):
            self.apply_injury(world, random.choice(possible))

    def apply_injury(self, world: "World", kind: str) -> None:
        injury = INJURIES[kind]

        self.life = clamp(self.life + injury.get("life", 0), 0, 100)
        self.energy = clamp(self.energy + injury.get("energy", 0), 0, self.energy_max)
        self.hunger = clamp(self.hunger + injury.get("hunger", 0), 0, 100)

        self.injuries.append(ActiveInjury(kind, injury.get("days", 1)))
        world.log(f"{self.name} sufre {kind}.")

        if "infection" in injury and chance(injury["infection"]):
            self.apply_injury(world, "enfermedad")

        if injury.get("may_disable", False) and chance(0.12):
            self.condition = {
                "id": "movilidad_reducida_adquirida",
                "name": "movilidad reducida adquirida",
                "effects": {"speed": -15, "energy_max": -8, "fall_risk": 0.10},
            }
            self.apply_condition_effects()
            world.log(f"{self.name} queda con movilidad reducida adquirida.")

        if self.life <= 0:
            self.alive = False
            world.log(f"{self.name} ha muerto.")


# ============================================================
# MUNDO
# ============================================================

class World:
    def __init__(self):
        self.map: List[List[str]] = [[GRASS for _ in range(MAP_W)] for _ in range(MAP_H)]
        self.hour = 7
        self.day = 1
        self.month = 1
        self.year = 1

        self.weather = "soleado"
        self.temperature = 18.0

        self.humans: List[Human] = []
        self.animals: List[Animal] = []
        self.plants: List[Plant] = []

        self.events: List[str] = []
        self.birth_count = 0
        self.buildings = 0

        self.tribe_knowledge = {
            "fuego": False,
            "cocina": False,
            "herramientas_piedra": False,
            "choza": False,
            "almacen": False,
            "agricultura": False,
            "casa_madera": False,
            "casa_piedra": False,
            "metalurgia": False,
            "ingenieria": False,
            "electricidad": False,
        }

        self.cave_pos = (MAP_W // 2, MAP_H // 2)

    # ----------------------------
    # Generación del mundo
    # ----------------------------

    def generate(self) -> None:
        self.generate_river()
        self.generate_forests()
        self.generate_mountains()
        self.generate_swamps()
        self.place_cave()
        self.spawn_plants(900)
        self.spawn_animals(260)

        cx, cy = self.cave_pos

        adam = Human("Adán", "hombre", cx, cy, age=18.0)
        eve = Human("Eva", "mujer", cx + 1, cy, age=18.0)

        # Memoria inicial mínima: conocen la cueva.
        for h in [adam, eve]:
            h.memory_places["shelter"].append(self.cave_pos)
            h.remember_surroundings(self)

        self.humans = [adam, eve]

        self.log("El mundo ha sido creado. Adán y Eva aparecen en la cueva.")

    def generate_river(self) -> None:
        x = random.randint(35, MAP_W - 35)

        for y in range(MAP_H):
            width = random.choice([1, 2, 2, 3])
            for w in range(width):
                xx = clamp(x + w, 0, MAP_W - 1)
                self.map[y][int(xx)] = WATER

            if chance(0.45):
                x += random.choice([-1, 0, 1])
                x = int(clamp(x, 3, MAP_W - 4))

    def generate_forests(self) -> None:
        for _ in range(22):
            cx = random.randint(0, MAP_W - 1)
            cy = random.randint(0, MAP_H - 1)
            radius = random.randint(8, 24)

            for y in range(max(0, cy - radius), min(MAP_H, cy + radius)):
                for x in range(max(0, cx - radius), min(MAP_W, cx + radius)):
                    d = abs(x - cx) + abs(y - cy)
                    if d < radius and chance(0.72) and self.map[y][x] == GRASS:
                        self.map[y][x] = FOREST

    def generate_mountains(self) -> None:
        for _ in range(8):
            cx = random.randint(0, MAP_W - 1)
            cy = random.randint(0, MAP_H - 1)
            radius = random.randint(6, 16)

            for y in range(max(0, cy - radius), min(MAP_H, cy + radius)):
                for x in range(max(0, cx - radius), min(MAP_W, cx + radius)):
                    d = abs(x - cx) + abs(y - cy)
                    if d < radius and chance(0.65) and self.map[y][x] in [GRASS, FOREST]:
                        self.map[y][x] = MOUNTAIN

    def generate_swamps(self) -> None:
        # Pantanos cerca de algunos tramos del río.
        water_cells = [(x, y) for y in range(MAP_H) for x in range(MAP_W) if self.map[y][x] == WATER]
        for _ in range(7):
            wx, wy = random.choice(water_cells)
            radius = random.randint(5, 12)

            for y in range(max(0, wy - radius), min(MAP_H, wy + radius)):
                for x in range(max(0, wx - radius), min(MAP_W, wx + radius)):
                    if self.map[y][x] == GRASS and chance(0.35):
                        self.map[y][x] = SWAMP

    def place_cave(self) -> None:
        # Intentamos poner la cueva cerca del centro y no encima del agua.
        for _ in range(1000):
            x = random.randint(MAP_W // 2 - 25, MAP_W // 2 + 25)
            y = random.randint(MAP_H // 2 - 25, MAP_H // 2 + 25)
            if self.map[y][x] != WATER:
                self.map[y][x] = CAVE
                self.cave_pos = (x, y)
                return

        self.map[MAP_H // 2][MAP_W // 2] = CAVE
        self.cave_pos = (MAP_W // 2, MAP_H // 2)

    def spawn_plants(self, amount: int) -> None:
        for _ in range(amount):
            x, y = self.random_land_cell()
            tile = self.map[y][x]

            # Más plantas en bosque y cerca del río.
            if tile == FOREST:
                if chance(0.85):
                    self.plants.append(Plant(x, y, random.choice(PLANTS)))
            elif self.tile_close(x, y, WATER, 3):
                if chance(0.55):
                    self.plants.append(Plant(x, y, random.choice(PLANTS)))
            elif tile == GRASS:
                if chance(0.20):
                    self.plants.append(Plant(x, y, random.choice(PLANTS)))

    def spawn_animals(self, amount: int) -> None:
        for _ in range(amount):
            x, y = self.random_land_cell()
            tile = self.map[y][x]

            # Pez cerca del río.
            if self.tile_close(x, y, WATER, 1):
                data = next(a for a in ANIMALS if a["id"] == "pez")
            elif tile == FOREST:
                data = random.choice(ANIMALS)
            else:
                data = random.choice(ANIMALS[:6])

            self.animals.append(Animal(x, y, data))

    def random_land_cell(self) -> Tuple[int, int]:
        while True:
            x = random.randint(0, MAP_W - 1)
            y = random.randint(0, MAP_H - 1)
            if self.map[y][x] != WATER:
                return x, y

    # ----------------------------
    # Tiempo, clima y desastres
    # ----------------------------

    def season(self) -> str:
        if self.month in [3, 4, 5]:
            return "primavera"
        if self.month in [6, 7, 8]:
            return "verano"
        if self.month in [9, 10, 11]:
            return "otoño"
        return "invierno"

    def is_night(self) -> bool:
        return self.hour >= 20 or self.hour <= 5

    def update_time(self) -> None:
        self.hour += 1

        if self.hour >= 24:
            self.hour = 0
            self.day += 1
            self.daily_update()

        if self.day > DAYS_PER_MONTH:
            self.day = 1
            self.month += 1

        if self.month > MONTHS_PER_YEAR:
            self.month = 1
            self.year += 1

    def daily_update(self) -> None:
        # Plantas vuelven a crecer poco a poco.
        season = self.season()
        grow = {
            "primavera": 35,
            "verano": 25,
            "otoño": 20,
            "invierno": 6,
        }[season]
        self.spawn_plants(grow)

        # Animales se recuperan algo.
        if len([a for a in self.animals if a.alive]) < 220:
            self.spawn_animals(12)

        # Procesar días de lesiones.
        for h in self.humans:
            new_injuries = []
            for inj in h.injuries:
                inj.days_left -= 1
                if inj.days_left > 0:
                    new_injuries.append(inj)
            h.injuries = new_injuries

        self.try_reproduction()
        self.try_technology_discovery()

    def update_weather(self) -> None:
        season = self.season()

        if season == "primavera":
            options = [
                ("soleado", 0.35),
                ("nublado", 0.20),
                ("lluvia", 0.25),
                ("lluvia_fuerte", 0.08),
                ("tormenta", 0.07),
                ("niebla", 0.05),
            ]
            self.temperature = random.uniform(8, 22)

        elif season == "verano":
            options = [
                ("soleado", 0.45),
                ("nublado", 0.12),
                ("lluvia", 0.10),
                ("tormenta", 0.12),
                ("ola_calor", 0.12),
                ("sequía", 0.09),
            ]
            self.temperature = random.uniform(18, 38)

        elif season == "otoño":
            options = [
                ("soleado", 0.25),
                ("nublado", 0.20),
                ("lluvia", 0.25),
                ("lluvia_fuerte", 0.12),
                ("tormenta", 0.08),
                ("niebla", 0.10),
            ]
            self.temperature = random.uniform(5, 20)

        else:
            options = [
                ("soleado", 0.20),
                ("nublado", 0.20),
                ("lluvia", 0.12),
                ("nieve", 0.20),
                ("frio_extremo", 0.16),
                ("niebla", 0.12),
            ]
            self.temperature = random.uniform(-8, 10)

        r = random.random()
        acc = 0.0
        for name, p in options:
            acc += p
            if r <= acc:
                self.weather = name
                break

        # Desastres raros según clima/estación/bioma.
        self.maybe_disaster()

    def maybe_disaster(self) -> None:
        season = self.season()

        # Tornados: más probables en pradera, primavera/verano con tormentas.
        if self.weather == "tormenta" and season in ["primavera", "verano"] and chance(0.006):
            self.weather = "tornado"
            self.log("Se forma un tornado.")

        # Terremoto raro, más daño en montaña.
        if chance(0.0015):
            self.weather = "terremoto"
            self.log("Hay un terremoto.")

        # Inundación si lluvia fuerte/tormenta.
        if self.weather in ["lluvia_fuerte", "tormenta"] and chance(0.007):
            self.log("Hay riesgo de inundación cerca del río.")
            for h in self.humans:
                if self.tile_close(h.x, h.y, WATER, 2) and not self.is_shelter(h.x, h.y):
                    h.apply_injury(self, "ahogamiento")

        # Incendio en bosque durante ola de calor o tormenta.
        if self.weather in ["ola_calor", "tormenta"] and season == "verano" and chance(0.004):
            self.forest_fire()

    def forest_fire(self) -> None:
        forest_cells = [(x, y) for y in range(MAP_H) for x in range(MAP_W) if self.map[y][x] == FOREST]
        if not forest_cells:
            return

        cx, cy = random.choice(forest_cells)
        radius = random.randint(4, 12)

        for y in range(max(0, cy - radius), min(MAP_H, cy + radius)):
            for x in range(max(0, cx - radius), min(MAP_W, cx + radius)):
                if self.map[y][x] == FOREST and chance(0.65):
                    self.map[y][x] = GRASS

        self.log("Un incendio forestal arrasa parte del bosque.")

        for h in self.humans:
            if distance((h.x, h.y), (cx, cy)) < radius + 2:
                h.apply_injury(self, "herida_grave")

    def temperature_normalized(self) -> float:
        return clamp((self.temperature + 15) / 55, 0, 1)

    # ----------------------------
    # Civilización
    # ----------------------------

    def try_technology_discovery(self) -> None:
        alive = self.living_humans()
        if not alive:
            return

        # Fuego: puede descubrirse cuando hay tormentas, frío o necesidad.
        if not self.tribe_knowledge["fuego"]:
            pressure = 0.0
            if self.weather in ["frio_extremo", "nieve"]:
                pressure += 0.03
            if any(h.life < 60 for h in alive):
                pressure += 0.01
            if len(alive) >= 2:
                pressure += 0.005

            if chance(pressure):
                self.tribe_knowledge["fuego"] = True
                self.log("La tribu descubre el fuego.")

        # Cocina: si ya conocen fuego y han sufrido enfermedades por comida.
        if self.tribe_knowledge["fuego"] and not self.tribe_knowledge["cocina"]:
            if chance(0.01):
                self.tribe_knowledge["cocina"] = True
                self.log("La tribu descubre que cocinar reduce enfermedades.")

        # Herramientas de piedra.
        if not self.tribe_knowledge["herramientas_piedra"]:
            mechanics_avg = sum(h.skills.get("mechanics", 50) for h in alive) / len(alive)
            if mechanics_avg > 60 and chance(0.008):
                self.tribe_knowledge["herramientas_piedra"] = True
                self.log("La tribu descubre herramientas de piedra.")

        # Primeras casas futuras: todavía no construimos ciudades, pero dejamos desbloqueos.
        if self.buildings >= 3 and not self.tribe_knowledge["almacen"]:
            if chance(0.02):
                self.tribe_knowledge["almacen"] = True
                self.log("La tribu aprende a almacenar recursos.")

    # ----------------------------
    # Búsquedas y comprobaciones
    # ----------------------------

    def in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < MAP_W and 0 <= y < MAP_H

    def walkable(self, x: int, y: int) -> bool:
        if not self.in_bounds(x, y):
            return False
        return self.map[y][x] != MOUNTAIN

    def is_shelter(self, x: int, y: int) -> bool:
        if not self.in_bounds(x, y):
            return False
        return self.map[y][x] in [CAVE, HUT, HOUSE, BUILDING]

    def tile_close(self, x: int, y: int, tile: str, radius: int) -> bool:
        return self.find_nearest_tile(x, y, tile, radius) is not None

    def find_nearest_tile(self, x: int, y: int, tile: str, radius: int) -> Optional[Tuple[int, int]]:
        best = None
        best_d = 999999

        for yy in range(max(0, y - radius), min(MAP_H, y + radius + 1)):
            for xx in range(max(0, x - radius), min(MAP_W, x + radius + 1)):
                if self.map[yy][xx] == tile:
                    d = distance((x, y), (xx, yy))
                    if d < best_d:
                        best = (xx, yy)
                        best_d = d

        return best

    def find_nearest_shelter(self, x: int, y: int, radius: int) -> Optional[Tuple[int, int]]:
        best = None
        best_d = 999999

        for yy in range(max(0, y - radius), min(MAP_H, y + radius + 1)):
            for xx in range(max(0, x - radius), min(MAP_W, x + radius + 1)):
                if self.is_shelter(xx, yy):
                    d = distance((x, y), (xx, yy))
                    if d < best_d:
                        best = (xx, yy)
                        best_d = d

        return best

    def shelter_close(self, x: int, y: int, radius: int) -> bool:
        return self.find_nearest_shelter(x, y, radius) is not None

    def find_nearest_plant(self, x: int, y: int, radius: int) -> Optional[Plant]:
        best = None
        best_d = 999999

        for p in self.plants:
            d = distance((x, y), (p.x, p.y))
            if d <= radius and d < best_d:
                best = p
                best_d = d

        return best

    def plant_close(self, x: int, y: int, radius: int) -> bool:
        return self.find_nearest_plant(x, y, radius) is not None

    def plant_at_or_near(self, x: int, y: int, radius: int) -> Optional[Plant]:
        for p in self.plants:
            if distance((x, y), (p.x, p.y)) <= radius + 0.5:
                return p
        return None

    def find_nearest_animal(self, x: int, y: int, radius: int) -> Optional[Animal]:
        best = None
        best_d = 999999

        for a in self.animals:
            if not a.alive:
                continue
            d = distance((x, y), (a.x, a.y))
            if d <= radius and d < best_d:
                best = a
                best_d = d

        return best

    def nearest_danger(self, x: int, y: int, radius: int) -> Optional[Animal]:
        best = None
        best_d = 999999

        for a in self.animals:
            if not a.alive or not a.aggressive:
                continue
            d = distance((x, y), (a.x, a.y))
            if d <= radius and d < best_d:
                best = a
                best_d = d

        return best

    def danger_close(self, x: int, y: int, radius: int) -> bool:
        return self.nearest_danger(x, y, radius) is not None

    def nearby_humans(self, human: Human, radius: int) -> List[Human]:
        return [
            h for h in self.humans
            if h.alive and h is not human and distance((human.x, human.y), (h.x, h.y)) <= radius
        ]

    def human_close(self, human: Human, radius: int) -> bool:
        return bool(self.nearby_humans(human, radius))

    def living_humans(self) -> List[Human]:
        return [h for h in self.humans if h.alive]

    # ----------------------------
    # Reproducción
    # ----------------------------

    def try_reproduction(self) -> None:
        adults = [
            h for h in self.living_humans()
            if h.age >= 16 and h.life > 70 and h.hunger > 60 and h.thirst > 60
        ]

        men = [h for h in adults if h.sex == "hombre"]
        women = [h for h in adults if h.sex == "mujer"]

        if not men or not women:
            return

        # Necesitan refugio cercano y estabilidad.
        if not any(self.shelter_close(h.x, h.y, 8) for h in adults):
            return

        # Para no explotar la población al principio.
        if len(self.living_humans()) > 80:
            return

        for woman in women:
            if woman.age < 16 or woman.age > 42:
                continue

            if chance(PROB_REPRODUCCION_DIARIA_BASE):
                father = random.choice(men)
                self.birth_count += 1

                sex = random_sex()
                name = f"Bebé_{self.birth_count}"
                brain = mix_brains(father.brain, woman.brain)

                baby = Human(
                    name=name,
                    sex=sex,
                    x=woman.x,
                    y=woman.y,
                    age=0.0,
                    brain=brain,
                )

                # Parte del conocimiento básico se transmite.
                for parent in [father, woman]:
                    for k, v in parent.memory_food.items():
                        if chance(0.35):
                            baby.memory_food[k] = v

                    for place_key in parent.memory_places:
                        if parent.memory_places[place_key] and chance(0.25):
                            baby.memory_places[place_key].append(random.choice(parent.memory_places[place_key]))

                self.humans.append(baby)
                father.children += 1
                woman.children += 1
                father.fitness += 2.0
                woman.fitness += 2.0

                self.log(f"Nace {name}, sexo: {sex}.")

    # ----------------------------
    # Eventos principales
    # ----------------------------

    def step(self) -> None:
        # Clima cambia cada 6 horas.
        if self.hour % 6 == 0:
            self.update_weather()

        # Animales.
        for a in self.animals:
            a.move(self)

        # Humanos.
        for h in list(self.humans):
            h.process_hour(self)

        for h in list(self.humans):
            h.act(self)

        # Ataques de animales cercanos.
        self.resolve_animal_attacks()

        # Tiempo.
        self.update_time()

        # Limpieza.
        self.animals = [a for a in self.animals if a.alive]
        self.humans = [h for h in self.humans if h.alive]

    def resolve_animal_attacks(self) -> None:
        for h in self.living_humans():
            for a in self.animals:
                if not a.alive or not a.aggressive:
                    continue

                d = distance((h.x, h.y), (a.x, a.y))

                # De noche atacan más.
                attack_radius = 1.8 if self.is_night() else 1.2
                attack_prob = 0.08 if self.is_night() else 0.03

                if d <= attack_radius and chance(attack_prob):
                    if a.data["id"] == "serpiente":
                        h.apply_injury(self, "mordedura")
                    elif a.strength > 70:
                        h.apply_injury(self, random.choice(["herida_grave", "fractura", "mordedura"]))
                    else:
                        h.apply_injury(self, random.choice(["mordedura", "corte", "herida_grave"]))

                    h.memory_places["danger"].append((a.x, a.y))
                    self.log(f"{h.name} fue atacado por {a.data['name']}.")

    # ----------------------------
    # Render
    # ----------------------------

    def render(self) -> None:
        clear_screen()

        living = self.living_humans()
        center = living[0] if living else None

        print(color("EDEN SIM 0.1 — IA neuronal evolutiva", "cyan"))
        print(
            f"Año {self.year} | Mes {self.month} | Día {self.day} | Hora {self.hour:02d}:00 | "
            f"Estación: {self.season()} | Clima: {self.weather} | Temp: {self.temperature:.1f}°C"
        )
        print(
            f"Humanos vivos: {len(living)} | Animales: {len(self.animals)} | Plantas: {len(self.plants)} | "
            f"Edificios: {self.buildings}"
        )

        techs = [k for k, v in self.tribe_knowledge.items() if v]
        print(f"Conocimiento: {', '.join(techs) if techs else 'ninguno'}")
        print()

        if center:
            self.render_view(center.x, center.y)
        else:
            print(color("La humanidad se ha extinguido.", "red"))

        print()
        print(color("Humanos destacados:", "yellow"))

        for h in sorted(living, key=lambda x: x.fitness, reverse=True)[:8]:
            cond = h.condition["name"] if h.condition else "sin condición"
            print(
                f"{h.name:10s} {h.sex:6s} edad {h.age:5.1f} | "
                f"vida {h.life:5.1f} hambre {h.hunger:5.1f} sed {h.thirst:5.1f} energía {h.energy:5.1f} | "
                f"hijos {h.children} | {cond}"
            )

        print()
        print(color("Eventos recientes:", "yellow"))
        for e in self.events[-8:]:
            print(" - " + e)

    def render_view(self, cx: int, cy: int) -> None:
        min_x = max(0, cx - VIEW_RADIUS)
        max_x = min(MAP_W - 1, cx + VIEW_RADIUS)
        min_y = max(0, cy - VIEW_RADIUS)
        max_y = min(MAP_H - 1, cy + VIEW_RADIUS)

        human_pos = {(h.x, h.y): h for h in self.living_humans()}
        animal_pos = {(a.x, a.y): a for a in self.animals if a.alive}
        plant_pos = {(p.x, p.y): p for p in self.plants}

        for y in range(min_y, max_y + 1):
            row = ""
            for x in range(min_x, max_x + 1):
                if (x, y) in human_pos:
                    row += human_pos[(x, y)].symbol()
                elif (x, y) in animal_pos:
                    a = animal_pos[(x, y)]
                    row += color("X" if a.aggressive else "a", "red" if a.aggressive else "yellow")
                elif (x, y) in plant_pos:
                    p = plant_pos[(x, y)]
                    row += color("f", "red" if p.data["poison"] else "lightgreen")
                else:
                    row += terrain_symbol(self.map[y][x])
            print(row)

        print()
        print("Leyenda: A Adán | E Eva | @ humanos | X animal agresivo | a animal | f planta | ~ río | T bosque | C cueva | h choza")

    def log(self, msg: str) -> None:
        self.events.append(msg)
        if len(self.events) > 60:
            self.events = self.events[-60:]


# ============================================================
# EJECUCIÓN
# ============================================================

def main() -> None:
    random.seed()
    np.random.seed()

    world = World()
    world.generate()

    total_hours = MAX_DAYS * HOURS_PER_DAY

    try:
        for _ in range(total_hours):
            if not world.living_humans():
                world.render()
                print()
                print(color("Fin: la humanidad se ha extinguido.", "red"))
                return

            world.step()

            # Render cada 3 horas para que no parpadee demasiado.
            if world.hour % 3 == 0:
                world.render()
                time.sleep(TURN_DELAY)

        world.render()
        print()
        print(color("Simulación terminada.", "cyan"))

    except KeyboardInterrupt:
        world.render()
        print()
        print(color("Simulación detenida por el usuario.", "yellow"))


if __name__ == "__main__":
    main()
