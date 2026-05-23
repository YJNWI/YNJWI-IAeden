#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
eden_sim_v1_11.py

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
    python eden_sim_v1_11.py

Controles:
    El simulador corre solo.
    Pulsa CTRL + C para detenerlo.

Notas:
    Esta es la versión 1.11 con corrección de help, nombres por generación, búsqueda por número de humano y estadísticas de muertos.
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
import threading
import queue
import sys
import select
import curses
import locale
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================

MAP_W = 256
MAP_H = 256

VIEW_RADIUS = 12
TURN_DELAY = 0.20
SHOW_START_LEGEND = True
WAIT_AFTER_START_LEGEND = True


HOURS_PER_DAY = 24
DAYS_PER_MONTH = 30
MONTHS_PER_YEAR = 12

MAX_DAYS = 365

PROB_CONDICION_NACIMIENTO = 0.04
PROB_REPRODUCCION_DIARIA_BASE = 0.08

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
    "fabricar",
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

    def outputs(self, inputs: List[float]) -> np.ndarray:
        x = np.array(inputs, dtype=float)

        if x.shape[0] != INPUT_SIZE:
            raise ValueError(f"Brain esperaba {INPUT_SIZE} entradas, recibió {x.shape[0]}.")

        h = np.tanh(x @ self.w1 + self.b1)
        return h @ self.w2 + self.b2

    def decide(self, inputs: List[float]) -> int:
        out = self.outputs(inputs)

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
    {"id": "manzana_silvestre", "name": "manzana silvestre", "symbol": "m", "poison": False, "food": 24, "bad": 0.10},
    {"id": "bayas_azules", "name": "bayas azules", "symbol": "u", "poison": False, "food": 16, "bad": 0.15},
    {"id": "higos", "name": "higos", "symbol": "i", "poison": False, "food": 22, "bad": 0.08},
    {"id": "nueces", "name": "nueces", "symbol": "n", "poison": False, "food": 18, "bad": 0.05},
    {"id": "raices_comestibles", "name": "raíces comestibles", "symbol": "r", "poison": False, "food": 18, "bad": 0.20},
    {"id": "bayas_rojas_venenosas", "name": "bayas rojas venenosas", "symbol": "v", "poison": True, "food": 4, "bad": 1.00},
    {"id": "hongos_morados", "name": "hongos morados", "symbol": "q", "poison": True, "food": 4, "bad": 1.00},
    {"id": "fruto_negro_brillante", "name": "fruto negro brillante", "symbol": "z", "poison": True, "food": 4, "bad": 1.00},
    {"id": "raiz_amarga", "name": "raíz amarga", "symbol": "y", "poison": True, "food": 4, "bad": 1.00},
]

ANIMALS = [
    {"id": "conejo", "name": "conejo", "symbol": "c", "aggressive": False, "strength": 8, "meat": 18, "raw_risk": 0.20, "night": False},
    {"id": "ciervo", "name": "ciervo", "symbol": "d", "aggressive": False, "strength": 22, "meat": 45, "raw_risk": 0.25, "night": False},
    {"id": "pez", "name": "pez", "symbol": "p", "aggressive": False, "strength": 2, "meat": 14, "raw_risk": 0.30, "night": False},
    {"id": "gallina", "name": "gallina salvaje", "symbol": "g", "aggressive": False, "strength": 6, "meat": 16, "raw_risk": 0.15, "night": False},
    {"id": "cabra", "name": "cabra salvaje", "symbol": "k", "aggressive": False, "strength": 18, "meat": 35, "raw_risk": 0.20, "night": False},
    {"id": "lobo", "name": "lobo", "symbol": "L", "aggressive": True, "strength": 45, "meat": 30, "raw_risk": 0.40, "night": True},
    {"id": "oso", "name": "oso", "symbol": "O", "aggressive": True, "strength": 85, "meat": 85, "raw_risk": 0.50, "night": True},
    {"id": "serpiente", "name": "serpiente", "symbol": "V", "aggressive": True, "strength": 35, "meat": 6, "raw_risk": 0.70, "night": True},
    {"id": "jabali", "name": "jabalí", "symbol": "J", "aggressive": True, "strength": 55, "meat": 50, "raw_risk": 0.35, "night": True},
    {"id": "pantera", "name": "pantera", "symbol": "P", "aggressive": True, "strength": 70, "meat": 45, "raw_risk": 0.45, "night": True},
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



def inherit_skills(father: "Human", mother: "Human", sex: str) -> Dict[str, float]:
    """Hereda fortalezas de ambos padres, pero también defectos si ambos son bajos."""
    base = create_skills(sex)
    skills: Dict[str, float] = {}

    for key in base:
        f = father.skills.get(key, 50)
        m = mother.skills.get(key, 50)
        best = max(f, m)
        avg = (f + m) / 2
        natural = base[key]

        value = avg * 0.60 + best * 0.25 + natural * 0.15

        if f < 35 and m < 35:
            value -= random.uniform(4, 12)

        value += random.gauss(0, 5)
        skills[key] = clamp(value, 0, 100)

    return skills


def inherit_personality(father: "Human", mother: "Human") -> Dict[str, float]:
    p = {}
    natural = create_personality()
    for key in natural:
        value = (father.personality.get(key, 50) + mother.personality.get(key, 50)) / 2
        value = value * 0.75 + natural[key] * 0.25 + random.gauss(0, 6)
        p[key] = clamp(value, 0, 100)
    return p


def inherit_learned_actions(father: "Human", mother: "Human") -> Dict[str, float]:
    learned = {}
    for action in ACTIONS:
        f = father.learned_actions.get(action, 0.0)
        m = mother.learned_actions.get(action, 0.0)
        learned[action] = clamp((f + m) / 2 + random.gauss(0, 0.25), -4.0, 4.0)
    return learned


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
    person_id: int = 0
    father_label: str = ""
    mother_label: str = ""
    child_number: int = 0
    generation: int = 1
    weapon: str = "ninguna"
    weapon_power: float = 0.0
    learned_actions: Dict[str, float] = field(default_factory=dict)
    known_techs: Dict[str, float] = field(default_factory=dict)
    death_reason: str = ""

    def label(self) -> str:
        if self.name in ["Adán", "Eva"]:
            return self.name

        # Formato v1.11:
        # padre,madre Npersona Ngeneracion
        # Ejemplo: Adán,Eva 3 2
        # Ejemplo avanzado: 8,5 17 4
        if self.father_label or self.mother_label:
            if self.mother_label:
                return f"{self.father_label},{self.mother_label} {self.person_id} {self.generation}"
            return f"{self.father_label} {self.person_id} {self.generation}"

        return self.name


    def __post_init__(self) -> None:
        if not self.skills:
            self.skills = create_skills(self.sex)
        if self.condition is None:
            self.condition = random_birth_condition()
        if not self.learned_actions:
            self.learned_actions = {action: 0.0 for action in ACTIONS}
        if not self.known_techs:
            self.known_techs = {}
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

        # Versión 0.1.1: consumo más suave para que la primera generación no muera demasiado rápido.
        self.hunger -= 0.32
        self.thirst -= 0.42
        self.energy -= 0.28

        if world.weather in ["ola_calor", "sequía"]:
            self.thirst -= 0.55
            self.energy -= 0.25

        if world.weather in ["frio_extremo", "nieve"] and not world.is_shelter(self.x, self.y):
            self.life -= 0.45
            self.energy -= 0.35

        if world.weather in ["lluvia_fuerte", "tormenta"] and not world.is_shelter(self.x, self.y):
            self.energy -= 0.25

        if self.hunger <= 0:
            self.life -= 0.65
        if self.thirst <= 0:
            self.life -= 0.90
        if self.energy <= 0:
            self.life -= 0.18

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
            if self.age > self.max_age():
                self.death_reason = "muerte por edad avanzada"
            elif self.thirst <= 0:
                self.death_reason = "muerte por deshidratación"
            elif self.hunger <= 0:
                self.death_reason = "muerte por hambre"
            else:
                self.death_reason = "muerte por pérdida de vida"
            world.log(f"{self.label()} ha muerto: {self.death_reason}.")

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
        # Instintos de supervivencia reforzados.
        # La red neuronal sigue decidiendo, pero estas reglas evitan muertes tontas al principio.
        if self.thirst < 35:
            if world.map[self.y][self.x] == WATER or world.tile_close(self.x, self.y, WATER, 1):
                return "beber"
            return "buscar_agua"

        if self.hunger < 38:
            if self.inventory["food"] > 0 or self.inventory["raw_meat"] > 0:
                return "comer"
            return "buscar_comida"
        if world.danger_close(self.x, self.y, 2) and (self.life < 70 or self.skills.get("strength", 50) < 75):
            return "huir"
        if world.weather in ["tormenta", "tornado", "terremoto", "frio_extremo"] and not world.is_shelter(self.x, self.y):
            return "volver_refugio"
        if self.energy < 25:
            return "descansar"

        outputs = self.brain.outputs(self.neural_inputs(world))

        for i, action_name in enumerate(ACTIONS):
            outputs[i] += self.learned_actions.get(action_name, 0.0)

        if self.generation <= 15:
            early_bias = {
                "socializar": 0.55,
                "buscar_comida": 0.35,
                "buscar_agua": 0.35,
                "volver_refugio": 0.35,
                "construir": 0.30,
                "fabricar": 0.20,
                "cazar": -0.15,
                "explorar": -0.05,
            }
            for i, action_name in enumerate(ACTIONS):
                outputs[i] += early_bias.get(action_name, 0.0)

        if random.random() < 0.025:
            action = random.choice(ACTIONS)
        else:
            action = ACTIONS[int(np.argmax(outputs))]

        if action == "construir":
            nearby_shelters = world.count_shelters_close(self.x, self.y, 10)
            if nearby_shelters >= max(2, len(world.living_humans()) // 3 + 1):
                return "explorar" if self.energy > 45 else "descansar"
            if self.energy < 55 or self.hunger < 55 or self.thirst < 55:
                return "buscar_comida" if self.hunger < self.thirst else "buscar_agua"

        if action == "fabricar":
            if self.energy < 45 or (self.inventory.get("wood", 0) < 1 and self.inventory.get("stone", 0) < 1):
                return "explorar"

        return action

    def act(self, world: "World") -> None:
        if not self.alive:
            return

        action = self.decide_action(world)

        before_life = self.life
        before_hunger = self.hunger
        before_thirst = self.thirst
        before_energy = self.energy
        before_food = self.inventory.get("food", 0)
        before_wood = self.inventory.get("wood", 0)
        before_stone = self.inventory.get("stone", 0)

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
        elif action == "fabricar":
            self.craft(world)

        # Aprendizaje en vida: refuerzo simple por resultado.
        reward = 0.0
        reward += (self.life - before_life) * 0.08
        reward += (self.hunger - before_hunger) * 0.025
        reward += (self.thirst - before_thirst) * 0.03
        reward += (self.energy - before_energy) * 0.012
        reward += (self.inventory.get("food", 0) - before_food) * 0.8
        reward += (self.inventory.get("wood", 0) - before_wood) * 0.12
        reward += (self.inventory.get("stone", 0) - before_stone) * 0.16

        if not self.alive:
            reward -= 20.0

        old_value = self.learned_actions.get(action, 0.0)
        self.learned_actions[action] = clamp(old_value * 0.92 + reward * 0.08, -6.0, 6.0)


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

        # Riesgo de accidente al explorar: ahora es mucho más bajo y solo sube
        # por motivos claros: noche, clima malo, cansancio o terreno difícil.
        risk = 0.001
        terrain = world.map[self.y][self.x]
        if terrain in [FOREST, SWAMP, MOUNTAIN]:
            risk += 0.006
        if world.is_night():
            risk += 0.012
        if world.weather in ["niebla", "lluvia_fuerte", "tormenta", "nieve"]:
            risk += 0.012
        if self.energy < 25:
            risk += 0.015

        if risk > 0.008:
            self.accident_risk(world, risk, ["corte", "torcedura", "fractura"], cause="explorar en condiciones difíciles")

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
            self.thirst = clamp(self.thirst + 70, 0, 100)
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
        success += self.weapon_power / 100
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
            self.accident_risk(world, risk, ["corte", "mordedura", "herida_grave", "fractura"], cause="fallar una caza peligrosa")

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
            self.inventory["food"] += 2
            self.memory_food[pid] = "segura"
            self.fitness += 0.25

    def craft(self, world: "World") -> None:
        # Fabricación básica de armas/herramientas.
        # No puede crear nada de la nada: necesita madera y/o piedra.
        if self.weapon != "ninguna":
            if world.tribe_knowledge.get("metalurgia") and self.inventory.get("stone", 0) >= 2 and self.inventory.get("wood", 0) >= 1:
                if self.weapon_power < 18:
                    self.inventory["stone"] -= 2
                    self.inventory["wood"] -= 1
                    self.weapon = "herramienta reforzada"
                    self.weapon_power = 18
                    world.log(f"{self.name} fabrica una herramienta reforzada.")
            else:
                self.explore(world)
            return

        tile = world.map[self.y][self.x]
        if tile == FOREST or world.tile_close(self.x, self.y, FOREST, 3):
            self.inventory["wood"] += 1
            self.energy -= 0.8

        if tile == MOUNTAIN or world.tile_close(self.x, self.y, MOUNTAIN, 3):
            self.inventory["stone"] += 1
            self.energy -= 1.0

        mechanics = self.skills.get("mechanics", 50)
        focus = self.skills.get("focus", 50)

        if self.inventory.get("wood", 0) >= 2 and mechanics >= 25:
            self.inventory["wood"] -= 2
            self.weapon = "palo"
            self.weapon_power = 5
            world.log(f"{self.name} fabrica un palo.")
            return

        if self.inventory.get("wood", 0) >= 3 and self.inventory.get("stone", 0) >= 1 and mechanics + focus >= 90:
            self.inventory["wood"] -= 3
            self.inventory["stone"] -= 1
            self.weapon = "lanza básica"
            self.weapon_power = 11
            world.tribe_knowledge["herramientas_piedra"] = True
            world.log(f"{self.name} fabrica una lanza básica.")
            return

        self.explore(world)

    def build(self, world: "World") -> None:
        # Construcción progresiva. Requiere materiales y conocimientos.
        tile = world.map[self.y][self.x]

        if tile == WATER:
            self.explore(world)
            return

        if self.energy < 50 or self.hunger < 45 or self.thirst < 45:
            self.rest(world)
            return

        if tile == FOREST or world.tile_close(self.x, self.y, FOREST, 4):
            self.inventory["wood"] += 1
            self.energy -= 1.0

        if tile == MOUNTAIN or world.tile_close(self.x, self.y, MOUNTAIN, 4):
            self.inventory["stone"] += 1
            self.energy -= 1.2

        nearby_shelters = world.count_shelters_close(self.x, self.y, 10)
        living = max(1, len(world.living_humans()))
        needed_shelters = max(1, math.ceil(living / 3))

        if nearby_shelters >= needed_shelters and chance(0.75):
            self.explore(world)
            return

        mechanics = self.skills.get("mechanics", 50)
        focus = self.skills.get("focus", 50)

        # 1) Choza de madera.
        if not world.tribe_knowledge.get("casa_madera"):
            hut_cost = 12
            if self.inventory.get("wood", 0) >= hut_cost and tile in [GRASS, FOREST]:
                self.inventory["wood"] -= hut_cost
                world.map[self.y][self.x] = HUT
                world.tribe_knowledge["choza"] = True
                world.buildings += 1
                world.log(f"{self.name} construyó una choza de madera.")
                self.fitness += 1.0
                return

        # 2) Casa de madera.
        if world.tribe_knowledge.get("choza") and world.tribe_knowledge.get("herramientas_piedra"):
            if self.inventory.get("wood", 0) >= 25 and mechanics + focus >= 110 and tile in [GRASS, FOREST, HUT]:
                self.inventory["wood"] -= 25
                world.map[self.y][self.x] = HOUSE
                world.tribe_knowledge["casa_madera"] = True
                world.buildings += 1
                world.log(f"{self.name} construyó una casa de madera.")
                self.fitness += 2.0
                return

        # 3) Casa de piedra / edificio simple.
        if world.tribe_knowledge.get("casa_madera"):
            if self.inventory.get("wood", 0) >= 10 and self.inventory.get("stone", 0) >= 25 and mechanics + focus >= 125:
                self.inventory["wood"] -= 10
                self.inventory["stone"] -= 25
                world.map[self.y][self.x] = BUILDING
                world.tribe_knowledge["casa_piedra"] = True
                world.tribe_knowledge["ingenieria"] = True
                world.buildings += 1
                world.log(f"{self.name} construyó una casa de piedra.")
                self.fitness += 3.0
                return

        if self.inventory.get("wood", 0) < 12:
            pos = world.find_nearest_tile(self.x, self.y, FOREST, 18)
            if pos:
                self.move_to(world, pos, steps=1)
                return

        if self.inventory.get("stone", 0) < 5:
            pos = world.find_nearest_tile(self.x, self.y, MOUNTAIN, 25)
            if pos:
                self.move_to(world, pos, steps=1)
                return

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

    def accident_risk(self, world: "World", base: float, possible: List[str], cause: str = "accidente") -> None:
        risk = base
        risk += max(0, 25 - self.energy) / 650
        risk += max(0, 25 - self.life) / 650
        risk -= self.skills.get("speed", 50) / 1800
        risk -= self.skills.get("spatial", 50) / 1900

        if self.condition:
            effects = self.condition.get("effects", {})
            risk += effects.get("accident_risk", 0) * 0.45
            risk += effects.get("fall_risk", 0) * 0.45
            risk += effects.get("fracture_risk", 0) * 0.22

        if world.weather in ["tormenta", "lluvia_fuerte", "niebla", "nieve"]:
            risk += 0.018

        risk = clamp(risk, 0, 0.55)

        if chance(risk):
            injury = random.choice(possible)
            world.log(f"{self.name} tiene un accidente por {cause}.")
            self.apply_injury(world, injury)

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
            self.death_reason = f"muerte por lesión: {kind}"
            world.log(f"{self.label()} ha muerto: {self.death_reason}.")


# ============================================================
# MUNDO
# ============================================================

class World:
    def __init__(self):
        self.map: List[List[str]] = [[GRASS for _ in range(MAP_W)] for _ in range(MAP_H)]
        self.hour = 7
        self.day = 1
        self.month = 1
        self.year = 0

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
        self.spawn_plants(1300)
        self.spawn_initial_animals()

        cx, cy = self.cave_pos

        adam = Human("Adán", "hombre", cx, cy, age=18.0)
        eve = Human("Eva", "mujer", cx + 1, cy, age=18.0)
        adam.person_id = 1
        eve.person_id = 2
        adam.generation = 1
        eve.generation = 1

        if not hasattr(self, "person_count"):
            self.person_count = 2
        self.all_humans = {1: adam, 2: eve}
        self.max_humans_alive = 2
        self.max_humans_time = f"Año {self.year}, Mes {self.month}, Día {self.day}, Hora {self.hour:02d}:00"

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

    def spawn_initial_animals(self) -> None:
        """
        Fauna inicial:
        - Muchos animales pacíficos para que haya comida y vida.
        - Muy pocos animales agresivos al principio para que Adán y Eva no mueran enseguida.
        """
        peaceful_counts = {
            "conejo": 70,
            "ciervo": 38,
            "pez": 80,
            "gallina": 55,
            "cabra": 35,
        }

        aggressive_counts = {
            "lobo": 5,
            "oso": 1,
            "serpiente": 4,
            "jabali": 5,
            "pantera": 1,
        }

        for animal_id, count in peaceful_counts.items():
            for _ in range(count):
                self.spawn_specific_animal(animal_id)

        for animal_id, count in aggressive_counts.items():
            for _ in range(count):
                self.spawn_specific_animal(animal_id)

        self.log(
            "Fauna inicial: muchos animales pacíficos y pocos depredadores."
        )

    def spawn_specific_animal(self, animal_id: str) -> None:
        data = next(a for a in ANIMALS if a["id"] == animal_id)

        for _ in range(200):
            x, y = self.random_land_cell()
            tile = self.map[y][x]

            if animal_id == "pez":
                if self.tile_close(x, y, WATER, 1):
                    self.animals.append(Animal(x, y, data))
                    return
            elif data["aggressive"]:
                # Los depredadores aparecen preferentemente lejos de la cueva.
                if distance((x, y), self.cave_pos) > 45 and tile in [FOREST, MOUNTAIN, SWAMP, GRASS]:
                    self.animals.append(Animal(x, y, data))
                    return
            else:
                if tile in [FOREST, GRASS, SWAMP]:
                    self.animals.append(Animal(x, y, data))
                    return

        # Fallback si no encuentra ubicación ideal.
        x, y = self.random_land_cell()
        self.animals.append(Animal(x, y, data))

    def spawn_daily_animals(self, amount: int) -> None:
        """
        Reproducción/entrada diaria de fauna.
        La mayoría son pacíficos. Los depredadores aparecen mucho menos.
        """
        peaceful_ids = ["conejo", "ciervo", "pez", "gallina", "cabra"]
        aggressive_ids = ["lobo", "oso", "serpiente", "jabali", "pantera"]

        for _ in range(amount):
            if chance(0.88):
                self.spawn_specific_animal(random.choice(peaceful_ids))
            else:
                self.spawn_specific_animal(random.choice(aggressive_ids))

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
            self.spawn_daily_animals(12)

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
                ("lluvia_fuerte", 0.06),
                ("tormenta", 0.04),
                ("niebla", 0.05),
            ]
            self.temperature = random.uniform(8, 22)

        elif season == "verano":
            options = [
                ("soleado", 0.45),
                ("nublado", 0.12),
                ("lluvia", 0.10),
                ("tormenta", 0.08),
                ("ola_calor", 0.07),
                ("sequía", 0.05),
            ]
            self.temperature = random.uniform(18, 38)

        elif season == "otoño":
            options = [
                ("soleado", 0.25),
                ("nublado", 0.20),
                ("lluvia", 0.25),
                ("lluvia_fuerte", 0.08),
                ("tormenta", 0.05),
                ("niebla", 0.10),
            ]
            self.temperature = random.uniform(5, 20)

        else:
            options = [
                ("soleado", 0.20),
                ("nublado", 0.20),
                ("lluvia", 0.12),
                ("nieve", 0.14),
                ("frio_extremo", 0.08),
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
        if self.weather == "tormenta" and season in ["primavera", "verano"] and chance(0.002):
            self.weather = "tornado"
            self.log("Se forma un tornado.")

        # Terremoto raro, más daño en montaña.
        if chance(0.0005):
            self.weather = "terremoto"
            self.log("Hay un terremoto.")

        # Inundación si lluvia fuerte/tormenta.
        if self.weather in ["lluvia_fuerte", "tormenta"] and chance(0.002):
            self.log("Hay riesgo de inundación cerca del río.")
            for h in self.humans:
                if self.tile_close(h.x, h.y, WATER, 2) and not self.is_shelter(h.x, h.y):
                    h.apply_injury(self, "ahogamiento")

        # Incendio en bosque durante ola de calor o tormenta.
        if self.weather in ["ola_calor", "tormenta"] and season == "verano" and chance(0.0015):
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

        population = len(alive)
        avg_mechanics = sum(h.skills.get("mechanics", 50) for h in alive) / population
        avg_memory = sum(h.skills.get("memory", 50) for h in alive) / population
        avg_social = sum(h.personality.get("social", 50) for h in alive) / population
        avg_focus = sum(h.skills.get("focus", 50) for h in alive) / population
        max_generation = max(h.generation for h in alive)

        if not self.tribe_knowledge["fuego"]:
            pressure = 0.0
            if self.weather in ["frio_extremo", "nieve"]:
                pressure += 0.035
            if any(h.life < 60 for h in alive):
                pressure += 0.012
            if population >= 3:
                pressure += 0.01
            if avg_focus > 60:
                pressure += 0.008
            if chance(pressure):
                self.tribe_knowledge["fuego"] = True
                self.log("La tribu descubre el fuego.")

        if self.tribe_knowledge["fuego"] and not self.tribe_knowledge["cocina"]:
            if avg_memory > 55 and chance(0.018):
                self.tribe_knowledge["cocina"] = True
                self.log("La tribu descubre que cocinar reduce enfermedades.")

        if not self.tribe_knowledge["herramientas_piedra"]:
            if avg_mechanics > 58 and chance(0.018):
                self.tribe_knowledge["herramientas_piedra"] = True
                self.log("La tribu descubre herramientas de piedra.")

        if not self.tribe_knowledge.get("lenguaje", False):
            if population >= 8 and max_generation >= 3 and avg_social > 55 and avg_memory > 50:
                if chance(0.012 + (population / 1000)):
                    self.tribe_knowledge["lenguaje"] = True
                    self.log("La tribu empieza a desarrollar un lenguaje común.")

        if self.buildings >= 3 and not self.tribe_knowledge["almacen"]:
            if chance(0.025):
                self.tribe_knowledge["almacen"] = True
                self.log("La tribu aprende a almacenar recursos.")

        if not self.tribe_knowledge.get("agricultura", False):
            if population >= 10 and self.tribe_knowledge.get("almacen") and (self.tribe_knowledge.get("lenguaje") or avg_memory > 65):
                if chance(0.010):
                    self.tribe_knowledge["agricultura"] = True
                    self.log("La tribu descubre agricultura básica.")

        if not self.tribe_knowledge.get("casa_madera", False):
            if self.tribe_knowledge.get("choza") and self.tribe_knowledge.get("herramientas_piedra") and self.buildings >= 4:
                if avg_mechanics > 60 and chance(0.012):
                    self.tribe_knowledge["casa_madera"] = True
                    self.log("La tribu aprende a construir casas de madera.")

        if not self.tribe_knowledge.get("ingenieria", False):
            if self.tribe_knowledge.get("casa_madera") and population >= 12 and avg_mechanics + avg_focus > 125:
                if chance(0.008):
                    self.tribe_knowledge["ingenieria"] = True
                    self.log("La tribu desarrolla principios de ingeniería.")


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

    def count_tiles_close(self, x: int, y: int, tile: str, radius: int) -> int:
        count = 0
        for yy in range(max(0, y - radius), min(MAP_H, y + radius + 1)):
            for xx in range(max(0, x - radius), min(MAP_W, x + radius + 1)):
                if self.map[yy][xx] == tile:
                    count += 1
        return count

    def count_shelters_close(self, x: int, y: int, radius: int) -> int:
        count = 0
        for yy in range(max(0, y - radius), min(MAP_H, y + radius + 1)):
            for xx in range(max(0, x - radius), min(MAP_W, x + radius + 1)):
                if self.is_shelter(xx, yy):
                    count += 1
        return count

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
        if len(self.living_humans()) > 120:
            return

        for woman in women:
            if woman.age < 16 or woman.age > 42:
                continue

            if chance(PROB_REPRODUCCION_DIARIA_BASE):
                father = random.choice(men)
                self.birth_count += 1
                if not hasattr(self, "person_count"):
                    self.person_count = len(self.humans)
                self.person_count += 1

                sex = random_sex()
                numero_hijo_madre = woman.children + 1
                father_label = str(father.person_id) if father.person_id else father.label()
                mother_label = str(woman.person_id) if woman.person_id else woman.label()
                # Nombre v1.11: padre,madre Npersona Ngeneracion
                child_generation = max(father.generation, woman.generation) + 1
                name = f"{father_label},{mother_label} {self.person_count} {child_generation}"
                brain = mix_brains(father.brain, woman.brain)

                baby = Human(
                    name=name,
                    sex=sex,
                    x=woman.x,
                    y=woman.y,
                    age=0.0,
                    brain=brain,
                )
                baby.person_id = self.person_count
                baby.father_label = father_label
                baby.mother_label = mother_label
                baby.child_number = numero_hijo_madre
                baby.generation = child_generation
                baby.skills = inherit_skills(father, woman, sex)
                baby.personality = inherit_personality(father, woman)
                baby.learned_actions = inherit_learned_actions(father, woman)
                baby.apply_condition_effects()

                # Parte del conocimiento básico se transmite.
                for parent in [father, woman]:
                    for k, v in parent.memory_food.items():
                        if chance(0.35):
                            baby.memory_food[k] = v

                    for place_key in parent.memory_places:
                        if parent.memory_places[place_key] and chance(0.25):
                            baby.memory_places[place_key].append(random.choice(parent.memory_places[place_key]))

                if not hasattr(self, "all_humans"):
                    self.all_humans = {}
                self.all_humans[baby.person_id] = baby
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

        # Estadística histórica de humanos vivos.
        living_now = len(self.living_humans())
        if not hasattr(self, "max_humans_alive"):
            self.max_humans_alive = living_now
            self.max_humans_time = f"Año {self.year}, Mes {self.month}, Día {self.day}, Hora {self.hour:02d}:00"
        elif living_now > self.max_humans_alive:
            self.max_humans_alive = living_now
            self.max_humans_time = f"Año {self.year}, Mes {self.month}, Día {self.day}, Hora {self.hour:02d}:00"

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
                attack_prob = 0.04 if self.is_night() else 0.015

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

        print(color("EDEN SIM 0.1.3b — ventana gráfica sin dependencia de pygame.font", "cyan"))
        print(
            f"Año {self.year} | Mes {self.month} | Día {self.day} | Hora {self.hour:02d}:00 | "
            f"Estación: {self.season()} | Clima: {self.weather} | Temp: {self.temperature:.1f}°C"
        )
        peaceful = sum(1 for a in self.animals if not a.aggressive)
        aggressive = sum(1 for a in self.animals if a.aggressive)
        print(
            f"Humanos vivos: {len(living)} | Animales pacíficos: {peaceful} | Depredadores: {aggressive} | "
            f"Plantas: {len(self.plants)} | Edificios: {self.buildings}"
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
                    row += color(a.data.get("symbol", "?"), "red" if a.aggressive else "yellow")
                elif (x, y) in plant_pos:
                    p = plant_pos[(x, y)]
                    row += color(p.data.get("symbol", "?"), "red" if p.data["poison"] else "lightgreen")
                else:
                    row += terrain_symbol(self.map[y][x])
            print(row)

        print()
        print("Leyenda rápida: A Adán | E Eva | @ humano | letras minúsculas/amarillas animales pacíficos | letras mayúsculas/rojas depredadores | letras verdes/rojas plantas")
        print("Pulsa CTRL+C para detener. La leyenda completa aparece al inicio y también abajo resumida.")

    def log(self, msg: str) -> None:
        self.events.append(msg)
        if len(self.events) > 60:
            self.events = self.events[-60:]


# ============================================================
# LEYENDA
# ============================================================

def print_full_legend() -> None:
    print(color("LEYENDA COMPLETA — EDEN SIM", "cyan"))
    print()
    print(color("Humanos", "yellow"))
    print("  A = Adán")
    print("  E = Eva")
    print("  @ = humano nacido después")
    print()
    print(color("Terreno y construcciones", "yellow"))
    print("  . = pradera")
    print("  T = bosque")
    print("  ~ = río / agua")
    print("  C = cueva")
    print("  M = montaña")
    print("  S = pantano")
    print("  h = choza")
    print("  H = casa futura")
    print("  B = edificio futuro")
    print()
    print(color("Animales pacíficos", "yellow"))
    for a in ANIMALS:
        if not a["aggressive"]:
            print(f"  {a['symbol']} = {a['name']}")
    print()
    print(color("Animales agresivos / depredadores", "yellow"))
    for a in ANIMALS:
        if a["aggressive"]:
            print(f"  {a['symbol']} = {a['name']}")
    print()
    print(color("Plantas y alimentos", "yellow"))
    for p in PLANTS:
        estado = "venenosa" if p["poison"] else "comestible"
        print(f"  {p['symbol']} = {p['name']} ({estado})")
    print()
    print(color("Nota de diseño", "cyan"))
    print("  Más adelante cambiaremos esta vista de terminal por una ventana gráfica:")
    print("  - mapa con colores y formas en vez de letras")
    print("  - botones para pausar")
    print("  - botones para acelerar o ralentizar el tiempo")
    print("  - panel lateral con datos de humanos, clima, edad, hambre, sed y civilización")
    print()





# ============================================================
# CONSOLA CURSES, COMANDOS Y MAPAS LIMPIOS
# ============================================================

TERMINAL_RENDER_EVERY_HOURS = 1
USE_COLOR_CURSES = True


def install_runtime_state(world: World) -> None:
    world.command_queue = queue.Queue()
    world.paused = False
    world.speed_steps = 1
    world.step_delay = TURN_DELAY
    world.running = True
    world.relevant_events = []
    world.extinction_report_printed = False
    world.last_status_line = ""
    world.person_count = 2
    if not hasattr(world, "all_humans"):
        world.all_humans = {}
    world.max_humans_alive = 2
    world.max_humans_time = "Año 0, Mes 1, Día 1, Hora 07:00"
    world.command_feedback = ""
    world.command_feedback_timer = 0
    world.map_scroll = 0


def world_time_string(world: World) -> str:
    return (
        f"Año {world.year} | Mes {world.month} | Día {world.day} | Hora {world.hour:02d}:00 | "
        f"Estación: {world.season()} | Clima: {world.weather} | Temp: {world.temperature:.1f}°C"
    )


def world_status_string(world: World) -> str:
    living = world.living_humans()
    peaceful = sum(1 for a in world.animals if not a.aggressive)
    aggressive = sum(1 for a in world.animals if a.aggressive)
    techs = [k for k, v in world.tribe_knowledge.items() if v]
    return (
        f"[TIEMPO] {world_time_string(world)} | "
        f"Humanos actuales: {len(living)} | Máx: {getattr(world, 'max_humans_alive', len(living))} "
        f"({getattr(world, 'max_humans_time', '-')}) | "
        f"Pacíficos: {peaceful} | Depredadores: {aggressive} | "
        f"Plantas: {len(world.plants)} | Edificios: {world.buildings} | Gen máx: {max([h.generation for h in living], default=0)} | "
        f"Velocidad: x{world.speed_steps} | "
        f"Estado: {'PAUSA' if world.paused else 'RUN'} | "
        f"Conocimiento: {', '.join(techs) if techs else 'ninguno'}"
    )


def patched_log(self: World, msg: str) -> None:
    self.events.append(msg)
    if len(self.events) > 60:
        self.events = self.events[-60:]

    if not hasattr(self, "relevant_events"):
        self.relevant_events = []

    stamp = f"Año {self.year}, Mes {self.month}, Día {self.day}, Hora {self.hour:02d}:00"
    line = f"[{stamp}] {msg}"
    self.relevant_events.append(line)

    if len(self.relevant_events) > 3000:
        self.relevant_events = self.relevant_events[-3000:]


World.log = patched_log


def print_relevant_logs_plain(world: World, amount: Optional[int] = None) -> str:
    logs = getattr(world, "relevant_events", [])
    if not logs:
        return "No hay registros relevantes todavía."

    selected = logs if amount is None else logs[-amount:]
    lines = ["REGISTROS RELEVANTES", "-" * 90]
    lines.extend(selected)
    lines.append("-" * 90)
    lines.append(world_status_string(world))
    return "\n".join(lines)


def normalize_sex(value: str) -> str:
    value = value.lower().strip()
    if value in ["man", "male", "hombre", "m"]:
        return "hombre"
    if value in ["woman", "female", "mujer", "f"]:
        return "mujer"
    return random_sex()


def find_human_by_name(world: World, name: str, include_dead: bool = False) -> Optional[Human]:
    target = name.lower().strip()

    # Permite buscar por número de humano/persona.
    # Ejemplo: estadistica 286
    if target.isdigit():
        pid = int(target)
        if include_dead and hasattr(world, "all_humans"):
            return world.all_humans.get(pid)
        for h in world.humans:
            if h.person_id == pid:
                return h
        return None

    pool = list(getattr(world, "all_humans", {}).values()) if include_dead and hasattr(world, "all_humans") else world.humans

    for h in pool:
        if h.name.lower() == target or h.label().lower() == target:
            return h
    return None

def spawn_humans_command(world: World, sex_word: str, age_word: str, count_word: str) -> str:
    sex = normalize_sex(sex_word)

    try:
        age = float(age_word)
        count = int(count_word)
    except ValueError:
        return "Uso: spawn human hombre 18 3"

    count = max(1, min(count, 300))
    x, y = world.cave_pos

    for _ in range(count):
        world.birth_count += 1
        if not hasattr(world, "person_count"):
            world.person_count = len(world.humans)
        world.person_count += 1
        name = f"spawn {world.person_count} 1"
        h = Human(name=name, sex=sex, x=x + random.randint(-2, 2), y=y + random.randint(-2, 2), age=age)
        h.person_id = world.person_count
        h.father_label = "spawn"
        h.mother_label = ""
        h.child_number = 0
        h.generation = 1
        if not hasattr(world, "all_humans"):
            world.all_humans = {}
        world.all_humans[h.person_id] = h
        h.memory_places["shelter"].append(world.cave_pos)
        h.remember_surroundings(world)
        world.humans.append(h)

    # Si la humanidad estaba extinta, este spawn reactiva la simulación.
    world.extinction_report_printed = False
    world.paused = False

    # Actualizar máximo histórico inmediatamente.
    living_now = len(world.living_humans())
    if living_now > getattr(world, "max_humans_alive", 0):
        world.max_humans_alive = living_now
        world.max_humans_time = f"Año {world.year}, Mes {world.month}, Día {world.day}, Hora {world.hour:02d}:00"

    world.log(f"Comando: aparecen {count} humano(s), sexo {sex}, edad {age}.")
    return f"Aparecen {count} humano(s), sexo {sex}, edad {age}. Simulación reactivada."


def spawn_balanced_humans_command(world: World, age_word: str, count_word: str) -> str:
    try:
        age = float(age_word)
        count = int(count_word)
    except ValueError:
        return "Uso: spawn humans 18 10  /  spawn balanced 18 10"

    count = max(1, min(count, 500))
    men = count // 2
    women = count - men

    if men > 0:
        spawn_humans_command(world, "hombre", str(age), str(men))
    if women > 0:
        spawn_humans_command(world, "mujer", str(age), str(women))

    return f"Aparecen {count} humanos equilibrados: {men} hombre(s), {women} mujer(es)."



def spawn_animals_command(world: World, animal_id: str, count_word: str) -> str:
    try:
        count = int(count_word)
    except ValueError:
        return "Uso: spawn animal conejo 10"

    aliases = {
        "wolf": "lobo",
        "bear": "oso",
        "snake": "serpiente",
        "boar": "jabali",
        "panther": "pantera",
        "rabbit": "conejo",
        "deer": "ciervo",
        "fish": "pez",
        "chicken": "gallina",
        "goat": "cabra",
        "jabalí": "jabali",
    }

    animal_id = aliases.get(animal_id.lower(), animal_id.lower())
    valid = [a["id"] for a in ANIMALS]

    if animal_id not in valid:
        return "Animal no válido. Usa: " + ", ".join(valid)

    count = max(1, min(count, 500))

    for _ in range(count):
        world.spawn_specific_animal(animal_id)

    world.log(f"Comando: aparecen {count} animal(es) de tipo {animal_id}.")
    return f"Aparecen {count} animal(es) de tipo {animal_id}."


def heal_command(world: World, target: str, amount_word: str) -> str:
    try:
        amount = float(amount_word)
    except ValueError:
        return "Uso: heal all 30  /  heal Adán 30"

    if target.lower() == "all":
        humans = world.living_humans()
    else:
        h = find_human_by_name(world, target)
        humans = [h] if h else []

    if not humans:
        return "No he encontrado humanos para curar."

    for h in humans:
        h.life = clamp(h.life + amount, 0, 100)

    world.log(f"Comando: +{amount} de vida a {len(humans)} humano(s).")
    return f"+{amount} de vida a {len(humans)} humano(s)."


def setlife_command(world: World, target: str, value_word: str) -> str:
    try:
        value = float(value_word)
    except ValueError:
        return "Uso: setlife all 100  /  setlife Eva 100"

    if target.lower() == "all":
        humans = world.living_humans()
    else:
        h = find_human_by_name(world, target)
        humans = [h] if h else []

    if not humans:
        return "No he encontrado humanos."

    for h in humans:
        h.life = clamp(value, 0, 100)

    world.log(f"Comando: vida fijada a {value} para {len(humans)} humano(s).")
    return f"Vida fijada a {value} para {len(humans)} humano(s)."


def resource_command(world: World, stat: str, target: str, value_word: str) -> str:
    try:
        value = float(value_word)
    except ValueError:
        return f"Uso: {stat} all 100"

    if target.lower() == "all":
        humans = world.living_humans()
    else:
        h = find_human_by_name(world, target)
        humans = [h] if h else []

    if not humans:
        return "No he encontrado humanos."

    attr = {"food": "hunger", "water": "thirst", "energy": "energy"}[stat]

    for h in humans:
        if attr == "energy":
            h.energy = clamp(value, 0, h.energy_max)
        else:
            setattr(h, attr, clamp(value, 0, 100))

    world.log(f"Comando: {stat} fijado a {value} para {len(humans)} humano(s).")
    return f"{stat} fijado a {value} para {len(humans)} humano(s)."


def give_command(world: World, target: str, item: str, amount_word: str) -> str:
    try:
        amount = int(amount_word)
    except ValueError:
        return "Uso: give all food 10  /  give Eva wood 20"

    valid_items = ["food", "raw_meat", "wood", "stone"]

    if item not in valid_items:
        return "Objeto no válido: " + ", ".join(valid_items)

    if target.lower() == "all":
        humans = world.living_humans()
    else:
        h = find_human_by_name(world, target)
        humans = [h] if h else []

    if not humans:
        return "No he encontrado humanos."

    for h in humans:
        h.inventory[item] = h.inventory.get(item, 0) + amount

    world.log(f"Comando: +{amount} {item} a {len(humans)} humano(s).")
    return f"+{amount} {item} a {len(humans)} humano(s)."


def set_weather_command(world: World, weather_name: str) -> str:
    valid = [
        "soleado", "nublado", "lluvia", "lluvia_fuerte", "tormenta", "niebla",
        "sequía", "ola_calor", "frio_extremo", "nieve", "tornado", "terremoto"
    ]
    if weather_name not in valid:
        return "Clima no válido: " + ", ".join(valid)

    world.weather = weather_name
    world.log(f"Comando: clima cambiado a {weather_name}.")
    return f"Clima cambiado a {weather_name}."


def disaster_command(world: World, disaster_name: str) -> str:
    if disaster_name == "incendio":
        world.forest_fire()
        return "Incendio provocado."

    if disaster_name not in ["tornado", "terremoto", "inundacion"]:
        return "Desastres: tornado, terremoto, inundacion, incendio"

    world.weather = "terremoto" if disaster_name == "terremoto" else "tornado"

    if disaster_name == "inundacion":
        for h in world.humans:
            if h.alive and world.tile_close(h.x, h.y, WATER, 2) and not world.is_shelter(h.x, h.y):
                h.apply_injury(world, "ahogamiento")

    world.log(f"Comando: desastre provocado: {disaster_name}.")
    return f"Desastre provocado: {disaster_name}."



def human_statistics_text(world: World, target: Optional[str] = None) -> str:
    if target and target.lower() != "all":
        h = find_human_by_name(world, target, include_dead=True)
        if not h:
            return f"No encuentro al humano {target}."
        humans = [h]
    else:
        # Por defecto mostramos vivos. Si quieres histórico total: estadistica all
        humans = world.living_humans()

    if target and target.lower() == "all":
        humans = sorted(getattr(world, "all_humans", {}).values(), key=lambda x: x.person_id or 999999)

    if not humans:
        return "No hay humanos para mostrar estadísticas."

    lines = ["ESTADÍSTICAS DE HUMANOS", "-" * 90]
    for h in sorted(humans, key=lambda x: x.person_id or 999999):
        cond = h.condition["name"] if h.condition else "sin condición"
        estado = "vivo" if h.alive else f"muerto ({h.death_reason or 'causa desconocida'})"
        lines.append(
            f"{h.label()} | id={h.person_id} | estado={estado} | sexo={h.sex} | gen={h.generation} | edad={h.age:.1f} | "
            f"vida={h.life:.1f} hambre={h.hunger:.1f} sed={h.thirst:.1f} energía={h.energy:.1f}"
        )
        lines.append(
            f"  fuerza={h.skills.get('strength',0):.1f} velocidad={h.skills.get('speed',0):.1f} "
            f"mecánica={h.skills.get('mechanics',0):.1f} memoria={h.skills.get('memory',0):.1f} "
            f"empatía={h.skills.get('empathy',0):.1f} foco={h.skills.get('focus',0):.1f}"
        )
        lines.append(
            f"  arma={h.weapon} poder_arma={h.weapon_power:.1f} | "
            f"food={h.inventory.get('food',0)} raw_meat={h.inventory.get('raw_meat',0)} "
            f"wood={h.inventory.get('wood',0)} stone={h.inventory.get('stone',0)} | {cond}"
        )
        best_actions = sorted(h.learned_actions.items(), key=lambda kv: kv[1], reverse=True)[:4]
        lines.append("  aprendizaje: " + ", ".join(f"{k}={v:.2f}" for k, v in best_actions))
        lines.append("")
    return "\n".join(lines)

def help_text() -> str:
    return (
        "COMANDOS\\n"
        "help | status | logs | logs 30 | pause | resume | speed 5 | delay 0.20 | render\\n"
        "spawn human hombre 18 3 | spawn human mujer 18 3 | spawn animal conejo 20 | spawn animal lobo 2\\n"
        "heal all 30 | setlife all 100 | food all 100 | water all 100 | energy all 100\\n"
        "give all food 10 | give all wood 30 | weather lluvia | disaster terremoto | quit\nFlechas arriba/abajo = mover vista vertical si hay muchas filas de mapas\nlogs abre una pantalla de registros; render vuelve al mapa"
    )


def process_command(world: World, command: str) -> str:
    command = command.strip()

    if not command:
        return ""

    parts = command.split()
    cmd = parts[0].lower()

    try:
        if cmd == "help":
            return help_text()

        elif cmd == "status":
            return world_status_string(world)

        elif cmd == "logs":
            amount = None
            if len(parts) >= 2:
                amount = int(parts[1])
            return print_relevant_logs_plain(world, amount)

        elif cmd == "pause":
            world.paused = True
            return "Simulación pausada."

        elif cmd == "resume":
            world.paused = False
            return "Simulación reanudada."

        elif cmd == "speed":
            if len(parts) < 2:
                return f"Velocidad actual: {world.speed_steps}"
            world.speed_steps = max(1, min(int(parts[1]), 200))
            return f"Velocidad: {world.speed_steps} hora(s) simuladas por ciclo."

        elif cmd == "delay":
            if len(parts) < 2:
                return f"Delay actual: {world.step_delay}"
            world.step_delay = max(0.0, min(float(parts[1]), 3.0))
            return f"Delay: {world.step_delay} segundos."

        elif cmd == "render":
            return "Render actualizado."

        elif cmd == "spawn":
            if len(parts) < 2:
                return "Uso: spawn human hombre 18 3 / spawn humans 18 10 / spawn animal conejo 10"
            elif parts[1].lower() == "human" and len(parts) >= 5:
                return spawn_humans_command(world, parts[2], parts[3], parts[4])
            elif parts[1].lower() in ["humans", "balanced", "balanceado"] and len(parts) >= 4:
                return spawn_balanced_humans_command(world, parts[2], parts[3])
            elif parts[1].lower() == "animal" and len(parts) >= 4:
                return spawn_animals_command(world, parts[2], parts[3])
            return "Uso: spawn human hombre 18 3 / spawn humans 18 10 / spawn animal conejo 10"

        elif cmd in ["estadistica", "estadisticas", "stats"]:
            target = parts[1] if len(parts) >= 2 else None
            return human_statistics_text(world, target)

        elif cmd == "heal" and len(parts) >= 3:
            return heal_command(world, parts[1], parts[2])

        elif cmd == "setlife" and len(parts) >= 3:
            return setlife_command(world, parts[1], parts[2])

        elif cmd in ["food", "water", "energy"] and len(parts) >= 3:
            return resource_command(world, cmd, parts[1], parts[2])

        elif cmd == "give" and len(parts) >= 4:
            return give_command(world, parts[1], parts[2], parts[3])

        elif cmd == "weather" and len(parts) >= 2:
            return set_weather_command(world, parts[1])

        elif cmd == "disaster" and len(parts) >= 2:
            return disaster_command(world, parts[1])

        elif cmd in ["quit", "exit"]:
            world.running = False
            return "Saliendo..."

        return "Comando no reconocido. Escribe: help"

    except Exception as e:
        return f"Error ejecutando comando: {e}"


def plain_entity_char_at(world: World, x: int, y: int) -> Tuple[str, int]:
    for h in world.living_humans():
        if h.x == x and h.y == y:
            if h.name == "Adán":
                return "A", 4
            if h.name == "Eva":
                return "E", 5
            return "@", 4 if h.sex == "hombre" else 5

    for a in world.animals:
        if a.alive and a.x == x and a.y == y:
            return a.data.get("symbol", "?"), 3 if a.aggressive else 6

    for p in world.plants:
        if p.x == x and p.y == y:
            return p.data.get("symbol", "?"), 3 if p.data["poison"] else 2

    tile = world.map[y][x]
    if tile == GRASS:
        return ".", 2
    if tile == FOREST:
        return "T", 2
    if tile == WATER:
        return "~", 4
    if tile == CAVE:
        return "C", 7
    if tile == MOUNTAIN:
        return "M", 7
    if tile == SWAMP:
        return "S", 6
    if tile in [HUT, HOUSE, BUILDING]:
        return tile, 5
    return tile, 1


def make_map_window(world: World, h: Human, radius: int) -> List[List[Tuple[str, int]]]:
    width = radius * 2 + 1
    title = f" Mapa de {h.label()} "
    border = "+" + "-" * width + "+"

    lines: List[List[Tuple[str, int]]] = []
    lines.append([(ch, 1) for ch in border])

    header = title[:width].center(width)
    lines.append([(ch, 1) for ch in ("|" + header + "|")])

    for y in range(h.y - radius, h.y + radius + 1):
        row: List[Tuple[str, int]] = [("|", 1)]
        for x in range(h.x - radius, h.x + radius + 1):
            if not world.in_bounds(x, y):
                row.append((" ", 1))
            else:
                row.append(plain_entity_char_at(world, x, y))
        row.append(("|", 1))
        lines.append(row)

    lines.append([(ch, 1) for ch in border])
    return lines


def sorted_living_humans(world: World) -> List[Human]:
    def key(h: Human):
        if h.name == "Adán":
            return (0, 1)
        if h.name == "Eva":
            return (1, 2)
        return (2, h.person_id or 999999)
    return sorted(world.living_humans(), key=key)


def addstr_safe(stdscr, y: int, x: int, text: str, attr=0) -> None:
    try:
        max_y, max_x = stdscr.getmaxyx()
        if y < 0 or y >= max_y or x >= max_x:
            return
        stdscr.addstr(y, x, text[: max(0, max_x - x - 1)], attr)
    except curses.error:
        pass


def draw_colored_window(stdscr, y0: int, x0: int, window: List[List[Tuple[str, int]]], color_pairs: Dict[int, int]) -> None:
    max_y, max_x = stdscr.getmaxyx()
    for dy, line in enumerate(window):
        y = y0 + dy
        if y >= max_y - 7:
            break
        x = x0
        for ch, pair_id in line:
            if x >= max_x - 1:
                break
            attr = curses.color_pair(color_pairs.get(pair_id, 1))
            try:
                stdscr.addstr(y, x, ch, attr)
            except curses.error:
                pass
            x += 1


def is_log_screen(text: str) -> bool:
    return text.startswith("REGISTROS RELEVANTES") or text.startswith("ESTADÍSTICAS DE HUMANOS")


def draw_log_screen(stdscr, world: World, command_buffer: str, last_output: str) -> None:
    stdscr.erase()
    max_y, max_x = stdscr.getmaxyx()

    addstr_safe(stdscr, 0, 0, "EDEN SIM v1.11 — PANEL", curses.color_pair(4))
    addstr_safe(stdscr, 1, 0, world_status_string(world), curses.color_pair(1))

    lines = last_output.splitlines() if last_output else ["No hay registros."]
    usable_top = 3
    usable_bottom = max_y - 3
    max_lines = max(0, usable_bottom - usable_top)

    # Para logs mostramos los últimos que caben, que normalmente son los más relevantes ahora.
    visible = lines[-max_lines:] if len(lines) > max_lines else lines

    y = usable_top
    for line in visible:
        attr = curses.color_pair(1)
        if line.startswith("REGISTROS"):
            attr = curses.color_pair(6)
        elif "ha muerto" in line or "extinguido" in line:
            attr = curses.color_pair(3)
        elif "Comando:" in line:
            attr = curses.color_pair(4)
        addstr_safe(stdscr, y, 0, line, attr)
        y += 1

    addstr_safe(stdscr, max_y - 2, 0, "Escribe render para volver al mapa | logs 200 para pedir más | quit para salir", curses.color_pair(6))
    addstr_safe(stdscr, max_y - 1, 0, f"cmd> {command_buffer}", curses.color_pair(4))
    try:
        stdscr.move(max_y - 1, min(5 + len(command_buffer), max_x - 2))
    except curses.error:
        pass

    stdscr.refresh()


def draw_ui(stdscr, world: World, command_buffer: str, last_output: str, color_pairs: Dict[int, int]) -> None:
    # Si el último comando fue logs, mostramos una pantalla de logs real,
    # no solo una línea de respuesta abajo.
    if is_log_screen(last_output):
        draw_log_screen(stdscr, world, command_buffer, last_output)
        return

    stdscr.erase()
    max_y, max_x = stdscr.getmaxyx()

    addstr_safe(stdscr, 0, 0, "EDEN SIM v1.11 — correcciones de nombres y estadísticas", curses.color_pair(4))
    addstr_safe(stdscr, 1, 0, world_status_string(world), curses.color_pair(1))

    living = sorted_living_humans(world)

    # Panel inferior fijo.
    panel_h = 7
    panel_y = max_y - panel_h

    if not living:
        addstr_safe(stdscr, 3, 0, "La humanidad se ha extinguido. Puedes usar: spawn human hombre 18 3", curses.color_pair(3))
    else:
        # Se dibujan TODOS los mapas en una parrilla lógica:
        # 6 por fila, luego segunda fila, tercera, etc.
        # Si la pantalla no da altura, puedes bajar/subir con flechas.
        maps_per_row = 6
        gap = 2
        min_radius = 3
        radius = VIEW_RADIUS

        # Ajustar radio para que entren 6 por fila en ancho.
        while radius > min_radius:
            window_width = radius * 2 + 3
            total_width = maps_per_row * window_width + (maps_per_row - 1) * gap
            if total_width <= max_x:
                break
            radius -= 1

        window_width = radius * 2 + 3
        window_height = radius * 2 + 4
        row_height = window_height + 1
        total_rows = (len(living) + maps_per_row - 1) // maps_per_row
        total_grid_height = total_rows * row_height

        visible_area_height = max(1, panel_y - 3)
        max_scroll = max(0, total_grid_height - visible_area_height)
        world.map_scroll = max(0, min(getattr(world, "map_scroll", 0), max_scroll))

        start_y = 3 - world.map_scroll

        for idx, h in enumerate(living):
            row = idx // maps_per_row
            col = idx % maps_per_row

            y0 = start_y + row * row_height
            x0 = col * (window_width + gap)

            # Dibuja la ventana aunque esté parcialmente fuera; draw_colored_window ya recorta.
            if y0 + window_height < 3:
                continue
            if y0 >= panel_y:
                continue

            window = make_map_window(world, h, radius)
            draw_colored_window(stdscr, y0, x0, window, color_pairs)

        if max_scroll > 0:
            addstr_safe(
                stdscr,
                2,
                0,
                f"Hay {len(living)} mapas en {total_rows} fila(s). Scroll: {world.map_scroll}/{max_scroll}. Usa flechas arriba/abajo.",
                curses.color_pair(6),
            )

    # Panel inferior fijo: nunca pisa mapas ni comando.
    for yy in range(panel_y, max_y):
        addstr_safe(stdscr, yy, 0, " " * (max_x - 1))

    addstr_safe(stdscr, panel_y, 0, "Humanos destacados:", curses.color_pair(6))
    top = sorted(living, key=lambda x: x.fitness, reverse=True)[:3]
    line_y = panel_y + 1
    for h in top:
        addstr_safe(
            stdscr,
            line_y,
            0,
            f"{h.label():24s} {h.sex:6s} edad {h.age:4.1f} vida {h.life:5.1f} hambre {h.hunger:5.1f} sed {h.thirst:5.1f} energía {h.energy:5.1f}",
            curses.color_pair(1),
        )
        line_y += 1

    recent = world.relevant_events[-1] if world.relevant_events else ""
    addstr_safe(stdscr, panel_y + 4, 0, f"Último evento: {recent}", curses.color_pair(6))

    if last_output:
        first_line = last_output.splitlines()[0]
        addstr_safe(stdscr, panel_y + 5, 0, f"Respuesta: {first_line}", curses.color_pair(5))

    addstr_safe(stdscr, panel_y + 6, 0, f"cmd> {command_buffer}", curses.color_pair(4))
    try:
        stdscr.move(panel_y + 6, min(5 + len(command_buffer), max_x - 2))
    except curses.error:
        pass

    stdscr.refresh()

def init_curses_colors() -> Dict[int, int]:
    curses.start_color()
    curses.use_default_colors()

    # ids lógicos -> pair de curses
    # 1 normal, 2 verde, 3 rojo, 4 cian, 5 magenta, 6 amarillo, 7 blanco/gris
    pairs = {
        1: 1,
        2: 2,
        3: 3,
        4: 4,
        5: 5,
        6: 6,
        7: 7,
    }

    curses.init_pair(1, curses.COLOR_WHITE, -1)
    curses.init_pair(2, curses.COLOR_GREEN, -1)
    curses.init_pair(3, curses.COLOR_RED, -1)
    curses.init_pair(4, curses.COLOR_CYAN, -1)
    curses.init_pair(5, curses.COLOR_MAGENTA, -1)
    curses.init_pair(6, curses.COLOR_YELLOW, -1)
    curses.init_pair(7, curses.COLOR_WHITE, -1)

    return pairs


def curses_simulation(stdscr, world: World) -> None:
    curses.curs_set(1)
    stdscr.nodelay(True)
    stdscr.keypad(True)

    color_pairs = init_curses_colors()
    command_buffer = ""
    last_output = help_text()
    last_step = time.time()
    last_render = 0.0

    while world.running:
        now = time.time()

        # Entrada de comandos persistente: ya no desaparecen las letras.
        try:
            ch = stdscr.getch()
        except curses.error:
            ch = -1

        if ch != -1:
            if ch in [10, 13]:
                cmd = command_buffer.strip()
                command_buffer = ""
                if cmd:
                    last_output = process_command(world, cmd)
            elif ch in [27]:  # ESC
                world.running = False
                last_output = "Saliendo..."
            elif ch in [curses.KEY_BACKSPACE, 127, 8]:
                command_buffer = command_buffer[:-1]
            elif ch == curses.KEY_DOWN:
                world.map_scroll = getattr(world, "map_scroll", 0) + 3
            elif ch == curses.KEY_UP:
                world.map_scroll = max(0, getattr(world, "map_scroll", 0) - 3)
            elif ch == curses.KEY_NPAGE:
                world.map_scroll = getattr(world, "map_scroll", 0) + 12
            elif ch == curses.KEY_PPAGE:
                world.map_scroll = max(0, getattr(world, "map_scroll", 0) - 12)
            elif 0 <= ch < 256:
                command_buffer += chr(ch)

        if not world.paused and now - last_step >= world.step_delay:
            for _ in range(world.speed_steps):
                if not world.living_humans():
                    if not world.extinction_report_printed:
                        world.log(
                            "La humanidad se ha extinguido. "
                            f"Máximo histórico: {getattr(world, 'max_humans_alive', 0)} humanos "
                            f"en {getattr(world, 'max_humans_time', '-')}"
                        )
                        # Abrimos automáticamente pantalla de registros al extinguirse.
                        last_output = print_relevant_logs_plain(world, amount=None)
                        world.extinction_report_printed = True
                        world.paused = True
                    break

                world.step()

            last_step = now

        if now - last_render >= 0.05:
            draw_ui(stdscr, world, command_buffer, last_output, color_pairs)
            last_render = now

        time.sleep(0.01)


def run_terminal_simulation(world: World) -> None:
    locale.setlocale(locale.LC_ALL, "")
    curses.wrapper(curses_simulation, world)

    # Al salir de curses, imprimimos resumen limpio.
    print("Simulación cerrada.")
    print(print_relevant_logs_plain(world, amount=80))


# ============================================================
# EJECUCIÓN
# ============================================================

def main() -> None:
    random.seed()
    np.random.seed()

    if SHOW_START_LEGEND:
        clear_screen()
        print_full_legend()
        print(color("NOVEDAD v1.11:", "cyan"))
        print("Si spawneas humanos tras una extinción, el tiempo vuelve a avanzar.")
        print("Corrección de bugs: help ya no abre logs, nombres por generación y estadísticas por número de humano.")
        print("Año inicial: 0.")
        print("Ahora puedes usar: estadistica 286 para ver el humano número 286, incluso si está muerto.")
        print("Ejemplos de nombres: Adán,Eva 3 2  /  8,5 17 4  /  spawn 8 1")
        print()
        if WAIT_AFTER_START_LEGEND:
            input("Pulsa ENTER para crear el mundo y empezar la simulación...")

    world = World()
    install_runtime_state(world)
    world.generate()

    # Aseguramos ids por si la generación se hizo antes del runtime.
    for h in world.humans:
        if h.name == "Adán":
            h.person_id = 1
        elif h.name == "Eva":
            h.person_id = 2

    run_terminal_simulation(world)


if __name__ == "__main__":
    main()
