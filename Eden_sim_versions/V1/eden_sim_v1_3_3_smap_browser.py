#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
eden_sim_v1_3_3_smap_browser.py

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
    python eden_sim_v1_3_3_smap_browser.py

Controles:
    El simulador corre solo.
    Pulsa CTRL + C para detenerlo.

Notas:
    Esta es la versión 1.3.3 + smap browser: misma IA/supervivencia que v1.3.3, solo añade mapa completo en navegador sin tkinter.
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
import struct
import zlib
import subprocess
import tempfile
import webbrowser
import socket
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
PROB_REPRODUCCION_DIARIA_BASE = 0.20

# Para que la evolución generacional sea visible en minutos y no en horas reales:
# 36 significa que 1 año biológico pasa en 10 días simulados aprox.
AGE_SPEED_MULTIPLIER = 36

# Edad mínima de reproducción en este universo acelerado.
REPRODUCTION_AGE = 12

# Los niños consumen menos recursos.
CHILD_CONSUMPTION_MULTIPLIER = 0.45
ADOLESCENT_CONSUMPTION_MULTIPLIER = 0.70

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
    "curarse",
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
BROWN_HOUSE = "R"

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
    BROWN_HOUSE: "casa segura",
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
    "brown": "\033[38;5;94m",
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
    if tile == BROWN_HOUSE:
        return color(tile, "brown")
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
    {"id": "trex", "name": "T-Rex", "symbol": "X", "aggressive": True, "strength": 240, "meat": 250, "raw_risk": 0.85, "night": True},
]


# Capas de población animal para que aumenten con el tiempo sin invadir el mapa.
PEACEFUL_ANIMAL_SOFT_CAP = 520
AGGRESSIVE_ANIMAL_SOFT_CAP = 75
PEACEFUL_ANIMAL_HARD_CAP = 720
AGGRESSIVE_ANIMAL_HARD_CAP = 110

ANIMAL_REPRODUCTION = {
    "conejo": 0.070,
    "gallina": 0.055,
    "pez": 0.060,
    "cabra": 0.028,
    "ciervo": 0.024,
    "lobo": 0.010,
    "serpiente": 0.010,
    "jabali": 0.012,
    "pantera": 0.006,
    "oso": 0.004,
    # T-Rex no se reproduce de forma natural. Solo existe si lo spawneas por comando.
    "trex": 0.000,
}


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
    "intoxicacion": {"life": -12, "energy": -18, "hunger": -6, "days": 5},
    "enfermedad": {"life": -9, "energy": -16, "days": 7},
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
        "hides": 0,
        "bones": 0,
        "herbs": 0,
        "seeds": 0,
        "clay": 0,
        "metal_ore": 0,
        "metal": 0,
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
    last_position: Tuple[int, int] = field(default_factory=lambda: (-1, -1))
    still_hours: int = 0
    pending_signals: List[Dict] = field(default_factory=list)
    clothing: str = "ninguna"
    cold_protection: float = 0.0
    home_pos: Optional[Tuple[int, int]] = None
    last_water_target: Optional[Tuple[int, int]] = None
    recent_targets: List[Tuple[int, int]] = field(default_factory=list)

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

    def queue_signal(self, kind: str, pos: Tuple[int, int], strength: float = 1.0, ttl_hours: int = 72) -> None:
        """Guarda un descubrimiento para comunicarlo luego al grupo.

        No avisa instantáneamente. Tiene que volver cerca de otros humanos
        o de un refugio/asentamiento para que el grupo lo aprenda.
        """
        x, y = int(pos[0]), int(pos[1])

        for signal in self.pending_signals:
            if signal["kind"] == kind and distance(signal["pos"], (x, y)) <= 4:
                signal["strength"] = min(10.0, signal.get("strength", 1.0) + strength)
                signal["ttl"] = max(signal.get("ttl", ttl_hours), ttl_hours)
                return

        self.pending_signals.append({
            "kind": kind,
            "pos": (x, y),
            "strength": strength,
            "ttl": ttl_hours,
        })
        self.pending_signals = self.pending_signals[-8:]

    def can_tell_group(self, world: "World") -> bool:
        return world.human_close(self, 5) or world.is_shelter(self.x, self.y)

    def deliver_pending_signals(self, world: "World") -> bool:
        if not self.pending_signals:
            return False
        if not self.can_tell_group(world):
            return False

        for signal in list(self.pending_signals):
            world.broadcast_signal(
                signal["kind"],
                signal["pos"],
                source=self,
                strength=signal.get("strength", 1.0),
                ttl_hours=signal.get("ttl", 72),
            )

        count = len(self.pending_signals)
        self.pending_signals.clear()
        self.fitness += 0.15 * count
        return True

    def should_return_to_report(self) -> bool:
        if not self.pending_signals:
            return False
        return self.thirst > 55 and self.hunger > 45 and self.energy > 35 and self.life > 45

    def try_use_herbs(self, world: "World") -> None:
        if self.inventory.get("herbs", 0) <= 0:
            return
        if self.life < 82 or self.injuries:
            # Con vida baja no puede ser aleatorio casi siempre: debe intentar salvarse.
            success = chance(0.82 if self.life < 55 else 0.68)
            self.inventory["herbs"] -= 1
            if success:
                heal_amount = 18 if self.life < 55 else 13
                self.life = clamp(self.life + heal_amount, 0, 100)
                if self.injuries and chance(0.70):
                    self.injuries.pop(0)
                world.tribe_knowledge["medicina_herbal"] = True
                self.queue_signal("medical", (self.x, self.y), strength=1.2, ttl_hours=100)
                world.log(f"{self.label()} usa hierbas medicinales y se recupera.")
            else:
                self.life = clamp(self.life + 4, 0, 100)
                world.log(f"{self.label()} prueba hierbas, pero apenas mejora.")

    def health_state(self) -> str:
        if self.life < 30:
            return "crítico"
        if self.life < 55:
            return "grave"
        if self.life < 75 or self.injuries:
            return "tocado"
        return "estable"

    def needs_healing(self) -> bool:
        return self.life < 75 or bool(self.injuries)

    def seek_healing(self, world: "World") -> None:
        """
        Modo médico:
        - Si tiene hierbas, las usa.
        - Si conoce señales médicas, va a ellas.
        - Si no, busca plantas/hierbas.
        - Si está muy débil, se refugia y descansa.
        - Si hay otros humanos cerca, pide ayuda.
        """
        if self.inventory.get("herbs", 0) > 0:
            self.try_use_herbs(world)
            if self.life >= 55 and not self.injuries:
                return

        med = world.best_shared_signal("medical", self.x, self.y, max_distance=120)
        if med:
            target = world.find_free_near_target(med[0], med[1], radius=5, human=self) or med
            self.move_to(world, target, steps=3)
            if distance((self.x, self.y), target) <= 3:
                self.rest(world)
            return

        helper = world.nearest_living_human_pos(self, radius=80)
        if helper and self.life < 55:
            target = world.find_free_near_target(helper[0], helper[1], radius=4, human=self) or helper
            self.move_to(world, target, steps=3)
            self.queue_signal("medical", (self.x, self.y), strength=1.0, ttl_hours=80)
            return

        plant = world.find_nearest_plant(self.x, self.y, 28)
        if plant and not plant.data.get("poison", False):
            self.move_to(world, (plant.x, plant.y), steps=3)
            if distance((self.x, self.y), (plant.x, plant.y)) <= 1.5:
                if plant in world.plants:
                    world.plants.remove(plant)
                if chance(0.55):
                    self.inventory["herbs"] += 1
                if chance(0.45):
                    self.inventory["food"] += 1
                self.memory_food[plant.data["id"]] = "segura"
                self.queue_signal("medical", (self.x, self.y), strength=1.0, ttl_hours=80)
                world.log(f"{self.label()} recolecta recursos medicinales.")
            return

        if self.life < 45:
            if not world.is_shelter(self.x, self.y):
                self.return_shelter(world)
            else:
                self.rest(world)
            return

        self.explore(world)

    def receive_help_from_group(self, world: "World") -> bool:
        """
        Otros humanos pueden ayudar si tienen hierbas o conocimientos.
        Esto evita que una colonia de 100 muera sin que nadie actúe como cuidador.
        """
        if not self.needs_healing():
            return False

        helpers = [
            h for h in world.living_humans()
            if h is not self
            and distance((h.x, h.y), (self.x, self.y)) <= 3
            and (h.inventory.get("herbs", 0) > 0 or world.tribe_knowledge.get("medicina_herbal"))
        ]

        if not helpers:
            return False

        helper = max(helpers, key=lambda h: h.inventory.get("herbs", 0) + h.skills.get("empathy", 50) / 100)
        if helper.inventory.get("herbs", 0) > 0:
            helper.inventory["herbs"] -= 1
            self.life = clamp(self.life + 16, 0, 100)
            if self.injuries and chance(0.55):
                self.injuries.pop(0)
            world.tribe_knowledge["medicina_herbal"] = True
            world.broadcast_signal("medical", (self.x, self.y), source=helper, strength=1.2, ttl_hours=90)
            world.log(f"{helper.label()} ayuda a curar a {self.label()} con hierbas.")
            return True

        if chance(0.35):
            self.life = clamp(self.life + 5, 0, 100)
            world.log(f"{helper.label()} cuida a {self.label()} y evita que empeore.")
            return True

        return False


    def try_primitive_farming(self, world: "World") -> bool:
        if not world.tribe_knowledge.get("agricultura", False):
            return False
        if self.inventory.get("seeds", 0) < 3:
            return False
        if world.map[self.y][self.x] not in [GRASS, FOREST]:
            return False
        if self.thirst < 50 or self.energy < 45:
            return False

        self.inventory["seeds"] -= 3
        planted = 0
        for _ in range(8):
            nx = int(clamp(self.x + random.randint(-4, 4), 0, MAP_W - 1))
            ny = int(clamp(self.y + random.randint(-4, 4), 0, MAP_H - 1))
            if world.map[ny][nx] in [GRASS, FOREST] and chance(0.65):
                world.plants.append(Plant(nx, ny, random.choice([p for p in PLANTS if not p["poison"]])))
                planted += 1

        if planted:
            world.log(f"{self.label()} planta semillas cerca del asentamiento.")
            self.fitness += 0.8
            return True
        return False

    def max_age(self) -> float:
        base = 65 + (self.skills.get("longevity", 50) - 50) * 0.35
        if self.sex == "mujer":
            base += 4
        return clamp(base, 35, 105)

    def process_hour(self, world: "World") -> None:
        if not self.alive:
            return

        self.age += AGE_SPEED_MULTIPLIER / (HOURS_PER_DAY * DAYS_PER_MONTH * MONTHS_PER_YEAR)

        # Consumo ajustado v1.12:
        # antes se morían demasiado rápido por sed/hambre antes de formar sociedad.
        consumption = 1.0
        if self.age < 4:
            consumption = CHILD_CONSUMPTION_MULTIPLIER
        elif self.age < REPRODUCTION_AGE:
            consumption = ADOLESCENT_CONSUMPTION_MULTIPLIER

        # v1.12.1:
        # Sed más realista, pero sin romper la simulación:
        # - sin beber, la sed cae a zona crítica en pocos días, no en 11+ días.
        # - el hambre baja más lento que la sed, como en la vida real.
        self.hunger -= 0.26 * consumption
        self.thirst -= 0.82 * consumption
        self.energy -= 0.20 * consumption

        # Diagnóstico de movimiento: registra cuántas horas lleva en la misma casilla.
        current_pos = (self.x, self.y)
        if self.last_position == current_pos:
            self.still_hours += 1
        else:
            self.still_hours = 0
            self.last_position = current_pos

        if world.weather in ["ola_calor", "sequía"] or world.temperature >= 36:
            self.thirst -= 0.90
            self.energy -= 0.35
            if world.temperature >= 40 and not world.tile_close(self.x, self.y, WATER, 3) and not world.is_shelter(self.x, self.y):
                self.life -= 0.18

        if (world.weather in ["frio_extremo", "nieve"] or world.temperature <= 0) and not world.is_shelter(self.x, self.y):
            cold_damage = 0.35
            if world.temperature <= -8:
                cold_damage = 0.70
            if world.tribe_knowledge.get("fuego"):
                cold_damage *= 0.55
            if self.cold_protection > 0:
                cold_damage *= max(0.20, 1.0 - self.cold_protection)
            self.life -= cold_damage
            self.energy -= 0.40

        if world.weather in ["lluvia_fuerte", "tormenta"] and not world.is_shelter(self.x, self.y):
            self.energy -= 0.25

        if self.hunger <= 0:
            self.life -= 0.42
        if self.thirst <= 0:
            self.life -= 1.45
        if self.energy <= 0:
            self.life -= 0.18

        # Lesiones activas.
        if self.injuries:
            self.energy -= 0.05 * len(self.injuries)
            if world.is_shelter(self.x, self.y) and chance(0.18):
                self.life = clamp(self.life + 1.2, 0, 100)

        self.receive_help_from_group(world)
        self.try_use_herbs(world)

        self.life = clamp(self.life, 0, 100)
        self.hunger = clamp(self.hunger, 0, 100)
        self.thirst = clamp(self.thirst, 0, 100)
        self.energy = clamp(self.energy, 0, self.energy_max)

        self.fitness += 0.01
        if self.life > 60 and self.hunger > 50 and self.thirst > 50:
            self.fitness += 0.02

        # Última oportunidad: si la vida cae demasiado, intenta una acción de emergencia
        # antes de morir por "pérdida de vida" sin haber buscado solución.
        if self.life <= 8 and self.age <= self.max_age():
            helped = self.receive_help_from_group(world)
            if not helped and self.inventory.get("herbs", 0) > 0:
                self.try_use_herbs(world)
            elif not helped and world.is_shelter(self.x, self.y):
                self.life = clamp(self.life + 3, 0, 100)

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
        # Prioridad absoluta: con sed crítica no socializa, no construye, no vuelve al spawn.
        if self.thirst < 92:
            if world.map[self.y][self.x] == WATER or world.tile_close(self.x, self.y, WATER, 1):
                return "beber"
            return "buscar_agua"

        # Modo supervivencia médica:
        # si pierde vida, no sigue con la rutina normal; busca solución.
        if self.life < 72 or self.injuries:
            return "curarse"

        if self.hunger < 68:
            if self.inventory["food"] > 0 or self.inventory["raw_meat"] > 0:
                return "comer"
            return "buscar_comida"
        if world.danger_close(self.x, self.y, 2) and (self.life < 70 or self.skills.get("strength", 50) < 75):
            return "huir"
        if (world.weather in ["tormenta", "tornado", "terremoto", "frio_extremo"] or world.temperature <= -2) and not world.is_shelter(self.x, self.y):
            return "volver_refugio"

        if world.temperature >= 38 and self.thirst < 88:
            if world.map[self.y][self.x] == WATER or world.tile_close(self.x, self.y, WATER, 1):
                return "beber"
            return "buscar_agua"
        if world.is_night() and not world.is_shelter(self.x, self.y) and world.shelter_close(self.x, self.y, 45):
            return "volver_refugio"
        if world.is_night() and world.is_shelter(self.x, self.y) and self.energy < 92:
            return "descansar"

        # Si está tocado por intoxicación/lesiones, prioriza sobrevivir.
        if self.life < 55 or self.injuries:
            if self.inventory.get("herbs", 0) > 0:
                return "fabricar"
            if world.is_shelter(self.x, self.y) or self.energy < 45:
                return "descansar"
            return "volver_refugio"

        if self.energy < 25:
            return "descansar"

        if self.thirst > 70 and self.should_return_to_report() and not self.can_tell_group(world):
            return "reunirse"

        if self.energy > 55 and self.hunger > 55 and self.thirst > 55:
            build_signal = world.best_shared_signal("build", self.x, self.y, max_distance=90)
            if build_signal and world.crowd_count(build_signal[0], build_signal[1], radius=7) < 8 and chance(0.32):
                return "construir"

        # Si lleva haciendo el mismo bucle de destinos, rompe patrón buscando otro recurso.
        if len(self.recent_targets) >= 6:
            unique_recent = len(set(self.recent_targets[-6:]))
            if unique_recent <= 2 and self.energy > 45 and self.thirst > 45 and self.hunger > 45 and chance(0.45):
                return random.choice(["buscar_comida", "fabricar", "explorar"])

        # Primeras generaciones: más espíritu aventurero.
        if self.generation <= 5:
            if not self.memory_places["water"] and chance(0.35):
                return "buscar_agua"
            if not self.memory_places["food"] and chance(0.18):
                return "buscar_comida"
            if self.name in ["Adán", "Eva"] and self.thirst < 95 and not self.memory_places["water"]:
                return "buscar_agua"

        outputs = self.brain.outputs(self.neural_inputs(world))

        for i, action_name in enumerate(ACTIONS):
            outputs[i] += self.learned_actions.get(action_name, 0.0)

        if self.generation <= 15:
            early_bias = {
                "socializar": 1.20,
                "buscar_comida": 0.70,
                "buscar_agua": 0.90,
                "volver_refugio": 0.55,
                "construir": 0.45,
                "fabricar": 0.25,
                "cazar": -0.45,
                "explorar": -0.15,
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

        if action == "cazar":
            # No queremos que las primeras generaciones se suiciden cazando sin arma.
            if self.weapon == "ninguna" and self.hunger > 18:
                return "buscar_comida"
            if self.energy < 45:
                return "descansar"

        if action == "fabricar":
            if self.energy < 45 or (self.inventory.get("wood", 0) < 1 and self.inventory.get("stone", 0) < 1):
                return "explorar"

        return action

    def act(self, world: "World") -> None:
        if not self.alive:
            return

        self.deliver_pending_signals(world)

        # Anti-bloqueo de emergencia:
        # si lleva demasiadas horas quieto y tiene sed, fuerza búsqueda de agua.
        if self.still_hours >= 8 and self.thirst < 55:
            self.seek_water(world)
            self.learned_actions["buscar_agua"] += 0.25
            return

        action = self.decide_action(world)

        before_life = self.life
        before_hunger = self.hunger
        before_thirst = self.thirst
        before_energy = self.energy
        before_food = self.inventory.get("food", 0)
        before_wood = self.inventory.get("wood", 0)
        before_stone = self.inventory.get("stone", 0)
        before_extra_resources = (
            self.inventory.get("hides", 0)
            + self.inventory.get("bones", 0)
            + self.inventory.get("herbs", 0)
            + self.inventory.get("seeds", 0)
            + self.inventory.get("clay", 0)
            + self.inventory.get("metal_ore", 0)
            + self.inventory.get("metal", 0)
        )

        if action == "explorar":
            self.explore(world)
        elif action == "buscar_agua":
            self.seek_water(world)
        elif action == "buscar_comida":
            self.seek_food(world)
        elif action == "curarse":
            self.seek_healing(world)
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
        elif action == "reunirse":
            self.meet_group(world)
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

        self.deliver_pending_signals(world)

        # Aprendizaje en vida: refuerzo simple por resultado.
        reward = 0.0
        reward += (self.life - before_life) * 0.08
        reward += (self.hunger - before_hunger) * 0.025
        reward += (self.thirst - before_thirst) * 0.03
        reward += (self.energy - before_energy) * 0.012
        reward += (self.inventory.get("food", 0) - before_food) * 0.8
        reward += (self.inventory.get("wood", 0) - before_wood) * 0.12
        reward += (self.inventory.get("stone", 0) - before_stone) * 0.16
        after_extra_resources = (
            self.inventory.get("hides", 0)
            + self.inventory.get("bones", 0)
            + self.inventory.get("herbs", 0)
            + self.inventory.get("seeds", 0)
            + self.inventory.get("clay", 0)
            + self.inventory.get("metal_ore", 0)
            + self.inventory.get("metal", 0)
        )
        reward += (after_extra_resources - before_extra_resources) * 0.20

        if not self.alive:
            reward -= 20.0

        old_value = self.learned_actions.get(action, 0.0)
        self.learned_actions[action] = clamp(old_value * 0.92 + reward * 0.08, -6.0, 6.0)


    def move_to(self, world: "World", target: Tuple[int, int], steps: int = 1) -> None:
        if not self.recent_targets or self.recent_targets[-1] != target:
            self.recent_targets.append(target)
            self.recent_targets = self.recent_targets[-8:]

        moved = False

        for _ in range(steps):
            tx, ty = target

            # Si el destino está ocupado, apunta a una casilla libre alrededor.
            if world.human_occupied(tx, ty, exclude=self):
                nearby = world.find_free_near_target(tx, ty, radius=5, human=self)
                if nearby:
                    tx, ty = nearby

            # Paso directo ideal.
            nx, ny = step_towards(self.x, self.y, tx, ty)

            if world.walkable_for_human(nx, ny, human=self):
                # Evita entrar en embudos muy llenos.
                if world.crowd_count(nx, ny, radius=2) >= 3 and chance(0.55):
                    smarter = world.best_step_towards(self, (tx, ty))
                    if smarter:
                        nx, ny = smarter

                self.x, self.y = nx, ny
                self.energy -= 0.25
                moved = True
                continue

            # Si el paso directo falla, no te quedes quieto:
            # busca el vecino caminable que más se acerque al destino.
            smarter = world.best_step_towards(self, (tx, ty))
            if smarter:
                self.x, self.y = smarter
                self.energy -= 0.35
                moved = True
                continue

            # Último recurso: cualquier casilla libre cercana.
            alternative = world.find_free_adjacent_cell(self.x, self.y, radius=3, human=self)
            if alternative:
                self.x, self.y = alternative
                self.energy -= 0.45
                moved = True
                continue

            break

        if not moved:
            # Si no se ha movido nada, gasta un poco de energía pero no queda atrapado
            # en una falsa "decisión" durante cientos de horas.
            self.energy -= 0.05


    def random_move(self, world: "World", radius: int = 1) -> None:
        candidates = []
        for _ in range(8):
            nx = self.x + random.randint(-radius, radius)
            ny = self.y + random.randint(-radius, radius)
            if world.walkable_for_human(nx, ny, human=self):
                candidates.append((nx, ny))

        if candidates:
            self.x, self.y = random.choice(candidates)
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
        # Busca acceso al río, no el mismo punto exacto para todos.
        shared = world.best_shared_signal("water", self.x, self.y, max_distance=180)

        radius = 48
        if self.generation <= 5 or self.name in ["Adán", "Eva"]:
            radius = 110
        if self.thirst < 35:
            radius = 170

        access = world.find_water_access_cell(self.x, self.y, radius=radius, human=self)

        if not access and shared:
            access = world.find_water_access_cell(shared[0], shared[1], radius=12, human=self) or shared

        if access:
            self.last_water_target = access
            self.memory_places["water"].append(access)
            steps = 5 if self.thirst < 45 or self.generation <= 5 else 3
            self.move_to(world, access, steps=steps)
            if world.tile_close(self.x, self.y, WATER, 1):
                self.drink(world)
            return

        if self.memory_places["water"]:
            remembered = random.choice(self.memory_places["water"])
            access = world.find_water_access_cell(remembered[0], remembered[1], radius=12, human=self) or remembered
            self.move_to(world, access, steps=4 if self.thirst < 55 else 3)
            if world.tile_close(self.x, self.y, WATER, 1):
                self.drink(world)
            return

        # Sin memoria ni señales:
        # exploración fuerte hacia el río, con dispersión vertical para no formar una línea.
        # En este mapa el río inicial está bastante a la izquierda, pero no van todos por el mismo y.
        if self.generation <= 5 or self.thirst < 70:
            target_x = int(clamp(self.x - random.randint(8, 16), 0, MAP_W - 1))
            target_y = int(clamp(self.y + random.randint(-14, 14), 0, MAP_H - 1))
            self.move_to(world, (target_x, target_y), steps=5 if self.thirst < 35 else 3)
            self.energy -= 0.8
            return

        self.explore(world)


    def seek_food(self, world: "World") -> None:
        shared = world.best_shared_signal("food", self.x, self.y, max_distance=120)
        plant = world.find_nearest_plant(self.x, self.y, 22)
        if plant:
            self.memory_places["food"].append((plant.x, plant.y))
            if world.count_plants_close(plant.x, plant.y, 5) >= 3:
                self.queue_signal("food", (plant.x, plant.y), strength=1.3, ttl_hours=96)
            self.move_to(world, (plant.x, plant.y), steps=2)
            if distance((self.x, self.y), (plant.x, plant.y)) <= 1.5:
                self.gather(world)
        elif shared:
            self.memory_places["food"].append(shared)
            self.move_to(world, shared, steps=2)
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
            self.thirst = clamp(self.thirst + 90, 0, 100)
            self.energy = clamp(self.energy + 2, 0, self.energy_max)
            self.memory_places["water"].append((self.x, self.y))
            # No avisa instantáneamente: tiene que volver al grupo/refugio.
            self.queue_signal("water", (self.x, self.y), strength=1.4, ttl_hours=160)
            self.fitness += 0.2
        else:
            self.seek_water(world)

    def rest(self, world: "World") -> None:
        tile = world.map[self.y][self.x] if world.in_bounds(self.x, self.y) else GRASS
        if tile == BROWN_HOUSE:
            bonus = 1.45
        elif tile in [HOUSE, BUILDING]:
            bonus = 1.25
        elif tile in [CAVE, HUT]:
            bonus = 1.0
        else:
            bonus = 0.45

        if world.is_night():
            bonus += 0.30

        self.energy = clamp(self.energy + 6.4 * bonus, 0, self.energy_max)
        heal = 0.40 * bonus
        if self.life < 60:
            heal += 1.15 * bonus
        if self.injuries and world.is_shelter(self.x, self.y):
            heal += 0.65 * bonus
            if chance(0.10 * bonus):
                self.injuries.pop(0)
        self.life = clamp(self.life + heal, 0, 100)

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

    def meet_group(self, world: "World") -> None:
        pos = world.nearest_living_human_pos(self, radius=140)
        if pos:
            target = world.find_free_near_target(pos[0], pos[1], radius=4, human=self) or pos
            self.move_to(world, target, steps=3)
            self.deliver_pending_signals(world)
            return
        self.return_shelter(world)

    def return_shelter(self, world: "World") -> None:
        # No volver siempre al spawn: elige refugio/asentamiento menos saturado.
        pos = world.best_home_for(self)

        if pos:
            self.home_pos = pos
            self.memory_places["shelter"].append(pos)
            self.move_to(world, pos, steps=4 if world.is_night() else 3)
            if distance((self.x, self.y), pos) <= 1.5 and world.is_night():
                self.rest(world)
            return

        if self.memory_places["shelter"]:
            shelter = random.choice(self.memory_places["shelter"])
            target = world.find_free_near_target(shelter[0], shelter[1], radius=5, human=self) or shelter
            self.move_to(world, target, steps=4 if world.is_night() else 3)
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
            meat_gain = max(1, int(animal.data["meat"] // 20))
            self.inventory["raw_meat"] += meat_gain

            self.inventory["bones"] += max(1, int(animal.strength // 25))
            if animal.data["id"] in ["ciervo", "cabra", "lobo", "oso", "jabali", "pantera"]:
                self.inventory["hides"] += max(1, int(animal.strength // 30))

            self.fitness += 0.6
            world.log(f"{self.name} cazó un/a {animal.data['name']} y obtiene carne/huesos/pieles.")
        else:
            # Los pacíficos también se defienden: ciervos/cabras/gallinas no son muñecos.
            risk = 0.08 + animal.strength / 180
            if animal.aggressive:
                risk += 0.22
                cause = "fallar una caza peligrosa contra depredador"
            else:
                defend_bonus = 0.00
                if animal.strength >= 18:
                    defend_bonus = 0.08
                if animal.data["id"] in ["ciervo", "cabra", "gallina"]:
                    defend_bonus += 0.04
                risk += defend_bonus
                cause = f"{animal.data['name']} se defiende durante la caza"

            self.accident_risk(world, risk, ["corte", "mordedura", "herida_grave", "fractura"], cause=cause)
            if not animal.aggressive and animal.strength >= 18 and chance(0.25):
                animal.move(world)

    def gather(self, world: "World") -> None:
        plant = world.plant_at_or_near(self.x, self.y, 1)
        if not plant:
            self.seek_food(world)
            return

        pid = plant.data["id"]

        # Si no está hambriento de verdad y no conoce esa planta,
        # mejor no arriesgarse a comerla.
        if pid not in self.memory_food and self.hunger > 45 and chance(0.65):
            self.memory_places["food"].append((plant.x, plant.y))
            self.explore(world)
            return

        if plant in world.plants:
            world.plants.remove(plant)

        if plant.data["poison"]:
            # v1.12: los humanos no se comen siempre toda fruta venenosa.
            # Pueden desconfiar, probar poco, aprender y evitar morir tan rápido.
            caution = self.personality.get("caution", 50) / 100
            memory = self.skills.get("memory", 50) / 100
            generation_bonus = min(0.25, max(0, self.generation - 1) * 0.03)
            avoid_chance = 0.25 + caution * 0.25 + memory * 0.20 + generation_bonus

            if chance(avoid_chance):
                self.memory_food[pid] = "venenosa"
                self.memory_places["danger"].append((plant.x, plant.y))
                self.queue_signal("medical", (plant.x, plant.y), strength=0.8, ttl_hours=120)
                world.log(f"{self.label()} evita una planta venenosa y aprende a reconocerla.")
                return

            self.hunger = clamp(self.hunger + 2, 0, 100)
            # Menos daño que antes: intoxicación puede matar, pero no casi siempre.
            if chance(0.70):
                self.apply_injury(world, "intoxicacion")
            else:
                self.apply_injury(world, "enfermedad")
            if self.generation <= 2:
                self.life = clamp(self.life + 8, 0, 100)
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
            self.memory_places["food"].append((plant.x, plant.y))
            if chance(0.35):
                self.inventory["seeds"] += 1
            herb_chance = 0.35 if self.needs_healing() else 0.18
            if chance(herb_chance):
                self.inventory["herbs"] += 1
                self.queue_signal("medical", (plant.x, plant.y), strength=1.0, ttl_hours=90)
            self.queue_signal("food", (plant.x, plant.y), strength=1.0, ttl_hours=72)
            self.fitness += 0.25

    def craft(self, world: "World") -> None:
        # Fabricación básica y nuevos recursos.
        tile = world.map[self.y][self.x]

        if tile == FOREST or world.tile_close(self.x, self.y, FOREST, 3):
            self.inventory["wood"] += 1
            self.energy -= 0.8

        if tile == MOUNTAIN or world.tile_close(self.x, self.y, MOUNTAIN, 3):
            self.inventory["stone"] += 1
            self.energy -= 1.0
            if chance(0.18):
                self.inventory["metal_ore"] += 1

        if tile == SWAMP or world.tile_close(self.x, self.y, SWAMP, 3) or world.tile_close(self.x, self.y, WATER, 2):
            if chance(0.45):
                self.inventory["clay"] += 1

        mechanics = self.skills.get("mechanics", 50)
        focus = self.skills.get("focus", 50)

        if self.inventory.get("herbs", 0) >= 1 and (self.life < 82 or self.injuries):
            self.inventory["herbs"] -= 1
            self.life = clamp(self.life + 16, 0, 100)
            if self.injuries and chance(0.55):
                self.injuries.pop(0)
            world.tribe_knowledge["medicina_herbal"] = True
            self.queue_signal("medical", (self.x, self.y), strength=1.2, ttl_hours=100)
            world.log(f"{self.label()} prepara hierbas medicinales.")
            return

        if self.clothing == "ninguna" and self.inventory.get("hides", 0) >= 2:
            self.inventory["hides"] -= 2
            self.clothing = "ropa de piel"
            self.cold_protection = 0.45
            world.tribe_knowledge["ropa_piel"] = True
            world.log(f"{self.label()} fabrica ropa de piel contra el frío.")
            return

        if self.clothing == "ropa de piel" and self.inventory.get("hides", 0) >= 3 and self.inventory.get("bones", 0) >= 1:
            self.inventory["hides"] -= 3
            self.inventory["bones"] -= 1
            self.clothing = "abrigo de piel"
            self.cold_protection = 0.70
            world.tribe_knowledge["ropa_piel"] = True
            world.log(f"{self.label()} mejora su ropa a abrigo de piel.")
            return

        if self.weapon == "ninguna" and self.inventory.get("bones", 0) >= 2 and self.inventory.get("wood", 0) >= 1:
            self.inventory["bones"] -= 2
            self.inventory["wood"] -= 1
            self.weapon = "herramienta de hueso"
            self.weapon_power = 7
            world.tribe_knowledge["herramientas_hueso"] = True
            world.log(f"{self.label()} fabrica una herramienta de hueso.")
            return

        if self.weapon == "ninguna" and self.inventory.get("wood", 0) >= 2 and mechanics >= 25:
            self.inventory["wood"] -= 2
            self.weapon = "palo"
            self.weapon_power = 5
            world.log(f"{self.label()} fabrica un palo.")
            return

        if self.weapon in ["ninguna", "palo", "herramienta de hueso"] and self.inventory.get("wood", 0) >= 3 and self.inventory.get("stone", 0) >= 1 and mechanics + focus >= 90:
            self.inventory["wood"] -= 3
            self.inventory["stone"] -= 1
            self.weapon = "lanza básica"
            self.weapon_power = 11
            world.tribe_knowledge["herramientas_piedra"] = True
            world.log(f"{self.label()} fabrica una lanza básica.")
            return

        if world.tribe_knowledge.get("fuego") and self.inventory.get("metal_ore", 0) >= 3 and self.inventory.get("wood", 0) >= 2:
            self.inventory["metal_ore"] -= 3
            self.inventory["wood"] -= 2
            self.inventory["metal"] += 1
            world.tribe_knowledge["metalurgia"] = True
            world.log(f"{self.label()} consigue procesar mineral metálico.")
            return

        if self.inventory.get("metal", 0) >= 2 and self.inventory.get("wood", 0) >= 2 and self.weapon_power < 18:
            self.inventory["metal"] -= 2
            self.inventory["wood"] -= 2
            self.weapon = "lanza metálica"
            self.weapon_power = 18
            world.tribe_knowledge["metalurgia"] = True
            world.log(f"{self.label()} fabrica una lanza metálica.")
            return

        if self.try_primitive_farming(world):
            return

        self.explore(world)


    def build(self, world: "World") -> None:
        # Construcción progresiva. Requiere materiales.
        tile = world.map[self.y][self.x]

        build_signal = world.best_shared_signal("build", self.x, self.y, max_distance=100)
        if build_signal and distance((self.x, self.y), build_signal) > 3 and self.energy > 45 and self.thirst > 45 and self.hunger > 45:
            self.move_to(world, build_signal, steps=2)
            return

        if tile == WATER:
            self.explore(world)
            return

        if self.energy < 50 or self.hunger < 45 or self.thirst < 45:
            self.rest(world)
            return

        # Recolecta materiales si está cerca de recursos.
        if tile == FOREST or world.tile_close(self.x, self.y, FOREST, 4):
            self.inventory["wood"] += 1
            self.energy -= 1.0
            if self.inventory.get("wood", 0) >= 4:
                self.queue_signal("build", (self.x, self.y), strength=0.8, ttl_hours=48)

        if tile == MOUNTAIN or world.tile_close(self.x, self.y, MOUNTAIN, 4):
            self.inventory["stone"] += 1
            self.energy -= 1.2

        nearby_shelters = world.count_shelters_close(self.x, self.y, 10)
        living = max(1, len(world.living_humans()))
        needed_shelters = max(1, math.ceil(living / 3))

        # Si ya hay suficientes refugios cerca, no spamea construcción.
        if nearby_shelters >= needed_shelters and chance(0.70):
            self.explore(world)
            return

        mechanics = self.skills.get("mechanics", 50)
        focus = self.skills.get("focus", 50)

        if self.inventory.get("clay", 0) >= 5 and world.tribe_knowledge.get("choza"):
            self.inventory["clay"] -= 5
            world.tribe_knowledge["almacen"] = True
            world.log(f"{self.label()} fabrica vasijas de arcilla para almacenar recursos.")
            self.fitness += 0.8
            return

        # Casa segura marrón 3x3:
        # RRR
        # R@R / RAR si Adán está dentro
        # RRR
        if self.inventory.get("wood", 0) >= 8:
            x0 = int(clamp(self.x - 1, 0, MAP_W - 3))
            y0 = int(clamp(self.y - 1, 0, MAP_H - 3))
            if world.can_place_structure(x0, y0, 3, 3):
                self.inventory["wood"] -= 8
                world.place_structure(x0, y0, 3, 3, BROWN_HOUSE)
                world.tribe_knowledge["choza"] = True
                world.tribe_knowledge["casa_segura"] = True
                world.buildings += 1
                world.log(f"{self.label()} construyó una casa segura R 3x3.")
                self.queue_signal("build", (self.x, self.y), strength=1.0, ttl_hours=36)
                self.fitness += 1.4
                return

        # Casa segura marrón grande 6x3:
        # RRRRRR
        # R@@@@R si hay personas dentro
        # RRRRRR
        if self.inventory.get("wood", 0) >= 16 and mechanics + focus >= 90:
            x0 = int(clamp(self.x - 2, 0, MAP_W - 6))
            y0 = int(clamp(self.y - 1, 0, MAP_H - 3))
            if world.can_place_structure(x0, y0, 6, 3):
                self.inventory["wood"] -= 16
                world.place_structure(x0, y0, 6, 3, BROWN_HOUSE)
                world.tribe_knowledge["casa_segura"] = True
                world.tribe_knowledge["casa_madera"] = True
                world.buildings += 1
                world.log(f"{self.label()} construyó una casa segura R 6x3.")
                self.queue_signal("build", (self.x, self.y), strength=1.0, ttl_hours=36)
                self.fitness += 2.2
                return

        # Choza simple de madera.
        if not world.tribe_knowledge.get("casa_madera"):
            hut_cost = 12
            if self.inventory.get("wood", 0) >= hut_cost and tile in [GRASS, FOREST]:
                self.inventory["wood"] -= hut_cost
                world.map[self.y][self.x] = HUT
                world.tribe_knowledge["choza"] = True
                world.buildings += 1
                world.log(f"{self.label()} construyó una choza de madera.")
                self.fitness += 1.0
                return

        # Casa de madera.
        if world.tribe_knowledge.get("choza") and world.tribe_knowledge.get("herramientas_piedra"):
            if self.inventory.get("wood", 0) >= 25 and mechanics + focus >= 110 and tile in [GRASS, FOREST, HUT]:
                self.inventory["wood"] -= 25
                world.map[self.y][self.x] = HOUSE
                world.tribe_knowledge["casa_madera"] = True
                world.buildings += 1
                world.log(f"{self.label()} construyó una casa de madera.")
                self.fitness += 2.0
                return

        # Casa de piedra / edificio simple.
        if world.tribe_knowledge.get("casa_madera"):
            if self.inventory.get("wood", 0) >= 10 and self.inventory.get("stone", 0) >= 25 and mechanics + focus >= 125:
                self.inventory["wood"] -= 10
                self.inventory["stone"] -= 25
                world.map[self.y][self.x] = BUILDING
                world.tribe_knowledge["casa_piedra"] = True
                world.tribe_knowledge["ingenieria"] = True
                world.buildings += 1
                world.log(f"{self.label()} construyó una casa de piedra.")
                self.fitness += 3.0
                return

        # Si no tiene materiales, busca el recurso que falta.
        if self.inventory.get("wood", 0) < 8:
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
                elif tile in [CAVE, HUT, HOUSE, BUILDING, BROWN_HOUSE]:
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
        self.spawn_plants(1700)
        self.spawn_initial_animals()

        cx, cy = self.cave_pos

        # Adán y Eva aparecen siempre en casillas caminables junto a la cueva.
        # Adán queda en la cueva; Eva a la derecha, nunca en la misma casilla.
        self.map[cy][cx] = CAVE
        adam = Human("Adán", "hombre", cx, cy, age=18.0)

        eve_x = int(clamp(cx + 1, 0, MAP_W - 1))
        eve_y = cy
        if not self.walkable(eve_x, eve_y) or self.map[eve_y][eve_x] == WATER:
            eve_x = int(clamp(cx + 2, 0, MAP_W - 1))
        if (eve_x, eve_y) == (cx, cy):
            eve_x = int(clamp(cx + 1, 0, MAP_W - 1))
        if self.map[eve_y][eve_x] == WATER:
            self.map[eve_y][eve_x] = GRASS

        eve = Human("Eva", "mujer", eve_x, eve_y, age=18.0)
        adam.person_id = 1
        eve.person_id = 2
        adam.generation = 1
        eve.generation = 1

        # v1.3.2: arranque menos letal.
        for starter in [adam, eve]:
            starter.inventory["food"] = 3
            starter.inventory["herbs"] = 1
            starter.hunger = 92
            starter.thirst = 88
            starter.energy = min(starter.energy_max, 92)
            starter.life = 100

        if not hasattr(self, "person_count"):
            self.person_count = 2
        self.all_humans = {1: adam, 2: eve}
        self.max_humans_alive = 2
        self.max_humans_time = f"Año {self.year}, Mes {self.month}, Día {self.day}, Hora {self.hour:02d}:00"

        # Memoria inicial mínima: conocen la cueva.
        for h in [adam, eve]:
            h.memory_places["shelter"].append(self.cave_pos)
            h.remember_surroundings(self)
            nearby_water = self.find_nearest_tile(h.x, h.y, WATER, 15)
            if nearby_water:
                h.memory_places["water"].append(nearby_water)

        self.humans = [adam, eve]
        self.person_count = 2
        self.all_humans = {1: adam, 2: eve}
        self.max_humans_alive = 2
        self.max_humans_time = f"Año {self.year}, Mes {self.month}, Día {self.day}, Hora {self.hour:02d}:00"

        # Quitamos plantas o animales justo encima del spawn inicial
        # para que A/E se vean desde el primer render.
        spawn_cells = {(adam.x, adam.y), (eve.x, eve.y)}
        self.plants = [p for p in self.plants if (p.x, p.y) not in spawn_cells]
        self.animals = [a for a in self.animals if (a.x, a.y) not in spawn_cells]

        # v1.3.2: zona inicial menos letal.
        safe_plants = [p for p in PLANTS if not p["poison"]]
        self.plants = [
            p for p in self.plants
            if not (distance((p.x, p.y), self.cave_pos) <= 12 and p.data.get("poison"))
        ]
        for _ in range(18):
            px = int(clamp(cx + random.randint(-10, 10), 0, MAP_W - 1))
            py = int(clamp(cy + random.randint(-10, 10), 0, MAP_H - 1))
            if self.map[py][px] in [GRASS, FOREST]:
                self.plants.append(Plant(px, py, random.choice(safe_plants)))

        self.log("El mundo ha sido creado. Adán y Eva aparecen en la cueva.")

    def generate_river(self) -> None:
        # v1.2.3:
        # Río vertical recolocado bastante más a la izquierda,
        # aproximadamente donde se marcó en azul en el mapa global.
        # Queda en torno a x=65 en un mundo de 256.
        x = MAP_W // 2 - 63

        for y in range(MAP_H):
            width = random.choice([2, 2, 3])
            for w in range(width):
                xx = clamp(x + w, 0, MAP_W - 1)
                self.map[y][int(xx)] = WATER

            if chance(0.26):
                x += random.choice([-1, 0, 1])
                x = int(clamp(x, MAP_W // 2 - 68, MAP_W // 2 - 58))

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
        # v1.2.2: cueva fija en el centro, con zona inicial despejada.
        # El río queda bastante más a la izquierda, como en la zona azul marcada.
        x = MAP_W // 2
        y = MAP_H // 2

        # Limpiamos una pequeña zona inicial para evitar que bosque/montaña/pantano
        # o restos de generación tapen el spawn.
        for yy in range(max(0, y - 3), min(MAP_H, y + 4)):
            for xx in range(max(0, x - 3), min(MAP_W, x + 4)):
                if self.map[yy][xx] != WATER:
                    self.map[yy][xx] = GRASS

        self.map[y][x] = CAVE
        self.cave_pos = (x, y)

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

    def spawn_animal_near(self, animal_id: str, x: int, y: int, radius: int = 8) -> bool:
        data = next(a for a in ANIMALS if a["id"] == animal_id)

        for _ in range(40):
            nx = int(clamp(x + random.randint(-radius, radius), 0, MAP_W - 1))
            ny = int(clamp(y + random.randint(-radius, radius), 0, MAP_H - 1))
            tile = self.map[ny][nx]

            if animal_id == "pez":
                if self.tile_close(nx, ny, WATER, 1):
                    self.animals.append(Animal(nx, ny, data))
                    return True
            elif data["aggressive"]:
                if distance((nx, ny), self.cave_pos) > 35 and tile in [FOREST, MOUNTAIN, SWAMP, GRASS]:
                    self.animals.append(Animal(nx, ny, data))
                    return True
            else:
                if tile in [FOREST, GRASS, SWAMP]:
                    self.animals.append(Animal(nx, ny, data))
                    return True

        return False

    def animal_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for a in self.animals:
            if a.alive:
                animal_id = a.data["id"]
                counts[animal_id] = counts.get(animal_id, 0) + 1
        return counts

    def animal_ecology_daily(self) -> None:
        """
        Reproducción animal controlada.
        No es como la humana: sube poco a poco y con límites.
        - Pacíficos crecen más.
        - Agresivos crecen mucho menos.
        - Si hay demasiados, se frena.
        """
        counts = self.animal_counts()
        peaceful_total = sum(1 for a in self.animals if a.alive and not a.aggressive)
        aggressive_total = sum(1 for a in self.animals if a.alive and a.aggressive)

        newborns = 0
        newborn_predators = 0

        for animal_id, base_rate in ANIMAL_REPRODUCTION.items():
            data = next(a for a in ANIMALS if a["id"] == animal_id)
            current = counts.get(animal_id, 0)
            if current < 2:
                continue

            is_aggressive = bool(data["aggressive"])

            if is_aggressive:
                if aggressive_total >= AGGRESSIVE_ANIMAL_HARD_CAP:
                    continue
                pressure = max(0.08, 1.0 - aggressive_total / AGGRESSIVE_ANIMAL_SOFT_CAP)
            else:
                if peaceful_total >= PEACEFUL_ANIMAL_HARD_CAP:
                    continue
                pressure = max(0.12, 1.0 - peaceful_total / PEACEFUL_ANIMAL_SOFT_CAP)

            season = self.season()
            season_mult = {
                "primavera": 1.25,
                "verano": 1.05,
                "otoño": 0.85,
                "invierno": 0.45,
            }[season]

            expected = current * base_rate * pressure * season_mult
            births = int(expected)
            if chance(expected - births):
                births += 1

            births = min(births, 3 if is_aggressive else 12)

            parents = [a for a in self.animals if a.alive and a.data["id"] == animal_id]
            for _ in range(births):
                if not parents:
                    break
                parent = random.choice(parents)
                if self.spawn_animal_near(animal_id, parent.x, parent.y, radius=10):
                    if is_aggressive:
                        newborn_predators += 1
                        aggressive_total += 1
                    else:
                        newborns += 1
                        peaceful_total += 1

        # Migración mínima para evitar mapas muertos.
        if peaceful_total < 180:
            for _ in range(8):
                self.spawn_daily_animals(1)
                newborns += 1

        if aggressive_total < 18 and chance(0.30):
            self.spawn_specific_animal(random.choice(["lobo", "serpiente", "jabali", "pantera", "oso"]))
            newborn_predators += 1

        if newborns or newborn_predators:
            self.log(f"Fauna: nacen/migran {newborns} pacíficos y {newborn_predators} depredadores.")

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
            "primavera": 60,
            "verano": 45,
            "otoño": 35,
            "invierno": 14,
        }[season]
        self.spawn_plants(grow)

        # Ecología animal: reproducción/migración lenta y controlada.
        self.animal_ecology_daily()

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
            # Primavera suave, con noches frías ocasionales.
            self.temperature = random.uniform(4, 24)

        elif season == "verano":
            options = [
                ("soleado", 0.45),
                ("nublado", 0.12),
                ("lluvia", 0.10),
                ("tormenta", 0.08),
                ("ola_calor", 0.07),
                ("sequía", 0.05),
            ]
            # Verano más duro: las olas de calor pueden ser peligrosas.
            self.temperature = random.uniform(20, 43)

        elif season == "otoño":
            options = [
                ("soleado", 0.25),
                ("nublado", 0.20),
                ("lluvia", 0.25),
                ("lluvia_fuerte", 0.08),
                ("tormenta", 0.05),
                ("niebla", 0.10),
            ]
            self.temperature = random.uniform(2, 22)

        else:
            options = [
                ("soleado", 0.20),
                ("nublado", 0.20),
                ("lluvia", 0.12),
                ("nieve", 0.14),
                ("frio_extremo", 0.08),
                ("niebla", 0.12),
            ]
            # Invierno más duro: frío real obliga a refugio/fuego.
            self.temperature = random.uniform(-14, 8)

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

        if self.tribe_knowledge.get("ropa_piel", False) and not self.tribe_knowledge.get("adaptacion_frio", False):
            if chance(0.020):
                self.tribe_knowledge["adaptacion_frio"] = True
                self.log("La tribu aprende a protegerse mejor del frío con pieles.")

        if self.tribe_knowledge.get("medicina_herbal", False) and not self.tribe_knowledge.get("botanica", False):
            if avg_memory > 52 and chance(0.016):
                self.tribe_knowledge["botanica"] = True
                self.log("La tribu empieza a reconocer hierbas medicinales.")

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

    def human_at(self, x: int, y: int, exclude: Optional[Human] = None) -> Optional[Human]:
        for h in self.living_humans():
            if exclude is not None and h is exclude:
                continue
            if h.x == x and h.y == y:
                return h
        return None

    def human_occupied(self, x: int, y: int, exclude: Optional[Human] = None) -> bool:
        return self.human_at(x, y, exclude=exclude) is not None

    def walkable_for_human(self, x: int, y: int, human: Optional[Human] = None) -> bool:
        if not self.walkable(x, y):
            return False
        return not self.human_occupied(x, y, exclude=human)

    def find_free_adjacent_cell(self, x: int, y: int, radius: int = 4, human: Optional[Human] = None) -> Optional[Tuple[int, int]]:
        # Busca primero las casillas más cercanas. Evita que se apilen infinitos humanos.
        # v1.3.1: no devuelve la misma casilla actual, porque eso causaba humanos "quietos".
        for r in range(1, max(1, radius) + 1):
            candidates = []
            for yy in range(y - r, y + r + 1):
                for xx in range(x - r, x + r + 1):
                    if abs(xx - x) != r and abs(yy - y) != r:
                        continue
                    if (xx, yy) == (x, y):
                        continue
                    if self.walkable_for_human(xx, yy, human=human):
                        candidates.append((xx, yy))
            if candidates:
                return random.choice(candidates)
        return None

    def crowd_count(self, x: int, y: int, radius: int = 3) -> int:
        return sum(1 for h in self.living_humans() if distance((x, y), (h.x, h.y)) <= radius)

    def find_free_near_target(self, x: int, y: int, radius: int = 5, human: Optional[Human] = None, prefer_uncrowded: bool = True) -> Optional[Tuple[int, int]]:
        candidates: List[Tuple[float, int, int]] = []

        for yy in range(max(0, y - radius), min(MAP_H, y + radius + 1)):
            for xx in range(max(0, x - radius), min(MAP_W, x + radius + 1)):
                if not self.walkable_for_human(xx, yy, human=human):
                    continue
                d = distance((x, y), (xx, yy))
                crowd = self.crowd_count(xx, yy, radius=2) if prefer_uncrowded else 0
                candidates.append((d + crowd * 2.5 + random.random() * 0.35, xx, yy))

        if not candidates:
            return None
        candidates.sort(key=lambda item: item[0])
        _, xx, yy = candidates[0]
        return (xx, yy)

    def find_water_access_cell(self, x: int, y: int, radius: int = 80, human: Optional[Human] = None) -> Optional[Tuple[int, int]]:
        """Busca una casilla libre junto al río para que no todos usen el mismo punto."""
        candidates: List[Tuple[float, int, int]] = []

        for yy in range(max(0, y - radius), min(MAP_H, y + radius + 1)):
            for xx in range(max(0, x - radius), min(MAP_W, x + radius + 1)):
                if self.map[yy][xx] != WATER:
                    continue

                for ay in range(max(0, yy - 1), min(MAP_H, yy + 2)):
                    for ax in range(max(0, xx - 1), min(MAP_W, xx + 2)):
                        if self.map[ay][ax] == WATER:
                            continue
                        if self.walkable_for_human(ax, ay, human=human):
                            d = distance((x, y), (ax, ay))
                            crowd = self.crowd_count(ax, ay, radius=2)
                            candidates.append((d + crowd * 4.0 + random.random(), ax, ay))

        if not candidates:
            return None
        candidates.sort(key=lambda item: item[0])
        _, ax, ay = candidates[0]
        return (ax, ay)

    def best_step_towards(self, human: Human, target: Tuple[int, int]) -> Optional[Tuple[int, int]]:
        """
        Paso local inteligente.
        Si el paso directo está bloqueado, prueba vecinos que reduzcan distancia.
        Esto evita quedarse parado 50+ horas contra una montaña/persona.
        """
        candidates: List[Tuple[float, int, int]] = []

        for yy in range(human.y - 1, human.y + 2):
            for xx in range(human.x - 1, human.x + 2):
                if (xx, yy) == (human.x, human.y):
                    continue
                if not self.walkable_for_human(xx, yy, human=human):
                    continue

                d = distance((xx, yy), target)
                crowd = self.crowd_count(xx, yy, radius=2)
                # Penalizamos mucho la aglomeración, pero priorizamos avanzar.
                score = d + crowd * 2.2 + random.random() * 0.25
                candidates.append((score, xx, yy))

        if not candidates:
            return None

        candidates.sort(key=lambda item: item[0])
        _, xx, yy = candidates[0]
        return (xx, yy)

    def nearest_living_human_pos(self, human: Human, radius: int = 120) -> Optional[Tuple[int, int]]:
        best = None
        best_d = 999999.0
        for other in self.living_humans():
            if other is human:
                continue
            d = distance((human.x, human.y), (other.x, other.y))
            if d <= radius and d < best_d:
                best = (other.x, other.y)
                best_d = d
        return best

    def best_home_for(self, human: Human) -> Optional[Tuple[int, int]]:
        candidates: List[Tuple[float, int, int]] = []

        for yy in range(MAP_H):
            for xx in range(MAP_W):
                if not self.is_shelter(xx, yy):
                    continue

                free = self.find_free_near_target(xx, yy, radius=4, human=human, prefer_uncrowded=True)
                if not free:
                    continue

                tile = self.map[yy][xx]
                shelter_score = 0
                if tile == BROWN_HOUSE:
                    shelter_score -= 12
                elif tile in [HOUSE, BUILDING]:
                    shelter_score -= 8
                elif tile == HUT:
                    shelter_score -= 4
                elif tile == CAVE:
                    shelter_score += 7

                crowd = self.crowd_count(xx, yy, radius=6)
                d = distance((human.x, human.y), (xx, yy))
                candidates.append((d + crowd * 5 + shelter_score + random.random(), free[0], free[1]))

        if candidates:
            candidates.sort(key=lambda item: item[0])
            _, x, y = candidates[0]
            return (x, y)

        social = self.nearest_living_human_pos(human, radius=120)
        if social:
            return self.find_free_near_target(social[0], social[1], radius=5, human=human)

        return self.find_free_near_target(self.cave_pos[0], self.cave_pos[1], radius=6, human=human)

    def can_place_structure(self, x: int, y: int, w: int, h: int) -> bool:
        if x < 0 or y < 0 or x + w > MAP_W or y + h > MAP_H:
            return False
        allowed = {GRASS, FOREST, HUT}
        for yy in range(y, y + h):
            for xx in range(x, x + w):
                if self.map[yy][xx] not in allowed:
                    return False
        return True

    def place_structure(self, x: int, y: int, w: int, h: int, tile: str) -> None:
        for yy in range(y, y + h):
            for xx in range(x, x + w):
                self.map[yy][xx] = tile

    def world_hour_index(self) -> int:
        return (((self.year * MONTHS_PER_YEAR + (self.month - 1)) * DAYS_PER_MONTH + (self.day - 1)) * HOURS_PER_DAY + self.hour)

    def broadcast_signal(self, kind: str, pos: Tuple[int, int], source: Optional[Human] = None, strength: float = 1.0, ttl_hours: int = 72) -> None:
        if not hasattr(self, "shared_signals"):
            self.shared_signals = {"water": [], "food": [], "build": [], "medical": []}
        if kind not in self.shared_signals:
            self.shared_signals[kind] = []

        now = self.world_hour_index()
        x, y = int(pos[0]), int(pos[1])

        for signal in self.shared_signals[kind]:
            sx, sy = signal["pos"]
            if distance((sx, sy), (x, y)) <= 4:
                signal["score"] = min(10.0, signal.get("score", 1.0) + strength)
                signal["expires"] = max(signal.get("expires", now), now + ttl_hours)
                return

        self.shared_signals[kind].append({
            "pos": (x, y),
            "score": strength,
            "expires": now + ttl_hours,
            "source": source.label() if source else "desconocido",
        })
        self.shared_signals[kind] = self.shared_signals[kind][-80:]

        if not hasattr(self, "last_signal_log"):
            self.last_signal_log = {}
        last = self.last_signal_log.get(kind, -99999)
        if now - last > 18:
            if kind == "water":
                self.log(f"{source.label() if source else 'Alguien'} vuelve al grupo y avisa de una zona con agua.")
            elif kind == "food":
                self.log(f"{source.label() if source else 'Alguien'} vuelve al grupo y avisa de una zona con comida.")
            elif kind == "build":
                self.log(f"{source.label() if source else 'Alguien'} vuelve al grupo y pide ayuda para construir/recolectar.")
            elif kind == "medical":
                self.log(f"{source.label() if source else 'Alguien'} comparte conocimiento o ayuda medicinal.")
            self.last_signal_log[kind] = now

    def clean_shared_signals(self) -> None:
        if not hasattr(self, "shared_signals"):
            self.shared_signals = {"water": [], "food": [], "build": [], "medical": []}
            return
        now = self.world_hour_index()
        for kind in list(self.shared_signals.keys()):
            self.shared_signals[kind] = [s for s in self.shared_signals[kind] if s.get("expires", 0) >= now]

    def best_shared_signal(self, kind: str, x: int, y: int, max_distance: float = 9999) -> Optional[Tuple[int, int]]:
        if not hasattr(self, "shared_signals"):
            return None
        self.clean_shared_signals()
        best = None
        best_value = -999999.0
        for signal in self.shared_signals.get(kind, []):
            pos = signal["pos"]
            d = distance((x, y), pos)
            if d > max_distance:
                continue
            value = signal.get("score", 1.0) * 12 - d
            if value > best_value:
                best_value = value
                best = pos
        return best

    def is_shelter(self, x: int, y: int) -> bool:
        if not self.in_bounds(x, y):
            return False
        return self.map[y][x] in [CAVE, HUT, HOUSE, BUILDING, BROWN_HOUSE]

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

    def count_plants_close(self, x: int, y: int, radius: int) -> int:
        count = 0
        for plant in self.plants:
            if distance((x, y), (plant.x, plant.y)) <= radius:
                count += 1
        return count

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
            if h.age >= REPRODUCTION_AGE and h.life > 50 and h.hunger > 42 and h.thirst > 42
        ]

        men = [h for h in adults if h.sex == "hombre"]
        women = [h for h in adults if h.sex == "mujer"]

        if not men or not women:
            return

        # Necesitan algún refugio cerca, pero si la población está en peligro,
        # permitimos reproducción aunque el refugio no sea perfecto.
        shelter_available = any(self.shelter_close(h.x, h.y, 10) for h in adults)
        if not shelter_available and len(self.living_humans()) > 6:
            return

        # Evitar explosión infinita, pero permitir sociedad.
        if len(self.living_humans()) > 180:
            return

        for woman in women:
            if woman.age < REPRODUCTION_AGE or woman.age > 65:
                continue

            father = random.choice(men)

            base = PROB_REPRODUCCION_DIARIA_BASE

            # Las primeras 15 generaciones tienen objetivo claro:
            # reproducirse y crear una sociedad.
            if max(father.generation, woman.generation) <= 15:
                base += 0.12

            # Primera pareja estable: más probabilidad si ambos siguen vivos y funcionales.
            if len(self.living_humans()) <= 2 and father.generation == 1 and woman.generation == 1:
                if father.thirst > 45 and woman.thirst > 45 and father.hunger > 45 and woman.hunger > 45:
                    base += 0.08

            # Si quedan pocos humanos, sube mucho la prioridad reproductiva.
            pop = len(self.living_humans())
            if pop < 8:
                base += 0.12
            elif pop < 20:
                base += 0.06

            # La sociabilidad ayuda.
            social_bonus = (father.personality.get("social", 50) + woman.personality.get("social", 50)) / 2000
            probabilidad_nacimiento = clamp(base + social_bonus, 0.02, 0.55)

            if random.random() < probabilidad_nacimiento:
                self.birth_count += 1
                if not hasattr(self, "person_count"):
                    self.person_count = len(self.humans)
                self.person_count += 1

                sex = random_sex()
                numero_hijo_madre = woman.children + 1
                father_label = str(father.person_id) if father.person_id else father.label()
                mother_label = str(woman.person_id) if woman.person_id else woman.label()
                child_generation = max(father.generation, woman.generation) + 1
                name = f"{father_label},{mother_label} {self.person_count} {child_generation}"
                brain = mix_brains(father.brain, woman.brain)

                baby_pos = self.find_free_adjacent_cell(woman.x, woman.y, radius=3)
                if baby_pos is None:
                    # No hay espacio físico alrededor; no nace este ciclo.
                    self.person_count -= 1
                    continue

                baby = Human(
                    name=name,
                    sex=sex,
                    x=baby_pos[0],
                    y=baby_pos[1],
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
                        if chance(0.45):
                            baby.memory_food[k] = v

                    for place_key in parent.memory_places:
                        if parent.memory_places[place_key] and chance(0.35):
                            baby.memory_places[place_key].append(random.choice(parent.memory_places[place_key]))

                if not hasattr(self, "all_humans"):
                    self.all_humans = {}
                self.all_humans[baby.person_id] = baby
                self.humans.append(baby)
                father.children += 1
                woman.children += 1
                father.fitness += 2.0
                woman.fitness += 2.0

                self.log(f"NACE {baby.label()} | sexo: {sex} | gen: {baby.generation} | padres: {father.label()} + {woman.label()}.")


    # ----------------------------
    # Eventos principales
    # ----------------------------

    def step(self) -> None:
        self.clean_shared_signals()

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

        record_population_history(self)

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
        self.resolve_human_collisions()

        # SMAP: exporta el mapa completo solo si el usuario abrió smap.
        smap_write_snapshot(self)

    def resolve_human_collisions(self) -> None:
        occupied: Dict[Tuple[int, int], Human] = {}
        for h in self.living_humans():
            pos = (h.x, h.y)
            if pos not in occupied:
                occupied[pos] = h
                continue
            free = self.find_free_adjacent_cell(h.x, h.y, radius=6, human=h)
            if free:
                h.x, h.y = free

    def resolve_animal_attacks(self) -> None:
        for h in self.living_humans():
            for a in self.animals:
                if not a.alive or not a.aggressive:
                    continue

                d = distance((h.x, h.y), (a.x, a.y))

                # De noche atacan más. v1.2.4: depredadores algo más peligrosos.
                attack_radius = 2.4 if self.is_night() else 1.5
                attack_prob = 0.085 if self.is_night() else 0.032

                # Si hay pocos pacíficos, los depredadores presionan más a humanos.
                peaceful_total = sum(1 for aa in self.animals if aa.alive and not aa.aggressive)
                if peaceful_total < 180:
                    attack_prob *= 1.35

                if a.data["id"] == "trex":
                    attack_radius = 3.2 if self.is_night() else 2.4
                    attack_prob = 0.45 if self.is_night() else 0.28

                if d <= attack_radius and chance(attack_prob):
                    if a.data["id"] == "trex":
                        h.apply_injury(self, random.choice(["herida_grave", "fractura", "mordedura"]))
                        if chance(0.35):
                            h.apply_injury(self, "herida_grave")
                        self.log(f"T-Rex ataca brutalmente a {h.name}.")
                    elif a.data["id"] == "serpiente":
                        h.apply_injury(self, "mordedura")
                    elif a.strength > 70:
                        h.apply_injury(self, random.choice(["herida_grave", "fractura", "mordedura"]))
                    else:
                        h.apply_injury(self, random.choice(["mordedura", "corte", "herida_grave"]))

                    h.memory_places["danger"].append((a.x, a.y))
                    if a.data["id"] != "trex":
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
    world.all_events = []
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
    world.log_scroll = -1
    world.view_mode = "maps"
    world.population_history_monthly = []
    world.last_population_record_key = None
    world.shared_signals = {"water": [], "food": [], "build": [], "medical": []}
    world.last_signal_log = {}

    # SMAP BROWSER: visor externo del mapa completo 256x256.
    # Usa navegador + servidor local. No usa tkinter y no afecta a la IA.
    world.smap_enabled = False
    world.smap_dir = os.path.join(tempfile.gettempdir(), "eden_smap_browser")
    world.smap_snapshot_name = "snapshot.txt"
    world.smap_viewer_name = "viewer.html"
    world.smap_server_path = os.path.join(world.smap_dir, "smap_server.py")
    world.smap_process = None
    world.smap_last_write = 0.0
    world.smap_port = 8765


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


def is_important_event(msg: str) -> bool:
    keys = [
        "NACE ", "ha muerto", "extinguido", "descubre", "aprende", "desarrolla",
        "construyó", "Comando: aparecen", "gráfica poblacional", "casa segura",
        "lenguaje", "agricultura", "ingenieria", "ingeniería"
    ]
    return any(k in msg for k in keys)


def patched_log(self: World, msg: str) -> None:
    self.events.append(msg)
    if len(self.events) > 60:
        self.events = self.events[-60:]

    if not hasattr(self, "all_events"):
        self.all_events = []
    if not hasattr(self, "relevant_events"):
        self.relevant_events = []

    stamp = f"Año {self.year}, Mes {self.month}, Día {self.day}, Hora {self.hour:02d}:00"
    line = f"[{stamp}] {msg}"

    # all_events = absolutamente todos los eventos registrados.
    self.all_events.append(line)

    # relevant_events = eventos importantes/relevantes.
    if is_important_event(msg):
        self.relevant_events.append(line)

    # Evitamos consumo infinito de memoria, pero dejamos muchísimo margen.
    if len(self.all_events) > 200000:
        self.all_events = self.all_events[-200000:]
    if len(self.relevant_events) > 50000:
        self.relevant_events = self.relevant_events[-50000:]


World.log = patched_log


def print_relevant_logs_plain(world: World, amount: Optional[int] = None) -> str:
    all_logs = getattr(world, "all_events", [])
    important_logs = getattr(world, "relevant_events", [])

    if not all_logs and not important_logs:
        return "No hay registros todavía."

    selected_all = all_logs if amount is None else all_logs[-amount:]
    selected_imp = important_logs if amount is None else important_logs[-amount:]

    lines = ["REGISTROS COMPLETOS", "-" * 90]
    lines.extend(selected_all if selected_all else ["No hay registros completos todavía."])
    lines.append("-" * 90)
    lines.append("REGISTROS RELEVANTES")
    lines.append("-" * 90)
    lines.extend(selected_imp if selected_imp else ["No hay registros relevantes todavía."])
    lines.append("-" * 90)
    lines.append(world_status_string(world))
    return "\n".join(lines)


def print_important_logs_plain(world: World, amount: Optional[int] = None) -> str:
    logs = getattr(world, "relevant_events", [])
    if not logs:
        return "No hay registros relevantes todavía."

    selected = logs if amount is None else logs[-amount:]
    lines = ["REGISTROS RELEVANTES", "-" * 90]
    lines.extend(selected)
    lines.append("-" * 90)
    lines.append(world_status_string(world))
    return "\n".join(lines)


def record_population_history(world: World) -> None:
    if not hasattr(world, "population_history_monthly"):
        world.population_history_monthly = []
    if not hasattr(world, "last_population_record_key"):
        world.last_population_record_key = None

    key = (world.year, world.month)
    if world.last_population_record_key == key:
        return

    men = sum(1 for h in world.living_humans() if h.sex == "hombre")
    women = sum(1 for h in world.living_humans() if h.sex == "mujer")
    world.population_history_monthly.append({
        "year": world.year,
        "month": world.month,
        "label": f"A{world.year}-M{world.month}",
        "men": men,
        "women": women,
        "total": men + women,
    })
    world.last_population_record_key = key


def _draw_pixel(img: List[List[Tuple[int, int, int]]], x: int, y: int, color_rgb: Tuple[int, int, int]) -> None:
    if 0 <= y < len(img) and 0 <= x < len(img[0]):
        img[y][x] = color_rgb


def _draw_line(img: List[List[Tuple[int, int, int]]], x0: int, y0: int, x1: int, y1: int, color_rgb: Tuple[int, int, int], thickness: int = 1) -> None:
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    x, y = x0, y0

    while True:
        for oy in range(-thickness // 2, thickness // 2 + 1):
            for ox in range(-thickness // 2, thickness // 2 + 1):
                _draw_pixel(img, x + ox, y + oy, color_rgb)

        if x == x1 and y == y1:
            break

        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x += sx
        if e2 <= dx:
            err += dx
            y += sy


_DIGITS = {
    "0": ["111", "101", "101", "101", "111"],
    "1": ["010", "110", "010", "010", "111"],
    "2": ["111", "001", "111", "100", "111"],
    "3": ["111", "001", "111", "001", "111"],
    "4": ["101", "101", "111", "001", "001"],
    "5": ["111", "100", "111", "001", "111"],
    "6": ["111", "100", "111", "101", "111"],
    "7": ["111", "001", "001", "001", "001"],
    "8": ["111", "101", "111", "101", "111"],
    "9": ["111", "101", "111", "001", "111"],
    "-": ["000", "000", "111", "000", "000"],
    "M": ["101", "111", "111", "101", "101"],
    "P": ["110", "101", "110", "100", "100"],
    "H": ["101", "101", "111", "101", "101"],
    "T": ["111", "010", "010", "010", "010"],
}


def _draw_text(img: List[List[Tuple[int, int, int]]], x: int, y: int, text_value: str, color_rgb: Tuple[int, int, int], scale: int = 2) -> None:
    cursor = x
    for ch in str(text_value):
        if ch == " ":
            cursor += 4 * scale
            continue

        pattern = _DIGITS.get(ch.upper())
        if not pattern:
            cursor += 4 * scale
            continue

        for yy, row in enumerate(pattern):
            for xx, val in enumerate(row):
                if val == "1":
                    for sy in range(scale):
                        for sx in range(scale):
                            _draw_pixel(img, cursor + xx * scale + sx, y + yy * scale + sy, color_rgb)

        cursor += 4 * scale


def _write_png(path: str, img: List[List[Tuple[int, int, int]]]) -> None:
    height = len(img)
    width = len(img[0]) if height else 0

    raw = bytearray()
    for row in img:
        raw.append(0)
        for r, g, b in row:
            raw.extend([r, g, b])

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    png = b"\x89PNG\r\n\x1a\n"
    png += chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(bytes(raw), 9))
    png += chunk(b"IEND", b"")

    out = os.path.expanduser(path)
    parent = os.path.dirname(out)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(out, "wb") as f:
        f.write(png)


def _population_series_by_month(world: World) -> Tuple[List[int], List[int], List[int], int]:
    record_population_history(world)

    current_month_index = max(1, world.year * MONTHS_PER_YEAR + world.month)
    by_index: Dict[int, Dict[str, int]] = {}
    for item in getattr(world, "population_history_monthly", []):
        idx = item["year"] * MONTHS_PER_YEAR + item["month"]
        by_index[idx] = item

    men_vals: List[int] = []
    women_vals: List[int] = []
    total_vals: List[int] = []

    last_m = 0
    last_w = 0
    last_t = 0

    for idx in range(1, current_month_index + 1):
        if idx in by_index:
            last_m = by_index[idx]["men"]
            last_w = by_index[idx]["women"]
            last_t = by_index[idx]["total"]

        men_vals.append(last_m)
        women_vals.append(last_w)
        total_vals.append(last_t)

    y_max = max(1, getattr(world, "max_humans_alive", 0), max(total_vals or [0]))
    return men_vals, women_vals, total_vals, y_max


def generate_population_chart(world: World, path: str) -> str:
    men_vals, women_vals, total_vals, y_max = _population_series_by_month(world)

    if "." not in os.path.basename(path):
        path = path + ".png"
    if not path.lower().endswith(".png"):
        path = path + ".png"

    width, height = 1400, 850
    bg = (17, 24, 39)
    axis = (229, 231, 235)
    grid_color = (55, 65, 81)
    blue = (59, 130, 246)
    pink = (236, 72, 153)
    grey = (156, 163, 175)
    white = (249, 250, 251)
    yellow = (250, 204, 21)

    img = [[bg for _ in range(width)] for __ in range(height)]

    pad_l, pad_r, pad_t, pad_b = 100, 50, 85, 125
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b
    n = max(1, len(total_vals))

    def px(i: int) -> int:
        if n == 1:
            return pad_l + plot_w // 2
        return int(pad_l + (i / (n - 1)) * plot_w)

    def py(v: int) -> int:
        return int(pad_t + plot_h - (v / y_max) * plot_h)

    _draw_line(img, pad_l, pad_t, pad_l, height - pad_b, axis, 2)
    _draw_line(img, pad_l, height - pad_b, width - pad_r, height - pad_b, axis, 2)

    ticks = [0]
    if y_max > 1:
        step = max(1, math.ceil(y_max / 6))
        ticks = list(range(0, y_max, step))
        if y_max not in ticks:
            ticks.append(y_max)

    for t in ticks:
        y = py(t)
        _draw_line(img, pad_l, y, width - pad_r, y, grid_color, 1)
        _draw_text(img, 20, y - 7, str(t), white, 2)

    for i in range(n):
        x = px(i)
        _draw_line(img, x, height - pad_b, x, height - pad_b + 8, axis, 1)
        _draw_text(img, x - 7, height - pad_b + 18, str(i + 1), white, 2)

    _draw_text(img, width // 2 - 80, height - 40, "M", white, 3)
    _draw_text(img, 28, 72, "P", white, 3)

    _draw_line(img, width - 300, 95, width - 235, 95, blue, 5)
    _draw_text(img, width - 225, 86, "H", white, 3)
    _draw_line(img, width - 300, 125, width - 235, 125, pink, 5)
    _draw_text(img, width - 225, 116, "M", white, 3)
    _draw_line(img, width - 300, 155, width - 235, 155, grey, 5)
    _draw_text(img, width - 225, 146, "T", white, 3)

    def draw_series(vals: List[int], color_rgb: Tuple[int, int, int], thickness: int) -> None:
        if len(vals) == 1:
            x = px(0)
            y = py(vals[0])
            _draw_line(img, x - 5, y, x + 5, y, color_rgb, thickness)
            return
        for i in range(len(vals) - 1):
            _draw_line(img, px(i), py(vals[i]), px(i + 1), py(vals[i + 1]), color_rgb, thickness)

    draw_series(total_vals, grey, 3)
    draw_series(men_vals, blue, 3)
    draw_series(women_vals, pink, 3)
    _draw_text(img, pad_l + 8, pad_t + 8, str(y_max), yellow, 2)

    _write_png(path, img)
    world.log(f"Comando: gráfica poblacional PNG generada en {os.path.expanduser(path)}.")
    return f"Imagen PNG generada en: {os.path.expanduser(path)}"


def quietos_text(world: World, min_hours: int = 12) -> str:
    humans = [h for h in world.living_humans() if getattr(h, "still_hours", 0) >= min_hours]

    if not humans:
        return f"No hay humanos quietos durante {min_hours}+ horas."

    lines = [f"HUMANOS QUIETOS {min_hours}+ HORAS", "-" * 90]
    for h in sorted(humans, key=lambda x: x.still_hours, reverse=True):
        lines.append(
            f"{h.label()} | id={h.person_id} | quieto={h.still_hours}h | "
            f"x={h.x} y={h.y} | vida={h.life:.1f} hambre={h.hunger:.1f} sed={h.thirst:.1f} energía={h.energy:.1f}"
        )
    return "\n".join(lines)



def find_human_by_name(world: World, name: str, include_dead: bool = False) -> Optional[Human]:
    target = str(name).strip().lower()

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


def human_statistics_text(world: World, target: Optional[str] = None) -> str:
    if target and target.lower() != "all":
        h = find_human_by_name(world, target, include_dead=True)
        if not h:
            return f"No encuentro al humano {target}."
        humans = [h]
    elif target and target.lower() == "all":
        humans = sorted(getattr(world, "all_humans", {}).values(), key=lambda x: x.person_id or 999999)
    else:
        humans = world.living_humans()

    if not humans:
        return "No hay humanos para mostrar estadísticas."

    lines = ["ESTADÍSTICAS DE HUMANOS", "-" * 90]
    for h in sorted(humans, key=lambda x: x.person_id or 999999):
        cond = h.condition["name"] if h.condition else "sin condición"
        estado = "vivo" if h.alive else f"muerto ({h.death_reason or 'causa desconocida'})"
        lines.append(
            f"{h.label()} | id={h.person_id} | estado={estado} | salud={h.health_state()} | sexo={h.sex} | gen={h.generation} | edad={h.age:.1f} | "
            f"vida={h.life:.1f} hambre={h.hunger:.1f} sed={h.thirst:.1f} energía={h.energy:.1f}"
        )
        lines.append(
            f"  fuerza={h.skills.get('strength',0):.1f} velocidad={h.skills.get('speed',0):.1f} "
            f"mecánica={h.skills.get('mechanics',0):.1f} memoria={h.skills.get('memory',0):.1f} "
            f"empatía={h.skills.get('empathy',0):.1f} foco={h.skills.get('focus',0):.1f}"
        )
        lines.append(
            f"  arma={h.weapon} poder_arma={h.weapon_power:.1f} | ropa={h.clothing} protección_frío={h.cold_protection:.2f} | "
            f"food={h.inventory.get('food',0)} raw_meat={h.inventory.get('raw_meat',0)} wood={h.inventory.get('wood',0)} stone={h.inventory.get('stone',0)} "
            f"hides={h.inventory.get('hides',0)} bones={h.inventory.get('bones',0)} herbs={h.inventory.get('herbs',0)} seeds={h.inventory.get('seeds',0)} "
            f"clay={h.inventory.get('clay',0)} metal_ore={h.inventory.get('metal_ore',0)} metal={h.inventory.get('metal',0)} | {cond}"
        )
        best_actions = sorted(h.learned_actions.items(), key=lambda kv: kv[1], reverse=True)[:4]
        lines.append("  aprendizaje: " + ", ".join(f"{k}={v:.2f}" for k, v in best_actions))
        lines.append("")
    return "\n".join(lines)


def spawn_humans_command(world: World, sex_word: str, age_word: str, count_word: str) -> str:
    sex_map = {
        "hombre": "hombre",
        "man": "hombre",
        "male": "hombre",
        "mujer": "mujer",
        "woman": "mujer",
        "female": "mujer",
    }

    sex = sex_map.get(sex_word.lower())
    if not sex:
        return "Sexo no reconocido. Usa hombre o mujer."

    try:
        age = float(age_word)
        count = int(count_word)
    except ValueError:
        return "Uso: spawn human hombre 18 3"

    count = max(1, min(count, 500))

    if not hasattr(world, "person_count"):
        world.person_count = max([h.person_id for h in getattr(world, "all_humans", {}).values()], default=len(world.humans))

    if not hasattr(world, "all_humans"):
        world.all_humans = {h.person_id: h for h in world.humans if h.person_id}

    cx, cy = world.cave_pos
    created = []

    for _ in range(count):
        world.person_count += 1
        pos = world.find_free_adjacent_cell(cx, cy, radius=10)
        if pos is None:
            world.person_count -= 1
            break
        x, y = pos

        name = f"spawn {world.person_count} 1"
        h = Human(name=name, sex=sex, x=x, y=y, age=age)
        h.person_id = world.person_count
        h.father_label = "spawn"
        h.mother_label = ""
        h.child_number = 0
        h.generation = 1
        h.memory_places["shelter"].append(world.cave_pos)
        water = world.find_nearest_tile(h.x, h.y, WATER, 18)
        if water:
            h.memory_places["water"].append(water)

        world.humans.append(h)
        world.all_humans[h.person_id] = h
        created.append(h)

    world.extinction_report_printed = False
    world.paused = False

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
        return "Uso: spawn humans 18 10"

    count = max(1, min(count, 500))
    men = count // 2
    women = count - men

    if men:
        spawn_humans_command(world, "hombre", str(age), str(men))
    if women:
        spawn_humans_command(world, "mujer", str(age), str(women))

    return f"Aparecen {count} humanos equilibrados: {men} hombre(s), {women} mujer(es)."



def spawn_trex_command(world: World) -> str:
    # Solo puede existir un T-Rex vivo.
    existing = [a for a in world.animals if a.alive and a.data["id"] == "trex"]
    if existing:
        trex = existing[0]
        return f"Ya existe un T-Rex vivo en x={trex.x}, y={trex.y}."

    data = next(a for a in ANIMALS if a["id"] == "trex")

    # Lo colocamos lejos de la cueva para que sea una amenaza del mundo,
    # no una muerte instantánea al spawn.
    for _ in range(300):
        x, y = world.random_land_cell()
        tile = world.map[y][x]
        if distance((x, y), world.cave_pos) > 70 and tile in [FOREST, MOUNTAIN, SWAMP, GRASS]:
            world.animals.append(Animal(x, y, data))
            world.log("Comando: aparece un único T-Rex hiper fuerte en el mundo.")
            return f"Aparece un T-Rex hiper fuerte en x={x}, y={y}."

    x, y = world.random_land_cell()
    world.animals.append(Animal(x, y, data))
    world.log("Comando: aparece un único T-Rex hiper fuerte en el mundo.")
    return f"Aparece un T-Rex hiper fuerte en x={x}, y={y}."


def spawn_animals_command(world: World, animal_id: str, count_word: str) -> str:
    try:
        count = int(count_word)
    except ValueError:
        return "Uso: spawn animal conejo 10"

    count = max(1, min(count, 1000))
    valid = [a["id"] for a in ANIMALS]
    if animal_id not in valid:
        return "Animal no reconocido. Válidos: " + ", ".join(valid)

    if animal_id == "trex":
        return spawn_trex_command(world)

    for _ in range(count):
        world.spawn_specific_animal(animal_id)

    world.log(f"Comando: aparecen {count} animal(es) de tipo {animal_id}.")
    return f"Aparecen {count} animal(es) de tipo {animal_id}."


def _target_humans(world: World, target: str) -> List[Human]:
    if target.lower() == "all":
        return world.living_humans()

    h = find_human_by_name(world, target, include_dead=False)
    return [h] if h else []


def heal_command(world: World, target: str, amount_word: str) -> str:
    try:
        amount = float(amount_word)
    except ValueError:
        return "Uso: heal all 30"

    humans = _target_humans(world, target)
    if not humans:
        return f"No encuentro humanos vivos para {target}."

    for h in humans:
        h.life = clamp(h.life + amount, 0, 100)

    world.log(f"Comando: curación aplicada a {len(humans)} humano(s).")
    return f"Curados {len(humans)} humano(s) +{amount} vida."


def setlife_command(world: World, target: str, value_word: str) -> str:
    try:
        value = float(value_word)
    except ValueError:
        return "Uso: setlife all 100"

    humans = _target_humans(world, target)
    if not humans:
        return f"No encuentro humanos vivos para {target}."

    for h in humans:
        h.life = clamp(value, 0, 100)
        if h.life > 0:
            h.alive = True

    return f"Vida fijada a {value} para {len(humans)} humano(s)."


def resource_command(world: World, resource: str, target: str, value_word: str) -> str:
    try:
        value = float(value_word)
    except ValueError:
        return f"Uso: {resource} all 100"

    humans = _target_humans(world, target)
    if not humans:
        return f"No encuentro humanos vivos para {target}."

    attr = {"food": "hunger", "water": "thirst", "energy": "energy"}[resource]
    for h in humans:
        setattr(h, attr, clamp(value, 0, 100 if attr != "energy" else h.energy_max))

    return f"{resource} fijado a {value} para {len(humans)} humano(s)."


def give_command(world: World, target: str, item: str, amount_word: str) -> str:
    try:
        amount = int(amount_word)
    except ValueError:
        return "Uso: give all wood 30"

    valid_items = ["food", "raw_meat", "wood", "stone", "hides", "bones", "herbs", "seeds", "clay", "metal_ore", "metal"]
    if item not in valid_items:
        return "Recurso no reconocido. Usa: " + ", ".join(valid_items)

    humans = _target_humans(world, target)
    if not humans:
        return f"No encuentro humanos vivos para {target}."

    for h in humans:
        h.inventory[item] = h.inventory.get(item, 0) + amount

    return f"Dado {amount} de {item} a {len(humans)} humano(s)."


def set_weather_command(world: World, weather: str) -> str:
    valid = ["soleado", "nublado", "lluvia", "lluvia_fuerte", "tormenta", "niebla", "sequía", "ola_calor", "frio_extremo", "nieve", "tornado", "terremoto"]
    if weather not in valid:
        return "Clima no reconocido. Válidos: " + ", ".join(valid)

    world.weather = weather
    world.log(f"Comando: clima cambiado a {weather}.")
    return f"Clima cambiado a {weather}."


def disaster_command(world: World, disaster: str) -> str:
    if disaster not in ["terremoto", "tornado", "incendio", "inundacion"]:
        return "Desastre no reconocido. Usa terremoto, tornado, incendio o inundacion."

    if disaster == "terremoto":
        world.weather = "terremoto"
        for h in world.living_humans():
            if chance(0.08):
                h.apply_injury(world, random.choice(["corte", "torcedura", "fractura"]))
        world.log("Comando: terremoto provocado.")
    elif disaster == "tornado":
        world.weather = "tornado"
        for h in world.living_humans():
            if chance(0.10):
                h.apply_injury(world, random.choice(["corte", "fractura", "herida_grave"]))
        world.log("Comando: tornado provocado.")
    elif disaster == "incendio":
        for h in world.living_humans():
            if world.map[h.y][h.x] == FOREST and chance(0.12):
                h.apply_injury(world, "herida_grave")
        world.log("Comando: incendio provocado.")
    elif disaster == "inundacion":
        for h in world.living_humans():
            if world.tile_close(h.x, h.y, WATER, 2) and chance(0.05):
                h.apply_injury(world, "ahogamiento")
        world.log("Comando: inundación provocada.")

    return f"Desastre provocado: {disaster}."


def smap_find_free_port(preferred: int = 8765) -> int:
    for port in range(preferred, preferred + 50):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    return preferred


def smap_full_map_lines(world: World) -> List[str]:
    """
    Mapa completo 256x256 en letras.
    Solo lectura: no altera IA, posiciones, vida ni simulación.
    """
    humans: Dict[Tuple[int, int], str] = {}
    for h in world.living_humans():
        if h.name == "Adán":
            humans[(h.x, h.y)] = "A"
        elif h.name == "Eva":
            humans[(h.x, h.y)] = "E"
        else:
            humans[(h.x, h.y)] = "@"

    animals: Dict[Tuple[int, int], str] = {}
    for a in world.animals:
        if a.alive:
            animals[(a.x, a.y)] = a.data.get("symbol", "?")

    plants: Dict[Tuple[int, int], str] = {}
    for p in world.plants:
        plants[(p.x, p.y)] = p.data.get("symbol", "?")

    lines: List[str] = []
    for y in range(MAP_H):
        row = []
        for x in range(MAP_W):
            pos = (x, y)
            if pos in humans:
                row.append(humans[pos])
            elif pos in animals:
                row.append(animals[pos])
            elif pos in plants:
                row.append(plants[pos])
            else:
                tile = world.map[y][x]
                row.append("R" if tile == BROWN_HOUSE else tile)
        lines.append("".join(row))
    return lines


def smap_write_snapshot(world: World, force: bool = False) -> None:
    if not getattr(world, "smap_enabled", False):
        return

    now = time.time()
    if not force and now - getattr(world, "smap_last_write", 0.0) < 0.60:
        return

    world.smap_last_write = now
    os.makedirs(world.smap_dir, exist_ok=True)

    header = [
        "EDEN SIM SMAP — MAPA COMPLETO 256x256",
        world_status_string(world),
        "Leyenda: A=Adán | E=Eva | @=humano | ~=río | T=bosque | C=cueva | M=montaña | S=pantano | R=refugio/casa | X=T-Rex",
        "-" * MAP_W,
    ]

    lines = header + smap_full_map_lines(world)
    path = os.path.join(world.smap_dir, world.smap_snapshot_name)
    tmp = path + ".tmp"

    with open(tmp, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
        f.write("\n")
    os.replace(tmp, path)


def smap_write_viewer_files(world: World) -> None:
    os.makedirs(world.smap_dir, exist_ok=True)

    with open(os.path.join(world.smap_dir, world.smap_viewer_name), "w", encoding="utf-8") as f:
        f.write('<!doctype html>\n<html lang="es">\n<head>\n<meta charset="utf-8">\n<title>EDEN SIM — SMAP 256x256</title>\n<style>\n  body {\n    margin: 0;\n    background: #050816;\n    color: #d1d5db;\n    font-family: Menlo, Monaco, Consolas, monospace;\n    overflow: hidden;\n  }\n  #top {\n    height: 46px;\n    box-sizing: border-box;\n    padding: 7px 10px;\n    background: #111827;\n    border-bottom: 1px solid #374151;\n    display: flex;\n    gap: 12px;\n    align-items: center;\n    font-size: 12px;\n  }\n  button {\n    background: #1f2937;\n    color: white;\n    border: 1px solid #4b5563;\n    border-radius: 6px;\n    padding: 4px 8px;\n    cursor: pointer;\n  }\n  button:hover { background: #374151; }\n  #status { white-space: nowrap; overflow: hidden; text-overflow: ellipsis; flex: 1; }\n  #mapwrap {\n    height: calc(100vh - 46px);\n    overflow: auto;\n    padding: 8px;\n    box-sizing: border-box;\n  }\n  #map {\n    font-size: 4px;\n    line-height: 1;\n    white-space: pre;\n    letter-spacing: 0;\n  }\n  .grass { color: #39d353; }\n  .forest { color: #00b050; }\n  .water { color: #1f6feb; }\n  .cave { color: #e5e7eb; }\n  .mountain { color: #f9fafb; }\n  .swamp { color: #d4b106; }\n  .house { color: #a16207; }\n  .human { color: #7dd3fc; font-weight: bold; }\n  .adam { color: #00ffff; font-weight: bold; }\n  .eve { color: #ff66ff; font-weight: bold; }\n  .plant { color: #a3e635; }\n  .animal { color: #facc15; }\n  .predator { color: #ff3333; font-weight: bold; }\n  .trex { color: #ff0000; font-weight: bold; background: #330000; }\n  .header { color: #ffffff; }\n</style>\n</head>\n<body>\n<div id="top">\n  <button onclick="zoom(-1)">- letra</button>\n  <button onclick="zoom(1)">+ letra</button>\n  <span id="status">Esperando snapshot...</span>\n</div>\n<div id="mapwrap"><div id="map"></div></div>\n\n<script>\nlet fontSize = 4;\nlet lastContent = "";\n\nfunction esc(ch) {\n  if (ch === "&") return "&amp;";\n  if (ch === "<") return "&lt;";\n  if (ch === ">") return "&gt;";\n  return ch;\n}\n\nfunction cls(ch) {\n  if (ch === ".") return "grass";\n  if (ch === "T") return "forest";\n  if (ch === "~") return "water";\n  if (ch === "C") return "cave";\n  if (ch === "M") return "mountain";\n  if (ch === "S") return "swamp";\n  if (ch === "R" || ch === "h" || ch === "H" || ch === "B") return "house";\n  if (ch === "A") return "adam";\n  if (ch === "E") return "eve";\n  if (ch === "@") return "human";\n  if ("munirvqzy".includes(ch)) return "plant";\n  if ("cdpgk".includes(ch)) return "animal";\n  if ("LOVJP".includes(ch)) return "predator";\n  if (ch === "X") return "trex";\n  return "header";\n}\n\nfunction render(content) {\n  const lines = content.split(/\\r?\\n/);\n  let sep = lines.findIndex(l => l.length >= 200 && /^-+$/.test(l.trim()));\n  if (sep < 0) {\n    document.getElementById("map").textContent = content;\n    return;\n  }\n\n  const header = lines.slice(0, sep + 1);\n  const grid = lines.slice(sep + 1, sep + 1 + 256);\n  document.getElementById("status").textContent = header[1] || "SMAP activo";\n\n  let html = "";\n  for (const line of header) html += `<span class="header">${line}</span>\\n`;\n\n  for (const row of grid) {\n    for (const ch of row) {\n      html += `<span class="${cls(ch)}">${esc(ch)}</span>`;\n    }\n    html += "\\n";\n  }\n\n  const wrap = document.getElementById("mapwrap");\n  const sx = wrap.scrollLeft;\n  const sy = wrap.scrollTop;\n\n  document.getElementById("map").innerHTML = html;\n\n  wrap.scrollLeft = sx;\n  wrap.scrollTop = sy;\n}\n\nasync function update() {\n  try {\n    const res = await fetch("snapshot.txt?ts=" + Date.now(), { cache: "no-store" });\n    const content = await res.text();\n    if (content && content !== lastContent) {\n      lastContent = content;\n      render(content);\n    }\n  } catch (e) {\n    document.getElementById("status").textContent = "Esperando servidor/snapshot...";\n  }\n}\n\nfunction zoom(delta) {\n  fontSize = Math.max(2, Math.min(12, fontSize + delta));\n  document.getElementById("map").style.fontSize = fontSize + "px";\n}\n\nsetInterval(update, 650);\nupdate();\n</script>\n</body>\n</html>\n')

    with open(world.smap_server_path, "w", encoding="utf-8") as f:
        f.write('#!/usr/bin/env python3\n# -*- coding: utf-8 -*-\nimport os\nimport sys\nfrom http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler\n\nport = int(sys.argv[1])\ndirectory = sys.argv[2]\n\nclass Handler(SimpleHTTPRequestHandler):\n    def __init__(self, *args, **kwargs):\n        super().__init__(*args, directory=directory, **kwargs)\n\n    def log_message(self, format, *args):\n        pass\n\nserver = ThreadingHTTPServer(("127.0.0.1", port), Handler)\nserver.serve_forever()\n')


def smap_open_window(world: World) -> str:
    proc = getattr(world, "smap_process", None)

    # Si ya hay servidor vivo, solo abre el navegador.
    if proc is not None and proc.poll() is None:
        world.smap_enabled = True
        smap_write_snapshot(world, force=True)
        webbrowser.open(f"http://127.0.0.1:{world.smap_port}/{world.smap_viewer_name}")
        return "SMAP ya estaba activo. He abierto/actualizado la ventana del navegador."

    world.smap_enabled = True
    world.smap_port = smap_find_free_port(getattr(world, "smap_port", 8765))
    smap_write_viewer_files(world)
    smap_write_snapshot(world, force=True)

    try:
        world.smap_process = subprocess.Popen(
            [sys.executable, world.smap_server_path, str(world.smap_port), world.smap_dir],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        webbrowser.open(f"http://127.0.0.1:{world.smap_port}/{world.smap_viewer_name}")
        world.log("Comando: smap abierto en navegador con mapa completo.")
        return "SMAP abierto en navegador: mapa completo 256x256 con colores."
    except Exception as exc:
        world.smap_enabled = False
        return f"No se pudo abrir SMAP: {exc}"


def smap_close_window(world: World) -> str:
    world.smap_enabled = False
    proc = getattr(world, "smap_process", None)
    if proc is not None and proc.poll() is None:
        try:
            proc.terminate()
        except Exception:
            pass
    world.smap_process = None
    return "SMAP cerrado/desactivado. Puedes cerrar la pestaña del navegador."


def help_text() -> str:
    return (
        "AYUDA DE COMANDOS\n"
        "help | status | logs | logs 30 | relevantes | relevantes 30 | pause | resume | speed 5 | delay 0.20 | render | mapa | smap | quietos\n"
        "spawn human hombre 18 3 | spawn human mujer 18 3 | spawn humans 18 10 | spawn animal conejo 20 | spawn animal lobo 2 | spawn trex\n"
        "estadistica 286 | estadistica all | heal all 30 | setlife all 100 | food all 100 | water all 100 | energy all 100\n"
        "give all food 10 | give all wood 30 | give all hides 5 | give all herbs 3 | weather lluvia | disaster terremoto | grafica /ruta/poblacion.png | quit\n"
        "logs = todos los registros + relevantes al final | relevantes = solo eventos importantes | mapa = mapa global reducido | smap = navegador 256x256 completo | render = vuelve al mapa normal"
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
            world.view_mode = "logs"
            world.log_scroll = -1
            if len(parts) >= 2:
                amount = int(parts[1])
            return print_relevant_logs_plain(world, amount)

        elif cmd in ["relevantes", "importantes", "important", "imp"]:
            amount = int(parts[1]) if len(parts) >= 2 else None
            world.view_mode = "logs"
            world.log_scroll = -1
            return print_important_logs_plain(world, amount)

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
            world.view_mode = "maps"
            return "Render actualizado."

        elif cmd in ["mapa", "worldmap", "overview"]:
            world.view_mode = "overview"
            return "MAPA GLOBAL"

        elif cmd == "smap":
            if len(parts) >= 2 and parts[1].lower() in ["off", "cerrar", "close"]:
                return smap_close_window(world)
            return smap_open_window(world)

        elif cmd in ["grafica", "graph", "chart"] and len(parts) >= 2:
            return generate_population_chart(world, " ".join(parts[1:]))

        elif cmd in ["quietos", "stuck"]:
            min_hours = int(parts[1]) if len(parts) >= 2 else 12
            return quietos_text(world, min_hours)

        elif cmd == "spawn":
            if len(parts) < 2:
                return "Uso: spawn human hombre 18 3 / spawn humans 18 10 / spawn animal conejo 10"
            elif parts[1].lower() == "human" and len(parts) >= 5:
                return spawn_humans_command(world, parts[2], parts[3], parts[4])
            elif parts[1].lower() in ["humans", "balanced", "balanceado"] and len(parts) >= 4:
                return spawn_balanced_humans_command(world, parts[2], parts[3])
            elif parts[1].lower() in ["trex", "t-rex", "tiranosaurio"]:
                return spawn_trex_command(world)
            elif parts[1].lower() == "animal" and len(parts) >= 4:
                return spawn_animals_command(world, parts[2], parts[3])
            return "Uso: spawn human hombre 18 3 / spawn humans 18 10 / spawn animal conejo 10 / spawn trex"

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
    # Prioridad visual: Adán/Eva nunca deben quedar tapados por otros humanos.
    living_here = [h for h in world.living_humans() if h.x == x and h.y == y]
    for h in living_here:
        if h.name == "Adán":
            return "A", 4
    for h in living_here:
        if h.name == "Eva":
            return "E", 5
    if living_here:
        h = living_here[0]
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
    if tile == BROWN_HOUSE:
        return "R", 8
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



def make_overview_grid(world: World, width: int, height: int) -> List[List[Tuple[str, int]]]:
    """Mapa completo comprimido en la pantalla."""
    width = max(20, width)
    height = max(8, height)
    cell_w = MAP_W / width
    cell_h = MAP_H / height

    humans = {(h.x, h.y): h for h in world.living_humans()}
    animals = {(a.x, a.y): a for a in world.animals if a.alive}

    grid: List[List[Tuple[str, int]]] = []
    for gy in range(height):
        row: List[Tuple[str, int]] = []
        y0 = int(gy * cell_h)
        y1 = max(y0 + 1, int((gy + 1) * cell_h))
        for gx in range(width):
            x0 = int(gx * cell_w)
            x1 = max(x0 + 1, int((gx + 1) * cell_w))

            chosen = None
            pair = 1

            # Prioridad: Adán/Eva > otros humanos > animales > terreno.
            # Antes, en el mapa global comprimido, un @ podía tapar a A/E
            # si estaban dentro del mismo bloque comprimido.
            found_human = None
            found_adam = None
            found_eve = None

            for yy in range(y0, min(y1, MAP_H)):
                for xx in range(x0, min(x1, MAP_W)):
                    if (xx, yy) in humans:
                        h = humans[(xx, yy)]
                        if h.name == "Adán":
                            found_adam = h
                        elif h.name == "Eva":
                            found_eve = h
                        elif found_human is None:
                            found_human = h

            if found_adam is not None:
                chosen, pair = "A", 4
            elif found_eve is not None:
                chosen, pair = "E", 5
            elif found_human is not None:
                chosen, pair = "@", 4 if found_human.sex == "hombre" else 5

            if not chosen:
                for yy in range(y0, min(y1, MAP_H)):
                    for xx in range(x0, min(x1, MAP_W)):
                        if (xx, yy) in animals:
                            a = animals[(xx, yy)]
                            chosen, pair = a.data.get("symbol", "?"), 3 if a.aggressive else 6
                            break
                    if chosen:
                        break

            if not chosen:
                counts: Dict[str, int] = {}
                for yy in range(y0, min(y1, MAP_H)):
                    for xx in range(x0, min(x1, MAP_W)):
                        t = world.map[yy][xx]
                        counts[t] = counts.get(t, 0) + 1
                tile = max(counts, key=counts.get) if counts else GRASS

                if tile == GRASS:
                    chosen, pair = ".", 2
                elif tile == FOREST:
                    chosen, pair = "T", 2
                elif tile == WATER:
                    chosen, pair = "~", 4
                elif tile == CAVE:
                    chosen, pair = "C", 7
                elif tile == MOUNTAIN:
                    chosen, pair = "M", 7
                elif tile == SWAMP:
                    chosen, pair = "S", 6
                elif tile in [HUT, HOUSE, BUILDING]:
                    chosen, pair = tile, 5
                elif tile == BROWN_HOUSE:
                    chosen, pair = "R", 8
                else:
                    chosen, pair = tile, 1

            row.append((chosen, pair))
        grid.append(row)
    return grid


def draw_overview_screen(stdscr, world: World, command_buffer: str, color_pairs: Dict[int, int]) -> None:
    stdscr.erase()
    max_y, max_x = stdscr.getmaxyx()

    addstr_safe(stdscr, 0, 0, "EDEN SIM v1.3.3 — MAPA GLOBAL", curses.color_pair(4))
    addstr_safe(stdscr, 1, 0, world_status_string(world), curses.color_pair(1))
    addstr_safe(stdscr, 2, 0, "Mapa completo comprimido. Usa render para volver.", curses.color_pair(6))

    grid_h = max(8, max_y - 10)
    grid_w = max(20, max_x - 2)
    grid = make_overview_grid(world, grid_w, grid_h)

    for y, row in enumerate(grid, start=4):
        if y >= max_y - 5:
            break
        x = 0
        for ch, pair_id in row:
            if x >= max_x - 1:
                break
            try:
                stdscr.addstr(y, x, ch, curses.color_pair(color_pairs.get(pair_id, 1)))
            except curses.error:
                pass
            x += 1

    legend_y = max_y - 5
    addstr_safe(stdscr, legend_y, 0, "Leyenda: A/E/@ humanos | c/d/p/g/k pacíficos | L/O/V/J/P depredadores", curses.color_pair(6))
    addstr_safe(stdscr, legend_y + 1, 0, ". pradera | T bosque | ~ agua | C cueva | M montaña | S pantano | h/H/B/R refugios/casas", curses.color_pair(6))
    addstr_safe(stdscr, legend_y + 2, 0, f"Último evento: {world.relevant_events[-1] if world.relevant_events else ''}", curses.color_pair(1))
    addstr_safe(stdscr, legend_y + 3, 0, f"cmd> {command_buffer}", curses.color_pair(4))
    try:
        stdscr.move(legend_y + 3, min(5 + len(command_buffer), max_x - 2))
    except curses.error:
        pass
    stdscr.refresh()


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
    return (
        text.startswith("REGISTROS COMPLETOS")
        or text.startswith("REGISTROS RELEVANTES")
        or text.startswith("ESTADÍSTICAS DE HUMANOS")
        or text.startswith("HUMANOS QUIETOS")
        or text.startswith("AYUDA DE COMANDOS")
    )


def draw_log_screen(stdscr, world: World, command_buffer: str, last_output: str) -> None:
    stdscr.erase()
    max_y, max_x = stdscr.getmaxyx()

    addstr_safe(stdscr, 0, 0, "EDEN SIM v1.3.3 — PANEL", curses.color_pair(4))
    addstr_safe(stdscr, 1, 0, world_status_string(world), curses.color_pair(1))

    lines = last_output.splitlines() if last_output else ["No hay registros."]
    usable_top = 3
    usable_bottom = max_y - 3
    max_lines = max(1, usable_bottom - usable_top)

    if getattr(world, "log_scroll", -1) < 0:
        world.log_scroll = max(0, len(lines) - max_lines)

    world.log_scroll = max(0, min(world.log_scroll, max(0, len(lines) - max_lines)))
    visible = lines[world.log_scroll:world.log_scroll + max_lines]

    y = usable_top
    for line in visible:
        attr = curses.color_pair(1)
        if line.startswith("REGISTROS"):
            attr = curses.color_pair(6)
        elif "NACE " in line:
            attr = curses.color_pair(2)
        elif "Comando: aparecen" in line and "humano" in line:
            attr = curses.color_pair(4)
        elif "ha muerto" in line or "extinguido" in line:
            attr = curses.color_pair(3)
        elif "Comando:" in line:
            attr = curses.color_pair(4)
        addstr_safe(stdscr, y, 0, line, attr)
        y += 1

    info = f"Scroll log: {world.log_scroll}/{max(0, len(lines)-max_lines)} | flechas/RePag/AvPag | render vuelve"
    addstr_safe(stdscr, max_y - 2, 0, info, curses.color_pair(6))
    addstr_safe(stdscr, max_y - 1, 0, f"cmd> {command_buffer}", curses.color_pair(4))
    try:
        stdscr.move(max_y - 1, min(5 + len(command_buffer), max_x - 2))
    except curses.error:
        pass

    stdscr.refresh()


def draw_ui(stdscr, world: World, command_buffer: str, last_output: str, color_pairs: Dict[int, int]) -> None:
    if getattr(world, "view_mode", "maps") == "overview":
        draw_overview_screen(stdscr, world, command_buffer, color_pairs)
        return

    # Si el último comando fue logs, mostramos una pantalla de logs real,
    # no solo una línea de respuesta abajo.
    if is_log_screen(last_output):
        draw_log_screen(stdscr, world, command_buffer, last_output)
        return

    stdscr.erase()
    max_y, max_x = stdscr.getmaxyx()

    addstr_safe(stdscr, 0, 0, "EDEN SIM v1.3.3 — modo médico y supervivencia real", curses.color_pair(4))
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
    recent_color = 6
    if "NACE " in recent:
        recent_color = 2
    elif "Comando: aparecen" in recent and "humano" in recent:
        recent_color = 4
    elif "ha muerto" in recent or "extinguido" in recent:
        recent_color = 3
    addstr_safe(stdscr, panel_y + 4, 0, f"Último evento: {recent}", curses.color_pair(recent_color))

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
    # 1 normal, 2 verde, 3 rojo, 4 cian, 5 magenta, 6 amarillo, 7 blanco/gris, 8 marrón aproximado
    pairs = {
        1: 1,
        2: 2,
        3: 3,
        4: 4,
        5: 5,
        6: 6,
        7: 7,
        8: 8,
    }

    curses.init_pair(1, curses.COLOR_WHITE, -1)
    curses.init_pair(2, curses.COLOR_GREEN, -1)
    curses.init_pair(3, curses.COLOR_RED, -1)
    curses.init_pair(4, curses.COLOR_CYAN, -1)
    curses.init_pair(5, curses.COLOR_MAGENTA, -1)
    curses.init_pair(6, curses.COLOR_YELLOW, -1)
    curses.init_pair(7, curses.COLOR_WHITE, -1)
    curses.init_pair(8, curses.COLOR_YELLOW, -1)

    return pairs


def curses_simulation(stdscr, world: World) -> None:
    curses.curs_set(1)
    stdscr.nodelay(True)
    stdscr.keypad(True)

    color_pairs = init_curses_colors()
    command_buffer = ""
    # v1.2.3: arrancamos enseñando el mapa, no el panel de ayuda.
    # Si quieres ayuda, escribe help.
    last_output = ""
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
                if is_log_screen(last_output):
                    world.log_scroll = max(0, getattr(world, "log_scroll", 0) + 1)
                else:
                    world.map_scroll = getattr(world, "map_scroll", 0) + 3
            elif ch == curses.KEY_UP:
                if is_log_screen(last_output):
                    world.log_scroll = max(0, getattr(world, "log_scroll", 0) - 1)
                else:
                    world.map_scroll = max(0, getattr(world, "map_scroll", 0) - 3)
            elif ch == curses.KEY_NPAGE:
                if is_log_screen(last_output):
                    world.log_scroll = max(0, getattr(world, "log_scroll", 0) + 15)
                else:
                    world.map_scroll = getattr(world, "map_scroll", 0) + 12
            elif ch == curses.KEY_PPAGE:
                if is_log_screen(last_output):
                    world.log_scroll = max(0, getattr(world, "log_scroll", 0) - 15)
                else:
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
                        world.view_mode = "logs"
                        world.log_scroll = -1
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
    print(print_relevant_logs_plain(world, amount=None))


# ============================================================
# EJECUCIÓN
# ============================================================

def main() -> None:
    random.seed()
    np.random.seed()

    if SHOW_START_LEGEND:
        clear_screen()
        print_full_legend()
        print(color("NOVEDAD v1.3.3:", "cyan"))
        print("Si spawneas humanos tras una extinción, el tiempo vuelve a avanzar.")
        print("Modo médico: si baja la vida buscan curarse, refugiarse, recolectar hierbas o pedir ayuda al grupo.")
        print("Año inicial: 0.")
        print("Nacimientos en verde; spawns en cian; casas seguras marrones como R.")
        print("Ejemplos de nombres: Adán,Eva 3 2  /  8,5 17 4  /  spawn 8 1")
        print()
        if WAIT_AFTER_START_LEGEND:
            input("Pulsa ENTER para crear el mundo y empezar la simulación...")

    world = World()
    install_runtime_state(world)
    world.generate()
    record_population_history(world)

    # Aseguramos ids por si la generación se hizo antes del runtime.
    for h in world.humans:
        if h.name == "Adán":
            h.person_id = 1
        elif h.name == "Eva":
            h.person_id = 2

    run_terminal_simulation(world)


if __name__ == "__main__":
    main()
