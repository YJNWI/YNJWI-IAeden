# YNJWI-IAeden
Experimental Python civilization simulator where humans start with only basic instincts and learn shelters, fire, agriculture, paths and society through survival, observation and cultural transmission.
# EDEN SIM

**EDEN SIM** is an experimental Python artificial-life and civilization simulator created by **YJNWI**.

In this project, humans do **not** start with civilization knowledge. They are not born knowing what fire, shelters, storage, agriculture, roads or society are. They begin only with basic instincts:

- I am hungry.
- I am thirsty.
- I am cold.
- There is danger.
- Something hurts.
- I want to stay close to others.
- I need a safe place to sleep.

From those instincts, civilization slowly emerges through survival, observation, mistakes, learning and teaching.

---

## Important: which file should I run?

This repository contains **all project files and historical versions** so the evolution of EDEN SIM can be studied.

However, the file you should execute is:

```bash
python3 eden_sim.py
```

The other files are previous versions, experiments, backups, tests or historical builds.

Recommended execution flow:

```bash
cd eden_sim
python3 eden_sim.py
```

If you are using a virtual environment:

```bash
cd eden_sim
source edensim/bin/activate
python3 eden_sim.py
```

---

## What type of AI is EDEN SIM?

EDEN SIM is **not a chatbot** and it does not use a large language model.

It is an **agent-based artificial-life simulation**.

Each human is an autonomous agent with:

- hunger
- thirst
- energy
- health
- age
- sex
- generation
- personality tendencies
- learned behaviors
- individual knowledge
- primitive concepts
- memories of places
- social reactions
- survival priorities

The AI is based on **emergent behavior**: simple rules, needs, memory, learning and environmental pressure create complex behavior over time.

Humans can learn from:

- hunger
- thirst
- danger
- cold
- injuries
- plants
- animals
- other humans
- discovered locations
- repeated movement
- trial and error
- cultural transmission

---

## Core idea

The goal is not to give humans pre-built intelligence.

The goal is to let them discover civilization.

For example:

- They do not know what a shelter is.
- First, they feel unsafe at night.
- Then they sleep under trees or in caves.
- Later, one of them may understand the idea of a safe sleeping place.
- Then they may improvise a primitive shelter.
- If it works, others may observe it.
- Eventually, the knowledge spreads.

The same idea applies to fire, storage, agriculture, paths and social organization.

---

## Main features

### Survival

Humans need to manage:

- hunger
- thirst
- energy
- temperature
- injuries
- illness
- predators
- safe sleeping places

If they fail, they can die from:

- hunger
- dehydration
- injuries
- loss of life
- predators
- environmental pressure

---

### Generations

The first generations are stronger to make early civilization possible:

```text
Generations 1-5: x2.0 survival boost
Generation 6:     x1.7 survival boost
Generation 7:     x1.4 survival boost
Generation 8+:    normal
```

This affects physical survival stats, not knowledge.

They still do not start knowing civilization.

---

### Knowledge and culture

Humans can develop concepts such as:

```text
refugio_seguro
guardar_comida
calor_seguro
plantar
caminos
```

Those concepts can later become practical knowledge:

```text
shelters
fire
campfires
storage
agriculture
paths
cooking
```

Knowledge can be:

- individual
- taught to nearby humans
- adopted as collective knowledge
- partially transmitted through cultural exposure to younger generations

---

### Agriculture

Humans can eventually learn to plant and harvest.

Possible agriculture includes:

- basic crops
- fruit trees
- apple-related planting
- figs
- berries
- roots

Map symbols:

```text
G = crop / farm
o = planted fruit tree
```

---

### Paths

Paths appear through use.

If many humans walk through the same place, the ground changes:

```text
. = grass
, = light trail
: = marked trail
= = strong path
```

Humans prefer paths because they are easier to move through.

If a path is abandoned, it slowly returns to grass.

---

### Fire and shelters

Humans do not start knowing fire or shelters.

They must discover concepts first:

```text
calor_seguro  -> fire/campfire usefulness
refugio_seguro -> shelters/houses
guardar_comida -> storage
```

Campfires, shelters and storage only become meaningful through experience.

---

### Animals

The world includes peaceful animals and predators.

There is also a T-Rex command:

```text
spawn trex
spawn trex 3
spawn animal trex 3
```

---

### SMAP full-map viewer

The simulator includes a full-map browser viewer called `smap`.

In `smap`:

```text
A = Adam
E = Eve
@ blue = man
@ pink = woman
```

This allows the full world to be viewed outside the terminal.

---

### Export system

The simulator can export a complete report folder with:

- full logs
- relevant logs
- population graph
- full map as colored letters
- weather/time/status data
- general statistics
- statistics for each human

Example:

```text
exportar /Users/YOUR_USER/Desktop/reporte_eden
```

---

## Starting options

When you run:

```bash
python3 eden_sim.py
```

the simulator asks for map size.

### Map size options

```text
1. Minúsculo   — 16 chunks — 128x128
2. Pequeño     — 24 chunks — 192x192
3. Estándar    — 32 chunks — 256x256
4. Grande      — 40 chunks — 320x320
5. Enorme      — 48 chunks — 384x384
6. Gigantesco  — 56 chunks — 448x448
7. Titánico    — 64 chunks — 512x512
8. Iceberg     — 72 chunks — 576x576
```

Recommended classic setup:

```text
Map: 3
Spawn points: 1
```

That means:

```text
Standard 256x256 map
Only Adam and Eve in the central spawn
```

---

## Spawn points

After selecting map size, the simulator asks for initial spawn points.

```text
1 = only Adam and Eve in the center
>1 = Adam and Eve in the center + extra starting couples far away
```

This allows different societies to appear in different parts of the map.

---

## Main commands

Inside the simulator, type commands and press ENTER.

### Help and status

```text
help
status
civilizacion
sociedad
agricultura
aprendizaje
hidratacion
```

What they do:

```text
help          = shows available commands
status        = current time, weather, population and world status
civilizacion  = general civilization statistics
sociedad      = social roles, fertility, protection and repopulation status
agricultura   = crop/agriculture knowledge and production
aprendizaje   = primitive concepts and learning progress
hidratacion   = hydration status and humans most at risk of thirst
```

---

### Logs

```text
logs
logs 30
relevantes
relevantes 30
```

What they do:

```text
logs          = full logs + relevant logs
logs 30       = last 30 full logs
relevantes    = important events only
relevantes 30 = last 30 important events
```

Important achievements appear in gold/yellow, births in green, deaths in red and commands in cyan.

---

### Simulation control

```text
pause
resume
speed 5
delay 0.20
quit
```

What they do:

```text
pause      = pauses simulation
resume     = resumes simulation
speed 5    = simulates 5 hours per loop
delay 0.20 = delay between loops
quit       = exits
```

---

### Map and rendering

```text
render
mapa
smap
smap off
quietos
```

What they do:

```text
render   = returns to normal individual-map render
mapa     = compressed global map inside terminal
smap     = opens full map in browser
smap off = disables/closes smap server
quietos  = shows humans that have been still for too long
```

---

### Spawning humans and animals

```text
spawn human hombre 18 3
spawn human mujer 18 3
spawn humans 18 10
spawn animal conejo 20
spawn animal lobo 2
spawn trex
spawn trex 3
spawn animal trex 3
```

Examples:

```text
spawn human hombre 18 3 = creates 3 men aged 18
spawn human mujer 18 3  = creates 3 women aged 18
spawn humans 18 10      = creates 10 balanced humans aged 18
spawn animal conejo 20  = creates 20 rabbits
spawn animal lobo 2     = creates 2 wolves
spawn trex 3            = creates 3 T-Rex
```

---

### Editing human/world values

```text
estadistica 286
estadistica all
heal all 30
setlife all 100
food all 100
water all 100
energy all 100
```

What they do:

```text
estadistica 286  = shows statistics for human id 286
estadistica all  = shows statistics for all humans
heal all 30      = heals all humans by 30
setlife all 100  = sets life of all humans to 100
food all 100     = sets hunger/food level of all humans to 100
water all 100    = sets thirst/water level of all humans to 100
energy all 100   = sets energy of all humans to 100
```

---

### Giving resources

```text
give all food 10
give all wood 30
give all hides 5
give all herbs 3
```

What they do:

```text
give all food 10  = gives 10 food to every human
give all wood 30  = gives 30 wood to every human
give all hides 5  = gives 5 hides to every human
give all herbs 3  = gives 3 herbs to every human
```

---

### Weather and disasters

```text
weather lluvia
disaster terremoto
```

Examples:

```text
weather lluvia      = forces rain
disaster terremoto  = triggers an earthquake
```

---

### Graph and export

```text
grafica /ruta/poblacion.png
exportar /ruta/carpeta
```

Examples:

```text
grafica /Users/YOUR_USER/Desktop/poblacion.png
exportar /Users/YOUR_USER/Desktop/reporte_eden
```

`grafica` creates a population graph.

`exportar` creates a full report folder.

---

## Map legend

```text
A = Adam
E = Eve
@ blue = man
@ pink = woman

. = grass
T = forest
~ = river / water
C = cave
M = mountain
S = swamp

, = light trail
: = marked trail
= = strong path

h = hut / simple shelter
H = future house
B = future building
R = safe shelter / brown house
F = campfire
D = tribal storage
G = crop / farm
o = planted fruit tree

X = T-Rex
```

---

## Recommended files in the repository

Keep all historical files if you want people to study the evolution of the project.

But make clear that:

```text
eden_sim.py
```

is the executable entry point.

Suggested structure:

```text
YJNWI-EDEN-SIM/
├── README.md
├── LICENSE
├── NOTICE
├── eden_sim.py
├── versions/
│   ├── eden_sim_v1_...
│   ├── eden_sim_v2_...
│   └── ...
├── docs/
│   └── README files / history
└── exports/
```

---

## License

This project is licensed under the Apache License 2.0.

You may use, modify, distribute and sell this software, but the original attribution to **YJNWI** and the EDEN SIM project must be preserved.

See:

```text
LICENSE
NOTICE
```

---

## Attribution

Original project, concept, simulation design and development by **YJNWI**.

Please preserve the original attribution in copies, forks, modified versions, distributions and commercial uses.

---

## Project description for GitHub

Short description:

```text
Experimental Python artificial-life civilization simulator where humans begin with only basic instincts and gradually discover shelters, fire, agriculture, paths and society through survival, learning and cultural transmission.
```

Topics:

```text
python
simulation
artificial-life
civilization
agent-based-simulation
survival-simulator
emergent-behavior
terminal-game
procedural-world
evolution
```
