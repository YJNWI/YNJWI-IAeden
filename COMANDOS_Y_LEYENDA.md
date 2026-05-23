COMANDOS EDEN SIM

Help:
Muestra la ayuda con todos los comandos disponibles.

Status:
Muestra el estado actual del mundo: año, mes, día, hora, estación, clima, temperatura, población, animales, velocidad y si los caminos están activados o desactivados.

Pause:
Pausa la simulación.

Resume:
Reanuda la simulación.

Quit:
Cierra el simulador.

Speed:
Se usa como speed 5, speed 20, speed 100, etc.
Sirve para decir cuántas horas/ciclos avanza la simulación por vuelta.
A mayor número, más rápido avanza el tiempo.
Ejemplo: speed 10 hace que avance más rápido que speed 1.

Delay:
Se usa como delay 0, delay 0.001, delay 0.05, delay 0.20, etc.
Sirve para cambiar la pausa entre ciclos.
A menor número, más rápido va la simulación.
delay 0 es lo más rápido.
delay 0.20 va más lento y permite ver mejor lo que pasa.

Render:
Vuelve al mapa normal de la terminal.

Mapa:
Muestra el mapa global reducido dentro de la terminal.
Sirve para ver el mundo completo, pero comprimido.

Smap:
Abre el mapa completo en una ventana/navegador.
Sirve para ver el mapa entero con letras pequeñas y colores.
En smap los hombres salen como @ azul y las mujeres como @ rosa.

Smap off:
Apaga el servidor/visor de smap.
Sirve si el navegador se queda pillado o quieres reiniciar el mapa completo.

Quietos:
Muestra humanos que llevan muchas horas sin moverse.
Sirve para detectar personas atascadas o en bucle.

Logs:
Muestra todos los registros de la simulación y al final los eventos relevantes.

Logs 30:
Muestra los últimos 30 registros.
Puedes cambiar el número.
Ejemplo: logs 10, logs 50, logs 100.

Relevantes:
Muestra solo los eventos importantes: nacimientos, muertes, descubrimientos, fuego, refugios, agricultura, extinción, etc.

Relevantes 30:
Muestra los últimos 30 eventos importantes.
Puedes cambiar el número.
Ejemplo: relevantes 10, relevantes 50, relevantes 100.

Civilizacion:
Muestra estadísticas generales de la civilización: población, máximo histórico, generaciones, construcciones, almacenes, cultivos, caminos, recursos y conocimientos.

Sociedad:
Muestra información social: roles, fertilidad, mujeres fértiles, protectores, cuidadores, repoblación y organización social.

Agricultura:
Muestra información agrícola: cultivos, frutales, semillas, producción, conocimiento agrícola y cultivos listos para cosechar.

Aprendizaje:
Muestra los conceptos que los humanos están aprendiendo, como refugio seguro, guardar comida, calor seguro, plantar y caminos si están activados.
Sirve para ver cómo avanza la civilización sin que nazcan sabiendo cosas.

Hidratacion:
Muestra los humanos con más sed y sus datos.
Sirve para detectar problemas de agua o bucles de deshidratación.

Spawn human hombre:
Se usa como spawn human hombre 18 3.
Crea hombres.
El primer número es la edad y el segundo la cantidad.
Ejemplo: spawn human hombre 18 3 crea 3 hombres de 18 años.

Spawn human mujer:
Se usa como spawn human mujer 18 3.
Crea mujeres.
El primer número es la edad y el segundo la cantidad.
Ejemplo: spawn human mujer 18 3 crea 3 mujeres de 18 años.

Spawn humans:
Se usa como spawn humans 18 10.
Crea humanos mezclando hombres y mujeres.
El primer número es la edad y el segundo la cantidad.
Ejemplo: spawn humans 18 10 crea 10 humanos de 18 años.

Spawn animal:
Se usa como spawn animal conejo 20, spawn animal lobo 2, spawn animal ciervo 10, etc.
Crea animales del tipo indicado.
El último número es la cantidad.

Spawn trex:
Se usa como spawn trex.
Crea 1 T-Rex.

Spawn trex 3:
Crea 3 T-Rex.
Puedes cambiar el número.
Ejemplo: spawn trex 10 crea 10 T-Rex.

Spawn animal trex:
Se usa como spawn animal trex 3.
También sirve para crear varios T-Rex.

Estadistica:
Se usa como estadistica 286.
Muestra las estadísticas de un humano concreto usando su ID.

Estadistica all:
Muestra las estadísticas de todos los humanos.

Heal all:
Se usa como heal all 30.
Cura a todos los humanos.
El número indica cuánta vida se recupera.
Ejemplo: heal all 30 cura 30 puntos.

Setlife all:
Se usa como setlife all 100.
Pone la vida de todos los humanos al valor indicado.
Ejemplo: setlife all 100 pone la vida de todos a 100.

Food all:
Se usa como food all 100.
Pone el nivel de comida/hambre de todos al valor indicado.
Ejemplo: food all 100 pone a todos con comida completa.

Water all:
Se usa como water all 100.
Pone el nivel de agua/sed de todos al valor indicado.
Ejemplo: water all 100 quita la sed a todos.

Energy all:
Se usa como energy all 100.
Pone la energía de todos al valor indicado.
Ejemplo: energy all 100 pone a todos con energía alta.

Give all food:
Se usa como give all food 10.
Da comida a todos los humanos.
El número es la cantidad.

Give all wood:
Se usa como give all wood 30.
Da madera a todos los humanos.
El número es la cantidad.

Give all hides:
Se usa como give all hides 5.
Da pieles a todos los humanos.
El número es la cantidad.

Give all herbs:
Se usa como give all herbs 3.
Da hierbas a todos los humanos.
El número es la cantidad.

Give all stone:
Se usa como give all stone 20.
Da piedra a todos los humanos.
El número es la cantidad.

Give all seeds:
Se usa como give all seeds 10.
Da semillas a todos los humanos.
El número es la cantidad.

Give all clay:
Se usa como give all clay 10.
Da arcilla a todos los humanos.
El número es la cantidad.

Weather:
Se usa como weather lluvia, weather soleado, weather tormenta, weather nieve, weather frio_extremo, weather ola_calor, etc.
Cambia el clima del mundo.

Disaster:
Se usa como disaster terremoto, disaster tornado, disaster incendio, disaster sequia, etc.
Provoca un desastre en el mundo.

Grafica:
Se usa como grafica /Users/ruta/del/archivo.png
Genera una gráfica de población en la ruta indicada.

Exportar:
Se usa como exportar /Users/ruta/del/archivo
Crea una carpeta con el informe completo de la simulación.
Incluye logs completos, logs relevantes, gráfica de población, mapa completo, datos de clima/tiempo, estadísticas generales y estadísticas de cada persona.

LEYENDA DEL MAPA

A:
Adán.

E:
Eva.

@ azul:
Hombre.

@ rosa:
Mujer.

.:
Pradera.

T:
Bosque.

~:
Río o agua.

C:
Cueva.

M:
Montaña.

S:
Pantano.

,:
Sendero leve.
Solo aparece si los caminos están activados.

::
Sendero marcado.
Solo aparece si los caminos están activados.

=:
Camino fuerte.
Solo aparece si los caminos están activados.

h:
Choza o refugio sencillo.

H:
Casa.

B:
Edificio.

R:
Refugio o casa segura.

F:
Hoguera.

D:
Almacén.

G:
Cultivo.

o:
Frutal plantado.

c:
Conejo.

d:
Ciervo.

p:
Pez.

g:
Gallina salvaje.

k:
Cabra salvaje.

L:
Lobo.

O:
Oso.

V:
Serpiente.

J:
Jabalí.

P:
Pantera.

X:
T-Rex.

m:
Manzana silvestre.

u:
Bayas azules.

i:
Higos.

n:
Nueces.

r:
Raíces comestibles.

v:
Bayas rojas venenosas.

q:
Hongos morados.

z:
Fruto negro brillante.

y:
Raíz amarga.

CONFIGURACIÓN RECOMENDADA AL INICIAR

Mapa:
3 para estándar.

Puntos de spawn:
1 para empezar solo con Adán y Eva.

Caminos:
ENTER o 1 para activarlos.
2 para desactivarlos.

COMANDOS ÚTILES PARA PROBAR

Crear población:
spawn humans 18 50

Dar recursos:
give all food 20
give all wood 30
give all herbs 5

Estabilizar humanos:
heal all 30
food all 100
water all 100
energy all 100

Ver problemas:
quietos
hidratacion
logs 50
relevantes 50

Exportar resultado:
exportar /Users/ruta/del/archivo
