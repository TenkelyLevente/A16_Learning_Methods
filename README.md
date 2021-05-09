# A16_Learning_Methods

## 1 Feladat megfogalmazása

A feladatunk során neurális hálókat tanítását vizsgáltuk különböző gradiens mentes módszerekkel. A vizsgálat során összehasonlítottuk a különböző megközelítésekkel kapott eredményeket.

## 2 Megvizsgált algoritmusok rövid ismertetése

A következő algoritmusokat választottuk ki a feladat elvégzésére. A választás fő oka az volt, hogy ezekhez találtunk megfelelő könyvtárat, ami kompatibilis volt a Keras könyvtárban létrehozott neurális hálókkal.

### 2.1 Nodal Genetic Algorithm (NGA)

A csomóponti genetikus algoritmus (NGA) hasonló a hagyományos genetikus algoritmushoz, de ebben az esetben a csomópontok száma (melyet a mutációs ráta határoz meg) véletlenszerűen kerül kiválasztásra és csak a csomópontoknak megfelelő súlyok és bias értékek mutálódnak normál eloszlású értékek hozzáadásával a szigma által meghatározott normális eloszlással.

### 2.2 Covariance Matrix Adaptation Evolution Strategy (CMA-ES)

Kovariancia Mátrix Adaptációs Evolúciós Stratégia (CMA-ES) egy olyan evolúciós algoritmus, amely alkalmas sztochasztikus vagy randomizált módszerek, illetve nemlineáris, s nem konvex függvények valós paraméteres (folyamatos tartományú) optimalizálására.
Előnye, hogy jól teljesít nem szeparábilis problémák esetén, ha az objektív függvény deriváltja nem meghatározható, illetve magas dimenziójú problémák és nagyon nagy keresési területek esetén. Hátránya viszont, hogy részlegesen szeparálható problémák esetén, vagy amikor az objektív függvény deriváltja könnyen meghatározható, illetve alacsony dimenziós problémák esetén gyengébben teljesít más algoritmusokkal szemben.

### 2.3 Genetikus algoritmus

![alt text](https://github.com/TenkelyLevente/A16_Learning_Methods/blob/main/images/1.png)

A genetikus algoritmus speciális evolúciós algoritmusok, szintén a természetből ellesett optimalizációs technika. Az algoritmus során kezdetben létrehozunk egy kezdeti populációt, majd a populációban lévő egyedeket rangsoroljuk az egyes egyedekre jellemző fitnesz érték (alkalmassági érték) alapján, hogy mennyire jó megoldást adnak a feladatra. Ezek után végrehajtunk a populáción egy szelekciót, a fitnesz érték alapján (minél jobb az egyed, annál nagyobb az esély a kiválasztására). Az így megmaradt egyedeket pedig keresztezzük egymással egy véletlenszerű keresztezési pont kiválasztása alapján. Végezetül pedig mutációt hajtunk végre a populáción, mégpedig úgy, hogy az egyedekhez tartozó gének értékeit véletlenszerűen egy p_m (mutációs arány) valószínűséggel megváltoztatjuk. És ezt a folyamatot ismételjük az egyedek rangsorolásától kezdve a populáció mutációjáig, ameddig megfelelő eredményre nem jutunk.

### 2.4 Gradient descent

A gradiens módszer lényege, hogy mindig a legnagyobb csökkenés irányába haladunk. Kétváltozós függvény esetén ez úgy képzelhető el, hogy a hegyről a legmeredekebb úton ereszkedünk le a völgybe. A térképen megszokott szintvonalakkal ez jól ábrázolható. A 2. ábra első részén egy kétváltozós függvény képe, a második részén pedig a szintvonalak és a negatív gradiensek láthatóak. A csökkenés iránya jól megfigyelhető.

![alt text](https://github.com/TenkelyLevente/A16_Learning_Methods/blob/main/images/2.png)
![alt text](https://github.com/TenkelyLevente/A16_Learning_Methods/blob/main/images/3.png)

## 3 Megvalósítás

A megvalósítás során a Google Colab segítségével írtuk meg és futtattuk a programunkat. A Tensorflow és Keras könyvtárakban hoztuk létre a modelljeinket.

### 3.1 Képi klasszifikációs feladat

Feladatnak a Fashion-MNIST adatgyűjteményt választottuk, amely képeket tartalmaz különböző ruhadarabokról. 10 darab különböző osztályból áll és ezt kell a hálónak megfelelően klasszifikálnia.

![alt text](https://github.com/TenkelyLevente/A16_Learning_Methods/blob/main/images/4.jpg)

### 3.1.1 Eredmények
Az előbb bemutatott algoritmusok alapján megvalósítuttok kódban is a klasszifikálást. A kapott eredmények alapján látható, hogy a neurális hálóknak rendkívül nagy a számítási igénye, mivel rendkívül sok paraméterből állnak. A gradiens alapú keresés elég gyorsan viszont elég jó minőségű megoldást ad, míg a többi algoritmus hozzá viszonyítva rendkívül lassan konvergál a megoldáshoz. Ezt azzal indokoljuk, hogy mivel rendkívül sok paramétert kell figyelemmel tartaniuk és nem használnak gradiens alapú keresést, így nagyon lassan tanulnak. Továbbá azt is meg kell említeni, hogy az előző algoritmusoknak elég sok állítható paramétere van, és minden egyes paraméter kombináció más-más tanulást eredményezhet. Esetlegesen előfordulhat, hogy adott paraméter kombinációkkal jobb eredményeket tudtunk volna elérni, de még így is látható, hogyha lassan is, de konvergálnak a megoldás felé.

![alt text](https://github.com/TenkelyLevente/A16_Learning_Methods/blob/main/images/5.png)

Ahogy az előző ábrákon is látható a gradiens alapú keresés rendkívül gyors a többi algoritmussal szemben. Kísérleteztünk azzal, hogy kombináltuk a gradiens alapú keresést a genetikus algoritmussal. Ebben az esetben láthatóan jobb eredményt tudtunk elérni, mint a sima gradiens alapú keresés során. Ezt azzal magyarázhatjuk, hogy a genetikus algoritmusnak köszönhetően még kevésbé tud lokális minimumban ragadni, mivel az mindig a mutációknak köszönhetően kimozdítja onnan.
Az alábbi ábrán látható a CAM algoritmuson elért eredményeink. Ez teljesített a feladat szempontjából a leggyengébben, de ennek a dokumentációjában is szerepel, hogy részlegesen szeparálható problémák esetén, vagy amikor az objektív függvény deriváltja könnyen meghatározható, akkor rendkívül gyengén teljesít a többi algoritmushoz viszonyítva.

![alt text](https://github.com/TenkelyLevente/A16_Learning_Methods/blob/main/images/6.png)

### 3.2 Konvolúciós rétegek alkalmazása CIFAR10 adatgyűjteményen

Egy konvolúciós neurális hálón is megnéztük a genetikus algoritmus és a gradiens módszer kombináltját a CIFAR10 adatgyűjteményen összehasonlítva a puszta gradiens alapú módszerrel. Az alábbi ábrán láthatók az eredmények. Összességében hasonlóak az előbbiekhez.
![alt text](https://github.com/TenkelyLevente/A16_Learning_Methods/blob/main/images/7.png)

### 3.3 Regressziós feladat
Elvégeztünk egy regressziós feladatot is, ahol az előző algoritmusokat ismételten összehasonlítottuk. A választott adatgyűjtemény az Auto MPG adatgyűjtemény (UCI Machine Learning Repository: Auto MPG Data Set), ahol az autó adatai alapján becsüljük meg a városi üzemanyag fogyasztását.

#### 3.3.1 Regresszión kapott eredmények
![alt text](https://github.com/TenkelyLevente/A16_Learning_Methods/blob/main/images/8.png)
![alt text](https://github.com/TenkelyLevente/A16_Learning_Methods/blob/main/images/9.png)
A felső ábrán látható, hogy a genetikus algoritmus milyen eredményt ért el az adathalmazon, illetve az alsón az, hogy a gradiens esés és az NGA milyen eredményeket értek el. A CAM ezen az adathalmazon még rosszabbul teljesített, mint az előzőn, így azt nem ábrázoltuk.

## 4 Konklúzió

A neurális hálókat tekintve a gradiens alapú algoritmusok sokkalta jobban teljesítenek, mint a nem gradiens alapúak, viszont sokkalta hajlamosabbak lokális minimumok ragadni. Ezzel szemben önmagukban ezekre a feladatokra nem ajánljuk saját tapasztalataink alapján a genetikus és evolúciós algoritmusokat, mivel a neurális háló rendkívül sok paraméterére nem tudnak olyan jól tanulni, mint gradiens alapú társaik. Azonban egy rendkívül jó ötletnek tartjuk ötvözni ezeket a módszereket
Ezzel szemben rendkívül jók olyan feladatokra a neurális hálókkal kapcsolatosan, amiket a gradiens alapú módszerek képtelenek lekezelni. A hiperparaméter optimalizálás egy pont ilyen téma. Azok a paraméterek, illetve a neurális háló szerekezete, illetve egyéb paraméterek, amelyekre nem tud hibát számolni a modell, így gradiensét se tudja meghatározni, azokra tökéletesek lehetnek az evolúciós és genetikus algorimtusok.

# How to use

Három kód található a repositoryban. A kódok között a fő különbség az, hogy milyen hálót, veszteségfüggvényt, adatgyűjteményt, illetve algoritmusokat tartalmaznak.

* ga_plusz_gd_kombo_cifar.py
* regression.py
* image_classification.py

Mindegyik kód hasonló felépítéssel bír. Első sorban mindig a még telepítendő könyvtárakat tesszük fel, amennyiben importálni szeretnénk a hozzá tartozó könyvtárakat. Erre például:
```pip install evolutionary-keras```
Ezután importáljuk a könyvtárakat, amiket esetlegesen felhasználunk (jelenleg van bár könyvtár, amit az előzőekben használtunk, de jeleleg nem használjuk őket, a későbbiekben esetleges módosítások miatt tartottuk meg őket:

```# Könyvtárak importálása
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
import numpy as np
from tensorflow.keras import Sequential, layers, models, datasets
from keras.layers import Dense, Conv2D , MaxPool2D , Flatten ,Input
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report, confusion_matrix
import datetime
import seaborn as sn
import pandas as pd
import random, time
import sys
from tensorflow.keras.datasets import mnist
from tensorflow.keras import backend as K
from evolutionary_keras.models import EvolModel
import evolutionary_keras.optimizers
from tensorflow.keras.models import Model
```
Ezután az adatgyűjteményeket beolvassuk és kiszedjük a hozzátartozó adatokat, illetve esetlegesen előfeldolgozzuk őket. Kiszedjük belőlük a tanító, illetve tesztelő adatokat.

```(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

train_images, test_images = train_images / 255.0, test_images / 255.0

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
```
Ezután létrehozzuk a hálót. Ezt többféleképpen valósítottuk meg. A Keras API segítségével, illetve csak pusztán a megszokott módon:

```
def modelbuilder():
    model = keras.Sequential([
      layers.Input(shape=(32, 32, 3)),
      layers.Conv2D(8, (3, 3), activation='sigmoid'),
      layers.MaxPool2D(),
      layers.Conv2D(8, (3, 3), activation='sigmoid'),
      layers.MaxPool2D(),
      layers.Conv2D(8, (3, 3), activation='sigmoid'),
      layers.MaxPool2D(),
      layers.Flatten(),
      layers.Dense(len(class_names),activation="softmax")
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(0.01),
              loss="sparse_categorical_crossentropy",
              metrics=['accuracy'])
    return model
```
Illetve az API-val:
```
inputs = Input(shape=(28, 28, 1))
flatten = Flatten()(inputs)
dense = Dense(8, activation="relu")(flatten)
dense = Dense(8, activation="relu")(dense)
prediction = Dense(10, activation="softmax")(dense)
modelGD=Model(inputs=inputs, outputs=prediction)
modelGD.compile(optimizer=tf.keras.optimizers.Adam(0.0001),
              loss="categorical_crossentropy",
              metrics=['accuracy'])
```
Az evolúciós algoritmusokhoz tartozó modelleket a Model helyett csak a EvolModel:
```
Model(inputs=inputs, outputs=prediction)
EvolModel(inputs=inputs, outputs=prediction)
```
Minden algoritmushoz külön modellt/eket hoztunk létre.

A tanítások minden algoritmus esetén eltérnek:
A gradiens alapú illetve az evolúciós algoritmusoknál a model.fit függvénnyel elkezdenek tanulni a modellek a hozzájuk tartozó algoritmusokkal (CMA,NGA,GD).
Ezekhez az eredmények a megfelelő historyX-ban lesz eltárolva.

Kipróbáltuk a PyGAD könyvtárat is regresszió esetén, ahol a rendkívül sok opció elérhető mutációkat és keresztezéseket tekintve. Itt a tanítás és a kiértékelés külön meg van írva a kommentekből logikusan következnek a megadott lépesek:
```
# Prepare the PyGAD parameters. Check the documentation for more information: https://pygad.readthedocs.io/en/latest/README_pygad_ReadTheDocs.html#pygad-ga-class
num_generations = 300 # Number of generations.
num_parents_mating = 30 # Number of solutions to be selected as parents in the mating pool.
initial_population = keras_ga.population_weights # Initial population of network weights
parent_selection_type = "sss" # Type of parent selection.
crossover_type = "two_points" # Type of the crossover operator.
mutation_type = "random" # Type of the mutation operator.
mutation_percent_genes = 10 # Percentage of genes to mutate. This parameter has no action if the parameter mutation_num_genes exists.
keep_parents = 1 # Number of parents to keep in the next population. -1 means keep all parents and 0 means keep nothing.

ga_instance = pygad.GA(num_generations=num_generations,
                       num_parents_mating=num_parents_mating,
                       initial_population=initial_population,
                       fitness_func=fitness_func,
                       parent_selection_type=parent_selection_type,
                       crossover_type=crossover_type,
                       mutation_type=mutation_type,
                       mutation_percent_genes=mutation_percent_genes,
                       keep_parents=keep_parents,
                       on_generation=callback_generation)

# Start the genetic algorithm evolution.
ga_instance.run()

# After the generations complete, some plots are showed that summarize how the outputs/fitness values evolve over generations.
ga_instance.plot_result(title="PyGAD & Keras - Iteration vs. Fitness", linewidth=4)

# Returning the details of the best solution.
solution, solution_fitness, solution_idx = ga_instance.best_solution()
print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
print("Index of the best solution : {solution_idx}".format(solution_idx=solution_idx))

# Fetch the parameters of the best solution.
best_solution_weights = pygad.kerasga.model_weights_as_matrix(model=model,
                                                              weights_vector=solution)
model.set_weights(best_solution_weights)
predictions = model.predict(data_inputs)
print("Predictions : \n", predictions)

mae = tensorflow.keras.losses.MeanAbsoluteError()
abs_error = mae(data_outputs, predictions).numpy()
print("Absolute Error : ", abs_error)
```
A genetikus algoritmus és a gradiens alapú keresés kombináltja, ahol még eltér a többi algotirmustól a megoldás.
A keresztezést egyszerűen kikérjük a 2 szülőt és 50-50% eséllyel tartjuk meg az egyik, illetve a másik paramétereit. Mutáció hasonlóan van megvalósítva, csak ott egy random értékkel módosítjuk a meglévő értékeket adott valószínűséggel (mutation_rate), illetve adott mértékben (mutation_power). A ```runtournament()``` függvényben kiválasztunk a modellek közül egy részhalmazt, majd azon belül a 2 legjobb egyeddel térünk vissza. A fitness érték kiszámítása a ```parallel_scoring()``` függvényben számolódik ki.
A ```parallel_muttion()``` függvényben határozzuk meg a
