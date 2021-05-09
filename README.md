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

A genetikus algoritmus speciális evolúciós algoritmusok, szintén a természetből ellesett optimalizációs technika. Az algoritmus során kezdetben létrehozunk egy kezdeti populációt, majd a populációban lévő egyedeket rangsoroljuk az egyes egyedekre jellemző fitnesz érték (alkalmassági érték) alapján, hogy mennyire jó megoldást adnak a feladatra. Ezek után végrehajtunk a populáción egy szelekciót, a fitnesz érték alapján (minél jobb az egyed, annál nagyobb az esély a kiválasztására). Az így megmaradt egyedeket pedig keresztezzük egymással egy véletlenszerű keresztezési pont kiválasztása alapján. Végezetül pedig mutációt hajtunk végre a populáción, mégpedig úgy, hogy az egyedekhez tartozó gének értékeit véletlenszerűen egy p_m (mutációs arány) valószínűséggel megváltoztatjuk. És ezt a folyamatot ismételjük az egyedek rangsorolásától kezdve a populáció mutációjáig, ameddig megfelelő eredményre nem jutunk.

### 2.4 Gradient descent

A gradiens módszer lényege, hogy mindig a legnagyobb csökkenés irányába haladunk. Kétváltozós függvény esetén ez úgy képzelhető el, hogy a hegyről a legmeredekebb úton ereszkedünk le a völgybe. A térképen megszokott szintvonalakkal ez jól ábrázolható. A 2. ábra első részén egy kétváltozós függvény képe, a második részén pedig a szintvonalak és a negatív gradiensek láthatóak. A csökkenés iránya jól megfigyelhető.

## 3 Megvalósítás

A megvalósítás során a Google Colab segítségével írtuk meg és futtattuk a programunkat. A Tensorflow és Keras könyvtárakban hoztuk létre a modelljeinket.

### 3.1 Képi klasszifikációs feladat

Feladatnak a Fashion-MNIST adatgyűjteményt választottuk, amely képeket tartalmaz különböző ruhadarabokról. 10 darab különböző osztályból áll és ezt kell a hálónak megfelelően klasszifikálnia.

### 3.1.1 Eredmények
Az előbb bemutatott algoritmusok alapján megvalósítuttok kódban is a klasszifikálást. A kapott eredmények alapján látható, hogy a neurális hálóknak rendkívül nagy a számítási igénye, mivel rendkívül sok paraméterből állnak. A gradiens alapú keresés elég gyorsan viszont elég jó minőségű megoldást ad, míg a többi algoritmus hozzá viszonyítva rendkívül lassan konvergál a megoldáshoz. Ezt azzal indokoljuk, hogy mivel rendkívül sok paramétert kell figyelemmel tartaniuk és nem használnak gradiens alapú keresést, így nagyon lassan tanulnak. Továbbá azt is meg kell említeni, hogy az előző algoritmusoknak elég sok állítható paramétere van, és minden egyes paraméter kombináció más-más tanulást eredményezhet. Esetlegesen előfordulhat, hogy adott paraméter kombinációkkal jobb eredményeket tudtunk volna elérni, de még így is látható, hogyha lassan is, de konvergálnak a megoldás felé.

Ahogy az előző ábrákon is látható a gradiens alapú keresés rendkívül gyors a többi algoritmussal szemben. Kísérleteztünk azzal, hogy kombináltuk a gradiens alapú keresést a genetikus algoritmussal. Ebben az esetben láthatóan jobb eredményt tudtunk elérni, mint a sima gradiens alapú keresés során. Ezt azzal magyarázhatjuk, hogy a genetikus algoritmusnak köszönhetően még kevésbé tud lokális minimumban ragadni, mivel az mindig a mutációknak köszönhetően kimozdítja onnan.
Az alábbi ábrán látható a CAM algoritmuson elért eredményeink. Ez teljesített a feladat szempontjából a leggyengébben, de ennek a dokumentációjában is szerepel, hogy részlegesen szeparálható problémák esetén, vagy amikor az objektív függvény deriváltja könnyen meghatározható, akkor rendkívül gyengén teljesít a többi algoritmushoz viszonyítva.

### 3.2 Konvolúciós rétegek alkalmazása CIFAR10 adatgyűjteményen

Egy konvolúciós neurális hálón is megnéztük a genetikus algoritmus és a gradiens módszer kombináltját a CIFAR10 adatgyűjteményen összehasonlítva a puszta gradiens alapú módszerrel. Az alábbi ábrán láthatók az eredmények. Összességében hasonlóak az előbbiekhez.

### 3.3 Regressziós feladat
Elvégeztünk egy regressziós feladatot is, ahol az előző algoritmusokat ismételten összehasonlítottuk. A választott adatgyűjtemény az Auto MPG adatgyűjtemény (UCI Machine Learning Repository: Auto MPG Data Set), ahol az autó adatai alapján becsüljük meg a városi üzemanyag fogyasztását.

#### 3.3.1 Regresszión kapott eredmények
 
A felső ábrán látható, hogy a genetikus algoritmus milyen eredményt ért el az adathalmazon, illetve az alsón az, hogy a gradiens esés és az NGA milyen eredményeket értek el. A CAM ezen az adathalmazon még rosszabbul teljesített, mint az előzőn, így azt nem ábrázoltuk.

## 4 Konklúzió

A neurális hálókat tekintve a gradiens alapú algoritmusok sokkalta jobban teljesítenek, mint a nem gradiens alapúak, viszont sokkalta hajlamosabbak lokális minimumok ragadni. Ezzel szemben önmagukban ezekre a feladatokra nem ajánljuk saját tapasztalataink alapján a genetikus és evolúciós algoritmusokat, mivel a neurális háló rendkívül sok paraméterére nem tudnak olyan jól tanulni, mint gradiens alapú társaik. Azonban egy rendkívül jó ötletnek tartjuk ötvözni ezeket a módszereket
Ezzel szemben rendkívül jók olyan feladatokra a neurális hálókkal kapcsolatosan, amiket a gradiens alapú módszerek képtelenek lekezelni. A hiperparaméter optimalizálás egy pont ilyen téma. Azok a paraméterek, illetve a neurális háló szerekezete, illetve egyéb paraméterek, amelyekre nem tud hibát számolni a modell, így gradiensét se tudja meghatározni, azokra tökéletesek lehetnek az evolúciós és genetikus algorimtusok.
