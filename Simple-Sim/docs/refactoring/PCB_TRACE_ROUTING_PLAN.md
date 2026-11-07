# Leiterbahnen bauteilgerecht generieren – Analyse & Umbauplan

Stand: 2026-11-06, Fassung 2 (Review der ersten Fassung gegen den Arbeitsstand nach v1.0.3,
Branch `1.0.3`). **Umsetzungsstand:** Phase 0 übersprungen, Phase 1 umgesetzt
(`simple_sim/board/`, `render.board`, Tests in `tests/test_board.py`, Doku in
`docs/guides/PHOTOREAL_RENDERING.md`); Abweichungen vom Plan siehe Abschnitt 7. Betrifft den `photoreal`-Stil des OpenCV-2D-Backends **und** das
Blender-Backend (dort wird dieselbe Platine via `render_board()` als Textur gebacken,
`simple_sim/generators/blender3d/bake.py`). Jede Änderung hier wirkt also auf beide Backends.

Ziel: Leiterbahnen, Vias und Kupferflächen sollen **elektrisch plausibel zum Bauteil passen**
(Pads werden angebunden, nichts schließt kurz, Design-Rules werden eingehalten) – und das so
modular, dass dieselben Bausteine später eine **komplette Platine** generieren können.

Code-Stellen werden hier über Funktionsnamen referenziert, nicht über Zeilennummern: die Datei
ändert sich laufend (Fassung 1 zitierte Zeilen, die weder zu `HEAD` noch zum Arbeitsstand passten).

### Änderungen gegenüber Fassung 1

- **P10 war überholt:** Bahnbreiten und Via-Radien skalieren im Arbeitsstand bereits mit der
  kleinsten Pad-Abmessung (`_feature()` / `_pad_ref()`); fest in Pixeln waren sie nur in `HEAD`.
  Offen bleibt: kein mm-Bezug, keine Netzklassen, keine Neck-down-Regel.
- **`mm_per_px` erreicht den Renderer nicht.** `render_photoreal_base()` bekommt nur
  `config["render"]` und ein `nominal` in Pixeln; `roi.mm_per_px` liegt in `config["roi"]`.
  Das mm-Datenmodell braucht also eine kleine API-Erweiterung (siehe 3.0). Profile sind
  grundsätzlich in Pixeln definiert (`geometry_ranges`), nicht in mm.
- **RNG-Trennung existiert im 3D-Backend schon** (`np.random.default_rng([seed, k])` in
  `bake.py` für Kamerapose und Kameramodell). Der Plan übernimmt diese Konvention statt
  `SeedSequence.spawn()`, und schaltet die Trennung an dieselbe Option wie den Router, sonst
  kann `routing: legacy` nicht bit-gleich bleiben (Widerspruch in Fassung 1 zwischen 3.10,
  3.11 und den Tests).
- **Widerspruch Datenmodell / Flächen / DRC aufgelöst:** Fassung 1 verlangte ein reines
  Datenmodell ohne OpenCV, rechnete Flächen aber als Raster-Masken und wollte DRC „ohne
  Rendern“. Jetzt: Modell = Daten, Flächen und DRC auf einem **mm-Raster fester Auflösung**
  (unabhängig von `mm_per_px` und Supersampling), Rendering rastert separat in Pixel.
- **Fehlende Definitionen ergänzt:** `Port`, `ThermalRelief`, `RouteResult`, Via-Lagenspanne.
- **Render-Fenster ≠ ROI:** `_board()` rendert bereits ein größeres Fenster (Phone-Preset
  `view_canvas()`, Blender `BOARD_SCALE`/`crowd`). Das Board-Modell muss das Fenster abdecken,
  nicht nur das ROI.
- **Konsumenten der Pad-Dicts benannt** (Blender-Szene, Lot, Defekt-Layout, `flat`-Stil,
  Profil-Editor) und ein Adapter eingeplant, statt sie alle gleichzeitig umzubauen.
- `meta.jsonl`: DRC-Ergebnis unter `render_meta.board_drc` statt als neues Top-Level-Feld
  (`MetaRow` ist eine Dataclass mit festen Feldern).
- Via-Anzahl im Arbeitsstand ist 2–16 (80 %) bzw. 0–2, skaliert mit der Fensterfläche; nicht 0–12.
- Zusätzliche Punkte: Thread-Sicherheit (paralleles Rendern), Kosten von Belegungschecks auf
  dem supersampelten Canvas, Referenzbild-Hashes sind OpenCV-versionsabhängig,
  `--extend` prüft nur den Profil-Hash, nicht die Render-Optionen.

---

## 1. Ist-Zustand

Alles steckt in einer Funktion: `_traces_and_vias()` in
`simple_sim/generators/opencv2d/photoreal.py`, aufgerufen aus `_board()`.

Reihenfolge in `_board()`:

1. Pads des Prüflings berechnen (`_compute_pad_positions`, `draw.py`) → Keep-out-Rechteck
   (Bounding-Box aus Pads **und** nominalem Body)
2. Substrat (oder echte Textur über `render.texture_dir` → dann **keine** Leiterbahnen)
3. `_traces_and_vias()` – Kupfer in **eine** Maske, dann einmal eingeblendet:
   - Kupferflächen (60 %): 1–2 Achtecke, nur das Keep-out-Rechteck (+3–6 px Spalt) wird ausgespart
   - 1–3 „freie“ Leiterbahnen von Bildrand zu Bildrand
   - pro Pad mit 35 % eine Leiterbahn vom Pad-Mittelpunkt Richtung Bildrand
   - Vias an Zufallspositionen außerhalb des Keep-outs: 2–16 (80 %) bzw. 0–2, skaliert mit der
     Fensterfläche relativ zu 512²
   - Maße: `_feature(lo, hi, lo_px, hi_px)` = Anteil der kleinsten Pad-Abmessung, in Pixel
     geklemmt (Bahn 0,10–0,35 ×, Via-Radius 0,22–0,36 ×)
4. Bestückungsdruck (Silkscreen)
5. Nachbarbauteile platzieren und zeichnen (`_neighbours`, `_draw_neighbours`)
6. Rework-Rückstände

Alle Größen sind Pixel im supersampelten Render-Fenster (`_ss()` ist thread-lokal). Das Fenster
ist das ROI mal einem Canvas-Faktor (Phone-Preset: `view_canvas()`; Blender-Bake:
`BOARD_SCALE` ≥ 2, Zoom/Tilt); `crowd` skaliert die Nachbardichte mit.

Die Leiterbahnen wissen also nichts von Netzen, von den Nachbarbauteilen, voneinander oder
von Abständen. Das reicht für eine „sieht nach Platine aus“-Textur, ist aber elektrisch oft
unmöglich – und das Modell kann solche Muster lernen.

---

## 2. Gefundene Probleme

Schwere: **H** = sichtbar falsch / Kurzschluss im Bild, **M** = unrealistisch, **L** = Wartung / Skalierung.

### 2.1 Logik / Elektrik

| # | Schwere | Stelle | Problem |
|---|---|---|---|
| P1 | H | `_traces_and_vias`, Block „Free traces“ | **Freie Leiterbahnen ignorieren Bauteil und Keep-out.** Sie laufen von Rand zu Rand quer durchs Bild, auch durch die Pads des Prüflings und zwischen ihnen hindurch (z. B. zwischen den zwei Pads eines 0603 oder quer durch die QFN-Pinreihe). Elektrisch ist das ein Kurzschluss. Unter dem Bauteilkörper verdeckt, aber bei `MISSING` und bei Verschiebungen sichtbar – genau bei den Defektklassen, auf die es ankommt. |
| P2 | H | Block „Traces leaving a few pads“ | **QFN-Thermalpad bekommt eine Leiterbahn.** Das Thermalpad liegt in der Bildmitte, `outward ≈ (0, 0)`, der Vergleich `abs(outward[0]) >= abs(outward[1])` ist dann wahr, also fällt es in den x-Zweig und die Bahn läuft waagerecht **durch die linke oder rechte Pinreihe**. Real: Thermalpad = GND, angebunden über Vias im Pad (via-in-pad) und/oder eine Kupferfläche, nie über eine einzelne Signalleitung quer durch die Pins. |
| P3 | H | `_polyline` | **Der 45°-Knick beginnt direkt am Pad-Mittelpunkt.** `_polyline` geht zuerst diagonal (`jog = min(|dx|, |dy|)`), dann gerade. Mit Zielversatz ±h/6 bzw. ±w/6 schneidet die Diagonale bei Fine-Pitch (QFN, SOIC) die **Nachbarpads derselben Reihe**. Real: ein gerades Stück senkrecht aus der Padkante heraus (Fanout), erst danach abknicken. |
| P4 | H | beide Trace-Blöcke | **Leiterbahnen kreuzen sich** auf derselben Lage (freie gegen freie, freie gegen Pad-Abgänge). Zwei Netze auf einer Lage dürfen sich nie berühren; Kreuzungen gibt es nur mit Lagenwechsel (Via). |
| P5 | H | Reihenfolge in `_board()` | **Nachbarbauteile werden nach dem Routing platziert.** Leiterbahnen und Vias laufen durch deren Pads (Kurzschluss) und unter deren Körpern; die Nachbarpads sind nie angebunden. `_neighbours` kennt nur das Keep-out und die anderen Nachbarn (`taken`), kein Kupfer. |
| P6 | M | Block „Traces leaving a few pads“ | **Netzbelegung ist reiner Zufall (35 % pro Pad).** Ein 0603-Widerstand mit nur einem angebundenen Pad, ein SOT-23 mit einem angebundenen Pin von dreien – das ist die Regel statt die Ausnahme. Real sind fast alle Pins verbunden (wenige NC-Pins bei ICs), oft über die Kupferfläche statt über eine Bahn. |
| P7 | M | Block „Copper pours“ | **Kupferfläche hat keine Isolationsabstände.** Freistellung nur um das Keep-out-Rechteck. Bahnen anderer Netze liegen *in* der Fläche (verschwinden optisch darin, statt einen sichtbaren Isolationsspalt zu haben), Nachbarpads und Vias haben keinen Abstand. Pads, die zum Flächennetz gehören, haben keine Wärmefallen (Thermal-Relief-Stege). |
| P8 | M | Via-Schleife | **Vias sind unverbunden und zufällig.** Sie hängen an keiner Bahn, können auf Leiterbahnen anderer Netze, aufeinander, auf Nachbarpads oder unter Silkscreen liegen. Real: Vias am Ende eines Fanouts, als Stitching-Raster in Flächen, im Thermalpad. |
| P9 | M | Pad-Abgänge | **Abgänge berücksichtigen den Bauteilkörper nicht.** Die Diagonale kann durchs Keep-out laufen (unter dem Body). Bei `MISSING` sieht man dann Kupfer, das zwischen den eigenen Pads hindurch zu einem anderen Pad führt. |
| P10 | M | `_feature()` | **Bahnbreite ohne mm-Bezug und ohne Netzklasse.** Breiten und Via-Radien sind Anteile der *kleinsten* Pad-Abmessung, in Pixel geklemmt (Bahn 1,5–10 px × ss). Das passt pro Package grob, kennt aber weder `roi.mm_per_px` (erreicht den Renderer nicht) noch Signal/Power-Klassen oder Neck-down am Fine-Pitch-Pad; bei QFN bestimmt die schmale Pinbreite auch die Breite der freien Bahnen. (Fassung 1 behauptete feste Pixelbreiten – das galt nur für `HEAD`.) |
| P11 | M | Block „Free traces“ | **Freie Bahnen gehen immer von Rand zu Rand** und haben immer genau einen Knick. Real enden Bahnen auch an Vias im Bild, laufen parallel als Bus mit gleichem Abstand, haben mehrere Knicke. |
| P12 | L | `_board()`, `texture_dir`-Zweig | Mit `texture_dir` passt das Kupfer der Fototextur nicht zu den gezeichneten Pads (Pads liegen auf zufälligem Kupfer). Nicht falsch, aber nicht bauteilgerecht. Optional: Textur nur als Substrat nehmen und eigenes Kupfer darüber legen. |

### 2.2 Struktur / Skalierbarkeit

| # | Schwere | Problem |
|---|---|---|
| S1 | L | **Geometrie, Logik und Rendering sind vermischt.** `_traces_and_vias()` erzeugt Geometrie und rastert sie sofort in eine Maske. Es gibt kein Datenmodell (Netze, Bahnen, Vias als Objekte), also nichts, was man prüfen, testen, exportieren oder für eine ganze Platine wiederverwenden kann. |
| S2 | L | **Pads sind `dict`s ohne Pin-Nummer, Netz, Seite oder Austrittsrichtung** (`draw.py`). Das Thermalpad wird hinterher per Flächen-Heuristik erraten (`thermal_pad_index`, auch von der Blender-Szene `scene["thermal_pad"]` genutzt). Footprints sind `if/elif` nach String-Präfix statt einer Bibliothek. Konsumenten der Pad-Dicts: `_pads_and_solder`, `solder_defect_layout`, `_leads`, `_draw_solder_bridge`, `body_geometry`, `_render_board` (Blender-Szene), `draw.py` (`flat`-Stil), `tools/profile_editor/preview_3d.py`, Tests. |
| S3 | L | **Pixelkoordinaten im Bildraum des Render-Fensters.** Profile definieren `geometry_ranges` in Pixeln bei einem bestimmten `roi.mm_per_px`; der Renderer bekommt `mm_per_px` gar nicht. Eine Gesamtplatine braucht mm-Koordinaten im Platinensystem; das Fenster ist dann nur ein Ausschnitt. |
| S4 | L | **Im 2D-Pfad ein gemeinsamer RNG-Strom für alles** (`_render_photoreal_base`: Kamerapose, Board, Lot, Körper, Kameramodell aus einem `rng`). Jede Änderung an der Anzahl der Zufallszüge im Routing verschiebt alle folgenden Züge. Das 3D-Backend trennt bereits Kamerapose (`default_rng([seed, 2])`) und Kameramodell (`[seed, 1]`) vom Bake (`default_rng(seed)`). |
| S5 | L | **Keine Tests** für Leiterbahnen. Es gibt nichts, was „kein Kurzschluss“ oder „Thermalpad hat keine Bahn“ absichert. |
| S6 | L | **Nichts ist konfigurierbar.** Wahrscheinlichkeiten und Breiten sind Konstanten im Code; Profile/Run-Configs können nicht zwischen „dichte Consumer-Platine“ und „großzügige Industrieplatine“ wählen. |
| S7 | L | `photoreal.py` hat ~1200 Zeilen und mischt Substrat, Kupfer, Silkscreen, Lot, Bauteile, Rework, Ansicht und Kamera. |
| S8 | L | **Rendern läuft parallel in Threads** (`simple_sim.parallel.imap_ordered`, thread-lokales `_ss()`). Neue Module dürfen keinen veränderlichen Modulzustand haben (Caches nur `lru_cache` auf unveränderlichen Argumenten). |

---

## 3. Zielarchitektur

Grundidee: **erst ein Platinenmodell bauen (Geometrie + Netze), prüfen, dann rastern.**
Der Renderer schattiert nur noch fertige Lagenmasken. Dasselbe Modell trägt vom Render-Fenster
bis zur kompletten Platine.

```
Footprint-Bibliothek ─► Platzierung ─► Netzliste ─► Routing ─► Flächen ─► DRC ─► Rasterung ─► Shading (photoreal)
   (Pads, Pins,         (Prüfling +   (wer mit    (Bahnen,   (Pours,          (Masken je    (bestehender Code)
    Austrittsrichtung)   Nachbarn)     wem)        Vias)      Thermals)        Lage)
```

### 3.0 Voraussetzung: Maßstab an den Renderer durchreichen

Das Modell rechnet in mm. Der Renderer kennt heute nur Pixel. Minimaler Eingriff:

- `dataset.py` legt `config["render"]["mm_per_px"] = config["roi"]["mm_per_px"]` ab (analog zu
  `merge_profile_render`); `blender.py`, `preview/render.py`, `bake.prepare_job` und der
  Profil-Editor tun dasselbe (sie haben `mm_per_px` bereits in der Hand).
- Fehlt der Wert (alte Aufrufer, Tests), fällt der Renderer auf einen Default zurück
  (`DEFAULT_MM_PER_PX`, aus `run_0001.yaml`) – der Look bleibt dann skaliert wie heute, nur
  die mm-Angaben in `render_meta` wären relativ.
- Der Prüfling bleibt wie bisher in Pixeln aus dem Profil (`nominal`); die Footprint-Bibliothek
  wandelt nach mm. Die Umstellung der Profile selbst auf mm ist ein eigener, späterer Schritt
  (eigene Profilversion `@3`).

### 3.1 Paketstruktur (Vorschlag)

```
simple_sim/board/
  __init__.py          # build_window_board(...), build_full_board(...) – öffentliche Einstiege
  model.py             # Dataclasses, alles in mm, Platinenkoordinaten
  rules.py             # DesignRules + Presets
  footprints.py        # Footprint-Bibliothek (ersetzt _compute_pad_positions) + Adapter zu Pad-Dicts
  placement.py         # Prüfling + Nachbarn (zieht _neighbours um), später Gesamtplatzierung
  netlist.py           # Netzzuweisung nach Bauteilfamilie / Schaltungsmustern
  routing/
    base.py            # Router-Protokoll, RouteResult
    escape.py          # Fanout-Router (Phase 1)
    grid.py            # Raster-/A*-Router mit 45°-Zügen (Phase 2)
  zones.py             # Kupferflächen: Clearance, Thermal-Relief, Stitching-Vias
  occupancy.py         # mm-Belegungsraster (gemeinsam für Router, Zonen, DRC)
  drc.py               # Prüfungen (Tests, validate, optional pro Sample)
  raster.py            # Board → Lagenmasken (px) für ein Fenster
```

`photoreal.py` behält Substrat, Shading, Lot, Bauteilkörper, Ansicht und Kamera und bekommt die
Masken aus `raster.py`. `bake.py` profitiert automatisch. Silkscreen bleibt in Phase 1 in
`photoreal.py` (nur Prüfling); ab Phase 3 liefert das Modell Refdes und Umrisse pro Bauteil.

### 3.2 Datenmodell (`model.py`)

```python
Vec = tuple[float, float]   # mm, Platinenkoordinaten (x rechts, y unten, wie im Bild)

@dataclass(frozen=True, slots=True)
class Pad:
    pin: str                       # "1", "2", "EP" (exposed/thermal)
    center: Vec
    size: Vec
    rotation: float = 0.0
    shape: Literal["rect", "roundrect", "circle"] = "rect"
    escape_dir: Vec = (0.0, 0.0)   # Normalenvektor der Padkante, nach außen; (0,0) = kein Abgang
    role: Literal["signal", "power", "gnd", "thermal", "nc"] = "signal"
    net: str | None = None

@dataclass(slots=True)
class Part:
    ref: str                       # "R12", "U3"
    footprint: str
    family: str                    # resistor | capacitor | sot | soic | qfn ...
    pose: tuple[float, float, float]   # x, y (mm), Winkel (°)
    pads: list[Pad]
    body: Vec                      # Länge × Breite mm (Keep-out)
    is_target: bool = False        # der Prüfling im Fenster

@dataclass(slots=True)
class Track:
    net: str; layer: str; points: list[Vec]; width: float

@dataclass(slots=True)
class Via:
    net: str; center: Vec; drill: float; diameter: float
    layers: tuple[str, str] = ("top", "bottom")   # Lagenspanne (Phase 2: blind/buried)
    tented: bool = True

@dataclass(frozen=True, slots=True)
class ThermalRelief:
    gap: float; spoke_width: float; spokes: int = 4

@dataclass(slots=True)
class Zone:
    net: str; layer: str; outline: list[Vec]; clearance: float
    thermal: ThermalRelief | None = None

@dataclass(frozen=True, slots=True)
class Port:
    """Stelle, an der ein Netz das Fenster verlässt (nur Fenster-Modus)."""
    net: str; point: Vec; layer: str

@dataclass(slots=True)
class Board:
    outline: list[Vec]             # Render-Fenster oder echte Platinenkontur
    layers: tuple[str, ...]        # ("top",) | ("top", "bottom") | ("top", "inner1", ..., "bottom")
    parts: list[Part]; tracks: list[Track]; vias: list[Via]; zones: list[Zone]
    ports: list[Port]
    rules: DesignRules
```

Eigenschaften: reine Daten, keine OpenCV-Abhängigkeit, serialisierbar (`dataclasses.asdict` →
JSON unter `render_meta.board` oder als Debug-Datei), unabhängig von Supersampling und
`mm_per_px`. `Literal`-Typen statt freier Strings, `slots=True` (Python ≥ 3.10) für viele
kleine Objekte.

### 3.3 Design-Rules (`rules.py`)

```python
@dataclass(frozen=True, slots=True)
class DesignRules:
    track_width: Mapping[str, tuple[float, float]]  # Netzklasse -> (min, max) mm, z. B. signal 0.10-0.25, power 0.3-0.8
    clearance: float          # mm Kupfer–Kupfer verschiedener Netze
    neck_down: float          # mm Bahnbreite direkt am Fine-Pitch-Pad
    fanout_stub: tuple[float, float]  # Länge des geraden Austritts in mm
    via_drill: float; via_diameter: float
    zone_clearance: float; thermal_spoke: float; thermal_gap: float
    stitch_pitch: float | None
    angle_mode: Literal["45", "90", "any"] = "45"
```

Presets (`consumer_dense`, `industrial`, `hdi`) per Name, auswählbar in Run-Config/Profil.
Werte als Bereiche, aus denen pro Board gezogen wird → Variation bleibt erhalten.
Obergrenze pro Package: Bahnbreite ≤ Padbreite quer (ersetzt den heutigen `_feature`-Klemmbereich).

### 3.4 Footprint-Bibliothek (`footprints.py`)

- Ein Eintrag pro Footprint (`chip_2pad`, `sot23`, `soic_16`, `qfn_N`), Pins mit Nummer,
  Rolle und `escape_dir` (bei QFN je Seite, Thermalpad `role="thermal"`, `escape_dir=(0,0)`).
- Eingabe bleibt das `nominal` in Pixeln aus dem Profil (plus `mm_per_px` für die Umrechnung),
  damit Profile unverändert bleiben.
- **Adapter `pads_px(part, window) -> list[dict]`** liefert exakt die heutigen Pad-Dicts
  (`x, y, w, h` als `int`, gleiche Rundung wie `_compute_pad_positions`), damit Lot, Defekt-Layout,
  Leads, Blender-Szene, `flat`-Stil und Profil-Editor in Phase 1 unverändert bleiben. Ein
  Regressionstest vergleicht alt gegen neu über alle Footprints und viele `nominal`-Werte,
  bevor `_compute_pad_positions` entfernt wird.
- `thermal_pad_index()` wird überflüssig (Rolle steht am Pad); `scene["thermal_pad"]` für Blender
  wird aus der Rolle abgeleitet (`render_batch.py` liest den Index weiterhin).
- Neue Footprints = ein neuer Eintrag, kein neuer `elif`-Zweig an mehreren Stellen.

### 3.5 Netzliste (`netlist.py`)

Regeln pro Familie statt 35 %-Münzwurf:

| Familie | Netzzuweisung (Beispielregeln) |
|---|---|
| Widerstand / Kondensator | beide Pads verbunden; Kondensator häufig ein Pad an GND (Flächenanbindung), eines an Versorgung/Signal |
| SOT-23 | alle 3 Pins verbunden, meist ein Pin an GND |
| SOIC-16 | ~90 % der Pins verbunden, 1–2 NC; je ein GND- und VCC-Pin |
| QFN-N | ~85 % verbunden, Thermalpad = GND (Vias im Pad, ggf. Fläche) |
| Nachbarn | teilen gezielt Netze mit dem Prüfling: Abblockkondensator an VCC/GND des ICs, Pull-up an einem Signalpin, Reihenwiderstand in einer Signalleitung |

Netze, die das Fenster verlassen, enden an einem **Port** auf dem Fensterrand (oder an einem Via
→ Lagenwechsel). Im Gesamtplatinen-Modus gibt es keine Ports, sondern echte Gegenstellen.

### 3.6 Routing (`routing/`)

Gemeinsames Protokoll – austauschbar per Config:

```python
@dataclass(slots=True)
class RouteResult:
    tracks: list[Track]; vias: list[Via]
    unrouted: list[str]            # Netze, die nicht gingen -> Port/Via-Fallback oder weglassen

class Router(Protocol):
    def route(self, board: Board, occupancy: Occupancy, rng: np.random.Generator) -> RouteResult: ...
```

`Occupancy` (`occupancy.py`) ist ein Belegungsraster in mm-Auflösung (z. B. `clearance / 2`,
also ~0,05–0,1 mm) mit Netz-ID pro Zelle. Es ist bewusst **nicht** das supersampelte
Pixelraster: ein 1024-px-Fenster bei 4× Supersampling wären 16 M Zellen pro Check; in mm sind
es bei 10 × 10 mm und 0,1 mm Raster 10 k. Router, Zonen und DRC teilen dieses Raster.

**Phase 1 – `EscapeRouter` (Fanout, reicht für Fenster):**
1. Für jedes verbundene Pad: gerader Stummel entlang `escape_dir` (Länge aus `fanout_stub`),
   bei Fine-Pitch mit `neck_down`-Breite → behebt P3.
2. Danach 45°-Knick und Weiterführung zum Port/Via; Diagonale erst *außerhalb* der Pinreihe
   (Stummellänge ≥ halbe Padlänge + Clearance).
3. Thermalpad: kein Track, sondern Via-Raster im Pad (+ optional Fläche) → behebt P2.
4. Jede neue Bahn wird gegen die Belegung (Pads, Bodies, vorhandene Bahnen, inflated um
   `clearance + width/2`) geprüft; Konflikt → anderer Knickpunkt / Via / Netz auslassen → behebt
   P1, P4, P9.
5. „Freie“ Bahnen werden zu Durchgangsnetzen zwischen zwei Ports, die denselben Belegungscheck
   durchlaufen; optional als paralleler Bus mit gleichem Pitch → behebt P11.

**Phase 2 – `GridRouter` (für dichte Szenen und Gesamtplatine):**
- A* auf dem `Occupancy`-Raster mit 8 Richtungen, Kosten für Knicke und Diagonalen (ergibt
  45°-Routing), Lagenwechsel über Via-Kosten. Danach Pfadglättung (kollineare Punkte entfernen).
- Netzreihenfolge: kurze Netze zuerst, Power zuletzt (gehen in Flächen); optional
  Rip-up-and-Retry für unroutbare Netze.
- Ausreichend für einige zehn bis hundert Netze; für sehr große Boards austauschbar gegen
  einen externen Router (z. B. Export → Freerouting), weil die Schnittstelle gleich bleibt.

### 3.7 Kupferflächen (`zones.py`)

- `Zone` im Modell ist nur Umriss + Netz + Parameter. Die tatsächliche Kupferfläche entsteht
  erst in `raster.py`: Umriss **minus** (alle Kupferobjekte fremder Netze, aufgeblasen um
  `zone_clearance`) → sichtbarer Isolationsspalt um Bahnen und Pads (behebt P7).
- Pads des Flächennetzes: Thermal-Relief (Spalt + 2–4 Stege) oder Vollanbindung, je nach Preset.
- Stitching-Vias im Raster `stitch_pitch` innerhalb der Fläche; Inseln ohne Anbindung entfernen
  (Zusammenhangskomponenten auf dem `Occupancy`-Raster).
- Polygon-Arithmetik: OpenCV-Dilate/Subtrahieren auf Masken – keine neue Abhängigkeit.
  Shapely nur, falls später exakte Polygone/Export nötig sind.

### 3.8 DRC (`drc.py`)

Prüfungen auf dem `Occupancy`-Raster (mm, feste Auflösung – damit unabhängig von Supersampling
und `mm_per_px`, aber nicht „ohne Raster“):

- keine Überlappung von Kupfer verschiedener Netze auf derselben Lage (inkl. `clearance`)
- jede Bahn endet an Pad, Via, Port oder anderer Bahn desselben Netzes
- kein Track auf `thermal`/`nc`-Pads, kein Via in Signalpads (außer erlaubtem via-in-pad)
- nichts im Keep-out des Prüflings außer dessen eigenen Anbindungen
- Bahnbreite innerhalb der Netzklasse

Ergebnis unter `render_meta.board_drc: {violations: N, kinds: {...}}` in `meta.jsonl`
(`MetaRow` hat feste Felder; `render_meta` ist der vorgesehene Platz für Backend-Details).
`validate` kann eine Warnung ausgeben, wenn der Anteil Samples mit Verletzungen über einer
Schwelle liegt.

### 3.9 Rasterung (`raster.py`)

- `rasterize(board, window, mm_per_px, ss) -> {"copper_top": mask, "copper_inner": mask, "drill": mask, ...}`
  mit `window` = Fenster in mm (Mitte, Größe). Masken sind `uint8` in supersampelten Pixeln.
- `photoreal.py` blendet `copper_top` wie heute mit `_COPPER_UNDER_MASK` ein – der Look bleibt.
  Vias werden weiter wie heute gezeichnet (Ring, Loch, Mask-Dome), aber an Modellpositionen.
- Das Render-Fenster ist das heutige `shape` von `_board()` (ROI × Canvas-Faktor), nicht das
  ROI. Nachbardichte `crowd` wird zur Fläche des Fensters.
- Im Gesamtplatinen-Modus wird dasselbe Board in viele Fenster geschnitten (ein Prüfling pro
  Fenster), oder als Ganzes für `board_detection`.

### 3.10 Reproduzierbarkeit (behebt S4)

Konvention aus `bake.py` übernehmen: ein Strom pro Stufe, über einen festen Schlüssel aus dem
Sample-Seed abgeleitet:

```python
STREAM_PLACE, STREAM_NET, STREAM_ROUTE, STREAM_ZONE = 10, 11, 12, 13   # 1 und 2 sind im 3D-Backend vergeben
rng_route = np.random.default_rng([seed, STREAM_ROUTE])
```

Feste Schlüssel sind robuster als `SeedSequence.spawn()`: eine neue Stufe bekommt einen neuen
Schlüssel, bestehende Ströme bleiben identisch. Der Board-Teil zieht dann nicht mehr aus dem
Haupt-`rng`; Lot, Körper und Kamera im 2D-Pfad bleiben in Phase 1 auf dem Haupt-`rng` (deren
Trennung ist ein eigener Schritt und ändert ebenfalls alle Bilder).

**Wichtig:** die Stromtrennung ist Teil des neuen Pfads und hängt an derselben Option wie der
Router. Unter `routing: legacy` läuft der alte Code mit dem alten, gemeinsamen `rng` weiter –
nur so bleibt `legacy` bit-gleich. Der Umstieg auf `escape`/`grid` ändert alle Bilder eines
Seeds einmalig (auch Lot und Kamera, weil die Zugfolge des Haupt-`rng` sich verschiebt).

### 3.11 Konfiguration

```yaml
render:
  board:
    routing: escape          # legacy | none | escape | grid   (Default vorerst: legacy)
    design_rules: consumer_dense
    pour_probability: 0.6
    neighbour_density: 1.0
    drc: warn                # off | warn | strict (Sample neu würfeln, max. N Versuche)
```

- Validierung in `simple_sim/config.py` (`_validate_render`, beide Backends), Konstanten wie
  `RENDER_STYLES` (`BOARD_ROUTERS`, `DESIGN_RULE_PRESETS`).
- `bake.prepare_job` reicht `render.board` wie `mask_gloss`/`rework_artifacts` an `render_board` durch.
- GUI: Pipeline-Tab und Profile-Preview-Tab bekommen Router/Preset als Auswahl; Filter-Popup
  nutzt den Run-Config-Pfad bereits.
- `--extend` prüft heute nur Profil-ID/-Hash, nicht die Render-Optionen. Damit ein Datensatz
  mit `legacy` nicht versehentlich mit `escape` verlängert wird, `render.board` in den
  Manifest-Vergleich aufnehmen (oder generell `render` aus `config.yaml` des Datensatzes mit der
  aktuellen Config abgleichen und bei Abweichung abbrechen).

---

## 4. Umsetzungsplan

Entscheidung zuerst: **Phase 0 oder Phase 1**, nicht beides. Phase 0 ist ein Patch am alten
Code und wird in Phase 1 weggeworfen. Empfehlung: Phase 0 nur mit den Punkten 1–3 (ein Tag,
größter sichtbarer Effekt), falls Phase 1 nicht innerhalb weniger Wochen beginnt; sonst direkt
Phase 1.

### Phase 0 – Sofortkorrekturen im bestehenden Code

Ändert die Bilder bei gleichem Seed → hinter `render.board.routing: escape` schalten
(Phase-0-Code = erster `escape`-Stand), Default bleibt `legacy`.

1. Nachbarbauteile **vor** dem Routing platzieren (`_neighbours` vor `_traces_and_vias`),
   Nachbarpads in die Belegung aufnehmen (P5).
2. Freie Bahnen und Pad-Abgänge gegen eine Belegungsmaske prüfen (Pads + Keep-out +
   vorhandene Bahnen, dilatiert um Abstand); bei Kollision verwerfen (P1, P4, P9). Die Maske
   auf `ss = 1`-Auflösung oder gröber führen, nicht auf dem supersampelten Canvas.
3. Thermalpad (`thermal_pad_index`) von Abgängen ausnehmen, stattdessen 4–9 Vias im Pad (P2).
4. Abgang: gerader Stummel aus der Padkante (Richtung = Seite der Pinreihe), dann Knick (P3).
5. Bahnbreite nach Netzklasse: Pad-Abgang ≤ Padbreite quer, freie Bahnen unabhängig von der
   Pinbreite des Prüflings (P10).
6. Pad-Abgänge pro Bauteil statt pro Pad würfeln: Chip/SOT alle Pads, ICs ~85 % (P6).
7. Kupferfläche: Belegung (Bahnen, Nachbarpads) dilatiert aus der Fläche ausschneiden (P7).
8. Vias bevorzugt an Bahnenden und in Flächen statt frei verteilt (P8).

### Phase 1 – Modell + Fanout-Router

- 3.0 (`mm_per_px` durchreichen) zuerst, als eigener kleiner Commit.
- `simple_sim/board/` mit `model`, `rules`, `footprints` (+ Adapter), `netlist`, `occupancy`,
  `routing/escape`, `zones`, `drc`, `raster`.
- `_board()` ruft `build_window_board(...)` + `rasterize(...)`; `_traces_and_vias` bleibt als `legacy`.
- Getrennte RNG-Ströme für den Board-Teil (3.10), nur im neuen Pfad.
- Config-Validierung, `bake.prepare_job`, GUI-Auswahl, `--extend`-Abgleich (3.11).
- Tests (siehe 5).
- Doku: Abschnitt in `docs/guides/PHOTOREAL_RENDERING.md` (Tabelle „Copper under mask“
  anpassen), `docs/architekture/simple_sim-Overview.md` (neues Paket), Eintrag in `CHANGELOG.md`.
- Erst nach A/B-Vergleich (Arena, `tools/sim_to_real.py` gegen echte ROIs): Default auf `escape`.

### Phase 2 – Raster-Router, mehrlagig

- `routing/grid.py` (A*, 45°, Via-Lagenwechsel), Netzklassen, Busse.
- Innenlagen nur als „Durchscheinen“ relevant (schwach sichtbar unter dem Lötstopplack)
  → eigener, schwächerer Blend-Faktor (`copper_inner`).
- Profile/Presets für Design-Rules; GUI-Option im Pipeline-Tab.

### Phase 3 – Komplette Platine

- `placement.py`: Bauteile nach Regeln platzieren (Gruppen: IC + Abblock-Cs, Spannungsregler,
  Steckerbereich), Rasterabstände, Ausrichtung 0/90°.
- `netlist.py`: Schaltungsvorlagen (Decoupling, Pull-up, Spannungsteiler, Bus) statt rein zufällig.
- Routing der ganzen Platine, Flächen pro Lage, Stitching; Silkscreen aus dem Modell.
- Defekte pro Bauteil auf der Platine (mehrere Prüflinge), Labels als Bounding-Boxen für
  `board_detection`; ROI-Datensätze werden durch Ausschneiden aus Gesamtplatinen erzeugt.
- Profile in mm (`@3`), Footprint-Bibliothek wird einzige Quelle für Pad-Geometrie.
- Optional: Export des `Board` nach KiCad/Gerber zur Sichtkontrolle in echten Tools.

---

## 5. Tests

| Test | Prüft |
|---|---|
| `test_footprints_match_legacy` | `pads_px()` liefert dieselben Pad-Dicts wie `_compute_pad_positions` (alle Footprints, viele `nominal`-Werte inkl. ungerader Größen) |
| `test_window_board_drc_clean` | für alle Footprints × Defektklassen × N Seeds: `drc(board)` ohne Verletzungen |
| `test_thermal_pad_no_track` | QFN: kein Track berührt das Thermalpad, Vias im Pad vorhanden |
| `test_no_copper_between_chip_pads` | Chip: kein fremdes Netz im Keep-out zwischen den Pads |
| `test_target_pads_connected` | Chip/SOT: alle Pads haben Track oder Flächenanbindung |
| `test_neighbour_pads_clear` | kein fremdes Kupfer unter Nachbarpads/-körpern |
| `test_routing_isolated_rng` | `escape` vs. `grid` bei gleichem Seed ⇒ Nachbarn, Netzliste identisch (eigene Ströme) |
| `test_legacy_routing_unchanged` | `routing: legacy` erzeugt pixelgleich dasselbe Bild wie der alte Pfad – Vergleich **im selben Prozess** gegen die unveränderte Funktion (ein gespeicherter Hash wäre von der OpenCV-Version abhängig: `LINE_AA`, `resize`) |
| `test_render_golden` (nachgetragen) | Review-Befund: `legacy` war *nicht* pixelgleich zu v1.0.3 (Realismus-Änderungen am selben Pfad, zusätzliche RNG-Ziehungen). Seitdem `RENDERER_VERSION` im Manifest (`--extend` verweigert Mischungen) und gespeicherte md5-Hashes der 2D-Ausgabe; der Test wird übersprungen, wenn OpenCV/NumPy nicht die beim Erzeugen verwendeten Versionen sind |
| `test_board_meta_roundtrip` | `Board` → JSON → `Board` verlustfrei |
| `test_parallel_render_identical` | bestehender Test (`test_parallel_render.py`) deckt den neuen Pfad mit ab (S8) |

Modell-Tests laufen ohne Pixel-Rendern auf `Board` + `Occupancy` (mm-Raster) und sind damit schnell.

---

## 6. Risiken

- **Datensatz-Kompatibilität:** jede Routing-Änderung verändert Bilder bei gleichem Seed.
  Lösung: neue Option, `legacy` bleibt Default, bis A/B-Training zeigt, dass die neue
  Generierung nicht schlechter ist (Arena-Vergleich, `sim_to_real`). `--extend` muss die
  Option mitprüfen (3.11).
- **Laufzeit:** Belegungschecks und Flächen-Subtraktion kosten Zeit. Auf dem mm-Raster billig;
  nur die finale Rasterung läuft in Pixeln. Die Blend-Optimierung (nur Bounding-Box der Maske)
  bleibt nutzbar. Messen mit `render-debug-previews` und dem Profile-Preview-Tab (img/s).
- **Thread-Sicherheit:** Rendern läuft parallel; kein veränderlicher Modulzustand in `board/`.
- **Überrealismus ist nicht das Ziel:** das Modell soll Defekte am Prüfling erkennen. Kupfer
  muss plausibel und vielfältig sein, keine perfekten Layouts. Zufallsbereiche in den Presets
  breit halten; `drc: strict` nur für Tests und Debug, nicht als Standard.

---

## 7. Umsetzung Phase 1 – Abweichungen vom Plan

- **Seed der Board-Ströme:** `_board()` kennt den Sample-Seed nicht (es bekommt nur ein `rng`).
  Darum zieht der neue Pfad genau *einen* Wert aus dem Haupt-`rng` (`board_seed`) und leitet
  daraus die Stufen-Ströme ab (`default_rng([board_seed, STREAM_*])`). Die Trennung der Stufen
  untereinander ist damit wie geplant; Lot, Körper und Kamera bleiben am Haupt-`rng`.
- **Nachbarn** werden im neuen Pfad aus dem Platzierungsstrom gezogen (vor dem Routing), im
  `legacy`-Pfad wie bisher aus dem Haupt-`rng` (nach dem Kupfer). Die Platzierungsfunktion ist
  dieselbe (`placement.neighbour_layout`, verschoben, nicht kopiert).
- **Belegung:** Keep-out = Pads + Body jedes Bauteils ist gesperrt; der Stummel eines Pads darf
  die Sperre seines eigenen Bauteils berühren (`ignore_blocked`), alles Weitere nicht. Innerhalb
  der Pinreihe gilt als Abstand, was der Pitch hergibt (Stummel bleibt in der Padspalte).
- **DRC:** Pads werden nicht gegeneinander geprüft (Footprint-Geometrie ist gegeben, der Pitch
  darf unter der Clearance liegen) und der Fan-out eines Netzes ist im Clearance-Umfeld seiner
  eigenen Pads ausgenommen. Ergebnis pro Sample liegt unter `render_meta.board_drc` in
  `meta.jsonl` (`{ok, violations, kinds, routing, design_rules, drc}`, nur bei `drc: warn | strict`):
  der 2D-Renderer gibt weiterhin nur das Bild zurück, `build_window_board` meldet den Report an
  einen Kontext-Sammler (`board.collect_drc`, pro Thread/Prozess), `opencv2d.render_record_with_meta`
  liefert `(Bild, Details)`, `generation.common.render_2d_records` gibt die Details zurück und
  `add_render_meta` trägt sie vor dem Schreiben in die Zeilen ein. Blender-Backend: noch offen.
- **`routing: none`** zusätzlich: kein Kupfer, Rest wie `legacy` (für Ablationen).
- **`--extend`-Abgleich** von `render.board` (3.11) ist noch nicht umgesetzt.
- **GUI-Auswahl** (Pipeline-/Profile-Preview-Tab): Router, DRC-Modus und Regel-Preset (und
  `render.look`) sind als Überschreibung der Run-Config wählbar (Standard „from config“); der
  Pipeline-Tab schreibt dafür eine temporäre Config unter `outputs/live`
  (`gui.utils.pipeline_helpers.apply_render_overrides`).
- Nicht umgesetzt (Phase 2/3): `GridRouter`, Innenlagen, Gesamtplatine, Profile in mm.
