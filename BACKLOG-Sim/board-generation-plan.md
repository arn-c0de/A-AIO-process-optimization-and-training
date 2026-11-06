# Plan: Board-Generierung mit mehreren Bauteilen

Status: Planung · Stand: 2026-11-06

Ziel: Statt einzelner, zentrierter ROI-Bilder ganze synthetische Boards mit mehreren,
logisch angeordneten Bauteilen erzeugen – inklusive exakter Labels pro Bauteil.
Grundlage ist der bestehende OpenCV-2D-Renderer; er wird wiederverwendet, nicht ersetzt.

Leitlinien: **YAGNI** (nur bauen, was die jeweilige Stufe braucht), **DRY** (`render_roi`
und Defekt-Sampling wiederverwenden), **KISS** (einfaches Packing statt Optimierer).

---

## 1) Ausgangslage

| Thema | Ist-Zustand | Konsequenz |
|---|---|---|
| Rendering | `render_roi()` (`Simple-Sim/simple_sim/generators/opencv2d/render.py`) rendert genau ein Bauteil mittig im ROI | Board = Komposition vieler ROIs |
| Pad-Geometrie | `_compute_pad_positions()` (`draw.py`) rechnet fest vom Bildmittelpunkt | Kein Umbau nötig, wenn ROIs eingesetzt statt direkt gezeichnet werden |
| Inferenz | `TwoStageClassifier.predict_array()` klassifiziert ein ROI | Board-Inferenz = ROIs ausschneiden + iterieren |
| Maßstab | Profile nutzen unterschiedliche `mm_per_px` (0,01 / 0,012 / 0,03) | Pro Profil Skalierung auf gemeinsamen Board-Maßstab |

**Maßstabs-Inkonsistenz:** QFN32 (5 mm ≙ ~167 px @ 0,03) und SOIC16 (9,9 mm ≙ ~990 px @ 0,01)
sind maßstabsgetreu, `chip_0603_resistor@1` nicht (60 px @ 0,01 = 0,6 mm statt 1,6 mm).

**Entscheidung:** Skalierungsfaktor pro Profil beim Einsetzen, berechnet aus
`nominal_dims_mm.length` / gerenderter Body-Länge in px. Bestehende Profile und Datensätze
bleiben gültig (kein Profil-Hash-Bruch, keine `@2`-Versionen nötig).

---

## 2) Architektur

```
Layout-Quelle                         (Stufe 1: Raster · Stufe 2: Block-Templates · Stufe 3: Pick&Place-CSV)
        ↓   Liste von Placements (refdes, profile_id, x_mm, y_mm, rot_deg)
Defekt-Sampling pro Placement         (bestehende Logik aus simple_sim/generation/common.py / defects.py)
        ↓
ROI-Rendering pro Placement           (bestehendes render_roi, unverändert)
        ↓
Compose: skalieren, drehen, maskiert einsetzen
        ↓
Board-Extras                          (Fiducials, Bestückungsdruck – ab Stufe 2)
        ↓
Ausgabe: board.png + board.json  +  ausgeschnittene ROIs im bestehenden Dataset-Format
```

Neues Paket `Simple-Sim/simple_sim/board/`:

| Modul | Aufgabe |
|---|---|
| `layout.py` | Placement-Datenklasse + Layout-Quellen (Raster, später Blocks, CSV) |
| `compose.py` | ROIs rendern, skalieren, drehen, mit weicher Maske einsetzen |
| `export.py` | `board.json` schreiben, ROIs ausschneiden und als Dataset ablegen |

Einstieg: `Simple-Sim/scripts/generate_boards.py` (gleiche `--json`-Konventionen wie die
übrigen Skripte über `cli_utils`).

---

## 3) Stufe 1 – Compose + Raster-Layout (MVP)

**Umfang**
- `Placement`: `refdes`, `profile_id`, `x_mm`, `y_mm`, `rot_deg` (0/90/180/270).
- Raster-Layout: Bauteile zeilenweise mit festem Sperrabstand (`clearance_mm`).
- Compose:
  1. Board-Canvas in Substratfarbe (`board.size_mm`, `board.mm_per_px`).
  2. Pro Placement `render_roi()` mit Profil-Geometrie + gesampelten Defekt-Parametern.
  3. Auf Board-Maßstab skalieren, um `rot_deg` drehen.
  4. Maskiert einsetzen: Vordergrund = Abweichung von Substratfarbe, Kanten per Gauss weich
     → keine sichtbaren Kachelkanten (wichtig für Photoreal-Stil mit eigener Textur pro ROI).
- Defekte: pro Board 0–N Fehler (`defects_per_board: [0, 3]`), Rest OK.

**Akzeptanzkriterien**
- Deterministisch bei gleichem Seed.
- Keine Überlappung der Bounding-Boxen (Test).
- Ausgeschnittene ROIs laufen ohne Änderung durch `train`/`eval`/Arena.

---

## 4) Stufe 2 – Logisch strukturierte Anordnung (Block-Templates)

Statt Zufallsstreuung typische Schaltungsgruppen, definiert in YAML:

```yaml
board:
  size_mm: [40, 30]
  mm_per_px: 0.02
  clearance_mm: 0.5
  fiducials: true
  defects_per_board: [0, 3]
blocks:
  - type: ic_with_decoupling      # IC + 0603-Kondensatoren an den Pinseiten
    ic: qfn32_ic@1
    decoupling: chip_0603_resistor@1
    count: [2, 4]
  - type: row                     # Reihe, gleiche Rotation, festes Pitch
    profile: chip_0603_resistor@1
    count: [4, 8]
    pitch_mm: 2.0
```

**Platzierung**
- Block erzeugt intern seine Placements relativ zum Block-Ursprung
  (Abblock-Cs 1–2 mm vom IC-Rand, Reihen mit gleichem Pitch).
- Block-Bounding-Box + Sperrabstand → zufällige Position, bei Kollision verwerfen und neu
  ziehen (Rejection Sampling, max. Versuche konfigurierbar). Kein Packing-Optimierer.
- Fortlaufende Bezeichner pro Typ (R1…Rn, C1…Cn, U1…Un, Q1…Qn).

**Board-Extras (nur das Nötige)**
- Fiducials in drei Ecken – Voraussetzung für Registrierung laut `BACKLOG-Production/plan.md`.
- Bestückungsdruck (`cv2.putText`) neben jedem Bauteil.

Start mit nur **zwei** Block-Typen (`ic_with_decoupling`, `row`). Weitere (z. B. `cluster`
für SOT-23, Leiterbahnen als Kulisse) erst bei konkretem Bedarf.

---

## 5) Stufe 3 – CAD-getriebenes Layout (Pick&Place)

- Eingabe: Pick&Place-CSV (`refdes, x_mm, y_mm, rot_deg, package`).
- Mapping `package → profile_id` in YAML (z. B. `0603: chip_0603_resistor@1`).
- Ersetzt nur die Layout-Quelle; Compose und Export bleiben identisch.
- Ergebnis: synthetische Varianten der **echten** Designs mit beliebigen Defekt-Kombinationen.

Erst angehen, wenn ein reales Pick&Place-File vorliegt.

---

## 6) Ausgabeformat

```
datasets/boards/<run_id>/
  boards/<board_id>/
    board.png
    board.json
  images/ meta.jsonl labels.jsonl splits/   # ausgeschnittene ROIs, bestehendes Format
```

`board.json` (an Metadaten-Schema aus `BACKLOG-Production/plan.md` angelehnt):

```json
{
  "board_id": "000042",
  "design_id": "synthetic_blocks_v1",
  "mm_per_px": 0.02,
  "size_px": [2000, 1500],
  "placements": [
    {"refdes": "U1", "profile_id": "qfn32_ic@1", "x_mm": 12.0, "y_mm": 9.5, "rot_deg": 90,
     "roi_bbox_px": [500, 380, 220, 220], "label": "OK", "defect_params": {}}
  ]
}
```

**Splits:** nach `board_id` aufteilen (nicht nach ROI), damit ROIs eines Boards nicht in
train und test landen.

---

## 7) GUI – Board-Detection-Tab

- Board-Bild + `board.json` laden.
- Pro Placement ROI ausschneiden → `predict_array()`.
- Ergebnis über `gui/components/overlay_renderer.py` als farbige Boxen (OK / NOK / unsicher).
- Damit ist die Produktionskette Board → ROI → Klassifikation → Entscheidung einmal
  vollständig simuliert.

---

## 8) Reihenfolge und Aufwand

| # | Paket | Ergebnis | Aufwand (grob) |
|---|---|---|---|
| 1 | Stufe 1 + Export + Tests | Boards im Raster, ROIs trainierbar | 1–2 Tage |
| 2 | Stufe 2 (2 Block-Typen, Fiducials, Druck) | Logisch strukturierte Boards | 1–2 Tage |
| 3 | GUI-Overlay im Board-Detection-Tab | Board-Inferenz sichtbar | ~1 Tag |
| 4 | Stufe 3 (Pick&Place-Import) | Synthetik der echten Designs | ~1 Tag, sobald CSV vorliegt |

## 9) Offene Punkte / Risiken

- **Kontext am ROI-Rand:** Ausgeschnittene ROIs enthalten Nachbarbauteile. Realistischer,
  kann aber die Accuracy bestehender Modelle zunächst senken → in der Arena vergleichen.
- **Photoreal-Übergänge:** Maskiertes Einsetzen muss visuell geprüft werden
  (Sample-Galerie); ggf. Substrat-Textur einmal pro Board statt pro ROI erzeugen.
- **Blender-3D-Backend:** bewusst außen vor; erst nach stabiler 2D-Pipeline prüfen.
- **Bildgröße:** 40×30 mm @ 0,01 mm/px = 4000×3000 px – für die Generierung unkritisch,
  Speicherbedarf pro Run beachten.
