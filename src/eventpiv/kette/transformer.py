#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
transformer.py -- BAUSTEIN [6b]: der Paar-Encoder und die TOKEN-DEFINITION.

  baue_tokens        aus zwei Knoten die normierten Tokens EINES Paares
  PaarTransformer    2n+1 Tokens -> ein Logit
  save_/load_checkpoint

WARUM DIE TOKENS HIER LIEGEN: dieselbe Begruendung wie bei
model.build_features. Die Normierungsvorschrift muss im Training und im
Einsatz Zeichen fuer Zeichen dieselbe sein; zwei Umsetzungen waeren nur
durch Sorgfalt verbunden.

TOKENS SIND EVENTS. Ein Paar liefert 2*n Event-Tokens plus ein CLS-Token.
Das Segment-Token ist ein ETIKETT, das auf jedes Token seines Tracklets
addiert wird -- es fasst nichts zusammen und belegt keine Position.

KEIN POSITIONAL ENCODING. In der Sprachverarbeitung traegt ein Token nur
die Wortidentitaet, und die Reihenfolge muss addiert werden. Hier traegt
jedes Token seine Koordinaten SELBST -- (t, x, y) IST die Position. Ein
Tracklet ist eine Punktwolke, keine Sequenz: die Reihenfolge, in der die
32 Events gezogen wurden, darf das Ergebnis nicht aendern. Attention ist
permutationsinvariant, und das ist hier genau die richtige Eigenschaft.
Ein Sequenzindex waere schaedlich, weil er die Ausgabe an die Ziehung
haengte. Das ist als Test festgehalten, nicht nur als Absicht.

NORMIERUNG JE PAAR, Ursprung auf dem FRUEHEREN Knoten:
  t  relativ zu dessen t_min, geteilt durch kandidaten.alpha
  xy relativ zu dessen Zentroid, geteilt durch kandidaten.r_max
Absolute Koordinaten sind der Fehler, an dem raw6 in Stufe 1 scheiterte.
Die beiden Nenner unterscheiden sich um vier Groessenordnungen -- ohne
sie dominierte die Zeitachse jede Distanz im Embedding.

DIE ORIENTIERUNG IST TRAGEND, NICHT BELIEBIG. Das Netz ist NICHT
symmetrisch unter dem Tausch der beiden Knoten: Segment-Etikett und
Ursprung haengen daran, wer Nummer 1 ist. c_ij = c_ji gilt trotzdem --
weil kandidatenpaare jede Kante genau einmal liefert, mit dem frueheren
t_min als Nummer 1. Die kanonische Ordnung ersetzt die Symmetrie; ohne
sie waere die Kantenkostenfunktion nicht wohldefiniert.
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from eventpiv.kern.runconfig import POOLINGS


def baue_tokens(ev, maske, i, j, t_min, x_c, y_c, *, alpha, r_max):
    """Normierte Tokens eines Paar-STAPELS. Reine Funktion.

    ev     (m, n, 3) alle Knoten des Runs; Spalte 0 ist die Zeit RELATIV
           zu t_min des eigenen Knotens (so liegt sie im Artefakt)
    maske  (m, n) bool, True = echtes Event
    i, j   (B,) Indizes des Stapels, i = frueherer Knoten
    t_min, x_c, y_c  (m,) aus der Knotentabelle

    Rueckgabe: (tokens (B, 2n, 3), tok_maske (B, 2n)).
    Die ersten n Spalten gehoeren zu i, die zweiten n zu j.

    Der ZEITVERSATZ ist die eigentliche Information des Paares: fuer i ist
    er null (der Ursprung liegt auf seinem t_min), fuer j ist er
    t_min[j] - t_min[i] >= 0. Ohne ihn saehen zwei Tracklets aus
    benachbarten Fenstern genauso aus wie zwei aus demselben.

    Fuell-Token werden am Ende auf null gesetzt. Fuer die Attention ist das
    gleichgueltig (die Maske nimmt sie aus dem Softmax), aber ein
    NaN oder eine grosse Zahl dort waere eine stille Falle bei jeder
    spaeteren Aenderung -- etwa beim Wechsel auf pooling="mean".
    """
    n = ev.shape[1]
    ev_i, ev_j = ev[i], ev[j]                        # (B, n, 3)
    m_i, m_j = maske[i], maske[j]

    dt = (t_min[j] - t_min[i]).unsqueeze(1)          # (B, 1), >= 0
    t = torch.cat([ev_i[:, :, 0], ev_j[:, :, 0] + dt], dim=1) / alpha

    ox, oy = x_c[i].unsqueeze(1), y_c[i].unsqueeze(1)
    x = torch.cat([ev_i[:, :, 1], ev_j[:, :, 1]], dim=1)
    y = torch.cat([ev_i[:, :, 2], ev_j[:, :, 2]], dim=1)
    x = (x - ox) / r_max
    y = (y - oy) / r_max

    tok_maske = torch.cat([m_i, m_j], dim=1)         # (B, 2n)
    tokens = torch.stack([t, x, y], dim=2)           # (B, 2n, 3)
    return tokens * tok_maske.unsqueeze(2), tok_maske


class PaarTransformer(nn.Module):
    """2n+1 Tokens -> ein LOGIT (kein Sigmoid).

    Wie EdgeMLP gibt das Netz Log-Odds aus, die UNVERAENDERT als
    Kantenkosten an RAMA gehen. Damit gilt die Begruendung der Schwelle 0
    aus metrics.py eine Ebene hoeher unveraendert.

    norm_first=True (Pre-LN): Post-LN braucht ein Aufwaermen der Lernrate,
    sonst divergiert der Verlust in den ersten Schritten. Pre-LN kommt
    ohne aus, und ein Aufwaermplan waere ein Parameter mehr, den niemand
    gewaehlt haette.

    DIE PADDING-MASKE IST PFLICHT, nicht Kosmetik: ohne sie zoegen
    Fuell-Token echte Aufmerksamkeit, und ein Knoten mit 15 Events
    bekaeme systematisch andere Ausgaben als einer mit 32. Das CLS-Token
    ist NIE maskiert -- eine vollstaendig maskierte Zeile ergaebe einen
    Softmax ueber lauter -inf und damit NaN. Bei min_events >= 2 kann das
    ohnehin nicht auftreten, aber die Bedingung steht hier und nicht im
    Kopf des Lesers.
    """

    def __init__(self, d_model=64, n_layers=2, n_heads=4, d_ff=256,
                 dropout=0.0, pooling="cls"):
        super().__init__()
        if pooling not in POOLINGS:
            raise ValueError(f"pooling muss in {POOLINGS} liegen")
        self.pooling = pooling
        self.d_model = d_model

        # Drei Zahlen sind eine sehr flache Darstellung; zwei Schichten
        # geben dem Netz die Moeglichkeit, daraus etwas zu bauen, bevor
        # die Attention beginnt.
        self.einbettung = nn.Sequential(
            nn.Linear(3, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        # Etikett je Tracklet, ADDIERT. Ohne es waere die Eingabe eine
        # ununterscheidbare Wolke aus 2n Punkten und die Frage "gehoeren
        # sie zusammen" nicht einmal formulierbar.
        self.segment = nn.Parameter(torch.empty(2, d_model))
        self.cls = nn.Parameter(torch.empty(1, 1, d_model))
        nn.init.normal_(self.segment, std=0.02)
        nn.init.normal_(self.cls, std=0.02)

        schicht = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, activation="gelu", batch_first=True,
            norm_first=True)
        # enable_nested_tensor abschalten: die Verdichtung gepolsterter
        # Stapel ist mit norm_first=True nicht implementiert, torch schaltet
        # sie ohnehin ab und warnt bei JEDEM Aufbau. Die Padding-Maske wirkt
        # davon unabhaengig. Explizit statt still, damit die Warnung nicht
        # zur Gewohnheit wird.
        self.encoder = nn.TransformerEncoder(schicht, n_layers,
                                             enable_nested_tensor=False)
        self.kopf = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))

    def forward(self, tokens, tok_maske):
        """tokens (B, 2n, 3), tok_maske (B, 2n) bool. Rueckgabe: (B,) Logits."""
        b, l, _ = tokens.shape
        n = l // 2
        h = self.einbettung(tokens)
        # Erste Haelfte Tracklet 1, zweite Haelfte Tracklet 2.
        h = h + torch.cat([self.segment[0].expand(b, n, -1),
                           self.segment[1].expand(b, n, -1)], dim=1)
        h = torch.cat([self.cls.expand(b, -1, -1), h], dim=1)

        # src_key_padding_mask: True heisst IGNORIEREN -- umgekehrt zu
        # unserer Maske. Das CLS-Token bleibt immer sichtbar.
        pad = torch.cat([torch.zeros(b, 1, dtype=torch.bool,
                                     device=tokens.device),
                         ~tok_maske], dim=1)
        h = self.encoder(h, src_key_padding_mask=pad)

        if self.pooling == "cls":
            z = h[:, 0]
        else:
            # Mittel NUR ueber echte Tokens. Ein Mittel ueber alle
            # verduennte kurze Tracklets systematisch.
            w = tok_maske.unsqueeze(2).to(h.dtype)
            z = (h[:, 1:] * w).sum(1) / w.sum(1).clamp(min=1.0)
        return self.kopf(z).squeeze(1)


def save_checkpoint(path, model, tf, kd, **meta):
    """Netz + NORMIERUNG + Modus in EINER Datei.

    alpha und r_max spielen hier die Rolle, die mu/sig in model.py
    spielen: das Netz hat auf DIESER Skala gelernt. Ein Checkpoint, der
    mit r_max = 15 trainiert wurde, auf Tokens mit r_max = 20 angewandt,
    sieht eine andere Welt -- und nichts wuerde es melden.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": {k: v.cpu()
                               for k, v in model.state_dict().items()},
                "arch": {"d_model": tf.d_model, "n_layers": tf.n_layers,
                         "n_heads": tf.n_heads, "d_ff": tf.d_ff,
                         "dropout": tf.dropout, "pooling": tf.pooling},
                "norm": {"alpha": float(kd.alpha), "r_max": float(kd.r_max),
                         "n_events": int(kd.n_events)},
                "meta": meta}, path)
    return path


def load_checkpoint(path, device="cpu"):
    """Rueckgabe: (model, norm, ckpt). norm ist ein dict mit alpha, r_max,
    n_events -- direkt an baue_tokens weiterzureichen.

    weights_only=False wie in model.py: die .pt ist unsere eigene Datei.
    """
    ckpt = torch.load(Path(path), map_location=device, weights_only=False)
    model = PaarTransformer(**ckpt["arch"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt["norm"], ckpt