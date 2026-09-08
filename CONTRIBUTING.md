# Contribuer

Ce dépôt est un outil de recherche personnelle sur un device précis (voir
le disclaimer en tête du `README.md`). Les contributions les plus utiles à
ce stade :

## Ce qui aide le plus

1. **Des traces d'un autre SoC Hexagon** (v75, v79, autre SM8xxx) — même
   si elles contredisent les calibrations actuelles. Voir
   `REPRODUCIBILITY.md` § 4 pour la commande de capture exacte
   (`GGML_HEXAGON_PROFILE=3` + `GGML_HEXAGON_VERBOSE=1`). Ouvrir une issue
   avec le log (ou un lien vers le log si volumineux) et le modèle utilisé.
2. **Des runs de calibration supplémentaires** pour `predict_from_hf.py` —
   la calibration actuelle (2 points dense, 4 points MoE,
   `predictor_v1_outcomes.jsonl`) est explicitement documentée comme trop
   mince (voir README).
3. **Comparaison avec le parseur upstream officiel**
   (`llama.cpp/scripts/snapdragon/ggml-hexagon-profile.py`) sur une même
   trace — aucune comparaison systématique n'a été faite à ce jour.

## Soumettre une trace ou un correctif

- Une trace : ouvrir une issue, décrire le device/modèle/commande utilisée
  (idéalement en suivant le format `REPRODUCIBILITY.md` § 4), joindre le
  log ou un lien de téléchargement s'il dépasse quelques Mo.
- Un correctif de code : PR classique. Lancer `python -m unittest discover
  tests` avant (voir `.github/workflows/tests.yml`, tourne aussi en CI).
  Si le correctif touche `parse_hexagon_profile.py`, ajouter un cas dans
  `tests/test_parse_hexagon_profile.py` plutôt que de valider seulement à
  l'œil sur une trace réelle — c'est exactement le défaut qui a produit le
  bug corrigé le 2026-09-08 (mapping resté bloqué à 56% sans qu'aucun test
  ne le signale).

## Ce qui n'est probablement pas le bon projet pour vous

- Si vous cherchez un profileur générique multi-SoC : ce dépôt ne l'est
  pas et ne prétend pas l'être (voir disclaimer README).
- Si vous voulez juste faire tourner un LLM sur Hexagon HTP sans profiler :
  regardez directement `ggml-hexagon` (le fork upstream), pas cet outil
  d'analyse.
