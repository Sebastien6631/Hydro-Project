from __future__ import annotations

from previ_r2d2.preprocessing.onegate.memorandum import flatten_flexibilite


def _tree_with_groupe(groupe_extra=None, racc_extra=None, centrale_extra=None):
    groupe = {"uuid": "g1", "nom_groupe": "G1", **(groupe_extra or {})}
    return [
        {
            "nom_amenagement": "Apas",
            "uuid": "amenagement-uuid",
            "centrales": [
                {
                    "nom_centrale": "apas",
                    "uuid": "centrale-uuid",
                    "adresse_lat": 43.1,
                    "adresse_lng": 0.9,
                    "adresse_alt": 288,
                    **(centrale_extra or {}),
                    "raccordements": [
                        {
                            "uuid": "racc-uuid",
                            "flexibilite_rte": "true",
                            "flexibilite_pool": "1",
                            "flexibilite_type": "aFRR",
                            "flex_strategy": "DEFAULT",
                            "groupes": [groupe],
                            **(racc_extra or {}),
                        }
                    ],
                }
            ],
        }
    ]


def test_flatten_flexibilite_adds_centrale_and_raccordement_extra_fields_when_present():
    tree = _tree_with_groupe(
        centrale_extra={"debit_reserve": 13, "debit_non_turbinable": 13.9},
        racc_extra={"facteur_debit": 1.0},
    )

    records = flatten_flexibilite(tree)

    assert records[0]["debit_reserve"] == 13
    assert records[0]["debit_non_turbinable"] == 13.9
    assert records[0]["facteur_debit"] == 1.0
    assert "periode_ete" not in records[0]


def test_flatten_flexibilite_splits_seasonal_debit_non_turbinable_haute_chute():
    tree = _tree_with_groupe(
        centrale_extra={"debit_non_turbinable": [0.150, 0.113]},
        racc_extra={"flex_strategy": "HAUTE_CHUTE"},
    )

    records = flatten_flexibilite(tree)

    assert "debit_non_turbinable" not in records[0]
    assert records[0]["q_non_turbinable_ete"] == 0.150
    assert records[0]["q_non_turbinable_hiver"] == 0.113
    assert records[0]["periode_ete"] == [[7, 1], [10, 31]]


def test_flatten_flexibilite_splits_seasonal_debit_non_turbinable_default_strategy():
    tree = _tree_with_groupe(
        centrale_extra={"debit_non_turbinable": [0.150, 0.113]},
        racc_extra={"flex_strategy": "DEFAULT"},
    )

    records = flatten_flexibilite(tree)

    assert records[0]["periode_ete"] == [[6, 1], [10, 31]]


def test_flatten_flexibilite_splits_comma_separated_seasonal_debit_non_turbinable():
    """OneGate envoie parfois debit_non_turbinable sans crochets, juste
    "valeur_ete,valeur_hiver" (ex. Melles) -- json.loads échoue dans ce cas,
    il faut aussi tenter un découpage par virgule comme `_split_stations`."""
    tree = _tree_with_groupe(
        centrale_extra={"debit_non_turbinable": "0.150,0.113"},
        racc_extra={"flex_strategy": "HAUTE_CHUTE"},
    )

    records = flatten_flexibilite(tree)

    assert "debit_non_turbinable" not in records[0]
    assert records[0]["q_non_turbinable_ete"] == 0.150
    assert records[0]["q_non_turbinable_hiver"] == 0.113
    assert records[0]["periode_ete"] == [[7, 1], [10, 31]]


def test_flatten_flexibilite_omits_periode_ete_when_single_value_non_turbinable():
    tree = _tree_with_groupe(
        centrale_extra={"debit_non_turbinable": 13.9},
        racc_extra={"flex_strategy": "HAUTE_CHUTE"},
    )

    records = flatten_flexibilite(tree)

    assert "periode_ete" not in records[0]


def test_flatten_flexibilite_omits_centrale_and_raccordement_extra_fields_when_absent():
    tree = _tree_with_groupe()

    records = flatten_flexibilite(tree)

    for key in ("debit_reserve", "debit_non_turbinable", "facteur_debit", "periode_ete"):
        assert key not in records[0]


def test_flatten_flexibilite_includes_only_present_centrale_extra_fields():
    tree = _tree_with_groupe(centrale_extra={"debit_reserve": 13})

    records = flatten_flexibilite(tree)

    assert records[0]["debit_reserve"] == 13
    assert "debit_non_turbinable" not in records[0]


def test_flatten_flexibilite_adds_groupe_fields_when_present():
    tree = _tree_with_groupe(groupe_extra={
        "priorite": 1,
        "debit_armement_turbine": 0.225,
        "debit_max": 0.720,
        "rendement": [33.846, 147.74, -102.49],
        "chute_disponible_polynome": [[79.432, -0.0534, -2.1407]],
        "chute_disponible_seuils": [[0, 500]],
    })

    records = flatten_flexibilite(tree)

    groupe = records[0]["groupes"][0]
    assert groupe == {
        "uuid": "g1",
        "nom_groupe": "G1",
        "priorite": 1,
        "debit_armement_turbine": 0.225,
        "debit_max": 0.720,
        "rendement": [33.846, 147.74, -102.49],
        "chute_disponible_polynome": [[79.432, -0.0534, -2.1407]],
        "chute_disponible_seuils": [[0, 500]],
    }


def test_flatten_flexibilite_omits_groupe_fields_when_absent():
    tree = _tree_with_groupe()

    records = flatten_flexibilite(tree)

    groupe = records[0]["groupes"][0]
    assert groupe == {"uuid": "g1", "nom_groupe": "G1"}


def test_flatten_flexibilite_decodes_string_encoded_numbers_and_lists():
    """OneGate stocke tout en texte (BDD) : nombres et listes arrivent en
    chaînes JSON-encodées, comme le "true" déjà géré par _is_true."""
    tree = _tree_with_groupe(groupe_extra={
        "priorite": "2",
        "debit_armement_turbine": 4,
        "debit_max": "20",
        "rendement": "[-5.01818242,19.98181038,-1.35935771,0.02911171,0.00002935]",
        "chute_disponible_polynome": "[7.23831976,-0.01151016,0.00003856,-0.00000007]",
    })

    records = flatten_flexibilite(tree)

    groupe = records[0]["groupes"][0]
    assert groupe["priorite"] == 2
    assert groupe["debit_armement_turbine"] == 4
    assert groupe["debit_max"] == 20
    assert groupe["rendement"] == [-5.01818242, 19.98181038, -1.35935771, 0.02911171, 0.00002935]
    assert groupe["chute_disponible_polynome"] == [7.23831976, -0.01151016, 0.00003856, -0.00000007]


def test_flatten_flexibilite_fixes_multi_segment_chute_disponible_missing_outer_brackets():
    """Cas réel Touzac : plusieurs segments séparés par virgule mais sans les
    crochets extérieurs -- json.loads échoue tel quel, il faut réessayer en
    enveloppant dans un tableau."""
    tree = _tree_with_groupe(groupe_extra={
        "chute_disponible_polynome": (
            "[3.2157950947, -0.0108591576, 0.0000069831],"
            "[2.7289060845, -0.0018425316, -0.0000018705, 0.0000000011]"
        ),
        "chute_disponible_seuils": "[0, 72],[72, 501]",
    })

    records = flatten_flexibilite(tree)

    groupe = records[0]["groupes"][0]
    assert groupe["chute_disponible_polynome"] == [
        [3.2157950947, -0.0108591576, 0.0000069831],
        [2.7289060845, -0.0018425316, -0.0000018705, 0.0000000011],
    ]
    assert groupe["chute_disponible_seuils"] == [[0, 72], [72, 501]]


def test_flatten_flexibilite_fixes_yaml_inf_in_chute_disponible_seuils():
    """Cas réel Melles : ".inf" est une syntaxe YAML, pas JSON. `Infinity`
    (que json.loads accepte nativement) n'est pas non plus du JSON standard
    (json.dumps l'écrirait tel quel, invalide pour un parseur strict) --
    remplacé par la chaîne "Inf" (JSON standard, sans feindre une valeur
    numérique finie)."""
    tree = _tree_with_groupe(groupe_extra={
        "chute_disponible_seuils": "[0, 0.15],[0.15, 1.65],[1.65, .inf]",
    })

    records = flatten_flexibilite(tree)

    groupe = records[0]["groupes"][0]
    assert groupe["chute_disponible_seuils"] == [[0, 0.15], [0.15, 1.65], [1.65, "Inf"]]


def test_flatten_flexibilite_fixes_negative_yaml_inf_in_chute_disponible_seuils():
    tree = _tree_with_groupe(groupe_extra={
        "chute_disponible_seuils": "[-.inf, 0],[0, 1.65]",
    })

    records = flatten_flexibilite(tree)

    groupe = records[0]["groupes"][0]
    assert groupe["chute_disponible_seuils"] == [["-Inf", 0], [0, 1.65]]


def test_flatten_flexibilite_treats_empty_string_as_absent():
    tree = _tree_with_groupe(groupe_extra={"chute_disponible_seuils": ""})

    records = flatten_flexibilite(tree)

    groupe = records[0]["groupes"][0]
    assert "chute_disponible_seuils" not in groupe


def test_flatten_flexibilite_decodes_string_encoded_centrale_and_raccordement_extra_fields():
    tree = _tree_with_groupe(
        centrale_extra={"debit_reserve": "13", "debit_non_turbinable": "13.9"},
        racc_extra={"facteur_debit": "1.0"},
    )

    records = flatten_flexibilite(tree)

    assert records[0]["debit_reserve"] == 13
    assert records[0]["debit_non_turbinable"] == 13.9
    assert records[0]["facteur_debit"] == 1.0


def test_flatten_flexibilite_splits_string_encoded_seasonal_debit_non_turbinable():
    tree = _tree_with_groupe(
        centrale_extra={"debit_non_turbinable": "[0.150, 0.113]"},
        racc_extra={"flex_strategy": "HAUTE_CHUTE"},
    )

    records = flatten_flexibilite(tree)

    assert "debit_non_turbinable" not in records[0]
    assert records[0]["q_non_turbinable_ete"] == 0.150
    assert records[0]["q_non_turbinable_hiver"] == 0.113
    assert records[0]["periode_ete"] == [[7, 1], [10, 31]]
