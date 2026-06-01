"""
add_family_labels.py

Adds 'family' and 'family_common' columns to the amphibian merged_df.
Extracts genus from scientific_name, maps to family using a comprehensive lookup.

Run this BEFORE the classifier:
    python add_family_labels.py

It reads merged_df.csv and writes merged_df_with_family.csv.
"""

import pandas as pd
import numpy as np

# =============================================================================
# GENUS → FAMILY MAPPING  (covers iNaturalist's most-recorded amphibians)
# =============================================================================

GENUS_TO_FAMILY = {
    # ===== ANURA (Frogs & Toads) =====

    # Ranidae — True Frogs
    **{g: "Ranidae" for g in [
        "Rana", "Lithobates", "Glandirana", "Pelophylax", "Hylarana",
        "Amietia", "Amnirana", "Babina", "Clinotarsus", "Meristogenys",
        "Odorrana", "Papurana", "Ptychadena", "Sanguirana", "Staurois",
        "Sylvirana", "Abavorana", "Sumaterana",
    ]},

    # Hylidae — Tree Frogs
    **{g: "Hylidae" for g in [
        "Hyla", "Dryophytes", "Pseudacris", "Acris", "Smilisca",
        "Agalychnis", "Dendropsophus", "Boana", "Scinax", "Trachycephalus",
        "Osteopilus", "Isthmohyla", "Charadrahyla", "Ecnomiohyla",
        "Exerodonta", "Plectrohyla", "Ptychohyla", "Rheohyla",
        "Triprion", "Anotheca", "Bromeliohyla", "Duellmanohyla",
        "Megastomatohyla", "Quilticohyla", "Sarsinohyla", "Atlantihyla",
        "Tlalocohyla", "Diaglena", "Hyliola",
    ]},

    # Bufonidae — True Toads
    **{g: "Bufonidae" for g in [
        "Anaxyrus", "Bufo", "Rhinella", "Incilius", "Sclerophrys",
        "Duttaphrynus", "Bufotes", "Epidalea", "Atelopus",
        "Amazophrynella", "Rhaebo", "Phrynoidis", "Ansonia",
        "Capensibufo", "Schismaderma", "Nannophryne",
    ]},

    # Eleutherodactylidae — Rain Frogs / Coquís
    **{g: "Eleutherodactylidae" for g in [
        "Eleutherodactylus", "Diasporus", "Syrrhophus",
    ]},

    # Craugastoridae — Fleshbelly Frogs
    **{g: "Craugastoridae" for g in [
        "Craugastor", "Haddadus", "Pristimantis", "Strabomantis",
        "Yunganastes", "Noblella", "Oreobates", "Psychrophrynella",
    ]},

    # Microhylidae — Narrow-mouthed Frogs
    **{g: "Microhylidae" for g in [
        "Gastrophryne", "Hypopachus", "Microhyla", "Kaloula",
        "Chiasmocleis", "Dermatonotus", "Elachistocleis", "Hamptophryne",
        "Kalophrynus", "Micryletta", "Ramanella", "Uperodon",
        "Breviceps", "Cophixalus", "Oreophryne",
    ]},

    # Leptodactylidae — Thin-toed Frogs
    **{g: "Leptodactylidae" for g in [
        "Leptodactylus", "Physalaemus", "Engystomops", "Pleurodema",
        "Adenomera", "Lithodytes",
    ]},

    # Dendrobatidae — Poison Dart Frogs
    **{g: "Dendrobatidae" for g in [
        "Dendrobates", "Oophaga", "Ranitomeya", "Epipedobates",
        "Ameerega", "Phyllobates", "Andinobates", "Colostethus",
        "Silverstoneia", "Hyloxalus",
    ]},

    # Scaphiopodidae — North American Spadefoot Toads
    **{g: "Scaphiopodidae" for g in [
        "Scaphiopus", "Spea",
    ]},

    # Pipidae — Tongueless Frogs
    **{g: "Pipidae" for g in [
        "Xenopus", "Pipa", "Hymenochirus", "Silurana",
    ]},

    # Bombinatoridae — Fire-bellied Toads
    **{g: "Bombinatoridae" for g in [
        "Bombina",
    ]},

    # Alytidae — Midwife Toads & Painted Frogs
    **{g: "Alytidae" for g in [
        "Alytes", "Discoglossus",
    ]},

    # Centrolenidae — Glass Frogs
    **{g: "Centrolenidae" for g in [
        "Centrolene", "Hyalinobatrachium", "Espadarana", "Nymphargus",
        "Sachatamia", "Teratohyla", "Cochranella",
    ]},

    # Rhacophoridae — Old World Tree Frogs / Shrub Frogs
    **{g: "Rhacophoridae" for g in [
        "Rhacophorus", "Polypedates", "Kurixalus", "Zhangixalus",
        "Chiromantis", "Feihyla", "Taruga", "Theloderma",
    ]},

    # Myobatrachidae — Australian Ground Frogs
    **{g: "Myobatrachidae" for g in [
        "Crinia", "Geocrinia", "Myobatrachus", "Pseudophryne",
        "Paracrinia", "Uperoleia",
    ]},

    # Pelodryadidae — Australasian Tree Frogs
    **{g: "Pelodryadidae" for g in [
        "Litoria", "Ranoidea", "Nyctimystes",
    ]},

    # Limnodynastidae
    **{g: "Limnodynastidae" for g in [
        "Limnodynastes", "Platyplectrum", "Adelotus", "Heleioporus",
        "Neobatrachus",
    ]},

    # Pelodytidae — Parsley Frogs
    **{g: "Pelodytidae" for g in [
        "Pelodytes",
    ]},

    # Pelobatidae — European Spadefoot Toads
    **{g: "Pelobatidae" for g in [
        "Pelobates",
    ]},

    # Arthroleptidae — Squeakers
    **{g: "Arthroleptidae" for g in [
        "Arthroleptis", "Leptopelis",
    ]},

    # Hyperoliidae — African Reed Frogs
    **{g: "Hyperoliidae" for g in [
        "Hyperolius", "Afrixalus", "Heterixalus", "Kassina", "Semnodactylus",
    ]},

    # Mantellidae — Malagasy Frogs
    **{g: "Mantellidae" for g in [
        "Mantella", "Boophis", "Mantidactylus", "Guibemantis", "Spinomantis",
    ]},

    # Hemisotidae — Shovelnose Frogs
    **{g: "Hemisotidae" for g in [
        "Hemisus",
    ]},

    # Ceratophryidae — Horned Frogs
    **{g: "Ceratophryidae" for g in [
        "Ceratophrys", "Chacophrys", "Lepidobatrachus",
    ]},

    # Dicroglossidae — Fork-tongued Frogs
    **{g: "Dicroglossidae" for g in [
        "Fejervarya", "Limnonectes", "Hoplobatrachus", "Euphlyctis",
        "Minervarya", "Occidozyga",
    ]},

    # Pyxicephalidae — African Bullfrogs
    **{g: "Pyxicephalidae" for g in [
        "Pyxicephalus", "Tomopterna", "Strongylopus", "Cacosternum",
        "Natalobatrachus",
    ]},

    # Phyllomedusidae — Leaf Frogs
    **{g: "Phyllomedusidae" for g in [
        "Phyllomedusa", "Agalychnis", "Pithecopus", "Callimedusa",
        "Cruziohyla",
    ]},

    # Rhinophrynidae — Mexican Burrowing Toad
    **{g: "Rhinophrynidae" for g in [
        "Rhinophrynus",
    ]},

    # ===== CAUDATA (Salamanders & Newts) =====

    # Ambystomatidae — Mole Salamanders
    **{g: "Ambystomatidae" for g in [
        "Ambystoma",
    ]},

    # Salamandridae — Newts & Fire Salamanders
    **{g: "Salamandridae" for g in [
        "Notophthalmus", "Taricha", "Triturus", "Lissotriton",
        "Ichthyosaura", "Ommatotriton", "Salamandra", "Salamandrina",
        "Cynops", "Paramesotriton", "Tylototriton", "Neurergus",
        "Calotriton", "Chioglossa", "Euproctus", "Pleurodeles",
        "Lyciasalamandra",
    ]},

    # Plethodontidae — Lungless Salamanders
    **{g: "Plethodontidae" for g in [
        "Plethodon", "Desmognathus", "Eurycea", "Pseudotriton",
        "Gyrinophilus", "Hemidactylium", "Aneides", "Batrachoseps",
        "Ensatina", "Hydromantes", "Speleomantes", "Bolitoglossa",
        "Chiropterotriton", "Nototriton", "Oedipina", "Pseudoeurycea",
        "Thorius", "Stereochilus", "Urspelerpes",
    ]},

    # Cryptobranchidae — Giant Salamanders
    **{g: "Cryptobranchidae" for g in [
        "Cryptobranchus", "Andrias",
    ]},

    # Proteidae — Mudpuppies & Olm
    **{g: "Proteidae" for g in [
        "Necturus", "Proteus",
    ]},

    # Sirenidae — Sirens
    **{g: "Sirenidae" for g in [
        "Siren", "Pseudobranchus",
    ]},

    # Amphiumidae — Amphiumas (Congo Eels)
    **{g: "Amphiumidae" for g in [
        "Amphiuma",
    ]},

    # Rhyacotritonidae — Torrent Salamanders
    **{g: "Rhyacotritonidae" for g in [
        "Rhyacotriton",
    ]},

    # Dicamptodontidae — Pacific Giant Salamanders
    **{g: "Dicamptodontidae" for g in [
        "Dicamptodon",
    ]},

    # Hynobiidae — Asiatic Salamanders
    **{g: "Hynobiidae" for g in [
        "Hynobius", "Onychodactylus", "Batrachuperus",
    ]},
}


# =============================================================================
# FAMILY → COMMON NAME  (what users will see)
# =============================================================================

FAMILY_COMMON = {
    # Frogs & Toads
    "Ranidae":               "True Frogs (Bullfrogs, Leopard Frogs)",
    "Hylidae":               "Tree Frogs (Spring Peepers, Gray Treefrogs)",
    "Bufonidae":             "True Toads (American Toad, Cane Toad)",
    "Eleutherodactylidae":   "Rain Frogs & Coquís",
    "Craugastoridae":        "Robber Frogs",
    "Microhylidae":          "Narrow-mouthed Frogs",
    "Leptodactylidae":       "Thin-toed Frogs & Whistling Frogs",
    "Dendrobatidae":         "Poison Dart Frogs",
    "Scaphiopodidae":        "Spadefoot Toads",
    "Pipidae":               "Tongueless Frogs (African Clawed Frog)",
    "Bombinatoridae":        "Fire-bellied Toads",
    "Alytidae":              "Midwife Toads & Painted Frogs",
    "Centrolenidae":         "Glass Frogs",
    "Rhacophoridae":         "Shrub Frogs & Flying Frogs",
    "Myobatrachidae":        "Australian Ground Frogs",
    "Pelodryadidae":         "Australasian Tree Frogs",
    "Limnodynastidae":       "Australian Swamp Frogs",
    "Pelodytidae":           "Parsley Frogs",
    "Pelobatidae":           "European Spadefoot Toads",
    "Arthroleptidae":        "Squeaker Frogs",
    "Hyperoliidae":          "African Reed Frogs",
    "Mantellidae":           "Malagasy Poison Frogs",
    "Hemisotidae":           "Shovelnose Frogs",
    "Ceratophryidae":        "Horned Frogs (Pacman Frogs)",
    "Dicroglossidae":        "Fork-tongued Frogs",
    "Pyxicephalidae":        "African Bullfrogs",
    "Phyllomedusidae":       "Leaf Frogs (Red-eyed Tree Frog)",
    "Rhinophrynidae":        "Mexican Burrowing Toad",

    # Salamanders & Newts
    "Ambystomatidae":        "Mole Salamanders (Tiger Salamander, Axolotl)",
    "Salamandridae":         "Newts & Fire Salamanders",
    "Plethodontidae":        "Lungless Salamanders",
    "Cryptobranchidae":      "Giant Salamanders (Hellbender)",
    "Proteidae":             "Mudpuppies & Waterdogs",
    "Sirenidae":             "Sirens",
    "Amphiumidae":           "Amphiumas (Congo Eels)",
    "Rhyacotritonidae":      "Torrent Salamanders",
    "Dicamptodontidae":      "Pacific Giant Salamanders",
    "Hynobiidae":            "Asiatic Salamanders",
}


def add_family_labels(csv_path, output_path=None):
    """Read merged_df, add family + family_common columns, save."""
    df = pd.read_csv(csv_path, index_col=0)
    print(f"Loaded {len(df)} rows.")

    # Extract genus (first word of scientific_name)
    df["genus"] = df["scientific_name"].astype(str).str.split().str[0]

    # Map genus → family
    df["family"] = df["genus"].map(GENUS_TO_FAMILY)

    # Map family → common name
    df["family_common"] = df["family"].map(FAMILY_COMMON)

    # Report coverage
    mapped = df["family"].notna().sum()
    unmapped = df["family"].isna().sum()
    print(f"\nFamily mapping: {mapped} mapped, {unmapped} unmapped ({unmapped/len(df)*100:.1f}%)")

    if unmapped > 0:
        unknown_genera = df.loc[df["family"].isna(), "genus"].value_counts()
        print(f"\nUnmapped genera ({len(unknown_genera)} unique):")
        print(unknown_genera.to_string())

    # Distribution
    print(f"\n{'='*60}")
    print("FAMILY DISTRIBUTION (this is what your classifier will learn)")
    print(f"{'='*60}")
    family_counts = df["family"].value_counts()
    for fam, count in family_counts.items():
        common = FAMILY_COMMON.get(fam, "?")
        print(f"  {fam:25s} {count:5d}  — {common}")

    print(f"\nTotal families: {df['family'].nunique()}")
    print(f"Families with ≥30 clips: {(family_counts >= 30).sum()}")
    print(f"Families with ≥50 clips: {(family_counts >= 50).sum()}")

    # Also show common_name distribution within top families
    print(f"\n{'='*60}")
    print("SPECIES WITHIN TOP 5 FAMILIES")
    print(f"{'='*60}")
    for fam in family_counts.head(5).index:
        print(f"\n  {fam} ({FAMILY_COMMON.get(fam, '?')}):")
        species = df[df["family"] == fam]["common_name"].value_counts()
        for sp, ct in species.head(10).items():
            print(f"    {sp:40s} {ct:4d}")

    # Save
    if output_path is None:
        output_path = csv_path.replace(".csv", "_with_family.csv")
    df.to_csv(output_path)
    print(f"\nSaved to {output_path}")

    return df


if __name__ == "__main__":
    add_family_labels("merged_df.csv")
