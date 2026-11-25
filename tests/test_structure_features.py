"""구조 feature의 기하학적 정의와 변이 매핑을 작은 CIF로 검증한다.

테스트의 최소 원자 좌표는 계산 규칙 검증용이며 실제 단백질 구조를 뜻하지 않는다.
실행: python -m unittest discover -s tests -v
"""

import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from features import CANDIDATES, STRUCTURE_COLUMNS, add_structure, design_matrix, variants
from structure_features import calculate_structure_features
from hbond import HBOND_COLUMNS


def write_cif(path, entries):
    """필요한 atom_site 필드를 갖는 최소 테스트 CIF를 만든다."""
    fields = ["group_PDB", "id", "type_symbol", "label_atom_id", "label_alt_id",
              "label_comp_id", "label_asym_id", "label_seq_id", "Cartn_x", "Cartn_y",
              "Cartn_z", "B_iso_or_equiv", "pdbx_PDB_model_num", "auth_seq_id"]
    rows = []
    for serial, entry in enumerate(entries, start=1):
        row = dict(group_PDB="ATOM", id=serial, type_symbol="C", label_atom_id="CA",
                   label_alt_id=".", label_comp_id="ALA", label_asym_id="A",
                   label_seq_id=serial, Cartn_x=0, Cartn_y=0, Cartn_z=0,
                   B_iso_or_equiv=80, pdbx_PDB_model_num=1, auth_seq_id=serial + 100)
        row.update(entry)
        rows.append(" ".join(str(row[field]) for field in fields))
    path.write_text("data_geometry\nloop_\n" + "\n".join("_atom_site." + field for field in fields)
                    + "\n" + "\n".join(rows) + "\n#\n", encoding="utf-8")


class StructureFeaturesTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.path = self.root / "wt.cif"
        self.entries = [{"label_seq_id": pos, "Cartn_x": x, "B_iso_or_equiv": 50 + pos}
                        for pos, x in enumerate((0, 4, 8, 20), start=1)]
        write_cif(self.path, self.entries)

    def tearDown(self):
        self.temp.cleanup()

    def test_neighbor_definition_and_sasa(self):
        table = calculate_structure_features(self.path)
        self.assertEqual(table.pos.tolist(), [1, 2, 3, 4])
        self.assertEqual(table.neighbor_count.tolist(), [2, 2, 2, 0])
        self.assertEqual(table.plddt.tolist(), [51, 52, 53, 54])
        # 고립된 탄소: 원자 반지름 1.7 Å + 탐침 1.4 Å인 구의 표면적과 같아야 한다.
        self.assertAlmostEqual(table.sasa.iloc[3], 4 * math.pi * (1.7 + 1.4) ** 2)
        self.assertLess(table.sasa.iloc[0], table.sasa.iloc[3])

    def test_other_atoms_do_not_inflate_neighbor_count(self):
        before = calculate_structure_features(self.path)
        extra = dict(self.entries[0], label_atom_id="N", type_symbol="N", Cartn_y=1)
        write_cif(self.path, self.entries + [extra])
        after = calculate_structure_features(self.path)
        self.assertEqual(after.neighbor_count.tolist(), before.neighbor_count.tolist())

    def test_chain_model_altloc_and_label_numbering(self):
        extra = [dict(self.entries[0], label_asym_id="B"),
                 dict(self.entries[0], pdbx_PDB_model_num=2),
                 dict(self.entries[1], label_alt_id="B", Cartn_x=100)]
        write_cif(self.path, self.entries + extra)
        self.assertEqual(calculate_structure_features(self.path).neighbor_count.tolist(), [2, 2, 2, 0])

    def test_mapping_and_group_columns(self):
        frame = variants(pd.DataFrame({"mutant": ["A2D", "A1V", "A2K"]}))
        mapped = add_structure(frame, self.path)
        self.assertEqual(mapped.mutant.tolist(), frame.mutant.tolist())
        self.assertEqual(mapped.plddt.tolist(), [52, 51, 52])
        self.assertEqual(mapped.neighbor_count.tolist(), [2, 2, 2])
        self.assertEqual(mapped.sasa.iloc[0], mapped.sasa.iloc[2])
        self.assertNotIn("sasa", frame)
        matrix = design_matrix(mapped, ["structure"])
        self.assertEqual(matrix.columns.tolist(), STRUCTURE_COLUMNS)
        self.assertEqual(matrix.shape, (3, 3))
        self.assertEqual(CANDIDATES, ("charge", "hydropathy", "hbond", "structure"))
        with self.assertRaises(ValueError):
            design_matrix(mapped, ["structure", "plddt"])
        with self.assertRaises(ValueError):
            add_structure(mapped, self.path)

    def test_wt_and_position_mismatch(self):
        for mutant in ("C1A", "A99D"):
            with self.subTest(mutant=mutant), self.assertRaises(ValueError):
                add_structure(variants(pd.DataFrame({"mutant": [mutant]})), self.path)

    def test_hbond_geometry_and_mapping(self):
        # K1 NZ → D2 OD1. 다른 원자는 멀리 둬 판정할 거리·각도 하나씩만 바꾼다.
        atoms = []
        def atom(pos, name, x):
            return dict(label_seq_id=pos, label_comp_id=("LYS", "ASP")[pos-1],
                        label_atom_id=name, type_symbol=name[0], Cartn_x=x)
        for pos in (1, 2):
            for name, offset in [("CA", 0), ("N", 1), ("C", 2), ("O", 3)]:
                atoms.append(atom(pos, name, 20 * pos + offset))
        atoms += [atom(1, "NZ", 0), atom(2, "OD2", 12)]
        source = variants(pd.DataFrame({"mutant": ["D2A", "K1V", "K1R"]}))
        for ce, od, cg, count in [(-1, 3.5, 4.5, 1), (-1, 3.51, 4.5, 0),
                                  (-1, 2, 4.5, 0), (1, 3, 4, 0), (-1, 3, 2, 0)]:
            with self.subTest(ce=ce, od=od, cg=cg):
                write_cif(self.path, atoms + [atom(1, "CE", ce), atom(2, "OD1", od), atom(2, "CG", cg)])
                mapped = add_structure(source, self.path, include_hbond=True)
                matrix = design_matrix(mapped, ["hbond"])
                self.assertEqual(matrix.columns.tolist(), HBOND_COLUMNS)
                np.testing.assert_array_equal(matrix, [[0, count], [count, 0], [count, 0]])
        # 주사슬 O를 상대 원자로 허용하되 D2 곁사슬 acceptor 수에는 넣지 않는다.
        backbone = [r for r in atoms if not (r["label_seq_id"] == 2 and r["label_atom_id"] in {"C", "O"})]
        write_cif(self.path, backbone + [atom(1, "CE", -1), atom(2, "C", 4), atom(2, "O", 3),
                                        atom(2, "OD1", 15), atom(2, "CG", 14)])
        mapped = add_structure(source, self.path, include_hbond=True)
        np.testing.assert_array_equal(design_matrix(mapped, ["hbond"]), [[0, 0], [1, 0], [1, 0]])

    def test_hbond_required_inputs_and_legacy(self):
        source = variants(pd.DataFrame({"mutant": ["K1D"]}))
        with self.assertRaisesRegex(ValueError, "CIF"):
            design_matrix(source, ["hbond"])
        with self.assertRaisesRegex(ValueError, "원자 누락"):
            calculate_structure_features(self.path, include_hbond=True)
        table = self.root / "hbond.csv"
        pd.DataFrame({"pos": [1], HBOND_COLUMNS[0]: [2], HBOND_COLUMNS[1]: [0]}).to_csv(table, index=False)
        mapped = add_structure(source, table, include_hbond=True)
        np.testing.assert_array_equal(design_matrix(mapped, ["hbond"]), [[2, 0]])
        for bad in (-1, 0.5, np.nan):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                design_matrix(mapped.assign(**{HBOND_COLUMNS[0]: bad}), ["hbond"])
        old = design_matrix(source, ["hbond"], legacy_hbond=True)
        self.assertEqual(old.columns.tolist(), ["delta_donor", "delta_acceptor"])
        np.testing.assert_array_equal(old, [[-1, 1]])

    def test_legacy_json_and_precomputed_table(self):
        source = variants(pd.DataFrame({"mutant": ["A1D"]}))
        path = self.root / "confidence.json"
        path.write_text(json.dumps({"residueNumber": [1], "confidenceScore": [51]}), encoding="utf-8")
        legacy = add_structure(source, path)
        self.assertEqual(design_matrix(legacy, ["plddt"]).iloc[0, 0], 51)
        with self.assertRaisesRegex(ValueError, "CIF"):
            design_matrix(legacy, ["structure"])
        table = calculate_structure_features(self.path)
        csv = self.root / "structure.csv"
        table.to_csv(csv, index=False)
        np.testing.assert_allclose(design_matrix(add_structure(source, csv), ["structure"]),
                                   design_matrix(add_structure(source, self.path), ["structure"]))

    def test_invalid_structure_and_feature_values(self):
        invalid = [dict(self.entries[0], Cartn_x="?"),
                   dict(self.entries[0], label_comp_id="UNK"),
                   dict(self.entries[0], label_atom_id="N"),
                   dict(self.entries[0], B_iso_or_equiv=101)]
        for entry in invalid:
            write_cif(self.path, [entry])
            with self.subTest(entry=entry), self.assertRaises(ValueError):
                calculate_structure_features(self.path)
        write_cif(self.path, self.entries + [self.entries[0]])
        with self.assertRaises(ValueError):
            calculate_structure_features(self.path)
        for column, value in (("sasa", -1), ("neighbor_count", -1), ("neighbor_count", 1.5)):
            frame = pd.DataFrame({"sasa": [10.0], "neighbor_count": [2], "plddt": [80]})
            frame[column] = value
            with self.subTest(column=column, value=value), self.assertRaises(ValueError):
                design_matrix(frame, ["structure"])


if __name__ == "__main__":
    unittest.main()
