"""AlphaFold WT CIF의 원자 좌표에서 위치별 구조 feature를 계산한다.

계산 대상은 첫 모델의 label_asym_id='A'인 표준 아미노산이다.
위치 번호는 label_seq_id를 사용하며, feature는 Mut 구조 변화량이 아닌 WT 환경이다.
"""

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from parse_cif import parse_cif


NEIGHBOR_RADIUS = 8.0  # Å. Cα 사이 거리가 이 값 이하인 다른 잔기의 수.
SASA_PROBE_RADIUS = 1.4  # Å. Shrake–Rupley의 용매 탐침 반지름.
SASA_POINTS = 100  # 원자 표면 샘플 수. 모든 학습/예측에서 같은 설정을 사용한다.
RESIDUE_CODES = dict(zip(
    "ALA CYS ASP GLU PHE GLY HIS ILE LYS LEU MET ASN PRO GLN ARG SER THR VAL TRP TYR".split(),
    "ACDEFGHIKLMNPQRSTVWY",
))


def calculate_structure_features(path):
    """WT CIF에서 pos, wt, sasa, neighbor_count, plddt 표를 반환한다.

    SASA는 선택한 체인 전체의 비수소 원자로 계산한 잔기별 노출 면적(Å²)이다.
    neighbor_count는 Cα 기준 8 Å 이내의 잔기 수이며 자기 자신은 제외한다.
    서열상 바로 이웃한 잔기는 포함한다. pLDDT는 각 잔기의 Cα B_iso_or_equiv다.
    모델/체인/원자 번호를 임의로 1..N으로 다시 매기지 않는다.
    """
    from Bio.PDB.Atom import Atom
    from Bio.PDB.Chain import Chain
    from Bio.PDB.Residue import Residue
    from Bio.PDB.SASA import ShrakeRupley

    atoms = parse_cif(path)
    required = {"group_PDB", "type_symbol", "label_atom_id", "label_comp_id",
                "label_asym_id", "label_seq_id", "B_iso_or_equiv"}
    if not required <= set(atoms):
        raise ValueError(f"구조 feature 계산에 필요한 CIF 열 누락: {sorted(required - set(atoms))}")

    if "pdbx_PDB_model_num" in atoms:
        if atoms.pdbx_PDB_model_num.isna().any():
            raise ValueError("CIF의 모델 번호가 누락됨")
        # 파일에 처음 등장하는 모델만 사용해 서로 다른 구조를 합산하지 않는다.
        atoms = atoms.loc[atoms.pdbx_PDB_model_num.eq(atoms.pdbx_PDB_model_num.iloc[0])]
    atoms = atoms.loc[atoms.group_PDB.eq("ATOM") & atoms.label_asym_id.eq("A")].copy()
    for name in ("type_symbol", "label_atom_id", "label_comp_id"):
        if atoms[name].isna().any() or atoms[name].isin([".", "?", ""]).any():
            raise ValueError(f"CIF의 {name}에 원자/잔기 정보가 누락됨")
    atoms = atoms.loc[~atoms.type_symbol.str.upper().isin(["H", "D"])].copy()
    if atoms.empty:
        raise ValueError("첫 모델의 chain A에 단백질 원자가 없음")
    if "label_alt_id" in atoms:
        # AlphaFold의 기본 위치 및 A 대체 위치만 허용한다. 중복 원자는 아래에서 거부한다.
        atoms = atoms.loc[atoms.label_alt_id.isin([".", "?", " ", "A"])].copy()
    if atoms.empty:
        raise ValueError("chain A에 사용 가능한 원자 대체 위치가 없음")
    if atoms.label_seq_id.isna().any() or not atoms.label_seq_id.gt(0).all():
        raise ValueError("CIF label_seq_id는 1-based 양의 정수여야 함")
    if atoms.duplicated(["label_seq_id", "label_atom_id"]).any():
        raise ValueError("같은 잔기에 동일한 원자가 중복됨. 원자 대체 위치를 확인할 것")
    numeric = atoms[["Cartn_x", "Cartn_y", "Cartn_z", "B_iso_or_equiv"]].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError("CIF 좌표 또는 B_iso_or_equiv에 결측/무한값이 있음")
    if not atoms.B_iso_or_equiv.between(0, 100).all():
        raise ValueError("AlphaFold CIF의 pLDDT는 0~100이어야 함")

    # 기존 원자 표를 선택한 체인 하나의 Bio.PDB 객체로 만들어 SASA 계산에 사용한다.
    chain = Chain("A")
    ca_coordinates = []
    records = []
    for pos, group in atoms.groupby("label_seq_id", sort=True):
        names = group.label_comp_id.unique()
        if len(names) != 1 or names[0] not in RESIDUE_CODES:
            raise ValueError(f"위치 {pos}: 표준 WT 잔기 이름이 하나여야 함")
        ca = group.loc[group.label_atom_id.eq("CA")]
        if len(ca) != 1 or ca.type_symbol.iloc[0].upper() != "C":
            raise ValueError(f"위치 {pos}: Cα 원자가 정확히 하나 필요함")
        residue = Residue((" ", int(pos), " "), names[0], " ")
        chain.add(residue)
        for serial, row in enumerate(group.itertuples(index=False), start=1):
            coord = np.array([row.Cartn_x, row.Cartn_y, row.Cartn_z], dtype=float)
            residue.add(Atom(row.label_atom_id, coord, float(row.B_iso_or_equiv), 1.0,
                             " ", row.label_atom_id.rjust(4), serial,
                             element=row.type_symbol.upper()))
        ca_coordinates.append(ca[["Cartn_x", "Cartn_y", "Cartn_z"]].iloc[0].to_numpy(dtype=float))
        records.append({"pos": int(pos), "wt": RESIDUE_CODES[names[0]],
                        "plddt": float(ca.B_iso_or_equiv.iloc[0])})

    ShrakeRupley(probe_radius=SASA_PROBE_RADIUS, n_points=SASA_POINTS).compute(chain, level="R")
    table = pd.DataFrame(records)
    table["sasa"] = [residue.sasa for residue in chain]
    coordinates = np.asarray(ca_coordinates)
    # 각 Cα를 한 번씩만 넣으므로 원자 수가 아닌 서로 다른 잔기 수가 된다.
    # 검색 결과에는 자기 자신도 포함되므로 1을 뺀다.
    table["neighbor_count"] = cKDTree(coordinates).query_ball_point(
        coordinates, NEIGHBOR_RADIUS, return_length=True) - 1
    return table[["pos", "wt", "sasa", "neighbor_count", "plddt"]]
