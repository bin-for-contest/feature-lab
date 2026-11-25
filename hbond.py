"""WT 단백질의 중원자 좌표로 곁사슬 수소결합 후보를 센다."""

import numpy as np


HBOND_COLUMNS = ["wt_sidechain_hbond_donor_count", "wt_sidechain_hbond_acceptor_count"]

# (극성 원자, 가능한 역할, 바로 결합된 중원자). D는 donor, A는 acceptor다.
# 일반적인 이온화 상태의 N/O만 다룬다. HIS는 tautomer를 결정할 수 없어 양쪽 가능성을 둔다.
SIDECHAIN_SITES = {
    "ARG": [("NE", "D", "CD CZ"), ("NH1", "D", "CZ"), ("NH2", "D", "CZ")],
    "ASN": [("ND2", "D", "CG"), ("OD1", "A", "CG")],
    "ASP": [("OD1", "A", "CG"), ("OD2", "A", "CG")],
    "GLN": [("NE2", "D", "CD"), ("OE1", "A", "CD")],
    "GLU": [("OE1", "A", "CD"), ("OE2", "A", "CD")],
    "HIS": [("ND1", "DA", "CG CE1"), ("NE2", "DA", "CD2 CE1")],
    "LYS": [("NZ", "D", "CE")],
    "SER": [("OG", "DA", "CB")],
    "THR": [("OG1", "DA", "CB")],
    "TRP": [("NE1", "D", "CD1 CE2")],
    "TYR": [("OH", "DA", "CZ")],
}


def sidechain_hbond_counts(chain):
    """체인의 잔기 순서대로 [곁사슬이 donor인 원자쌍 수, acceptor인 원자쌍 수]를 반환한다."""
    from scipy.spatial import cKDTree

    donors, acceptors = [], []
    counts = np.zeros((len(chain), 2), dtype=int)
    for index, residue in enumerate(chain):
        pos = residue.id[1]
        sites = list(SIDECHAIN_SITES.get(residue.resname, ()))
        sites.append(("O", "A", "C"))
        if "OXT" in residue:
            sites.append(("OXT", "A", "C"))
        if residue.resname != "PRO" or pos == 1:
            sites.append(("N", "D", "CA CD" if residue.resname == "PRO" else "CA"))
        for name, roles, neighbors in sites:
            try:
                atom = residue[name]
                bonded = [residue[n].coord for n in neighbors.split()]
            except KeyError as exc:
                raise ValueError(f"hbond 계산에 필요한 원자 누락: {residue.resname}{pos} {exc}") from exc
            # 펩타이드 결합이 연결된 backbone N의 방향에는 앞 잔기의 C도 관여한다.
            if name == "N" and pos - 1 in chain:
                previous = chain[pos - 1]
                if "C" in previous and previous["C"] - atom <= 1.8:
                    bonded.append(previous["C"].coord)
            vectors = np.asarray(bonded) - atom.coord
            if atom.element != name[0] or np.any(np.linalg.norm(vectors, axis=1) == 0):
                raise ValueError(f"hbond 원소 또는 결합 좌표를 확인할 것: {residue.resname}{pos} {name}")
            sidechain = name not in {"N", "O", "OXT"}
            site = (index, atom.coord, vectors, sidechain)
            if "D" in roles:
                donors.append(site)
            if "A" in roles:
                acceptors.append(site)

    # 다른 잔기의 donor→acceptor 원자쌍을 방향별로 한 번만 센다.
    # 두 끝 모두 주사슬인 경우에는 변이 위치 곁사슬의 환경에 기여하지 않는다.
    tree = cKDTree([site[1] for site in acceptors])
    for di, donor, d_bonds, d_sidechain in donors:
        for j in tree.query_ball_point(donor, 3.5):
            ai, acceptor, a_bonds, a_sidechain = acceptors[j]
            if di == ai or not (d_sidechain or a_sidechain):
                continue
            direction = acceptor - donor
            if np.linalg.norm(direction) < 2.4:
                continue
            # 양 끝에서 '결합된 중원자–극성 원자–상대 원자' 각도가 모두 90° 이상이어야 한다.
            # 두 벡터의 내적이 양수이면 예각이다. 수소 위치를 가정하지 않는다.
            if (d_bonds @ direction > 0).any() or (a_bonds @ -direction > 0).any():
                continue
            if d_sidechain:
                counts[di, 0] += 1
            if a_sidechain:
                counts[ai, 1] += 1
    return counts
