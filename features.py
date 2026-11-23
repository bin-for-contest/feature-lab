"""변이 식별자를 검증하고 선택한 feature block을 고정된 순서로 조립함."""

from pathlib import Path

import numpy as np
import pandas as pd


AA = "ACDEFGHIKLMNPQRSTVWY"
VARIANT = rf"^([{AA}])([1-9][0-9]*)([{AA}])$"
EMBEDDING = [f"esm_emb_{i}" for i in range(1280)]
ESM = [*EMBEDDING, "esm_llr", "esm_cos_sim"]

# Kyte–Doolittle hydropathicity. 큰 값일수록 소수성이 강함.
HYDROPATHY = dict(zip(AA, [
    1.8, 2.5, -3.5, -3.5, 2.8, -0.4, -3.2, 4.5, -3.9, 3.8,
    1.9, -3.5, -1.6, -3.5, -4.5, -0.8, -0.7, 4.2, -0.9, -1.3,
]))
GROUPS = ("esm", "embedding", "llr", "cosine", "charge", "hydropathy",
          "hbond", "polarity", "plddt")


def read_table(path):
    """CSV/TSV 또는 기존 추출기의 Parquet를 같은 DataFrame으로 읽음."""
    suffix = Path(path).suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in {".csv", ".tsv"}:
        return pd.read_csv(path, sep="\t" if suffix == ".tsv" else ",")
    raise ValueError(f"지원 형식: .csv, .tsv, .parquet ({path})")


def variants(frame):
    """mutant 표기에서 wt/pos/mut를 복원하고 기존 메타데이터와 대조함."""
    if frame.empty or "mutant" not in frame:
        raise ValueError("mutant 열과 한 개 이상의 변이가 필요함")
    df = frame.reset_index(drop=True).copy()
    parsed = df["mutant"].astype("string").str.extract(VARIANT)
    parsed.columns = ["wt", "pos", "mut"]
    invalid = parsed.isna().any(axis=1) | parsed.wt.eq(parsed.mut)
    if invalid.any():
        raise ValueError(f"단일 missense 표기를 확인할 것: {df.loc[invalid, 'mutant'].iloc[0]}")
    parsed["pos"] = parsed.pos.astype("int64")
    for key in parsed:
        if key in df:
            existing = pd.to_numeric(df[key], errors="coerce") if key == "pos" else df[key]
            if not existing.eq(parsed[key]).fillna(False).all():
                raise ValueError(f"mutant 표기와 {key} 열이 일치하지 않음")
        df[key] = parsed[key]
    if df.mutant.duplicated().any():
        raise ValueError("mutant가 중복됨. 단백질 치환당 한 행으로 정리할 것")
    if df.groupby("pos").wt.nunique().gt(1).any():
        raise ValueError("한 위치에 서로 다른 WT가 있음. 참조 서열을 확인할 것")
    return df


def add_structure(df, path):
    """BRCA1 기준 위치(pos)별 pLDDT를 매핑해 변이 행 수와 순서를 보존함."""
    if Path(path).suffix.lower() == ".pdb":
        # AlphaFold WT 단량체 PDB의 chain A에서 Cα 하나씩 읽음.
        # 잔기 번호/아미노산을 함께 대조해 isoform·번호 체계 오류를 발견할 수 있음.
        names = "ALA CYS ASP GLU PHE GLY HIS ILE LYS LEU MET ASN PRO GLN ARG SER THR VAL TRP TYR"
        code = dict(zip(names.split(), AA))
        residues = []
        for line in Path(path).read_text(encoding="ascii").splitlines():
            if line.startswith("ENDMDL"):
                break
            if (line.startswith("ATOM  ") and line[12:16].strip() == "CA"
                    and line[21:22] == "A" and line[16:17] in {" ", "A"}):
                if line[26:27] != " " or line[17:20] not in code:
                    raise ValueError("표준 잔기 번호를 쓰는 AlphaFold WT PDB가 필요함")
                residues.append((int(line[22:26]), code[line[17:20]], float(line[60:66])))
        table = pd.DataFrame(residues, columns=["pos", "wt", "plddt"])
    else:
        table = read_table(path)
    if not {"pos", "plddt"} <= set(table):
        raise ValueError("구조 표에는 pos, plddt 열이 필요함")
    if "plddt" in df:
        raise ValueError("입력에 plddt가 이미 있음. --structure와 중복 지정됨")
    pos = pd.to_numeric(table["pos"], errors="raise")
    if (pos.isna().any() or pos.duplicated().any() or
            not ((pos > 0) & (pos % 1 == 0)).all()):
        raise ValueError("구조 표 pos는 중복 없는 1-based 양의 정수여야 함")
    indexed = table.set_index(pos.astype("int64"))
    if "wt" in indexed:
        aligned = df.pos.map(indexed.wt)
        if not aligned.eq(df.wt).fillna(False).all():
            raise ValueError("구조 표의 wt 또는 위치가 변이 표와 일치하지 않음")
    return df.assign(plddt=df.pos.map(indexed.plddt))


def design_matrix(df, groups):
    """정답이나 pos를 자동 선택하지 않음. 허용 목록에 속한 수치 열만 X에 넣음."""
    if not groups or set(groups) - set(GROUPS) or len(groups) != len(set(groups)):
        raise ValueError(f"feature 이름 또는 중복을 확인할 것: {groups}")
    if "esm" in groups and set(groups) & {"embedding", "llr", "cosine"}:
        raise ValueError("esm에는 embedding, llr, cosine이 이미 포함됨")

    blocks = []
    for group in groups:
        esm_columns = {"esm": ESM, "embedding": EMBEDDING, "llr": ["esm_llr"],
                       "cosine": ["esm_cos_sim"], "plddt": ["plddt"]}
        if group in esm_columns:
            columns = esm_columns[group]
            missing = [c for c in columns if c not in df]
            if missing:
                raise ValueError(f"{group}: 필요한 열 {len(missing)}개 누락. 예: {missing[:3]}")
            blocks.append(df[columns])
            continue

        # 속성별 Mut - WT 차이를 사용함. 수소결합은 donor와 acceptor를 따로 계산함.
        # 집합의 소속 여부는 0/1, charge는 간단한 정수 근사치로 코딩함.
        if group == "hydropathy":
            scales = {"hydropathy": HYDROPATHY}
        elif group == "charge":
            scales = {"charge": {a: int(a in "KR") - int(a in "DE") for a in AA}}
        elif group == "polarity":
            scales = {"polarity": {a: int(a in "DENQHKRSTY") for a in AA}}
        else:
            scales = {"donor": {a: int(a in "RKWNQHSTY") for a in AA},
                      "acceptor": {a: int(a in "DENQHSTY") for a in AA}}
        for name, lookup in scales.items():
            delta = df.mut.map(lookup) - df.wt.map(lookup)
            blocks.append(delta.rename(f"delta_{name}"))

    X = pd.concat(blocks, axis=1).astype("float32")
    if not np.isfinite(X.to_numpy()).all():
        bad = X.columns[~np.isfinite(X.to_numpy()).all(axis=0)]
        raise ValueError(f"결측/무한 feature를 확인할 것: {list(bad[:5])}")
    if "plddt" in X and not X.plddt.between(0, 100).all():
        raise ValueError("plddt 범위는 0~100임")
    return X
