"""변이 식별자를 검증하고 선택한 feature block을 고정된 순서로 조립함."""

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd


AA = "ACDEFGHIKLMNPQRSTVWY"
VARIANT = rf"^([{AA}])([1-9][0-9]*)([{AA}])$"
EMBEDDING = [f"esm_emb_{i}" for i in range(1280)]
ESM = [*EMBEDDING, "esm_cos_sim"]

CHARGE_PH = 7.0
CHARGE_PK_SCALE = "Lehninger"
HYDROPATHY_SCALE = "KyteDoolittle"
CANDIDATES = ("charge", "hydropathy", "hbond", "plddt")
GROUPS = ("esm", "embedding", "cosine", *CANDIDATES)


@lru_cache(maxsize=2)
def peptide_scale(group):
    """peptides의 단일 잔기 descriptor를 캐싱함. 구조/수소결합은 이 API의 대상이 아님."""
    from peptides import Peptide

    if group == "charge":
        return {a: Peptide(a).charge(pH=CHARGE_PH, pKscale=CHARGE_PK_SCALE) for a in AA}
    if group == "hydropathy":
        return {a: Peptide(a).hydrophobicity(scale=HYDROPATHY_SCALE) for a in AA}
    raise ValueError(f"peptides로 계산하는 feature가 아님: {group}")


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
    if "esm" in groups and set(groups) & {"embedding", "cosine"}:
        raise ValueError("esm에는 embedding, cosine이 이미 포함됨")

    blocks = []
    for group in groups:
        esm_columns = {"esm": ESM, "embedding": EMBEDDING,
                       "cosine": ["esm_cos_sim"], "plddt": ["plddt"]}
        if group in esm_columns:
            columns = esm_columns[group]
            missing = [c for c in columns if c not in df]
            if missing:
                raise ValueError(f"{group}: 필요한 열 {len(missing)}개 누락. 예: {missing[:3]}")
            blocks.append(df[columns])
            continue

        # 치환 잔기의 Mut - WT 차이. 전하와 소수성은 peptides에서 계산함.
        if group in {"charge", "hydropathy"}:
            # 전하 정의가 달라졌으므로 기존 정수 전하로 학습한 모델과 열 이름을 구분함.
            name = "charge_ph7_lehninger" if group == "charge" else "hydropathy"
            scales = {name: peptide_scale(group)}
        else:
            # 기존의 donor/acceptor 성향 근사이며 peptides 값이나 실제 결합 수가 아님.
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
