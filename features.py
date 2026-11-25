"""입력 표 검증과 LightGBM용 feature 행렬 생성을 담당하는 공통 모듈.

한 행은 한 단백질의 단일 아미노산 치환을 뜻한다. 예를 들어 C39Y는
WT 잔기(residue) C, 전체 서열의 1-based 위치 39, 변이 잔기(residue) Y로 해석한다.
이 모듈은 모델을 학습하지 않으며, 추출기와 실험 CLI 양쪽에서 사용한다.
정답 열은 입력 표에 함께 있어도 feature로 자동 선택하지 않는다.
"""

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd


# 허용하는 표준 아미노산 20종. 정규식과 PDB의 3문자 코드 변환에 같은 순서를 쓴다.
AA = "ACDEFGHIKLMNPQRSTVWY"
# 문자열 전체가 'WT 1-based위치 Mut' 형식이어야 한다. 0번 위치와 다중 치환은 제외한다.
# 정규식
VARIANT = rf"^([{AA}])([1-9][0-9]*)([{AA}])$"
# ESM-2 650M 마지막 레이어의 잔기(residue) 벡터는 1,280차원이다.
# 열 이름을 미리 고정해 파일의 물리적인 열 순서에 영향을 받지 않게 한다.
EMBEDDING = [f"esm_emb_{i}" for i in range(1280)]
# 기본 baseline: 변이 잔기(residue) 임베딩 + WT/Mut 잔기(residue) 벡터의 코사인 유사도.
# esm_llr은 포함하지 않는다. 입력에 이 열이 남아 있어도 선택되지 않는다.
ESM = [*EMBEDDING, "esm_cos_sim"]

# 전하의 수치는 pH와 pKa 척도에 따라 달라지므로 실험 전체에서 같은 설정을 쓴다.
CHARGE_PH = 7.0
CHARGE_PK_SCALE = "Lehninger"
HYDROPATHY_SCALE = "KyteDoolittle"
# compare는 추가 후보만 받는다. train은 ESM의 일부만 쓰는 실험도 허용한다.
CANDIDATES = ("charge", "hydropathy", "hbond", "structure")
# plddt 단독 그룹은 기존 모델의 추론 및 개별 train 실험과의 호환용으로 유지한다.
GROUPS = ("esm", "embedding", "cosine", *CANDIDATES, "plddt")
STRUCTURE_COLUMNS = ["sasa", "neighbor_count", "plddt"]


@lru_cache(maxsize=2)
def peptide_scale(group):
    """추가 feature 이름을 받아 {아미노산: descriptor 값} dictionary를 반환한다.

    peptides.Peptide에 한 글자 서열을 넣어 잔기(residue)별 값을 계산한다. 이후 변이별
    Mut - WT 차이를 사용하므로, 전체 단백질의 평균 소수성/총전하와 구분해야 한다.
    두 척도 각각의 사전을 한 번 계산해 재사용하며 반환한 사전은 수정하지 않는다.
    구조와 수소결합은 peptides 계산 대상이 아니다.
    """
    # ESM feature만 사용하는 실행에서는 peptides를 불러올 필요가 없다.
    from peptides import Peptide

    if group == "charge":
        return {a: Peptide(a).charge(pH=CHARGE_PH, pKscale=CHARGE_PK_SCALE) for a in AA}
    if group == "hydropathy":
        return {a: Peptide(a).hydrophobicity(scale=HYDROPATHY_SCALE) for a in AA}
    raise ValueError(f"peptides로 계산하는 feature가 아님: {group}")


def read_table(path):
    """파일 확장자로 CSV/TSV/Parquet를 구분해 pandas DataFrame으로 반환한다.

    여기서는 파일 형식만 처리한다. 변이 표기와 feature 값의 의미 검증은
    각각 variants(), design_matrix()가 담당한다.
    """
    suffix = Path(path).suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in {".csv", ".tsv"}:
        return pd.read_csv(path, sep="\t" if suffix == ".tsv" else ",")
    raise ValueError(f"지원 형식: .csv, .tsv, .parquet ({path})")


def variants(frame):
    """단일 치환 표를 검증하고 wt/pos/mut 열을 갖는 복사본을 반환한다.

    필수 입력은 비어 있지 않은 mutant 열이다. wt/pos/mut가 이미 있으면
    mutant에서 파싱한 값과 일치해야 한다. 동일 mutant 중복, WT와 Mut가 같은
    표기, 한 위치에서 WT가 서로 다른 데이터는 허용하지 않는다.
    한 호출의 입력은 한 단백질/참조 서열을 전제로 한다.
    """
    if frame.empty or "mutant" not in frame:
        raise ValueError("mutant 열과 한 개 이상의 변이가 필요함")
    # 기존 인덱스가 중복/비연속이어도 후속 concat과 NumPy 배열이 행 단위로 정렬되게 한다.
    # 원본 DataFrame에는 파생 열을 덧붙이지 않는다.
    df = frame.reset_index(drop=True).copy()
    parsed = df["mutant"].astype("string").str.extract(VARIANT)
    parsed.columns = ["wt", "pos", "mut"]
    # 정규식 불일치는 추출 결과의 결측으로 나타난다. C39C 같은 동의적 표기도 거부한다.
    invalid = parsed.isna().any(axis=1) | parsed.wt.eq(parsed.mut)
    if invalid.any():
        raise ValueError(f"단일 missense 표기를 확인할 것: {df.loc[invalid, 'mutant'].iloc[0]}")
    parsed["pos"] = parsed.pos.astype("int64")
    for key in parsed:
        if key in df:
            # pos는 '39'와 39를 같게 취급하지만 잔기(residue) 문자는 그대로 대조한다.
            existing = pd.to_numeric(df[key], errors="coerce") if key == "pos" else df[key]
            if not existing.eq(parsed[key]).fillna(False).all():
                raise ValueError(f"mutant 표기와 {key} 열이 일치하지 않음")
        df[key] = parsed[key]
    if df.mutant.duplicated().any():
        raise ValueError("mutant가 중복됨. 단백질 치환당 한 행으로 정리할 것")
    # 서로 다른 단백질/isoform의 변이가 같은 위치 번호로 섞인 경우를 일부 탐지한다.
    # 완전한 참조 서열 일치 검사는 ESM 추출기의 reference_sequence()가 수행한다.
    if df.groupby("pos").wt.nunique().gt(1).any():
        raise ValueError("한 위치에 서로 다른 WT가 있음. 참조 서열을 확인할 것")
    return df


def read_confidence_json(path):
    """AlphaFold confidence JSON을 pos, plddt 두 열의 표로 변환한다.

    residueNumber와 confidenceScore는 같은 길이의 숫자 배열이어야 한다.
    배열의 순서가 잔기 번호 순서라는 가정 없이 두 배열의 같은 항목끼리 대응시킨다.
    confidenceCategory는 점수에서 유도된 등급이므로 feature로 사용하지 않는다.
    위치 번호의 양의 정수 여부와 중복 검사는 add_structure()에서 공통으로 수행한다.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    required = ("residueNumber", "confidenceScore")
    if not isinstance(data, dict) or not set(required) <= set(data):
        raise ValueError("AlphaFold JSON에는 residueNumber, confidenceScore 배열이 필요함")

    positions, scores = (data[key] for key in required)
    if (not isinstance(positions, list) or not isinstance(scores, list)
            or not positions or len(positions) != len(scores)):
        raise ValueError("residueNumber와 confidenceScore는 길이가 같은 비어 있지 않은 배열이어야 함")
    for key, values in zip(required, (positions, scores)):
        # JSON의 true/false는 Python에서 정수처럼 취급되므로 숫자와 명시적으로 구분한다.
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in values):
            raise ValueError(f"AlphaFold JSON의 {key}에는 숫자만 들어가야 함")

    table = pd.DataFrame({"pos": positions, "plddt": scores})
    # 사용하지 않는 위치에 잘못된 값이 있어도 불완전한 confidence 파일은 거부한다.
    if not np.isfinite(table.plddt.to_numpy(dtype=float)).all() or not table.plddt.between(0, 100).all():
        raise ValueError("AlphaFold JSON의 confidenceScore는 0~100의 유한한 숫자여야 함")
    return table


def add_structure(df, path):
    """WT 구조 파일 또는 위치별 표에서 구조 feature를 가져와 변이 표에 연결한다.

    CIF/mmCIF는 SASA, 주변 잔기 수, pLDDT를 모두 계산한다. JSON과 기존 PDB 경로는
    pLDDT만 제공하며, CSV/TSV/Parquet는 미리 계산한 구조 열을 포함할 수 있다.
    df는 variants()를 통과한 표여야 한다. wt 열이 있는 구조 입력은 WT 문자도 대조한다.
    pLDDT는 WT 구조 예측의 신뢰도이며 변이 구조의 변화량이나 좌표 표현이 아니다.
    같은 pos의 여러 치환은 같은 pLDDT를 받는다.
    """
    if Path(path).suffix.lower() in {".cif", ".mmcif"}:
        from structure_features import calculate_structure_features

        table = calculate_structure_features(path)
    elif Path(path).suffix.lower() == ".json":
        # JSON에 제공된 실제 residueNumber로 매핑한다. 배열 인덱스 + 1로 추정하지 않는다.
        table = read_confidence_json(path)
    elif Path(path).suffix.lower() == ".pdb":
        # AlphaFold WT 단량체 PDB의 chain A에서 Cα 하나씩 읽음.
        # 잔기(residue) 번호/아미노산을 함께 대조해 isoform·번호 체계 오류를 발견할 수 있음.
        names = "ALA CYS ASP GLU PHE GLY HIS ILE LYS LEU MET ASN PRO GLN ARG SER THR VAL TRP TYR"
        code = dict(zip(names.split(), AA))
        residues = []
        for line in Path(path).read_text(encoding="ascii").splitlines():
            # 다중 모델 PDB에서 같은 잔기(residue)를 여러 번 읽지 않도록 첫 모델까지만 사용한다.
            if line.startswith("ENDMDL"):
                break
            # PDB는 고정 너비 형식이다. 원자명 12:16, chain 21, altLoc 16을 읽는다.
            # 잔기(residue)마다 Cα를 하나만 선택하며, 기본 또는 A 대체 위치만 허용한다.
            if (line.startswith("ATOM  ") and line[12:16].strip() == "CA"
                    and line[21:22] == "A" and line[16:17] in {" ", "A"}):
                # 삽입 코드가 있으면 단순 정수 pos와 일대일 대응하지 않으므로 중단한다.
                if line[26:27] != " " or line[17:20] not in code:
                    raise ValueError("표준 잔기(residue) 번호를 쓰는 AlphaFold WT PDB가 필요함")
                # AlphaFold PDB의 B-factor 칸(60:66)을 pLDDT로 해석한다.
                # 일반 실험 PDB의 B-factor를 넣으면 같은 의미가 되지 않는다.
                residues.append((int(line[22:26]), code[line[17:20]], float(line[60:66])))
        table = pd.DataFrame(residues, columns=["pos", "wt", "plddt"])
    else:
        table = read_table(path)
    if not {"pos", "plddt"} <= set(table):
        raise ValueError("구조 표에는 pos, plddt 열이 필요함")
    columns = [name for name in STRUCTURE_COLUMNS if name in table]
    overlap = [name for name in columns if name in df]
    if overlap:
        raise ValueError(f"입력에 구조 열이 이미 있음. --structure와 중복 지정됨: {overlap}")
    pos = pd.to_numeric(table["pos"], errors="raise")
    if (pos.isna().any() or pos.duplicated().any() or
            not ((pos > 0) & (pos % 1 == 0)).all()):
        raise ValueError("구조 표 pos는 중복 없는 1-based 양의 정수여야 함")
    # 위치를 사전의 key처럼 사용한다. 중복 위치를 위에서 거부해 매핑의 모호함을 없앤다.
    indexed = table.set_index(pos.astype("int64"))
    if "wt" in indexed:
        aligned = df.pos.map(indexed.wt)
        if not aligned.eq(df.wt).fillna(False).all():
            raise ValueError("구조 표의 wt 또는 위치가 변이 표와 일치하지 않음")
    # merge로 행을 늘리지 않고 map으로 원래 순서와 개수를 보존한다.
    # wt 없는 표에서 위치가 누락되면 NaN이 생기며, 해당 그룹 선택 시 design_matrix가 거부한다.
    return df.assign(**{name: df.pos.map(indexed[name]) for name in columns})


def design_matrix(df, groups):
    """선택한 그룹 순서대로 float32 feature 행렬 X(N행, D열)를 만든다.

    df는 variants()로 검증한 표이며, groups는 ['esm', 'charge'] 같은 목록이다.
    이미 추출한 ESM/pLDDT 열은 읽고, 잔기(residue) 속성 차이는 여기서 계산한다.
    DMS_score, DMS_score_bin, pos와 임의의 다른 입력 열은 자동 포함하지 않는다.
    반환 열 이름과 순서는 모델 저장 및 추론 시 동일성 확인에 사용한다.
    """
    if not groups or set(groups) - set(GROUPS) or len(groups) != len(set(groups)):
        raise ValueError(f"feature 이름 또는 중복을 확인할 것: {groups}")
    # esm은 embedding/cosine을 이미 포함하므로 중복 열을 만들어 학습하지 않게 한다.
    if "esm" in groups and set(groups) & {"embedding", "cosine"}:
        raise ValueError("esm에는 embedding, cosine이 이미 포함됨")
    if "structure" in groups and "plddt" in groups:
        raise ValueError("structure에는 plddt가 이미 포함됨")

    # DataFrame 블록과 Series 블록을 모아 마지막에 열 방향(axis=1)으로 연결한다.
    blocks = []
    for group in groups:
        esm_columns = {"esm": ESM, "embedding": EMBEDDING,
                       "cosine": ["esm_cos_sim"], "structure": STRUCTURE_COLUMNS,
                       "plddt": ["plddt"]}
        if group in esm_columns:
            columns = esm_columns[group]
            # 누락 열을 임의로 0으로 채우지 않는다. 잘못된 입력을 학습 전에 드러낸다.
            missing = [c for c in columns if c not in df]
            if missing:
                if group == "structure":
                    raise ValueError(f"structure 열 누락: {missing}. --structure에 AlphaFold WT CIF 또는 "
                                     "pos,sasa,neighbor_count,plddt 표를 지정할 것. confidence JSON만으로는 계산 불가")
                raise ValueError(f"{group}: 필요한 열 {len(missing)}개 누락. 예: {missing[:3]}")
            blocks.append(df[columns])
            continue

        # 치환 잔기(residue)의 Mut - WT 차이. 전하와 소수성은 peptides에서 계산함.
        if group in {"charge", "hydropathy"}:
            # 전하 정의가 달라졌으므로 기존 정수 전하로 학습한 모델과 열 이름을 구분함.
            name = "charge_ph7_lehninger" if group == "charge" else "hydropathy"
            scales = {name: peptide_scale(group)}
        else:
            # 기존의 donor/acceptor 성향 근사이며 peptides 값이나 실제 결합 수가 아님.
            # 각 집합에 속하면 1, 아니면 0이다. hbond 그룹은 두 개의 열을 만든다.
            scales = {"donor": {a: int(a in "RKWNQHSTY") for a in AA},
                      "acceptor": {a: int(a in "DENQHSTY") for a in AA}}
        for name, lookup in scales.items():
            # 양수는 해당 성질의 증가, 음수는 감소다. 수소결합 근사는 -1/0/1이 된다.
            delta = df.mut.map(lookup) - df.wt.map(lookup)
            blocks.append(delta.rename(f"delta_{name}"))

    # ESM과 추가 속성의 dtype을 맞추고 결측/무한값을 일괄 검사한다.
    # LightGBM 자체의 결측 허용 여부와 별개로, 이 실험은 완전한 feature를 요구한다.
    X = pd.concat(blocks, axis=1).astype("float32")
    if not np.isfinite(X.to_numpy()).all():
        bad = X.columns[~np.isfinite(X.to_numpy()).all(axis=0)]
        raise ValueError(f"결측/무한 feature를 확인할 것: {list(bad[:5])}")
    if "plddt" in X and not X.plddt.between(0, 100).all():
        raise ValueError("plddt 범위는 0~100임")
    if "sasa" in X and not X.sasa.ge(0).all():
        raise ValueError("sasa는 0 이상의 면적이어야 함")
    if "neighbor_count" in X and not (X.neighbor_count.ge(0) & X.neighbor_count.mod(1).eq(0)).all():
        raise ValueError("neighbor_count는 0 이상의 정수여야 함")
    return X
