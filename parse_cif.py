"""mmCIF 파일을 읽고 _atom_site 블록을 원자별 표로 변환하는 독립 파서.

필요 패키지: python -m pip install biopython pandas
사용 예: python parse_cif.py --input AF-P38398-F1-model_v6.cif
CSV 저장: python parse_cif.py --input model.cif --out atoms.csv

거리/SASA/수소결합 계산이나 모델 입력 연결은 수행하지 않는다.
_atom_site의 한 행은 잔기 하나가 아니라 원자 하나다.
"""

import argparse
from pathlib import Path

import pandas as pd


def read_cif(path):
    """CIF 전체를 {_category.field: 값 목록} 형태의 사전으로 반환한다.

    공백으로 직접 split하지 않고 Biopython을 사용해 loop_, 따옴표 문자열,
    여러 줄 텍스트를 처리한다. _atom_site 외의 메타데이터도 이 사전에 남긴다.
    원본의 '.'(해당 없음), '?'(알 수 없음) 표기는 그대로 보존한다.
    """
    try:
        from Bio.PDB.MMCIF2Dict import MMCIF2Dict
    except ImportError as exc:
        raise ImportError("biopython 설치 필요: python -m pip install biopython") from exc

    with Path(path).open(encoding="utf-8-sig") as handle:
        try:
            return dict(MMCIF2Dict(handle))
        except StopIteration as exc:
            raise ValueError("CIF 파일이 비어 있거나 올바른 mmCIF 형식이 아님") from exc


def atom_site_table(cif):
    """read_cif()의 결과에서 _atom_site 전체 열을 pandas 표로 반환한다.

    접두사 '_atom_site.'만 제거하므로 label_atom_id, label_comp_id, label_seq_id,
    Cartn_x/y/z, B_iso_or_equiv 등 원본에 존재하는 모든 원자 열을 유지한다.
    체인, 모델, ATOM/HETATM, 대체 원자 위치를 임의로 필터링하지 않는다.
    """
    prefix = "_atom_site."
    columns = {key[len(prefix):]: values for key, values in cif.items() if key.startswith(prefix)}
    if not columns:
        raise ValueError("CIF에 _atom_site 블록이 없음")
    if any(not isinstance(values, list) for values in columns.values()):
        raise ValueError("_atom_site의 각 열은 값 목록이어야 함")
    lengths = {len(values) for values in columns.values()}
    if len(lengths) != 1 or not next(iter(lengths)):
        raise ValueError("_atom_site의 열 길이가 다르거나 원자 데이터가 비어 있음")

    atoms = pd.DataFrame(columns)
    missing = [name for name in ("Cartn_x", "Cartn_y", "Cartn_z") if name not in atoms]
    if missing:
        raise ValueError(f"원자 좌표 열이 없음: {missing}")

    # 숫자 열에서만 CIF 결측 표기를 pandas의 결측값으로 바꾼다.
    # 문자열/식별자 열은 원본 표기를 유지해 auth_seq_id 등의 번호 체계를 보존한다.
    for name in ("Cartn_x", "Cartn_y", "Cartn_z", "occupancy", "B_iso_or_equiv"):
        if name in atoms:
            try:
                values = atoms[name].replace({".": pd.NA, "?": pd.NA})
                atoms[name] = pd.to_numeric(values, errors="raise").astype("Float64")
            except (TypeError, ValueError) as exc:
                raise ValueError(f"_atom_site.{name}에 숫자로 해석할 수 없는 값이 있음") from exc

    for name in ("label_seq_id", "pdbx_PDB_model_num"):
        if name in atoms:
            try:
                values = atoms[name].replace({".": pd.NA, "?": pd.NA})
                atoms[name] = pd.to_numeric(values, errors="raise").astype("Int64")
            except (TypeError, ValueError) as exc:
                raise ValueError(f"_atom_site.{name}에 정수로 해석할 수 없는 값이 있음") from exc

    # B_iso_or_equiv는 원래 열 이름을 유지한다. AlphaFold 파일에서는 pLDDT지만
    # 일반 실험 구조 파일에서는 B-factor이므로 이 파서에서 일괄 plddt로 이름을 바꾸지 않는다.
    return atoms


def parse_cif(path):
    """CIF 경로 하나를 받아 원자별 DataFrame을 반환하는 간단한 호출 인터페이스."""
    return atom_site_table(read_cif(path))


def main():
    """파싱한 표를 미리 보고, --out을 지정한 경우에만 새 CSV 파일로 저장한다."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="입력 .cif 또는 .mmcif 파일")
    parser.add_argument("--out", type=Path, help="선택적 원자 표 CSV 출력 경로. 기존 파일은 덮어쓰지 않음")
    args = parser.parse_args()
    if args.out and args.out.suffix.lower() != ".csv":
        parser.error("--out에는 .csv 파일 경로를 지정할 것")
    if args.out and args.out.exists():
        parser.error(f"출력 파일이 이미 있음: {args.out}")

    try:
        atoms = parse_cif(args.input)
        print(f"원자 표 크기: {atoms.shape[0]}행, {atoms.shape[1]}열")
        print(f"열 이름: {atoms.columns.tolist()}")
        print(atoms.head().to_string(index=False))
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            atoms.to_csv(args.out, index=False, mode="x")
            print(f"저장 완료: {args.out}")
    except (OSError, ValueError, ImportError) as exc:
        parser.exit(2, f"오류: {exc}\n")


if __name__ == "__main__":
    main()
