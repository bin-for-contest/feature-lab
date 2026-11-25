"""변이 단백질 서열을 ESM-2 기반 수치 feature 표로 변환하는 CLI.

입력 한 행은 한 단일 치환이며 mutant와 전체 mutated_sequence를 사용한다.
mutated_sequence가 없으면 같은 assay의 WT FASTA로 서열을 생성할 수 있다.
출력은 변이 위치의 1,280차원 Mut 임베딩과 WT/Mut 벡터의 코사인 유사도다.
esm_llr은 계산/저장하지 않는다. ESM-2는 고정된 추출기이며 여기서 학습하지 않는다.
"""

import argparse
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd

from features import AA, EMBEDDING, read_table, variants


def get_window_seq(sequence, pos, half=510):
    """변이 주변 서열과 그 서열 안의 잔기(residue) 위치를 (window, offset)으로 반환한다.

    pos는 전체 단백질의 1-based 위치다. 기본 window는 변이 잔기(residue)와 양옆 최대
    510개씩, 총 최대 1,021개 잔기(residue)다. 서열 양 끝에서는 길이가 짧아진다.
    전체 서열을 stride로 훑어 pooling하지 않고, 변이 위치마다 국소 window를 만든다.
    """
    # 생물학적 잔기(residue) 번호에서 Python 문자열 인덱스로 변환한다.
    center = pos - 1
    start = max(0, center - half)
    # 슬라이스의 끝은 미포함이므로 +1한다. 문자열 끝을 넘으면 Python이 자동으로 자른다.
    # 전체 pos를 그대로 쓰면 window 시작점이 0이 아닌 경우 잘못된 잔기(residue)를 선택하게 된다.
    return sequence[start:center + half + 1], center - start


def reference_sequence(df):
    """모든 mutated_sequence가 같은 WT의 단일 치환인지 확인하고 WT를 반환한다.

    입력은 variants()가 검증한 표다. 첫 행의 변이 잔기(residue)만 WT로 되돌려 참조를 만든 뒤
    각 행의 전체 서열이 그 참조에 정확히 한 번 치환한 결과와 같은지 검사한다.
    위치별 WT 문자만 검사하는 것보다 강한 검증으로, 다른 배경 변이도 거부한다.
    """
    # 첫 변이를 WT로 되돌려 공통 참조를 복구한 뒤 모든 행을 검증함.
    # 부분 서열이나 다른 isoform이 섞이면 같은 pos가 다른 잔기(residue)를 가리킬 수 있음.
    if "mutated_sequence" not in df:
        raise ValueError("mutated_sequence 열이 필요함")
    first = df.iloc[0]
    sequence = first.mutated_sequence
    if not isinstance(sequence, str) or first.pos > len(sequence):
        raise ValueError("첫 변이의 전체 길이 서열 및 위치를 확인할 것")
    i = first.pos - 1
    # 예: C2A의 mutated_sequence가 AADEFG라면 두 번째 A를 C로 되돌려 ACDEFG를 얻는다.
    reference = sequence[:i] + first.wt + sequence[i + 1:]
    if set(reference) - set(AA):
        raise ValueError("참조 서열에 표준 아미노산 이외의 문자가 있음")
    for row in df.itertuples(index=False):
        i = row.pos - 1
        if (i >= len(reference) or reference[i] != row.wt or
                row.mutated_sequence != reference[:i] + row.mut + reference[i + 1:]):
            raise ValueError(f"{row.mutant}: 단일 치환과 전체 서열이 일치하지 않음")
    return reference


def with_fasta(df, path):
    """단일 레코드 WT FASTA에서 각 변이의 전체 mutated_sequence를 만든다.

    FASTA는 여러 줄 서열을 허용하지만 헤더는 하나여야 한다. 입력 표에 이미 서열이
    있으면 덮어쓰기 전에 FASTA에서 생성한 결과와 일치하는지 검사한다.
    단백질 이름만 같은 다른 isoform을 넣지 않도록 각 위치의 WT도 확인한다.
    """
    # UTF-8 BOM과 빈 줄을 처리한 뒤, 헤더를 제외한 서열 줄을 연결한다.
    lines = [line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if not lines or not lines[0].startswith(">") or any(s.startswith(">") for s in lines[1:]):
        raise ValueError("단일 레코드 WT FASTA가 필요함")
    reference = "".join(lines[1:]).upper()
    if not reference or set(reference) - set(AA):
        raise ValueError("WT FASTA 서열을 확인할 것")
    sequences = []
    for row in df.itertuples(index=False):
        i = row.pos - 1
        if i >= len(reference) or reference[i] != row.wt:
            raise ValueError(f"{row.mutant}: FASTA의 잔기(residue)와 WT가 일치하지 않음")
        # 삽입/결실 없이 해당 위치의 한 글자만 바꿔 모든 행의 서열 길이를 유지한다.
        sequences.append(reference[:i] + row.mut + reference[i + 1:])
    if "mutated_sequence" in df and not df.mutated_sequence.eq(sequences).fillna(False).all():
        raise ValueError("입력 mutated_sequence와 FASTA가 일치하지 않음")
    return df.assign(mutated_sequence=sequences)


def extract(df, batch_size, device):
    """검증한 변이 표에서 ESM 임베딩/유사도를 추출해 새 DataFrame을 반환한다.

    batch_size는 한 번에 모델에 넣을 window 수이며, 큰 값은 메모리 사용량을 늘린다.
    device='auto'는 CUDA 사용 가능 시 GPU, 그렇지 않으면 CPU를 선택한다.
    DMS 정답은 계산에 쓰지 않고 존재하는 열만 결과에 그대로 옮긴다.
    """
    # 학습 CLI는 이 의존성을 요구하지 않음. 추출할 때만 GPU 모델을 로드함.
    import esm
    import torch
    from tqdm import tqdm

    # 모델을 로드하기 전에 서열 일관성을 확인해 잘못된 입력에 큰 연산을 시작하지 않는다.
    reference = reference_sequence(df)
    device = torch.device(device if device != "auto" else
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    # 지정된 사전학습 가중치를 로드한다. 로컬 캐시에 없으면 다운로드가 필요할 수 있다.
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    # eval()은 추론 동작으로 전환한다. 아래 inference_mode()와 함께 고정 feature를 추출한다.
    model = model.to(device).eval()
    # 문자열 목록을 특수 토큰과 padding을 포함하는 정수 토큰 행렬로 바꾸는 함수다.
    convert = alphabet.get_batch_converter()

    def encode(windows, offsets):
        """(이름, window) 목록에서 각 변이 위치의 벡터만 골라 (B, 1280)로 반환한다."""
        # <cls>가 앞에 붙으므로 residue offset에 1을 더함. Padding 구간은 선택하지 않음.
        _, _, tokens = convert(windows)
        # tokens: (B, 배치의 최대 서열 길이 + 특수 토큰).
        # representations[33]: (B, 토큰 수, 1280). contact map은 요청하지 않는다.
        result = model(tokens.to(device), repr_layers=[33], return_contacts=False)
        rows = torch.arange(len(windows), device=device)
        sites = torch.tensor(offsets, device=device) + 1
        # [rows, sites]는 각 샘플마다 다른 잔기(residue) 위치를 동시에 선택하는 인덱싱이다.
        # GPU 텐서는 float32 CPU 배열로 옮겨 pandas/NumPy 계산 및 파일 저장에 사용한다.
        return result["representations"][33][rows, sites].float().cpu().numpy()

    positions = sorted(df.pos.unique())
    # cache[pos]는 WT의 그 위치에서 추출한 1,280차원 벡터다.
    # WT 벡터 자체를 전부 feature로 넣지 않고 코사인 유사도 계산에만 사용한다.
    cache = {}
    n = len(df)
    embeddings = np.empty((n, len(EMBEDDING)), dtype=np.float32)
    cosine = np.empty(n, dtype=np.float32)
    # gradient를 저장하지 않아 메모리를 줄인다. 이 추출 과정은 ESM fine-tuning이 아니다.
    with torch.inference_mode():
        # 한 위치에 여러 치환이 있어도 동일한 WT window는 한 번만 인코딩함.
        for start in tqdm(range(0, len(positions), batch_size), desc="WT positions"):
            batch = positions[start:start + batch_size]
            windows, offsets = zip(*(get_window_seq(reference, p) for p in batch))
            vectors = encode([(str(p), w) for p, w in zip(batch, windows)], offsets)
            cache.update(zip(batch, vectors))
        # Mut 임베딩은 위치뿐 아니라 치환 잔기(residue)에도 달라지므로 변이 행마다 계산한다.
        for start in tqdm(range(0, n, batch_size), desc="Mutants"):
            batch = df.iloc[start:start + batch_size]
            windows, offsets = zip(*(get_window_seq(r.mutated_sequence, r.pos)
                                     for r in batch.itertuples()))
            vectors = encode(list(zip(batch.mutant, windows)), offsets)
            # 마지막 배치는 batch_size보다 짧을 수 있으므로 실제 배치 길이를 사용한다.
            embeddings[start:start + len(batch)] = vectors
            for i, row, mutant in zip(range(start, start + len(batch)), batch.itertuples(), vectors):
                wild = cache[row.pos]
                # 방향 유사도 = 내적 / 두 벡터 길이의 곱. 작은 epsilon은 0으로 나누기를 완화한다.
                cosine[i] = np.dot(wild, mutant) / (np.linalg.norm(wild) * np.linalg.norm(mutant) + 1e-8)

    # N행을 유지하고 식별자/선택적 정답 + 코사인 열 + 1,280개 임베딩 열을 연결한다.
    # mutated_sequence는 결과에 저장하지 않는다. 학습은 추출된 수치 feature 표를 사용한다.
    meta = ["mutant", "wt", "pos", "mut", *[k for k in ("DMS_score", "DMS_score_bin") if k in df]]
    return pd.concat([df[meta].assign(esm_cos_sim=cosine),
                      pd.DataFrame(embeddings, columns=EMBEDDING)], axis=1)


def main():
    """CLI 입력을 검증하고, 추출 성공 후 완성된 CSV/Parquet를 지정 경로에 공개한다."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--reference-fasta", type=Path, help="mutated_sequence 생략 시 WT FASTA")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()
    if args.batch_size < 1 or args.output.suffix.lower() not in {".parquet", ".csv"}:
        parser.error("양의 batch-size와 .parquet 또는 .csv 출력을 지정할 것")
    if args.output.exists():
        parser.error("출력 파일이 이미 있음. 새 파일명을 지정할 것")
    # 변이 표기 검증 → 선택적 FASTA 보완 → WT/Mut 서열 검증 및 ESM 추출 순서다.
    df = variants(read_table(args.input))
    if args.reference_fasta:
        df = with_fasta(df, args.reference_fasta)
    result = extract(df, args.batch_size, args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # 저장 중 중단된 파일이 정상 결과처럼 보이지 않게 같은 폴더의 임시 파일에 먼저 쓴다.
    # UUID를 사용해 서로 다른 실행의 임시 파일 이름이 겹치지 않도록 한다.
    temp = args.output.with_name(f".{uuid4().hex}.tmp")
    try:
        if args.output.suffix.lower() == ".parquet":
            result.to_parquet(temp, index=False)
        else:
            result.to_csv(temp, index=False)
        # 쓰기가 끝난 뒤에만 최종 파일명으로 바꾼다.
        temp.rename(args.output)
    finally:
        # 실패 시 남은 임시 파일을 정리한다. rename에 성공해 이미 없어진 경우도 허용한다.
        temp.unlink(missing_ok=True)
    print(f"Saved {result.shape} -> {args.output}")


if __name__ == "__main__":
    main()
