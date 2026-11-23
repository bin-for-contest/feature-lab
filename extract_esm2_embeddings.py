"""팀의 ESM-2 추출기를 경로·batch 크기를 지정해 재사용할 수 있게 정리한 버전."""

import argparse
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd

from features import AA, EMBEDDING, read_table, variants


def get_window_seq(sequence, pos, half=510):
    """전체 서열의 1-based 위치를 window 내부 0-based offset으로 변환함."""
    center = pos - 1
    start = max(0, center - half)
    return sequence[start:center + half + 1], center - start


def reference_sequence(df):
    # 첫 변이를 WT로 되돌려 공통 참조를 복구한 뒤 모든 행을 검증함.
    # 부분 서열이나 다른 isoform이 섞이면 같은 pos가 다른 잔기를 가리킬 수 있음.
    if "mutated_sequence" not in df:
        raise ValueError("mutated_sequence 열이 필요함")
    first = df.iloc[0]
    sequence = first.mutated_sequence
    if not isinstance(sequence, str) or first.pos > len(sequence):
        raise ValueError("첫 변이의 전체 길이 서열 및 위치를 확인할 것")
    i = first.pos - 1
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
    """mutated_sequence 없는 ProteinGym 배포본은 동일 assay의 WT FASTA로 보완함."""
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
            raise ValueError(f"{row.mutant}: FASTA의 잔기와 WT가 일치하지 않음")
        sequences.append(reference[:i] + row.mut + reference[i + 1:])
    if "mutated_sequence" in df and not df.mutated_sequence.eq(sequences).fillna(False).all():
        raise ValueError("입력 mutated_sequence와 FASTA가 일치하지 않음")
    return df.assign(mutated_sequence=sequences)


def extract(df, batch_size, device):
    # 학습 CLI는 이 의존성을 요구하지 않음. 추출할 때만 GPU 모델을 로드함.
    import esm
    import torch
    from tqdm import tqdm

    reference = reference_sequence(df)
    device = torch.device(device if device != "auto" else
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model = model.to(device).eval()
    convert = alphabet.get_batch_converter()

    def encode(windows, offsets, need_probs):
        # <cls>가 앞에 붙으므로 residue offset에 1을 더함. Padding 구간은 선택하지 않음.
        _, _, tokens = convert(windows)
        result = model(tokens.to(device), repr_layers=[33], return_contacts=False)
        rows = torch.arange(len(windows), device=device)
        sites = torch.tensor(offsets, device=device) + 1
        vectors = result["representations"][33][rows, sites].float().cpu().numpy()
        if need_probs:
            logits = result["logits"][rows, sites]
            return vectors, logits.float().log_softmax(-1).cpu().numpy()
        return vectors, None

    positions = sorted(df.pos.unique())
    cache = {}
    n = len(df)
    embeddings = np.empty((n, len(EMBEDDING)), dtype=np.float32)
    llr = np.empty(n, dtype=np.float32)
    cosine = np.empty(n, dtype=np.float32)
    with torch.inference_mode():
        # 한 위치에 여러 치환이 있어도 동일한 WT window는 한 번만 인코딩함.
        for start in tqdm(range(0, len(positions), batch_size), desc="WT positions"):
            batch = positions[start:start + batch_size]
            windows, offsets = zip(*(get_window_seq(reference, p) for p in batch))
            vectors, probs = encode([(str(p), w) for p, w in zip(batch, windows)], offsets, True)
            cache.update({p: (v, lp) for p, v, lp in zip(batch, vectors, probs)})
        for start in tqdm(range(0, n, batch_size), desc="Mutants"):
            batch = df.iloc[start:start + batch_size]
            windows, offsets = zip(*(get_window_seq(r.mutated_sequence, r.pos)
                                     for r in batch.itertuples()))
            vectors, _ = encode(list(zip(batch.mutant, windows)), offsets, False)
            embeddings[start:start + len(batch)] = vectors
            for i, row, mutant in zip(range(start, start + len(batch)), batch.itertuples(), vectors):
                wild, lp = cache[row.pos]
                # 원본과 동일한 unmasked WT-marginal 점수. WT 잔기가 보이는 입력을 사용함.
                llr[i] = lp[alphabet.get_idx(row.mut)] - lp[alphabet.get_idx(row.wt)]
                cosine[i] = np.dot(wild, mutant) / (np.linalg.norm(wild) * np.linalg.norm(mutant) + 1e-8)

    meta = ["mutant", "wt", "pos", "mut", *[k for k in ("DMS_score", "DMS_score_bin") if k in df]]
    return pd.concat([df[meta].assign(esm_llr=llr, esm_cos_sim=cosine),
                      pd.DataFrame(embeddings, columns=EMBEDDING)], axis=1)


def main():
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
    df = variants(read_table(args.input))
    if args.reference_fasta:
        df = with_fasta(df, args.reference_fasta)
    result = extract(df, args.batch_size, args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp = args.output.with_name(f".{uuid4().hex}.tmp")
    try:
        if args.output.suffix.lower() == ".parquet":
            result.to_parquet(temp, index=False)
        else:
            result.to_csv(temp, index=False)
        temp.rename(args.output)
    finally:
        temp.unlink(missing_ok=True)
    print(f"Saved {result.shape} -> {args.output}")


if __name__ == "__main__":
    main()
