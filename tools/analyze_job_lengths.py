import json
import os
import statistics

def load_jobs(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def text_from_job(job):
    parts = []
    # common keys to include
    for key in ('position_title','position_summary','primary_responsibilities','requirements','company_description','position_description','description','job_description'):
        if key in job:
            val = job[key]
            if isinstance(val, list):
                parts.append(' '.join(val))
            elif isinstance(val, str):
                parts.append(val)
    # if none found, join all string values
    if not parts:
        strs = [v for v in job.values() if isinstance(v, str)]
        parts = strs
    return '\n'.join(parts)


def words_count(text):
    return len(text.split())


def analyze(path):
    jobs = load_jobs(path)
    counts = []
    for job in jobs:
        t = text_from_job(job)
        counts.append(words_count(t))

    counts_sorted = sorted(counts)
    total = len(counts)
    mean = statistics.mean(counts)
    median = statistics.median(counts)
    p90 = counts_sorted[int(len(counts_sorted)*0.9)-1] if len(counts_sorted)>0 else 0
    p75 = counts_sorted[int(len(counts_sorted)*0.75)-1] if len(counts_sorted)>0 else 0
    p25 = counts_sorted[int(len(counts_sorted)*0.25)-1] if len(counts_sorted)>0 else 0
    mn = counts_sorted[0] if counts_sorted else 0
    mx = counts_sorted[-1] if counts_sorted else 0

    print(f"Jobs: {total}")
    print(f"Words per job: mean={mean:.1f}, median={median}, min={mn}, 25%={p25}, 75%={p75}, 90%={p90}, max={mx}")

    # recommendations
    print('\nRecommendations:')
    # If most jobs shorter than 1 chunk, suggest larger chunk_size so each JD is one chunk
    if p90 <= 300:
        print("- Most job descriptions are <=300 words. Consider using chunk_size=500-1000 and overlap=0 to keep each JD intact.")
    elif p75 <= 300:
        print("- Many job descriptions are <=300 words but some are longer. Consider chunk_size=400-600, overlap=50 (approx 10-15%).")
    else:
        print("- Job descriptions tend to be long. For fine-grained retrieval use chunk_size=150-300 and overlap=30-60.")

    print("- If you want per-JD retrieval (one vector per job), set chunk_size to a large value (e.g., 1000) or skip chunking and embed the whole JD.")
    print("- For QA/finer retrieval, use smaller chunks (150-300 words) with 15-25% overlap (e.g., overlap=30-50 words).")

    # estimate embeddings
    example_chunk = 300
    stride = example_chunk - 50
    est_chunks_per_job = [(c + stride - 1)//stride for c in counts]
    total_chunks = sum(est_chunks_per_job)
    print(f"\nEstimated chunks if chunk_size={example_chunk} and overlap=50: {total_chunks} (avg per job {statistics.mean(est_chunks_per_job):.2f})")
    dims = 384
    mem_bytes = total_chunks * dims * 4
    print(f"Estimated vector memory (~float32) for {total_chunks} vectors at dim={dims}: {mem_bytes/1024:.1f} KB")


if __name__ == '__main__':
    path = os.path.join('data','job_description','job_descriptions_100.json')
    analyze(path)
